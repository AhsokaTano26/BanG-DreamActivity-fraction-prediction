"""Independent recomputation and checkpoint replay, without test data access."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import shutil
import statistics
import joblib
import numpy as np
import pandas as pd
from analysis.holdout_dataset import ROOT,HoldoutDataset,sha256,write_json
from analysis.event_features import EventFeatures
from analysis.holdout_evaluate import case_metadata,score_predictions
from analysis import quality_irregular_benchmark as b
from analysis import quality_hierarchy as h


def audit(run,output):
    if output.exists():raise FileExistsError(output)
    data=HoldoutDataset();b.verify_run(run,data);config=json.loads((run/'run_manifest.json').read_text())
    if config['python']!=platform.python_version():raise ValueError('Python version changed')
    metadata=EventFeatures.load(data.metadata_directory);folds=json.loads((run/'folds.json').read_text())
    held={e for fold in folds for e in fold['held_events']};metrics={};independent_differences=[];all_predictions={}
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__),output/'audit_source.py')
    for split,filename,metricfile in [('train','training_predictions.csv.gz','training_metrics.json'),
            ('oof','training_oof_predictions.csv','training_oof_metrics.json'),
            ('validation','validation_predictions.csv','validation_metrics.json')]:
        ss=data.inputs('train' if split=='oof' else split);answers=data.labels('train' if split=='oof' else split)
        if split=='oof':ss=[s for s in ss if s['curve'].event in held]
        lookup={s['case_id']:s for s in ss};answers={s['case_id']:answers[s['case_id']] for s in ss}
        cases=case_metadata(ss,metadata);pred=pd.read_csv(run/filename);reported=json.loads((run/metricfile).read_text());scores={}
        for name,group in pred.groupby('method'):
            for row in group.itertuples():
                s=lookup[row.case_id]
                assert row.event==s['curve'].event and row.rank==s['curve'].rank
                assert row.stage==s['stage'] and row.observed_at==s['now'] and row.cutoff==s['cutoff']
                assert row.predicted>=s['score'] and row.current==s['score']
            scores[name]=score_predictions(cases,answers,group.to_dict('records'))
            for scope in scores[name]:
                if scope=='overall':pairs=[(scores[name][scope],reported[name][scope])]
                else:pairs=[(v,reported[name][scope][key]) for key,v in scores[name][scope].items()]
                for first,second in pairs:
                    for key,value in first.items():assert np.isclose(value,second[key],rtol=1e-10,atol=1e-8),(split,name,scope,key)
            events=defaultdict(list)
            for row in group.itertuples():events[lookup[row.case_id]['curve'].event].append(abs(row.predicted/answers[row.case_id]-1)*100)
            independent=statistics.mean(statistics.mean(v) for v in events.values())
            diff=abs(independent-scores[name]['overall']['event_macro_mape_pct']);assert diff<1e-10;independent_differences.append(diff)
        metrics[split]=scores;all_predictions[split]=pred
    samples=data.inputs('train');validation=data.inputs('validation');pool=data.train_curves()
    train_base,train_quality=b.frames(samples,pool,metadata);val_base,val_quality=b.frames(validation,pool,metadata)
    replay={};train_frame=all_predictions['train'];val_frame=all_predictions['validation']
    def compare(name,split,ss,pred):
        frame=all_predictions[split];expected=frame[frame.method==name].set_index('case_id').loc[[s['case_id'] for s in ss],'predicted'].to_numpy()
        diff=float(np.max(abs(pred/expected-1)));assert diff<1e-10,(split,name,diff);replay[f'{split}:{name}']=diff
    for name in h.RECIPES:
        model=joblib.load(run/'models'/f'final_{name}.joblib')
        compare(name,'train',samples,h.predict(model,samples,train_base,train_quality))
        compare(name,'validation',validation,h.predict(model,validation,val_base,val_quality))
    prep=joblib.load(run/'models/neural_preprocessor.joblib');b.torch.set_num_threads(4)
    train=b.arrays(samples,train_base,prep,config['neural_settings']['nodes']);val=b.arrays(validation,val_base,prep,config['neural_settings']['nodes'])
    for name in config['neural_recipes']:
        for seed in config['seeds']:
            model=b.load_model(run/'models'/f'{name}_{seed}.pt',config['run_id'])
            compare(f'{name}_seed{seed}','train',samples,b.decode(samples,b.infer(model,train),'remaining'))
            compare(f'{name}_seed{seed}','validation',validation,b.decode(validation,b.infer(model,val),'remaining'))
        for split,ss in [('train',samples),('validation',validation)]:
            frame=all_predictions[split].pivot(index='case_id',columns='method',values='predicted')
            expected=frame[[f'{name}_seed{s}' for s in config['seeds']]].mean(axis=1)
            assert np.allclose(expected,frame[f'{name}_ensemble3'],rtol=1e-12,atol=1e-7)
    val_wide=val_frame.pivot(index='case_id',columns='method',values='predicted')
    ref,identity=b.reference_values(Path(config['reference']['directory']),validation,data)
    assert identity==config['reference'],'Reference identity changed'
    np.testing.assert_allclose(ref,val_wide.loc[[s['case_id'] for s in validation],'current_reference'],rtol=1e-12,atol=1e-7)
    for name in h.RECIPES+[f'{n}_ensemble3' for n in config['neural_recipes']]:
        assert np.allclose(val_wide[f'blend25_{name}'],.25*val_wide[name]+.75*val_wide.current_reference,rtol=1e-12,atol=1e-7)
    old=pd.read_csv(ROOT/'analysis/deadline_research/prototypes_v2/validation_predictions.csv')
    old=old[old.method=='remaining_rank'].set_index('case_id').predicted
    assert np.array_equal(old.loc[val_wide.index],val_wide.remaining_rank)
    table=[]
    for name in metrics['validation']:
        record={'method':name}
        for split in ['train','oof','validation']:
            record[split+'_event_mape']=metrics[split].get(name,{}).get('overall',{}).get('event_macro_mape_pct',np.nan)
        table.append(record)
    pd.DataFrame(table).to_csv(output/'comparison.csv',index=False)
    fold_errors=[];oof=all_predictions['oof'];labels=data.labels('train');case_map={s['case_id']:s for s in samples}
    for (fold,name),group in oof.groupby(['fold','method']):
        group=group.copy();group['actual']=[labels[c] for c in group.case_id]
        group['bias_pct']=(group.predicted/group.actual-1)*100
        event_bias=group.groupby('event').bias_pct.mean()
        fold_errors.append({'fold':int(fold),'method':name,'activity_bias_pct':float(event_bias.mean()),
            'activity_mape_pct':float(group.assign(ape=group.bias_pct.abs()).groupby('event').ape.mean().mean())})
    pd.DataFrame(fold_errors).to_csv(output/'fold_errors.csv',index=False)
    quality_summary={}
    for split,ss,frame in [('train',samples,train_quality),('validation',validation,val_quality)]:
        quality_summary[split]={'cases':len(ss),'events':len({s['curve'].event for s in ss}),
            'reliability_median':float(frame.quality_shape_reliability.median()),
            'reliability_quartiles':frame.quality_shape_reliability.quantile([.25,.75]).tolist(),
            'zero_reliability_cases':int((frame.quality_shape_reliability==0).sum())}
    write_json(output/'quality_summary.json',quality_summary)
    expert_model=joblib.load(run/'models/final_quality_type_tree.joblib')
    write_json(output/'expert_shares.json',expert_model['shares'])
    # Prespecified paired ablations; these estimates do not select parameters.
    pairwise={}
    for split in ['oof','validation']:
        ss=[s for s in samples if s['curve'].event in held] if split=='oof' else validation
        answers=data.labels('train' if split=='oof' else 'validation')
        frame=all_predictions[split].pivot(index='case_id',columns='method',values='predicted')
        values={n:frame.loc[[s['case_id'] for s in ss],n].to_numpy() for n in h.RECIPES}
        pairwise[split]={name:b.bootstrap(ss,answers,values,base)[name] for base,name in
            [('additive_spline','hierarchical_spline'),('hierarchical_spline','quality_spline'),('quality_shared_tree','quality_type_tree')]}
    write_json(output/'paired_ablations.json',pairwise)
    write_json(output/'verified_metrics.json',metrics)
    write_json(output/'verification.json',{'dataset_id':config['dataset_id'],'run_id':config['run_id'],
        'source_and_artifact_seals_valid':True,'baseline_exact_replay':True,'no_test_inputs_or_labels_requested':True,
        'independent_metric_max_difference':max(independent_differences),'checkpoint_replay_max_relative_differences':replay,
        'python_version_verified':platform.python_version(),
        'frozen_database_snapshot_sha256':data.manifest['artifacts']['snapshot.sqlite3'],
        'unused_live_database_sha256':sha256(ROOT/'data/db.sqlite3'),
        'frozen_train_curves_sha256':data.manifest['artifacts']['train_curves.jsonl'],
        'audit_source_sha256':sha256(Path(__file__))})
    write_json(output/'audit_manifest.json',{'artifacts':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}})
    print('Verified all cases, metrics, models, seed means, mixtures and baseline; no test data read.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();audit(args.run,args.output)
