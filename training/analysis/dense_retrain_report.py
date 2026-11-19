"""Collect independently scored dense retraining runs without model selection on test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.holdout_dataset import ROOT, sha256, write_json

V3 = ROOT/'analysis/rolling_results/v3_dense_full_retrain_20261115'
CORE_TRAIN = ROOT/'analysis/dense_retrain_records/20261115/core_training'
IDS = {'V1':'80e354c3a5340df6724a40ef1df490f92c08007311d55849342de509eaf5ee50',
       'V2 two-rank':'d057beca6597dd28b59fde7da08f665ae74af41d96f940da04268c881864a215',
       'V2 19-rank':'8ffd781fc9f4ebcad9ea691839777458e5e4c7cbeade58978dca15bce2440ad8',
       'V3':'85f5924e0b90deb0e1a30f3aa27d2237640cca4d968d96612dda8fee1d9c72ec'}
EXPECTED_RUNS={f'analysis/holdout_results/dense_{family}_20261115' for family in
              ['core','relationship','structured_mps','deadline','quality','quality_extended','innovation']}
EXPECTED_RUNS|={f'analysis/temporal_results/dense_{family}_20261115' for family in
               ['base','paired','research','innovation','next','19rank']}
EXPECTED_RUNS.add(str(V3.relative_to(ROOT)))


def require_inventory(actual, expected):
    if set(actual)!=set(expected):raise ValueError('Run inventory differs from registration')


def require_suite_choices(suite):
    rows=suite['runner_selections']
    expected=[min([r for r in rows if r['protocol']==protocol],key=lambda r:(r['validation_mape_pct'],r['run'])) for protocol in IDS]
    if suite['selected']!=expected:raise ValueError('Suite choices differ from registered validation rule')


def require_evaluation_binding(receipt, evaluation, seal, blind_hash):
    identity=receipt.get('identity',{})
    bound=identity.get('blind_manifest_sha256',identity.get('blind_test_manifest_sha256'))
    if (bound!=blind_hash or evaluation.get('evaluation_id')!=receipt.get('evaluation_id')
            or evaluation.get('dataset_id')!=seal['dataset_id']):
        raise ValueError('Independent evaluation identity differs from the run')


def require_training_binding(replay, seal, blind_hash, run):
    bound=replay.get('sources',{}).get(str(run.resolve()),replay)
    if (replay.get('dataset_id')!=seal['dataset_id'] or bound.get('run_id')!=seal['run_id']
            or bound.get('blind_seal_sha256')!=blind_hash):
        raise ValueError('training replay identity differs from the run')


def require_selection_binding(run, sealfile, selected, registered):
    if (sha256(sealfile)!=registered['blind_seal_sha256']
            or sha256(run/'selection.json')!=registered['selection_sha256']
            or selected!=registered['selected_method']):
        raise ValueError('Run differs from frozen suite selection')


def check_receipt(path):
    doc=json.loads(path.read_text())
    for name,digest in doc['artifacts'].items():
        if sha256(path.parent/name)!=digest:
            raise ValueError('Artifact changed: '+str(path.parent/name))
    return doc


def collect(output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    suite_path=ROOT/'analysis/dense_retrain_records/20261115/pretest_selection.json'
    suite=json.loads(suite_path.read_text())
    require_suite_choices(suite)
    expected_runs={item['run']:item for item in suite['runner_selections']}
    specs=[]
    for run in sorted((ROOT/'analysis/holdout_results').glob('dense_*_20261115')):
        if (run/'blind_test_manifest.json').exists(): specs.append(('V1',run))
    for run in sorted((ROOT/'analysis/temporal_results').glob('dense_*_20261115')):
        if (run/'blind_manifest.json').exists():
            specs.append(('V2 19-rank' if run.name=='dense_19rank_20261115' else 'V2 two-rank',run))
    specs.append(('V3',V3))
    require_inventory(expected_runs,EXPECTED_RUNS)
    require_inventory([str(run.relative_to(ROOT)) for _,run in specs],expected_runs)
    rows=[];rankrows=[];stagerows=[];selectedrows=[];provenance=[]
    for protocol,run in specs:
        sealfile=run/('blind_test_manifest.json' if protocol=='V1' else 'blind_manifest.json')
        seal=check_receipt(sealfile)
        registered=expected_runs[str(run.relative_to(ROOT))]
        if sha256(sealfile)!=registered['blind_seal_sha256']:
            raise ValueError('Suite blind seal differs from pretest selection')
        if seal['dataset_id']!=IDS[protocol]:raise ValueError('Wrong dense cohort')
        receipt=check_receipt(run/'evaluation_manifest.json')
        evaluation=json.loads((run/'evaluation.json').read_text())
        require_evaluation_binding(receipt,evaluation,seal,sha256(sealfile))
        selection=json.loads((run/'selection.json').read_text())
        selected=selection.get('selected_recipe',selection.get('selected_on_validation'))
        if selected!=registered['selected_method'] or sha256(run/'selection.json')!=registered['selection_sha256']:
            raise ValueError('Selected method changed after suite lock')
        score_splits=evaluation.get('splits',{})
        if protocol=='V1':
            score_splits={'test':evaluation['test']}
            if run.name=='dense_core_20261115':
                check_receipt(CORE_TRAIN/'manifest.json')
                require_training_binding(json.loads((CORE_TRAIN/'replay.json').read_text()),seal,sha256(sealfile),run)
                score_splits.update(json.loads((CORE_TRAIN/'metrics.json').read_text())['splits'])
            else:
                from analysis.current_benchmark import load_dataset
                from analysis.holdout_evaluate import case_metadata, score_predictions
                from analysis.event_features import EventFeatures
                data=load_dataset('v1',seal['dataset_directory'])
                meta=EventFeatures.load(data.metadata_directory)
                for split in ['train','validation']:
                    frame=pd.read_csv(run/f'{split}_predictions.csv')
                    samples=data.inputs(split)
                    cases=case_metadata(samples,meta);labels=data.labels(split)
                    score_splits[split]={name:score_predictions(cases,labels,g.to_dict('records')) for name,g in frame.groupby('method')}
                training_audit=ROOT/'analysis/dense_retrain_records/20261115/v1_training'/run.name
                check_receipt(training_audit/'manifest.json')
                require_training_binding(json.loads((training_audit/'replay.json').read_text()),seal,sha256(sealfile),run)
                score_splits['train']=json.loads((training_audit/'metrics.json').read_text())
        for split,methods in score_splits.items():
            for method,score in methods.items():
                item={'protocol':protocol,'run':run.name,'method':method,'split':split,'selected_within_run':method==selected,**score['overall']}
                rows.append(item)
                for rank,metric in score.get('by_rank',{}).items():rankrows.append(item|metric|{'rank':int(rank)})
                for stage,metric in score.get('by_stage',{}).items():stagerows.append(item|metric|{'stage':stage})
        record={'protocol':protocol,'run':run.name,'selected_method':selected}
        for split in ['train','validation','test']:
            metric=score_splits.get(split,{}).get(selected,{}).get('overall',{})
            record[split+'_mape_pct']=metric.get('event_macro_mape_pct')
        for rank in [1000,2000]:
            record[f'test_T{rank}_mape_pct']=score_splits['test'][selected].get('by_rank',{}).get(str(rank),{}).get('event_macro_mape_pct')
        if not np.isclose(record['validation_mape_pct'],registered['validation_mape_pct'],rtol=0,atol=1e-9):
            raise ValueError('Validation score differs from frozen suite selection')
        selectedrows.append(record)
        provenance.append({'protocol':protocol,'run':str(run.relative_to(ROOT)),'run_id':seal['run_id'],
            'dataset_id':seal['dataset_id'],'evaluation_id':receipt['evaluation_id'],'methods':len(score_splits['test']),
            'blind_seal_sha256':sha256(sealfile),'evaluation_manifest_sha256':sha256(run/'evaluation_manifest.json')})
    selected_frame=pd.DataFrame(selectedrows)
    winners={(r['protocol'],Path(r['run']).name,r['selected_method']) for r in suite['selected']}
    overview=selected_frame.loc[[(r.protocol,r.run,r.selected_method) in winners for r in selected_frame.itertuples()]]
    if len(overview)!=len(IDS):raise ValueError('Overview does not match frozen suite choices')
    output.mkdir(parents=True)
    for name,frame in [('all_metrics',pd.DataFrame(rows)),('by_rank',pd.DataFrame(rankrows)),('by_stage',pd.DataFrame(stagerows)),('selected_by_run',selected_frame),('overview',overview)]:
        frame.to_csv(output/f'{name}.csv',index=False)
    wide=pd.DataFrame(rows).pivot(index=['protocol','run','method','selected_within_run'],columns='split',values='event_macro_mape_pct').reset_index()
    wide=wide.rename(columns={s:s+'_mape_pct' for s in ['train','validation','test']})
    wide.sort_values(['protocol','run','validation_mape_pct','method']).to_csv(output/'all_models.csv',index=False)
    write_json(output/'provenance.json',provenance)
    write_json(output/'pretest_selection.json',suite)
    (output/'source.py').write_text(Path(__file__).read_text())
    print(overview.to_string(index=False))
    return overview

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    collect(p.parse_args().output)
