"""Frozen-development experiment: type pooling, history quality, observed-time networks.

This runner never requests test inputs or labels. Old experiments are immutable.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from analysis.holdout_dataset import ROOT,HoldoutDataset,sha256,write_json
from analysis.event_features import EventFeatures
from analysis.relationship_benchmark import prep_for,weights,target_values,decode,rows,verify_software
from analysis.relationship_features import relationship_frame,select_features
from analysis.deadline_features import history_features,SETTINGS
from analysis.deadline_benchmark import reference_values,macro,TREE_PARAMETERS
from analysis.holdout_evaluate import case_metadata,score_predictions,paired_event_errors
from analysis import quality_hierarchy as hierarchy
from analysis.irregular_prefix import observed_path
import torch
from torch.nn import functional as F
from analysis.irregular_neural import IrregularForecaster

FOLD_LEFT=[98,123,148,173]
FOLD_SIZE=25
SEEDS=[42,123,2026]
NEURAL=['time_gru','neural_cde']
NEURAL_SETTINGS={'width':32,'nodes':32,'epochs':80,'minimum_epochs':30,'patience':20,
                 'batch_size':256,'lr':.001,'weight_decay':.005,'dropout':.2,'gradient_clip':1}
SOURCES=['training/analysis/quality_irregular_benchmark.py','training/analysis/quality_hierarchy.py',
         'training/analysis/irregular_prefix.py','training/analysis/irregular_neural.py','training/analysis/deadline_features.py',
         'training/analysis/deadline_benchmark.py','training/analysis/deadline_diagnostics.py','training/analysis/relationship_features.py',
         'training/analysis/relationship_statistics.py','training/analysis/relationship_benchmark.py',
         'training/analysis/holdout_dataset.py','training/analysis/holdout_evaluate.py','training/analysis/backtest.py',
         'training/analysis/event_features.py','training/legacy/forecasting.py']


def frames(samples,pool,metadata):
    base=select_features(relationship_frame(samples,metadata),'invariant').reset_index(drop=True)
    quality=pd.concat([base,hierarchy.quality_features(history_features(samples,pool))],axis=1)
    return base,quality


def arrays(samples,frame,prep,nodes):
    paths,masks=zip(*(observed_path(s,nodes) for s in samples))
    return {'path':torch.as_tensor(np.stack(paths)), 'mask':torch.as_tensor(np.stack(masks)),
            'tab':torch.as_tensor(np.asarray(prep.transform(frame),np.float32)),
            'rank':torch.tensor([0 if s['curve'].rank==1000 else 1 for s in samples])}


def forward(model,values):return model(values['path'],values['mask'],values['tab'],values['rank'])


def infer(model,values):
    model.eval();result=[]
    with torch.no_grad():
        for begin in range(0,len(values['tab']),256):
            result.append(forward(model,{k:v[begin:begin+256] for k,v in values.items()}).numpy())
    output=np.concatenate(result).astype(float)
    if not np.isfinite(output).all():raise ValueError('Nonfinite neural output')
    return output


def load_model(path,run_id):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['run_id']!=run_id:raise ValueError('Model identity differs')
    model=IrregularForecaster(saved['architecture'],saved['features'],saved['width'])
    model.load_state_dict(saved['state']);return model.eval()


def fit_neural(name,seed,config,train,val,samples,validation,labels,val_labels,run):
    settings=config['neural_settings'];torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed)
    model=IrregularForecaster(name,train['tab'].shape[1],settings['width'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    y=torch.tensor(target_values(samples,labels,'remaining'),dtype=torch.float32)
    w=torch.tensor(weights(samples),dtype=torch.float32)
    remaining=torch.tensor([(1-s['progress'])/s['progress'] for s in samples],dtype=torch.float32)
    actual=torch.tensor([labels[s['case_id']]/s['score'] for s in samples],dtype=torch.float32)
    best=float('inf');best_epoch=0;state=None;trace=[];started=time.perf_counter()
    for epoch in range(1,settings['epochs']+1):
        model.train();order=torch.randperm(len(samples),generator=rng);total=0.
        for begin in range(0,len(samples),settings['batch_size']):
            indices=order[begin:begin+settings['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            out=forward(model,{k:v[indices] for k,v in train.items()})
            relative=(1+remaining[indices]*out.clamp(-6,4).exp())/actual[indices]-1
            loss=.8*torch.sqrt(relative.square()+1e-4)+.2*F.smooth_l1_loss(out,y[indices],beta=.2,reduction='none')
            loss=(loss*w[indices]).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']);optimizer.step()
            total+=loss.item()*len(indices)
        score=macro(validation,val_labels,decode(validation,infer(model,val),'remaining'))
        trace.append({'epoch':epoch,'training_loss':total/len(samples),'validation_mape':score})
        if score<best:
            best=score;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%10==0:print(json.dumps({'method':name,'seed':seed,'epoch':epoch,'best_epoch':best_epoch,'validation_mape':best}),flush=True)
        if epoch>=settings['minimum_epochs'] and epoch-best_epoch>=settings['patience']:break
    path=run/'models'/f'{name}_{seed}.pt'
    torch.save({'run_id':config['run_id'],'architecture':name,'features':train['tab'].shape[1],
                'width':settings['width'],'seed':seed,'state':state},path)
    model.load_state_dict(state);before=infer(model,val)
    reloaded=load_model(path,config['run_id']);after=infer(reloaded,val)
    if not np.array_equal(before,after):raise ValueError('Checkpoint replay differs')
    validation_prediction=decode(validation,after,'remaining')
    training_prediction=decode(samples,infer(reloaded,train),'remaining')
    record={'method':name,'seed':seed,'epochs':epoch,'best_epoch':best_epoch,'validation_mape':best,
            'training_mape':macro(samples,labels,training_prediction),'trace':trace,'seconds':time.perf_counter()-started,
            'model_sha256':sha256(path),'run_id':config['run_id'],'solver_sensitivity':{}}
    if name=='neural_cde':
        for steps in [2,4]:
            reloaded.substeps=steps;p=decode(validation,infer(reloaded,val),'remaining')
            record['solver_sensitivity'][str(steps)]={'mean_absolute_relative_change_pct':float(np.mean(abs(p/validation_prediction-1))*100),
                'max_absolute_relative_change_pct':float(np.max(abs(p/validation_prediction-1))*100),
                'validation_mape':macro(validation,val_labels,p)}
    write_json(run/'models'/f'{name}_{seed}.json',record)
    print('FIT '+json.dumps({k:v for k,v in record.items() if k!='trace'}),flush=True)
    return training_prediction,validation_prediction


def metric_report(samples,labels,values,metadata):
    cases=case_metadata(samples,metadata);answers={s['case_id']:labels[s['case_id']] for s in samples}
    return {name:score_predictions(cases,answers,rows(samples,name,pred)) for name,pred in values.items()}


def bootstrap(samples,labels,values,reference):
    cases=case_metadata(samples);answers={s['case_id']:labels[s['case_id']] for s in samples}
    base=paired_event_errors(cases,answers,rows(samples,reference,values[reference]));rng=np.random.default_rng(42)
    indices=rng.integers(0,len(base),(5000,len(base)));result={}
    for name,pred in values.items():
        errors=paired_event_errors(cases,answers,rows(samples,name,pred)).reindex(base.index)
        delta=(errors-base).to_numpy();ci=np.quantile(delta[indices].mean(axis=1),[.025,.975])
        result[name]={'delta_mape_pp':float(delta.mean()),'ci95_pp':ci.tolist(),'events_improved':int((delta<0).sum()),'events':len(base)}
    return result


def develop(run,data,reference):
    if run.exists():raise FileExistsError(run)
    torch.set_num_threads(4)
    samples=data.inputs('train');validation=data.inputs('validation');labels=data.labels('train');val_labels=data.labels('validation')
    ref,identity=reference_values(reference,validation,data)
    config={'dataset_id':data.manifest['dataset_id'],'dataset_directory':str(data.root.resolve()),
        'statistical_recipes':hierarchy.RECIPES,'neural_recipes':NEURAL,'seeds':SEEDS,'neural_settings':NEURAL_SETTINGS,
        'tree_parameters':TREE_PARAMETERS,'history_settings':SETTINGS,'fold_left':FOLD_LEFT,'fold_size':FOLD_SIZE,
        'hierarchy':{'ridge_alpha':20,'type_penalty_factor':4,'type_rank_penalty_factor':16,'expert_shrinkage_events':20},
        'quality':'n_eff/(n_eff+10)/(1+span_h/2); unknown shape deviation zero; no activity reweighting',
        'neural_solver':'piecewise-linear controls; Heun one step per retained interval; 2/4 steps sensitivity only',
        'loss':'.8*sqrt(relative_final_error^2+1e-4)+.2*Huber(log_remaining_rate,beta=.2)',
        'selection':'fixed validation only; all statistical / neural seed means and 25% candidate + 75% frozen reference; never single seed',
        'weights':'global equal activity total across ranks, no subgroup renormalization',
        'reference':identity,'scope':'development only; no test inputs, labels, predictions or scores',
        'previous_test_exposure':True,'metadata_limitation':'retrospective snapshot without archived point-in-time provenance',
        'python':platform.python_version(),'software':{n:importlib.metadata.version(n) for n in
            ['numpy','pandas','scipy','scikit-learn','catboost','lightgbm','joblib','torch']},
        'source_hashes':{n:sha256(ROOT/n) for n in SOURCES},
        'plan_sha256':sha256(ROOT/'docs/superpowers/plans/2026-11-11-quality-irregular.md')}
    config['run_id']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    run.mkdir(parents=True);(run/'models').mkdir();(run/'sources').mkdir();write_json(run/'run_manifest.json',config)
    for name in SOURCES:shutil.copyfile(ROOT/name,run/'sources'/Path(name).name)
    shutil.copyfile(ROOT/'docs/superpowers/plans/2026-11-11-quality-irregular.md',run/'sources/experiment_plan.md')
    metadata=EventFeatures.load(data.metadata_directory);history=data.train_curves();ids=data.manifest['splits']['train']['event_ids']
    oof_rows=[];held_samples=[];oof={name:[] for name in hierarchy.RECIPES};folds=[]
    for fold,left in enumerate(FOLD_LEFT):
        fit_ids=set(ids[:left]);held_ids=set(ids[left:left+FOLD_SIZE])
        if len(held_ids)!=FOLD_SIZE or fit_ids&held_ids:raise ValueError('Invalid fold')
        fit_samples=[s for s in samples if s['curve'].event in fit_ids];held=[s for s in samples if s['curve'].event in held_ids]
        if max(s['curve'].end for s in fit_samples)>=min(s['curve'].start for s in held):raise ValueError('Fold temporal overlap')
        pool=[c for c in history if c.event in fit_ids]
        fit_base,fit_quality=frames(fit_samples,pool,metadata);held_base,held_quality=frames(held,pool,metadata)
        scores={}
        for name in hierarchy.RECIPES:
            bundle=hierarchy.fit(fit_samples,labels,fit_base,fit_quality,name)
            path=run/'models'/f'fold{fold}_{name}.joblib';joblib.dump(bundle,path)
            pred=hierarchy.predict(joblib.load(path),held,held_base,held_quality);oof[name].append(pred)
            oof_rows.extend(row|{'fold':fold} for row in rows(held,name,pred));scores[name]=macro(held,labels,pred)
        held_samples.extend(held);folds.append({'fold':fold,'fit_events':ids[:left],'held_events':ids[left:left+FOLD_SIZE],'scores':scores})
        print('FOLD '+json.dumps(folds[-1]),flush=True)
    write_json(run/'folds.json',folds);pd.DataFrame(oof_rows).to_csv(run/'training_oof_predictions.csv',index=False)
    oof_values={name:np.concatenate(parts) for name,parts in oof.items()}
    write_json(run/'training_oof_metrics.json',metric_report(held_samples,labels,oof_values,metadata))
    write_json(run/'training_oof_bootstrap.json',bootstrap(held_samples,labels,oof_values,'remaining_rank'))
    train_base,train_quality=frames(samples,history,metadata);val_base,val_quality=frames(validation,history,metadata)
    train_values={};val_values={'current_reference':ref}
    for name in hierarchy.RECIPES:
        bundle=hierarchy.fit(samples,labels,train_base,train_quality,name)
        path=run/'models'/f'final_{name}.joblib';joblib.dump(bundle,path);bundle=joblib.load(path)
        train_values[name]=hierarchy.predict(bundle,samples,train_base,train_quality)
        val_values[name]=hierarchy.predict(bundle,validation,val_base,val_quality)
    prep=prep_for(train_base);prep.fit(train_base);joblib.dump(prep,run/'models/neural_preprocessor.joblib')
    train=arrays(samples,train_base,prep,NEURAL_SETTINGS['nodes']);val=arrays(validation,val_base,prep,NEURAL_SETTINGS['nodes'])
    for name in NEURAL:
        for seed in SEEDS:
            label=f'{name}_seed{seed}'
            train_values[label],val_values[label]=fit_neural(name,seed,config,train,val,samples,validation,labels,val_labels,run)
        for values in [train_values,val_values]:values[f'{name}_ensemble3']=np.mean([values[f'{name}_seed{s}'] for s in SEEDS],axis=0)
    eligible=list(hierarchy.RECIPES)+[f'{n}_ensemble3' for n in NEURAL]
    for name in eligible:val_values[f'blend25_{name}']=.25*val_values[name]+.75*ref
    for filename,values,ss in [('training_predictions.csv.gz',train_values,samples),('validation_predictions.csv',val_values,validation)]:
        pd.DataFrame([r for name,pred in values.items() for r in rows(ss,name,pred)]).to_csv(run/filename,index=False)
    train_metrics=metric_report(samples,labels,train_values,metadata);val_metrics=metric_report(validation,val_labels,val_values,metadata)
    write_json(run/'training_metrics.json',train_metrics);write_json(run/'validation_metrics.json',val_metrics)
    write_json(run/'validation_bootstrap.json',bootstrap(validation,val_labels,val_values,'current_reference'))
    eligible+=['current_reference']+[f'blend25_{n}' for n in eligible]
    selected=min(eligible,key=lambda n:(val_metrics[n]['overall']['event_macro_mape_pct'],n))
    selection={'run_id':config['run_id'],'dataset_id':config['dataset_id'],'selected_on_validation':selected,
               'validation_scores':{n:v['overall']['event_macro_mape_pct'] for n,v in val_metrics.items()},'test_evaluated':False}
    write_json(run/'selection.json',selection)
    write_json(run/'development_seal.json',{'run_id':config['run_id'],'artifacts':{
        str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file()}})
    verify_run(run,data);print('VALIDATION '+json.dumps(selection),flush=True)


def verify_run(run,data):
    config=json.loads((run/'run_manifest.json').read_text());seal=json.loads((run/'development_seal.json').read_text())
    if config['dataset_id']!=data.manifest['dataset_id']:raise ValueError('Dataset changed')
    raw={k:v for k,v in config.items() if k!='run_id'}
    if hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()!=config['run_id']:raise ValueError('Run ID differs')
    if seal['run_id']!=config['run_id']:raise ValueError('Seal identity differs')
    for name,digest in seal['artifacts'].items():
        if sha256(run/name)!=digest:raise ValueError('Artifact changed: '+name)
    for name,digest in config['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Source changed: '+name)
    verify_software(config['software'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=ROOT/'analysis/holdout_results/structured_neural_v1')
    parser.add_argument('--verify',action='store_true');args=parser.parse_args();data=HoldoutDataset()
    if args.verify:verify_run(args.run,data)
    else:develop(args.run,data,args.reference)
