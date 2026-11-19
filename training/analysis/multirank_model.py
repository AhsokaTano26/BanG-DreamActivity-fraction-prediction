"""Shared rank-aware baselines; no test labels, SQLite, or full evaluation curves."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from analysis.backtest import ROOT
from analysis.event_features import EventFeatures
from analysis.holdout_dataset import sha256,write_json
from analysis.holdout_evaluate import score_predictions
from analysis.multirank_dataset import MultirankDataset,DEFAULT,identity

SOURCES=['training/analysis/multirank_model.py','training/analysis/multirank_dataset.py','training/analysis/backtest.py',
    'training/analysis/event_features.py','training/analysis/holdout_dataset.py','training/analysis/holdout_evaluate.py']
METHODS=['linear','catboost_rank','catboost_context']

def feature_row(sample,metadata=None):
    row={k:v for k,v in sample['features'].items() if k!='rank'}
    row['log_rank']=float(np.log(sample['curve'].rank))
    # All interpolation coordinates are at or before the last observed time.
    times,scores=sample['times'],sample['scores'];start=sample['curve'].start
    for i,fraction in enumerate(np.linspace(0,1,33)):
        t=start+fraction*(sample['now']-start)
        row[f'prefix_{i:02}']=float(np.interp(t,times,scores)/sample['score']) if t>=times[0] else None
    if metadata is not None:row.update(metadata.static(sample['curve'],sample['now']))
    return {k:(None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in row.items()}

def event_weights(events):
    counts=Counter(events);w=np.array([1/counts[e] for e in events],dtype=float)
    return w/w.mean()

def metadata_for(data):
    metadata=EventFeatures.load(data.metadata_directory)
    train_ids=data.manifest['splits']['train']['event_ids']
    metadata.character_ids=sorted({c for e in train_ids for c in metadata.parties[e]['characters']})
    metadata.band_ids=sorted({b for e in train_ids for b in metadata.parties[e]['bands']})
    return metadata

def prepare(samples,metadata):
    rows=[];base_rows=[];cases=[];current=[];progress=[]
    for s in samples:
        base=feature_row(s);context=feature_row(s,metadata)
        rows.append(context);base_rows.append(base);current.append(s['score']);progress.append(s['progress'])
        cases.append({'case_id':s['case_id'],'event':s['curve'].event,'rank':s['curve'].rank,
            'stage':s['stage'],'event_type':s['curve'].kind,'band_group':metadata.band_only(s['curve'])['band_group'],
            'cutoff':s['cutoff'],'observed_at':s['now'],'current':s['score']})
    return {'context':pd.DataFrame(rows),'rank':pd.DataFrame(base_rows),'cases':cases,
        'current':np.array(current),'progress':np.array(progress)}

def make_preprocessor(frame):
    cats=[k for k in frame if not pd.api.types.is_numeric_dtype(frame[k]) and frame[k].notna().any()]
    nums=[k for k in frame if k not in cats]
    return ColumnTransformer([('num',SimpleImputer(strategy='median',keep_empty_features=True),nums),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])

def model_prediction(bundle,arrays):
    pred=bundle['model'].predict(bundle['preprocessor'].transform(arrays[bundle['features']]))
    return arrays['current']*np.exp(np.clip(-np.log(arrays['progress'])+pred,0,4))

def records(arrays,name,pred):
    if len(pred)!=len(arrays['cases']) or not np.isfinite(pred).all():raise ValueError('Invalid predictions')
    return [c|{'method':name,'predicted':float(p)} for c,p in zip(arrays['cases'],pred)]

def verify_run(run):
    run=Path(run);seal=json.loads((run/'blind_test_manifest.json').read_text())
    for name,digest in seal['artifacts'].items():
        if sha256(run/name)!=digest:raise ValueError('Sealed artifact changed: '+name)
    return seal

def run_experiment(data,run,iterations=600):
    run=Path(run)
    if run.exists():raise FileExistsError(run)
    config={'dataset_directory':str(data.root.resolve()),'dataset_id':data.manifest['dataset_id'],
        'base_dataset_id':data.manifest['base_dataset_id'],'methods':METHODS,'iterations':iterations,
        'tree_depth':5,'learning_rate':.035,'l2_leaf_reg':12,'seed':42,
        'weights':'equal total weight per activity','selection':'minimum validation event-macro MAPE; no refit',
        'scope':'supplementary 19-rank benchmark, not replacement/improvement claim for holdout_v1',
        'python':platform.python_version(),'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','catboost','joblib']},
        'source_hashes':{n:sha256(ROOT/n) for n in SOURCES}}
    config['run_id']=identity(config);run.mkdir(parents=True);(run/'sources').mkdir();(run/'models').mkdir()
    write_json(run/'run_manifest.json',config)
    shutil.copyfile(data.root/'coverage.csv',run/'models/rank_coverage.csv')
    for name in SOURCES:shutil.copyfile(ROOT/name,run/'sources'/Path(name).name)
    metadata=metadata_for(data);joblib.dump(metadata,run/'models/metadata.joblib')
    train=prepare(data.inputs('train'),metadata);valid=prepare(data.inputs('validation'),metadata)
    labels={s:data.labels(s) for s in ('train','validation')}
    actual=np.array([labels['train'][c['case_id']] for c in train['cases']])
    y=np.log(actual/train['current'])+np.log(train['progress'])
    weights=event_weights([c['event'] for c in train['cases']]);validation_results={};training_results={}
    predictions={'train':[],'validation':[]};bundles={}
    for method in METHODS:
        if method=='linear':
            tp=train['current']/train['progress'];vp=valid['current']/valid['progress']
        else:
            features='context' if method.endswith('context') else 'rank';prep=make_preprocessor(train[features])
            x=prep.fit_transform(train[features]);model=CatBoostRegressor(iterations=iterations,depth=5,learning_rate=.035,
                l2_leaf_reg=12,random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
            model.fit(x,y,sample_weight=weights)
            bundle={'model':model,'preprocessor':prep,'features':features};bundles[method]=bundle
            joblib.dump(bundle,run/'models'/f'{method}.joblib')
            tp=model_prediction(bundle,train);vp=model_prediction(bundle,valid)
        tr=records(train,method,tp);vr=records(valid,method,vp)
        predictions['train'].extend(tr);predictions['validation'].extend(vr)
        training_results[method]=score_predictions(train['cases'],labels['train'],tr)
        validation_results[method]=score_predictions(valid['cases'],labels['validation'],vr)
        print(json.dumps({'method':method,'validation':validation_results[method]['overall']},ensure_ascii=False),flush=True)
    selected=min(METHODS,key=lambda m:validation_results[m]['overall']['event_macro_mape_pct'])
    selection={'selected_recipe':selected,'reported_methods':METHODS,'criterion':'validation event-macro MAPE',
        'validation':validation_results,'training_fit_not_generalization':training_results}
    write_json(run/'selection.json',selection)
    for s,rows in predictions.items():pd.DataFrame(rows).to_csv(run/f'{s}_predictions.csv',index=False)
    # No test label is read; recipes and training are already finished.
    test=prepare(data.inputs('test'),metadata);test_rows=[]
    for method in METHODS:
        p=test['current']/test['progress'] if method=='linear' else model_prediction(bundles[method],test)
        test_rows.extend(records(test,method,p))
    pd.DataFrame(test_rows).to_csv(run/'test_predictions.csv',index=False)
    seal={k:config[k] for k in ['run_id','dataset_id','dataset_directory']}
    seal['artifacts']={str(p.relative_to(run)):sha256(p) for p in sorted(run.rglob('*')) if p.is_file()}
    write_json(run/'blind_test_manifest.json',seal)
    return selection

def predict_samples(run,samples):
    run=Path(run);verify_run(run)
    config=json.loads((run/'run_manifest.json').read_text())
    for name,digest in config['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Prediction code differs from sealed run: '+name)
    metadata=joblib.load(run/'models/metadata.joblib');arrays=prepare(samples,metadata)
    method=json.loads((run/'selection.json').read_text())['selected_recipe']
    if method=='linear':return arrays['current']/arrays['progress']
    return model_prediction(joblib.load(run/'models'/f'{method}.joblib'),arrays)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DEFAULT)
    p.add_argument('--run',type=Path,required=True);a=p.parse_args();run_experiment(MultirankDataset(a.dataset),a.run)
