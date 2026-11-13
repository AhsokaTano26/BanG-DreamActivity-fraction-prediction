"""Statistical/structural candidates on the unchanged holdout. Develop then predict.

python -m analysis.relationship_benchmark develop --run analysis/holdout_results/relationships_v1
python -m analysis.relationship_benchmark predict --run analysis/holdout_results/relationships_v1
python -m analysis.holdout_evaluate --run analysis/holdout_results/relationships_v1
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,SplineTransformer

from analysis.holdout_dataset import ROOT,HoldoutDataset,sha256,write_json
from analysis.holdout_evaluate import score_predictions,case_metadata
from analysis.event_features import EventFeatures
from analysis.relationship_features import relationship_frame,select_features

VARIANTS=['full','no_band','no_peer','no_type','no_quality','invariant','invariant_rank']
RECIPES={f'{backend}_{v}':{'backend':backend,'variant':v,'target':'residual','loss':'RMSE'}
         for backend in ['catboost','lightgbm'] for v in VARIANTS}
RECIPES.update({f'catboost_mae_{v}':{'backend':'catboost','variant':v,'target':'residual','loss':'MAE'} for v in ['full','invariant']})
RECIPES.update({f'{backend}_remaining_{v}':{'backend':backend,'variant':v,'target':'remaining','loss':'RMSE'}
                for backend in ['catboost','lightgbm'] for v in ['invariant','invariant_rank']})
RECIPES.update({f'gam_ridge_{a}':{'backend':'gam','variant':'invariant','target':'residual','alpha':a} for a in [1,10,100]})
SOURCE_NAMES=['analysis/relationship_features.py','analysis/relationship_statistics.py','analysis/relationship_benchmark.py',
    'analysis/holdout_dataset.py','analysis/backtest.py','analysis/event_features.py','analysis/holdout_evaluate.py']


def weights(samples):
    events=np.array([s['curve'].event for s in samples]);_,inv,counts=np.unique(events,return_inverse=True,return_counts=True)
    w=1/counts[inv];return w/w.mean()


def target_values(samples,labels,target):
    p=np.array([s['progress'] for s in samples]);ratio=np.array([labels[s['case_id']]/s['score'] for s in samples])
    if target=='remaining':return np.clip(np.log(np.maximum(ratio-1,1e-4)*p/(1-p)),-6,4)
    return np.log(np.maximum(1,ratio))+np.log(p)


def decode(samples,values,target):
    p=np.array([s['progress'] for s in samples]);score=np.array([s['score'] for s in samples])
    if target=='remaining':return score*(1+np.exp(np.clip(values,-6,4))*(1-p)/p)
    return score*np.exp(np.clip(values-np.log(p),0,4))


def prep_for(frame,gam=False):
    cats=[c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])];nums=[c for c in frame if c not in cats]
    if gam:
        nums=[c for c in ['progress','log_remaining_h','duration_h','rate_6h_relative','rate_24h_relative',
            'momentum_6_24','gap_log_1000_2000','gap_change_24h','peer_rate_24h','quality_median_gap_h'] if c in frame]
        numeric=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
            SplineTransformer(n_knots=4,degree=3,include_bias=False,extrapolation='constant'),StandardScaler())
    else:numeric=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True),StandardScaler())
    return ColumnTransformer([('numeric',numeric,nums),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])


def expanded(x,samples,gam):
    if not gam:return x
    r=np.array([s['curve'].rank==1000 for s in samples],dtype=float)
    return np.concatenate([x,x*r[:,None]],axis=1)


def fit_recipe(recipe,frame,samples,labels):
    frame=select_features(frame,recipe['variant']);gam=recipe['backend']=='gam';prep=prep_for(frame,gam)
    x=expanded(prep.fit_transform(frame),samples,gam);y=target_values(samples,labels,recipe['target'])
    def learner():
        if gam:return Ridge(alpha=recipe['alpha'])
        if recipe['backend']=='catboost':return CatBoostRegressor(iterations=500,depth=4,learning_rate=.03,
            l2_leaf_reg=15,loss_function=recipe['loss'],random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
        return LGBMRegressor(n_estimators=500,max_depth=3,num_leaves=7,learning_rate=.025,
                            min_child_samples=40,reg_lambda=15,random_state=42,n_jobs=4,verbosity=-1)
    rank=np.array([s['curve'].rank for s in samples]);models={}
    for key in ([1000,2000] if recipe['variant']=='invariant_rank' else ['all']):
        mask=np.ones(len(samples),dtype=bool) if key=='all' else rank==key
        subset=[s for s,m in zip(samples,mask) if m]
        model=learner();model.fit(x[mask],y[mask],sample_weight=weights(subset));models[key]=model
    return {'recipe':recipe,'prep':prep,'models':models,'feature_columns':list(frame)}


def predict_bundle(bundle,frame,samples):
    recipe=bundle['recipe'];frame=select_features(frame,recipe['variant'])
    x=expanded(bundle['prep'].transform(frame),samples,recipe['backend']=='gam');values=np.empty(len(samples))
    rank=np.array([s['curve'].rank for s in samples])
    for key,model in bundle['models'].items():
        mask=np.ones(len(samples),dtype=bool) if key=='all' else rank==key
        values[mask]=model.predict(x[mask])
    return decode(samples,values,recipe['target'])


def rows(samples,name,prediction):
    if len(samples)!=len(prediction) or not np.isfinite(prediction).all():raise ValueError('Invalid predictions')
    return [{'method':name,'case_id':s['case_id'],'event':s['curve'].event,'rank':s['curve'].rank,'stage':s['stage'],
             'event_type':s['curve'].kind,'cutoff':s['cutoff'],'observed_at':s['now'],'current':s['score'],'predicted':float(p)}
            for s,p in zip(samples,prediction)]


def score(samples,labels,prediction):
    return score_predictions(case_metadata(samples),labels,[{'case_id':s['case_id'],'predicted':float(p)} for s,p in zip(samples,prediction)])['overall']['event_macro_mape_pct']


def old_reference(split,samples):
    old=ROOT/'analysis/holdout_results/v1';seal=json.loads((old/'blind_test_manifest.json').read_text())
    name=f'{split}_predictions.csv'
    if sha256(old/name)!=seal['artifacts'][name]:raise ValueError('Old reference fingerprint mismatch')
    frame=pd.read_csv(old/name);frame=frame[frame.method=='validation_top3_ensemble'].set_index('case_id')
    if set(frame.index)!={s['case_id'] for s in samples}:raise ValueError('Reference case IDs differ')
    return frame.loc[[s['case_id'] for s in samples],'predicted'].to_numpy()


def development(run,data):
    if run.exists():raise FileExistsError(run)
    run.mkdir(parents=True);(run/'models').mkdir();(run/'sources').mkdir()
    config={'dataset_id':data.manifest['dataset_id'],'recipes':RECIPES,'source_hashes':{n:sha256(ROOT/n) for n in SOURCE_NAMES},
        'dataset_directory':str(data.root.resolve()),'weights':'equal total weight per activity',
        'selection':'minimum validation event MAPE among registered recipes/top3 equal blend/old reference/25,50,75 percent new blend',
        'tree_parameters':'CatBoost 500 depth4 lr.03 L2=15; LightGBM 500 depth3/leaves7 lr.025 min40 L2=15; seed42',
        'target_remaining':'log((final/current-1)*progress/(1-progress)); ratio floor1e-4, clip[-6,4]',
        'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','catboost','lightgbm','joblib']},
        'cv':'training-only chronological: first98 then holdout25/25/25/25; last boundary capped at198; no test data',
        'previous_test_exposure':True}
    config['run_id']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    write_json(run/'run_manifest.json',config)
    for n in SOURCE_NAMES:shutil.copyfile(ROOT/n,run/'sources'/Path(n).name)
    samples=data.inputs('train');validation=data.inputs('validation');labels=data.labels('train');val_labels=data.labels('validation')
    meta=EventFeatures.load(data.metadata_directory);train_frame=relationship_frame(samples,meta);val_frame=relationship_frame(validation,meta)
    from analysis.relationship_statistics import analyze
    analyze(data,train_frame,samples,meta,run)
    # Sequential out-of-fold checks within train; folds never span event groups.
    ids=data.manifest['splits']['train']['event_ids'];cv_rows=[];audit=[]
    for fold,left in enumerate([98,123,148,173]):
        fit_ids=set(ids[:left]);eval_ids=set(ids[left:left+25])
        ti=[i for i,s in enumerate(samples) if s['curve'].event in fit_ids];vi=[i for i,s in enumerate(samples) if s['curve'].event in eval_ids]
        ts=[samples[i] for i in ti];vs=[samples[i] for i in vi]
        assert max(s['curve'].end for s in ts)<min(s['curve'].start for s in vs)
        audit.append({'fold':fold,'train_events':ids[:left],'evaluation_events':ids[left:left+25]})
        for variant in ['full','no_band','no_peer','no_type','no_quality','invariant']:
            bundle=fit_recipe(RECIPES[f'catboost_{variant}'],train_frame.iloc[ti],ts,labels)
            pred=predict_bundle(bundle,train_frame.iloc[vi],vs)
            cv_rows.extend([r|{'fold':fold,'actual':labels[r['case_id']]} for r in rows(vs,variant,pred)])
        print(json.dumps({'training_cv_fold':fold,'events':len(eval_ids)}),flush=True)
    pd.DataFrame(cv_rows).to_csv(run/'training_cv_predictions.csv',index=False);write_json(run/'training_cv_folds.json',audit)
    values={'old_v1_reference':old_reference('validation',validation)};records=[]
    for name,recipe in RECIPES.items():
        started=time.perf_counter();bundle=fit_recipe(recipe,train_frame,samples,labels)
        path=run/f'models/{name}.joblib';joblib.dump(bundle,path)
        pred=predict_bundle(joblib.load(path),val_frame,validation);values[name]=pred
        record={'method':name,'validation_mape':score(validation,val_labels,pred),'seconds':time.perf_counter()-started,
                'model_sha256':sha256(path),'feature_columns':bundle['feature_columns']}
        records.append(record);print(json.dumps(record|{'feature_columns':len(bundle['feature_columns'])}),flush=True)
    scores={n:score(validation,val_labels,p) for n,p in values.items()}
    top3=sorted(RECIPES,key=lambda n:(scores[n],n))[:3]
    values['new_top3']=np.mean([values[n] for n in top3],axis=0)
    for weight in [.25,.5,.75]:values[f'blend_new_{weight}']=weight*values['new_top3']+(1-weight)*values['old_v1_reference']
    scores={n:score(validation,val_labels,p) for n,p in values.items()};selected=min(scores,key=lambda n:(scores[n],n))
    selection={'dataset_id':data.manifest['dataset_id'],'run_id':config['run_id'],'selected_recipe':selected,
        'top3_members':top3,'validation_scores':scores,'reported_methods':sorted(values),'test_labels_used_for_selection':False}
    write_json(run/'selection.json',selection);write_json(run/'training_records.json',records)
    pd.DataFrame([r for n,p in values.items() for r in rows(validation,n,p)]).to_csv(run/'validation_predictions.csv',index=False)
    write_json(run/'development_seal.json',{'artifacts':{str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file()}})
    print('SELECTION '+json.dumps(selection),flush=True)


def blind_prediction(run,data):
    if (run/'blind_test_manifest.json').exists():raise FileExistsError('Blind predictions already sealed')
    config=json.loads((run/'run_manifest.json').read_text());selection=json.loads((run/'selection.json').read_text())
    if config['dataset_id']!=data.manifest['dataset_id']:raise ValueError('Dataset changed')
    for n,h in config['source_hashes'].items():
        if sha256(ROOT/n)!=h:raise ValueError('Code changed after development')
    for n,h in json.loads((run/'development_seal.json').read_text())['artifacts'].items():
        if sha256(run/n)!=h:raise ValueError('Development artifacts changed')
    samples=data.inputs('test');frame=relationship_frame(samples,EventFeatures.load(data.metadata_directory))
    values={'old_v1_reference':old_reference('test',samples)}
    for name in RECIPES:values[name]=predict_bundle(joblib.load(run/f'models/{name}.joblib'),frame,samples)
    values['new_top3']=np.mean([values[n] for n in selection['top3_members']],axis=0)
    for w in [.25,.5,.75]:values[f'blend_new_{w}']=w*values['new_top3']+(1-w)*values['old_v1_reference']
    pd.DataFrame([r for n,p in values.items() for r in rows(samples,n,p)]).to_csv(run/'test_predictions.csv',index=False)
    write_json(run/'blind_test_manifest.json',{'run_id':config['run_id'],'dataset_id':data.manifest['dataset_id'],
        'dataset_directory':str(data.root.resolve()),'artifacts':{str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file()},'test_cases':len(samples)})
    print('Blind predictions sealed; test labels were not read.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['develop','predict']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();data=HoldoutDataset()
    (development if args.phase=='develop' else blind_prediction)(args.run,data)


if __name__=='__main__':main()
