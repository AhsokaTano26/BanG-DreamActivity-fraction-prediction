"""Training folds and validation only: deadline priors and honest residual correction."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from analysis.holdout_dataset import ROOT, HoldoutDataset, sha256, write_json
from analysis.event_features import EventFeatures
from analysis.relationship_features import relationship_frame, select_features
from analysis.relationship_benchmark import prep_for, weights, target_values, decode, rows
from analysis.deadline_features import history_features, prior_offset, SETTINGS

FOLD_LEFT=[98,123,148,173]
FOLD_SIZE=25
RECIPES=['remaining_rank','deadline_features','deadline_residual']
SHRINKAGES=[.25,.5,1.]
TREE_PARAMETERS={'iterations':500,'depth':4,'learning_rate':.03,'l2_leaf_reg':15,'loss_function':'RMSE',
                 'random_seed':42,'thread_count':4,'verbose':False,'allow_writing_files':False}
SOURCES=['analysis/deadline_benchmark.py','analysis/deadline_features.py','analysis/deadline_diagnostics.py',
         'analysis/relationship_benchmark.py','analysis/relationship_features.py','analysis/relationship_statistics.py',
         'analysis/holdout_dataset.py','analysis/backtest.py','analysis/event_features.py']


def fit(samples,labels,frame,recipe):
    prep=prep_for(frame);x=prep.fit_transform(frame);y=target_values(samples,labels,'remaining')
    if recipe=='deadline_residual':y=y-prior_offset(frame)
    rank=np.array([s['curve'].rank for s in samples]);w=weights(samples);models={}
    for r in [1000,2000]:
        mask=rank==r
        model=CatBoostRegressor(**TREE_PARAMETERS);model.fit(x[mask],y[mask],sample_weight=w[mask]);models[r]=model
    return {'prep':prep,'models':models,'columns':list(frame),'recipe':recipe}


def predict(bundle,samples,frame):
    x=bundle['prep'].transform(frame);rank=np.array([s['curve'].rank for s in samples]);values=np.empty(len(samples))
    for r,model in bundle['models'].items():
        mask=rank==r;values[mask]=model.predict(x[mask])
    if bundle['recipe']=='deadline_residual':values+=prior_offset(frame)
    return decode(samples,values,'remaining')


def calibration_frame(samples):
    records=[]
    for s in samples:
        p=s['progress'];remaining=(s['curve'].end-s['now'])/3600000;r=float(s['curve'].rank==1000)
        phase='last6' if remaining<=6 else 'last24' if remaining<=24 else 'early' if p<.25 else 'middle' if p<.75 else 'late'
        records.append({'progress':p,'progress_squared':p*p,'log_remaining':np.log1p(remaining),
                        'rank_progress':r*p,'rank_log_remaining':r*np.log1p(remaining),
                        'type':s['curve'].kind,'rank':str(s['curve'].rank),
                        'rank_type_phase':f'{s["curve"].rank}_{s["curve"].kind}_{phase}'})
    return pd.DataFrame(records)


def fit_calibration(samples,labels,prediction):
    frame=calibration_frame(samples);prep=prep_for(frame);x=prep.fit_transform(frame)
    y=np.log(np.array([labels[s['case_id']] for s in samples])/prediction)
    model=Ridge(alpha=100);model.fit(x,y,sample_weight=weights(samples))
    return {'prep':prep,'model':model,'fit_events':sorted({s['curve'].event for s in samples})}


def calibrated(bundle,samples,prediction,shrinkage):
    if bundle is None:return prediction.copy()
    adjustment=np.clip(bundle['model'].predict(bundle['prep'].transform(calibration_frame(samples))),-.25,.25)
    return np.maximum([s['score'] for s in samples],prediction*np.exp(shrinkage*adjustment))


def macro(samples,labels,prediction):
    _,inverse,counts=np.unique([s['curve'].event for s in samples],return_inverse=True,return_counts=True)
    actual=np.array([labels[s['case_id']] for s in samples])
    return float(np.mean(np.bincount(inverse,weights=abs(prediction/actual-1)*100)/counts))


def frame_for(samples,history,metadata):
    ordinary=select_features(relationship_frame(samples,metadata),'invariant').reset_index(drop=True)
    extended=pd.concat([ordinary,history_features(samples,history)],axis=1)
    return ordinary,extended


def reference_values(root,samples,data):
    seal=json.loads((root/'blind_test_manifest.json').read_text())
    if seal['dataset_id']!=data.manifest['dataset_id']:raise ValueError('Reference dataset differs')
    for name in ['validation_predictions.csv','selection.json']:
        if sha256(root/name)!=seal['artifacts'][name]:raise ValueError('Reference changed')
    selection=json.loads((root/'selection.json').read_text())
    frame=pd.read_csv(root/'validation_predictions.csv');frame=frame[frame.method==selection['selected_recipe']].set_index('case_id')
    ids=[s['case_id'] for s in samples]
    if not frame.index.is_unique or set(frame.index)!=set(ids):raise ValueError('Reference cases differ')
    identity={'directory':str(root.resolve()),'manifest_sha256':sha256(root/'blind_test_manifest.json'),
              'run_id':seal['run_id'],'selected':selection['selected_recipe']}
    return frame.loc[ids,'predicted'].to_numpy(),identity


def develop(run,data,reference):
    if run.exists():raise FileExistsError(run)
    samples=data.inputs('train');validation=data.inputs('validation');labels=data.labels('train');val_labels=data.labels('validation')
    ref,ref_identity=reference_values(reference,validation,data)
    config={'dataset_id':data.manifest['dataset_id'],'dataset_directory':str(data.root.resolve()),'recipes':RECIPES,
            'fold_left':FOLD_LEFT,'fold_size':FOLD_SIZE,'tree_parameters':TREE_PARAMETERS,'history':SETTINGS,
            'residual_calibration':{'alpha':100,'clip':.25,'shrinkages':SHRINKAGES,'training':'earlier OOF folds only; first fold unchanged'},
            'reference':ref_identity,'scope':'training folds and frozen validation only; no test prediction or scoring',
            'global_weights':'equal activity total across ranks','python':platform.python_version(),
            'source_hashes':{name:sha256(ROOT/name) for name in SOURCES},
            'software':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','catboost','scikit-learn','joblib']}}
    config['run_id']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    run.mkdir(parents=True);(run/'models').mkdir();(run/'sources').mkdir();write_json(run/'run_manifest.json',config)
    for name in SOURCES:shutil.copyfile(ROOT/name,run/'sources'/Path(name).name)
    history=data.train_curves();metadata=EventFeatures.load(data.metadata_directory);ids=data.manifest['splits']['train']['event_ids']
    oof_records=[];past_samples=[];past_pred=[];folds=[]
    for fold,left in enumerate(FOLD_LEFT):
        fit_ids=set(ids[:left]);eval_ids=set(ids[left:left+FOLD_SIZE])
        if len(eval_ids)!=FOLD_SIZE:raise ValueError('Invalid fold boundary')
        train=[s for s in samples if s['curve'].event in fit_ids];hold=[s for s in samples if s['curve'].event in eval_ids]
        assert max(s['curve'].end for s in train)<min(s['curve'].start for s in hold)
        pool=[c for c in history if c.event in fit_ids]
        fit_base,fit_extra=frame_for(train,pool,metadata);eval_base,eval_extra=frame_for(hold,pool,metadata)
        predictions={}
        for recipe in RECIPES:
            fit_frame,eval_frame=(fit_base,eval_base) if recipe=='remaining_rank' else (fit_extra,eval_extra)
            bundle=fit(train,labels,fit_frame,recipe);joblib.dump(bundle,run/f'models/fold{fold}_{recipe}.joblib')
            predictions[recipe]=predict(bundle,hold,eval_frame)
        calibration=fit_calibration(past_samples,labels,np.concatenate(past_pred)) if past_samples else None
        if calibration is not None:
            assert not set(calibration['fit_events'])&eval_ids
            assert max(s['curve'].end for s in past_samples)<min(s['curve'].start for s in hold)
            joblib.dump(calibration,run/f'models/fold{fold}_calibration.joblib')
        for shrink in SHRINKAGES:
            predictions[f'oof_correction_{shrink}']=calibrated(calibration,hold,predictions['remaining_rank'],shrink)
        for name,prediction in predictions.items():
            oof_records.extend(row|{'fold':fold,'actual':labels[row['case_id']]} for row in rows(hold,name,prediction))
        folds.append({'fold':fold,'fit_events':ids[:left],'held_events':ids[left:left+FOLD_SIZE],
                      'calibration_events':[] if calibration is None else calibration['fit_events']})
        past_samples.extend(hold);past_pred.append(predictions['remaining_rank'])
        print(json.dumps({'fold':fold,'scores':{n:macro(hold,labels,p) for n,p in predictions.items()}}),flush=True)
    pd.DataFrame(oof_records).to_csv(run/'training_oof_predictions.csv',index=False);write_json(run/'folds.json',folds)
    train_base,train_extra=frame_for(samples,history,metadata);val_base,val_extra=frame_for(validation,history,metadata)
    values={'current_reference':ref}
    for recipe in RECIPES:
        fit_frame,eval_frame=(train_base,val_base) if recipe=='remaining_rank' else (train_extra,val_extra)
        bundle=fit(samples,labels,fit_frame,recipe);path=run/f'models/final_{recipe}.joblib';joblib.dump(bundle,path)
        values[recipe]=predict(joblib.load(path),validation,eval_frame)
    calibration=fit_calibration(past_samples,labels,np.concatenate(past_pred));joblib.dump(calibration,run/'models/final_calibration.joblib')
    for shrink in SHRINKAGES:values[f'oof_correction_{shrink}']=calibrated(calibration,validation,values['remaining_rank'],shrink)
    scores={name:macro(validation,val_labels,pred) for name,pred in values.items()}
    choice=min(scores,key=lambda name:(scores[name],name))
    selection={'run_id':config['run_id'],'dataset_id':config['dataset_id'],'validation_scores':scores,
               'selected_on_validation':choice,'test_evaluated':False,'reference_is_frozen':True}
    write_json(run/'selection.json',selection)
    pd.DataFrame([row for name,pred in values.items() for row in rows(validation,name,pred)]).to_csv(run/'validation_predictions.csv',index=False)
    write_json(run/'development_seal.json',{'artifacts':{str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file()}})
    print('VALIDATION '+json.dumps(selection),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=ROOT/'analysis/holdout_results/structured_neural_v1')
    args=parser.parse_args();develop(args.run,HoldoutDataset(),args.reference)
