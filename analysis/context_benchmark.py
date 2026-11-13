"""Ablate activity/band context on the same chronological benchmark cases.

Frozen-block context uses only that block's training events for evaluation
heat statistics. Peer inputs use only the other tier's observed prefix.
Development selects the recipe; all later comparisons remain exploratory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import torch

from analysis.backtest import ROOT, load_curves, make_samples
from analysis.advanced_models import prefix_sequence
from analysis.advanced_benchmark import training_samples, prepare_arrays, tree_fit, neural_fit, prediction_rows
from analysis.event_features import EventFeatures

OUT=ROOT/'analysis/context_results'
RECIPES={
    'lgbm_baseline':('baseline','lgbm'),
    'lgbm_band':('band','lgbm'),
    'lgbm_static':('static','lgbm'),
    'lgbm_history':('history','lgbm'),
    'lgbm_peer':('peer','lgbm'),
    'catboost_history':('history','catboost'),
    'catboost_peer':('peer','catboost'),
    'catboost_peer_regularized':('peer','catboost_regularized'),
    'lgbm_type_experts':('peer','type_experts'),
    'tcn_context':('peer','tcn'),
}


class ContextBuilder:
    def __init__(self, curves):
        self.curves=curves
        self.metadata=EventFeatures.load()
        self.static_cache={}
        self.training_history_cache={}

    def augment(self, samples, mode, history, *, training=False):
        if mode=='baseline':return samples
        result=[]
        for sample in samples:
            curve=sample['curve'];key=(curve.event,curve.rank,sample['now'])
            if key not in self.static_cache:
                self.static_cache[key]=self.metadata.static(curve,sample['now'])
            features=sample['features'].copy()
            features.update(self.metadata.band_only(curve) if mode=='band' else self.static_cache[key])
            if mode in ['history','peer']:
                if training:
                    if key not in self.training_history_cache:
                        self.training_history_cache[key]=self.metadata.historical(sample,self.curves)
                    features.update(self.training_history_cache[key])
                else:
                    features.update(self.metadata.historical(sample,history))
            if mode=='peer':features.update(self.metadata.peer(sample,self.curves))
            result.append(sample|{'features':features})
        return result


def arrays_for(builder,train,test,history,mode):
    if mode=='baseline':return prepare_arrays(train,test)
    train=builder.augment(train,mode,history,training=True)
    test=builder.augment(test,mode,history)
    a=pd.DataFrame([s['features'] for s in train]);b=pd.DataFrame([s['features'] for s in test])
    categorical=[column for column in a if not pd.api.types.is_numeric_dtype(a[column])]
    numerical=[column for column in a if column not in categorical]
    prep=ColumnTransformer([
        ('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),numerical),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
    train_tab=prep.fit_transform(a).astype(np.float32);test_tab=prep.transform(b).astype(np.float32)
    train_base=-np.log(np.array([s['progress'] for s in train],dtype=np.float32))
    test_base=-np.log(np.array([s['progress'] for s in test],dtype=np.float32))
    growth=np.array([s['target'] for s in train],dtype=np.float32)
    return {'train_tab':train_tab,'test_tab':test_tab,
        'train_seq':np.stack([prefix_sequence(s) for s in train]),
        'test_seq':np.stack([prefix_sequence(s) for s in test]),
        'train_growth':growth,'train_y':growth-train_base,'test_base':test_base,
        'feature_columns':list(a),'categorical_columns':categorical}


def fit_context(backend,arrays,train,test):
    if backend=='lgbm':return tree_fit('lightgbm_sequence_residual',arrays)
    x=np.concatenate([arrays['train_tab'],arrays['train_seq'].reshape(len(train),-1)],axis=1)
    z=np.concatenate([arrays['test_tab'],arrays['test_seq'].reshape(len(test),-1)],axis=1)
    started=time.perf_counter()
    if backend.startswith('catboost'):
        model=CatBoostRegressor(iterations=600,depth=5 if backend=='catboost_regularized' else 4,
            learning_rate=.03,l2_leaf_reg=30 if backend=='catboost_regularized' else 10,
            random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
        model.fit(x,arrays['train_y']);pred=model.predict(z)
        return pred,{'seconds':time.perf_counter()-started}
    def learner():
        return LGBMRegressor(n_estimators=500,max_depth=4,num_leaves=15,learning_rate=.03,
            min_child_samples=30,reg_lambda=10,random_state=42,n_jobs=4,verbosity=-1)
    model=learner();model.fit(x,arrays['train_y']);prediction=model.predict(z)
    types=np.array([s['curve'].kind for s in train]);test_types=np.array([s['curve'].kind for s in test])
    for kind in sorted(set(test_types)):
        mask=types==kind;evaluate=test_types==kind
        events=len({s['curve'].event for i,s in enumerate(train) if mask[i]})
        if events<8:continue
        expert=learner();expert.fit(x[mask],arrays['train_y'][mask])
        weight=events/(events+20.)
        prediction[evaluate]=(1-weight)*prediction[evaluate]+weight*expert.predict(z[evaluate])
    return prediction,{'seconds':time.perf_counter()-started}


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    reference=json.load(open(ROOT/'analysis/advanced_results/benchmark.json'))
    curves,_=load_curves();evaluation,_=make_samples(curves);builder=ContextBuilder(curves)
    starts={c.event:c.start for c in curves};boundary=starts[266]
    history=[c for c in curves if c.end+15*60000<boundary]
    train=training_samples(history);test=[s for s in evaluation if 266<=s['curve'].event<=291]
    phases=[('development',-1,history,train,test)]
    for info in reference['fold_audit']:
        history=[c for c in curves if c.end+15*60000<info['test_earliest_start']]
        phases.append(('test',info['fold'],history,training_samples(history),
                       [s for s in evaluation if s['curve'].event in info['test_events']]))
    rows=[];records=[];features={};development_scores={};tcn_epochs=None
    for phase,fold,history,train,test in phases:
        array_cache={}
        for name,(mode,backend) in RECIPES.items():
            if mode not in array_cache:array_cache[mode]=arrays_for(builder,train,test,history,mode)
            arrays=array_cache[mode];features[mode]=arrays.get('feature_columns',[])
            seeds=[42,123,2026] if backend=='tcn' and phase=='test' else [42]
            values=[]
            for seed in seeds:
                started=time.perf_counter()
                if backend=='tcn':
                    pred,record=neural_fit('tcn',arrays,seed=seed,device='mps',max_epochs=80 if phase=='development' else tcn_epochs,
                        batch_size=256,validation_growth=np.array([s['target'] for s in test]) if phase=='development' else None)
                    if phase=='development':tcn_epochs=record['selected_epochs']
                else:pred,record=fit_context(backend,arrays,train,test)
                values.append(pred)
                record.update({'phase':phase,'fold':fold,'method':name,'seed':seed,
                    'training_events':len({c.event for c in history}),'training_latest_end':max(c.end for c in history),
                    'test_earliest_start':min(s['curve'].start for s in test),'wall_seconds':time.perf_counter()-started})
                records.append(record)
                rows.extend(prediction_rows(test,pred,arrays['test_base'],phase=phase,method=name,seed=seed,fold=fold))
                print(json.dumps({'completed':f'{phase}-{fold}-{name}-{seed}',
                    'seconds':round(record['wall_seconds'],2)}),flush=True)
            aggregate=np.mean(values,axis=0)
            if backend=='tcn' and phase=='test':
                rows.extend(prediction_rows(test,aggregate,arrays['test_base'],phase=phase,method=name+'_ensemble3',seed=-1,fold=fold))
            if phase=='development':
                truth=np.array([s['target'] for s in test]);development_scores[name]=float(np.mean(abs(np.exp(np.clip(arrays['test_base']+aggregate,0,4)-truth)-1))*100)
        pd.DataFrame(rows).to_csv(OUT/'predictions.csv',index=False)
        if phase=='development':
            selection={'development_mape':development_scores,'selected_recipe':min(development_scores,key=development_scores.get),'tcn_epochs':tcn_epochs}
            (OUT/'selection.json').write_text(json.dumps(selection,indent=2));print('SELECTION '+json.dumps(selection),flush=True)
    frame=pd.DataFrame(rows);frame['ape']=abs(frame.predicted/frame.actual-1)*100
    frame.groupby(['phase','method','seed','rank','stage']).ape.agg(['count','mean','median']).to_csv(OUT/'metrics.csv')
    report={'created_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'database_sha256':hashlib.sha256((ROOT/'data/db.sqlite3').read_bytes()).hexdigest(),
        'metadata_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/metadata').glob('*.json')},
        'selection':selection,'training_runs':records,'feature_columns':features,'recipes':RECIPES,
        'historical_statistics_policy':'for test samples use only that fold training events; for training samples use only events ended before the target began',
        'peer_policy':'other rank observed prefix at or before target latest observation; never its final score'}
    (OUT/'benchmark.json').write_text(json.dumps(report,indent=2))
    print(frame[frame.phase=='test'].groupby(['method','seed']).ape.mean().unstack('seed').round(3).to_string())


if __name__=='__main__':main()
