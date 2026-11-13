"""Attribute gains to pooled historical context, with a compact neural check."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from analysis.backtest import ROOT,load_curves,make_samples
from analysis.advanced_benchmark import training_samples,neural_fit,prediction_rows
from analysis.context_benchmark import ContextBuilder,arrays_for,fit_context

OUT=ROOT/'analysis/context_ablation_results'
GLOBAL={'history_rank_events','historical_heat_global','historical_correction_global',
        'recent_heat_shift','recent_correction_shift'}
RECIPES={'cb_baseline':'baseline','cb_global_history':'global','cb_type_history':'type',
         'cb_band_history':'band','cb_character_history':'character','cb_compact_peer':'peer',
         'tcn_compact':'compact','tcn_conditional_prior':'compact_prior'}


class FilteredBuilder(ContextBuilder):
    def __init__(self,curves):
        super().__init__(curves);self.policy='global'

    def augment(self,samples,mode,history,*,training=False):
        augmented=super().augment(samples,mode,history,training=training)
        output=[]
        for original,sample in zip(samples,augmented):
            base=set(original['features']);allowed=set(GLOBAL)
            groups=[]
            if self.policy in ['type','band','character','peer','compact','compact_prior']:groups.append('type')
            if self.policy in ['band','character','peer','compact','compact_prior']:groups+=['band','type_band']
            if self.policy in ['character','peer','compact','compact_prior']:groups.append('character')
            if self.policy in ['peer','compact','compact_prior']:groups.append('attribute')
            for group in groups:
                allowed.update(f'history_{group}_{suffix}' for suffix in ['weight','heat_delta','correction'])
            if self.policy in ['peer','compact','compact_prior']:
                allowed.update(k for k in sample['features'] if k.startswith('peer_') or k=='log_peer_current_ratio')
            if self.policy in ['compact','compact_prior']:
                allowed.update(k for k in sample['features'] if not k.startswith(('participant_','featured_character_','history_'))
                               and k!='featured_character')
            output.append(sample|{'features':{k:v for k,v in sample['features'].items() if k in base|allowed}})
        return output


def conditional_prior(samples,builder,history,training):
    augmented=builder.augment(samples,'peer',history,training=training)
    return np.array([.5*s['features']['history_type_correction']+.25*s['features']['history_band_correction']
                     +.25*s['features']['history_character_correction'] for s in augmented],dtype=np.float32)


def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True,parents=True)
    curves,_=load_curves();evaluation,_=make_samples(curves);builder=FilteredBuilder(curves)
    reference=json.load(open(ROOT/'analysis/advanced_results/benchmark.json'))
    start=next(c.start for c in curves if c.event==266)
    h=[c for c in curves if c.end+15*60000<start]
    phases=[('development',-1,h,training_samples(h),[s for s in evaluation if 266<=s['curve'].event<=291])]
    for info in reference['fold_audit']:
        h=[c for c in curves if c.end+15*60000<info['test_earliest_start']]
        phases.append(('test',info['fold'],h,training_samples(h),
                       [s for s in evaluation if s['curve'].event in info['test_events']]))
    rows=[];records=[];epoch_counts={};development_scores={}
    for phase,fold,history,train,test in phases:
        for name,policy in RECIPES.items():
            builder.policy=policy
            arrays=arrays_for(builder,train,test,history,'baseline' if policy=='baseline' else 'peer')
            if policy=='compact_prior':
                train_prior=conditional_prior(train,builder,history,True)
                test_prior=conditional_prior(test,builder,history,False)
                arrays=arrays|{'train_y':arrays['train_y']-train_prior,'test_base':arrays['test_base']+test_prior}
            neural=name.startswith('tcn')
            seeds=[42,123,2026] if neural and phase=='test' else [42]
            values=[]
            for seed in seeds:
                started=time.perf_counter()
                if neural:
                    pred,record=neural_fit('tcn',arrays,seed=seed,device='mps',
                        max_epochs=80 if phase=='development' else epoch_counts[name],batch_size=256,
                        validation_growth=np.array([s['target'] for s in test]) if phase=='development' else None)
                    if phase=='development':epoch_counts[name]=record['selected_epochs']
                else:pred,record=fit_context('catboost',arrays,train,test)
                values.append(pred)
                record.update({'method':name,'phase':phase,'fold':fold,'seed':seed,
                    'seconds':time.perf_counter()-started,'training_latest_end':max(c.end for c in history),
                    'test_earliest_start':min(s['curve'].start for s in test),
                    'feature_columns':arrays.get('feature_columns',[])})
                records.append(record)
                rows.extend(prediction_rows(test,pred,arrays['test_base'],phase=phase,method=name,seed=seed,fold=fold))
                print(json.dumps({'completed':f'{phase}-{fold}-{name}-{seed}',
                    'seconds':round(record['seconds'],2)}),flush=True)
            if neural and phase=='test':
                rows.extend(prediction_rows(test,np.mean(values,axis=0),arrays['test_base'],phase=phase,
                    method=name+'_ensemble3',seed=-1,fold=fold))
            if phase=='development':
                target=np.array([s['target'] for s in test]);development_scores[name]=float(np.mean(
                    abs(np.exp(np.clip(arrays['test_base']+np.mean(values,axis=0),0,4)-target)-1))*100)
        pd.DataFrame(rows).to_csv(OUT/'predictions.csv',index=False)
        if phase=='development':
            selection={'scores':development_scores,'selected_recipe':min(development_scores,key=development_scores.get),
                       'epochs':epoch_counts}
            (OUT/'selection.json').write_text(json.dumps(selection,indent=2));print('SELECTION '+json.dumps(selection),flush=True)
    frame=pd.DataFrame(rows);frame['ape']=abs(frame.predicted/frame.actual-1)*100
    frame.groupby(['phase','method','seed','rank','stage']).ape.agg(['count','mean','median']).to_csv(OUT/'metrics.csv')
    (OUT/'benchmark.json').write_text(json.dumps({'database_sha256':hashlib.sha256((ROOT/'data/db.sqlite3').read_bytes()).hexdigest(),
        'selection':selection,'recipes':RECIPES,'training_runs':records,
        'conditional_prior':'0.5 type + 0.25 band + 0.25 character; all smoothed prior-event log-growth corrections',
        'exploratory':True},indent=2))
    print(frame[frame.phase=='test'].groupby(['method','seed']).ape.mean().unstack('seed').round(3).to_string())


if __name__=='__main__':main()
