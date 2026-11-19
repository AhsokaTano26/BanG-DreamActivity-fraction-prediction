"""Test historical transfer without using any later test labels.

Adds two prespecified TCN strategies to the exact advanced-benchmark folds:
older-regime pretraining followed by recent-regime fine-tuning, and joint
training with older snapshots weighted at 0.25. TCN was selected as a cheap
member of the development-selected ensemble, before reviewing test scores.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from analysis.backtest import ROOT, Curve, load_curves, make_samples, snapshot
from analysis.advanced_models import Forecaster
from analysis.advanced_benchmark import training_samples, prepare_arrays, prediction_rows

OUT = ROOT / 'analysis' / 'transfer_results'


def older_curves():
    with sqlite3.connect((ROOT/'data/db.sqlite3').as_uri()+'?mode=ro',uri=True) as connection:
        connection.row_factory=sqlite3.Row
        rows=connection.execute('SELECT * FROM event WHERE EventID<226 ORDER BY StartAt,Rank').fetchall()
    result=[]
    for row in rows:
        points=ast.literal_eval(row['PointRank'])
        if len(points)<20 or not -300000<=row['EndAt']-points[-1]['time']<=900000:
            continue
        result.append(Curve(row['EventID'],row['Rank'],row['EventType'],row['StartAt'],row['EndAt'],
            np.array([p['time'] for p in points]),np.array([p['ep'] for p in points],dtype=float)))
    return result


def sparse_samples(curves):
    result=[]
    for curve in curves:
        bins={}
        for timestamp in curve.times:
            progress=(timestamp-curve.start)/curve.duration
            if not .2<=progress<=.98:
                continue
            value=snapshot(curve,int(timestamp))
            if value is not None:
                value['target']=np.log(max(1.,curve.final/value['score']))
                bins[int(progress*20)]=value
        result.extend(bins.values())
    return result


def train_transfer(arrays,n_old,mode,seed,epochs,device='mps'):
    torch.manual_seed(seed)
    if device=='mps':torch.mps.manual_seed(seed)
    model=Forecaster('tcn',arrays['train_tab'].shape[1]).to(device)
    sequence=torch.from_numpy(arrays['train_seq']).to(device)
    tab=torch.from_numpy(arrays['train_tab']).to(device)
    y=torch.from_numpy(arrays['train_y']).to(device)
    generator=torch.Generator().manual_seed(seed)
    if mode=='pretrain_finetune_tcn':
        phases=[(np.arange(n_old),30,.001,None),
                (np.arange(n_old,len(y)),epochs,.0003,None)]
    else:
        weights=torch.ones(len(y),device=device);weights[:n_old]=.25
        phases=[(np.arange(len(y)),epochs,.001,weights)]
    started=time.perf_counter()
    for indices,count,lr,weights in phases:
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
        model.train()
        for epoch in range(count):
            order=torch.tensor(indices[torch.randperm(len(indices),generator=generator).numpy()],device=device)
            for start in range(0,len(order),256):
                batch=order[start:start+256]
                optimizer.zero_grad(set_to_none=True)
                loss=F.smooth_l1_loss(model(sequence[batch],tab[batch]),y[batch],beta=.2,reduction='none')
                loss=loss.mean() if weights is None else (loss*weights[batch]).sum()/weights[batch].sum()
                if not torch.isfinite(loss):raise ValueError('Nonfinite transfer loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    model.eval()
    with torch.no_grad():
        test_seq=torch.from_numpy(arrays['test_seq']).to(device)
        test_tab=torch.from_numpy(arrays['test_tab']).to(device)
        prediction=np.concatenate([model(test_seq[start:start+256],test_tab[start:start+256]).cpu().numpy()
            for start in range(0,len(test_tab),256)])
    if device=='mps':torch.mps.synchronize()
    return prediction,time.perf_counter()-started


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    original=json.load(open(ROOT/'analysis/advanced_results/benchmark.json'))
    device=original['configuration']['device'];seeds=original['configuration']['seeds']
    epochs=original['selection']['epochs']['tcn']
    old=older_curves();recent,_=load_curves();evaluation,_=make_samples(recent)
    rows=[];audit=[];runs=[]
    for fold_info in original['fold_audit']:
        fold=fold_info['fold'];boundary=fold_info['test_earliest_start']
        older=[c for c in old if c.end+15*60000<boundary]
        latest=[c for c in recent if c.end+15*60000<boundary]
        test=[s for s in evaluation if s['curve'].event in fold_info['test_events']]
        first=sparse_samples(older);second=training_samples(latest)
        arrays=prepare_arrays(first+second,test)
        for mode in ['pretrain_finetune_tcn','weighted_history_tcn']:
            predictions=[]
            for seed in seeds:
                values,seconds=train_transfer(arrays,len(first),mode,seed,epochs,device)
                predictions.append(values)
                rows.extend(prediction_rows(test,values,arrays['test_base'],phase='test',method=mode,seed=seed,fold=fold))
                runs.append({'fold':fold,'mode':mode,'seed':seed,'seconds':seconds})
                print(json.dumps(runs[-1]),flush=True)
            rows.extend(prediction_rows(test,np.mean(predictions,axis=0),arrays['test_base'],phase='test',
                method=mode+f'_ensemble{len(seeds)}',seed=-1,fold=fold))
        audit.append({'fold':fold,'old_curves':len(older),'old_events':len({c.event for c in older}),
            'recent_curves':len(latest),'old_snapshots':len(first),'recent_snapshots':len(second),
            'latest_training_end':max(c.end for c in older+latest),'earliest_test_start':boundary})
        pd.DataFrame(rows).to_csv(OUT/'predictions.csv',index=False)
    frame=pd.DataFrame(rows);frame['ape']=abs(frame.predicted/frame.actual-1)*100
    frame.groupby(['method','seed','rank','stage']).ape.agg(['count','mean','median']).to_csv(OUT/'metrics.csv')
    (OUT/'transfer.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),
        'database_sha256':hashlib.sha256((ROOT/'data/db.sqlite3').read_bytes()).hexdigest(),
        'old_curves':len(old),'old_events':len({c.event for c in old}),
        'configuration':{'pretrain_epochs':30,'finetune_epochs':epochs,'old_weight':.25,
                         'pretrain_lr':.001,'finetune_lr':.0003,'device':device,'seeds':seeds},
        'fold_audit':audit,'training_runs':runs},indent=2))
    print(frame.groupby(['method','seed']).ape.mean().unstack('seed').round(3).to_string())


if __name__=='__main__':
    main()
