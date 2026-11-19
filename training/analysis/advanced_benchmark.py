"""Chronological comparison of tree and task-adapted neural forecasters.

Run: .venv/bin/python -m analysis.advanced_benchmark
Development data alone chooses neural epoch counts and ensemble membership.
Four expanding-window test blocks use identical available training events.
No event's labels enter the model used to predict it. Completed earlier
test blocks become training history for later blocks; current/later labels
never enter fitting, early stopping, preprocessing, or selection.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import torch
from torch.nn import functional as F

from analysis.backtest import ROOT, CAT, STAGES, load_curves, make_samples, snapshot, templates
from analysis.advanced_models import ARCHITECTURES, Forecaster, prefix_sequence

OUT = ROOT / 'analysis' / 'advanced_results'
TREE_METHODS = ['xgb_tab_growth', 'xgb_sequence_residual', 'catboost_sequence_residual',
                'lightgbm_sequence_residual', 'extra_trees_sequence_residual']


def training_samples(curves):
    samples = []
    for curve in curves:
        cutoffs = [curve.start + float(p) * curve.duration for p in np.arange(.25, .951, .05)]
        cutoffs += [curve.end - h * 3600000 for h in [24, 6]]
        seen = set()
        for cutoff in cutoffs:
            value = snapshot(curve, cutoff)
            if value is None or value['now'] in seen:
                continue
            seen.add(value['now'])
            value['target'] = np.log(max(1., curve.final / value['score']))
            samples.append(value)
    return samples


def prepare_arrays(train, evaluation):
    train_frame = pd.DataFrame([s['features'] for s in train])
    test_frame = pd.DataFrame([s['features'] for s in evaluation])
    numerical = [column for column in train_frame if column not in CAT]
    prep = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), numerical),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT),
    ])
    train_tab = prep.fit_transform(train_frame).astype(np.float32)
    test_tab = prep.transform(test_frame).astype(np.float32)
    train_sequence = np.stack([prefix_sequence(s) for s in train])
    test_sequence = np.stack([prefix_sequence(s) for s in evaluation])
    train_base = -np.log(np.array([s['progress'] for s in train], dtype=np.float32))
    test_base = -np.log(np.array([s['progress'] for s in evaluation], dtype=np.float32))
    train_growth = np.array([s['target'] for s in train], dtype=np.float32)
    # Evaluation labels deliberately do not leave the evaluation/reporting layer.
    return {'train_tab': train_tab, 'test_tab': test_tab,
            'train_seq': train_sequence, 'test_seq': test_sequence,
            'train_growth': train_growth, 'train_y': train_growth - train_base,
            'test_base': test_base}


def device_sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def neural_fit(architecture, arrays, *, seed, device, max_epochs, batch_size,
               validation_growth=None):
    torch.manual_seed(seed)
    if device == 'mps':
        torch.mps.manual_seed(seed)
    model = Forecaster(architecture, arrays['train_tab'].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    train_seq = torch.from_numpy(arrays['train_seq']).to(device)
    train_tab = torch.from_numpy(arrays['train_tab']).to(device)
    train_y = torch.from_numpy(arrays['train_y']).to(device)
    test_seq = torch.from_numpy(arrays['test_seq']).to(device)
    test_tab = torch.from_numpy(arrays['test_tab']).to(device)
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state = float('inf'), max_epochs, None
    trace = []
    device_sync(device);started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        indices = torch.randperm(len(train_y), generator=generator).to(device)
        losses = []
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(train_seq[batch], train_tab[batch])
            loss = F.smooth_l1_loss(pred, train_y[batch], beta=.2)
            if not torch.isfinite(loss):
                raise ValueError(f'{architecture}: nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.detach())
        if validation_growth is not None:
            model.eval()
            with torch.no_grad():
                residual = predict_tensor(model, test_seq, test_tab, batch_size)
            predicted_growth = np.clip(arrays['test_base'] + residual, 0, 4)
            score = float(np.mean(abs(np.exp(predicted_growth - validation_growth) - 1)))
            trace.append({'epoch': epoch, 'train_loss': float(torch.stack(losses).mean().item()),
                          'validation_mape': score * 100})
            if score < best:
                best, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch >= 25 and epoch - best_epoch >= 18:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = predict_tensor(model, test_seq, test_tab, batch_size)
    device_sync(device)
    record = {'architecture': architecture, 'seed': seed, 'device': device,
              'parameters': sum(p.numel() for p in model.parameters()),
              'epochs_run': epoch, 'selected_epochs': best_epoch,
              'seconds': time.perf_counter() - started, 'trace': trace}
    del model, optimizer, train_seq, train_tab, train_y, test_seq, test_tab
    if device == 'mps':
        torch.mps.empty_cache()
    return prediction, record


def predict_tensor(model, sequence, tab, batch_size):
    return np.concatenate([model(sequence[start:start + batch_size], tab[start:start + batch_size])
                           .detach().cpu().numpy() for start in range(0, len(tab), batch_size)])


def tree_fit(name, arrays):
    if name == 'xgb_tab_growth':
        model = XGBRegressor(n_estimators=180, max_depth=2, learning_rate=.04,
            min_child_weight=12, reg_lambda=10, subsample=.9, colsample_bytree=.9,
            random_state=42, n_jobs=4)
        train, test, target = arrays['train_tab'], arrays['test_tab'], arrays['train_growth']
    else:
        train = np.concatenate([arrays['train_tab'], arrays['train_seq'].reshape(len(arrays['train_tab']), -1)], axis=1)
        test = np.concatenate([arrays['test_tab'], arrays['test_seq'].reshape(len(arrays['test_tab']), -1)], axis=1)
        target = arrays['train_y']
        if name == 'xgb_sequence_residual':
            model = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=.03,
                min_child_weight=12, reg_lambda=10, subsample=.9, colsample_bytree=.8,
                random_state=42, n_jobs=4)
        elif name == 'catboost_sequence_residual':
            model = CatBoostRegressor(iterations=500, depth=4, learning_rate=.03,
                l2_leaf_reg=10, random_seed=42, thread_count=4, verbose=False,
                allow_writing_files=False)
        elif name == 'lightgbm_sequence_residual':
            model = LGBMRegressor(n_estimators=500, max_depth=4, num_leaves=15,
                learning_rate=.03, min_child_samples=30, reg_lambda=10,
                random_state=42, n_jobs=4, verbosity=-1)
        else:
            model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                max_features=.8, max_depth=12, random_state=42, n_jobs=4)
    started = time.perf_counter();model.fit(train, target)
    values = model.predict(test)
    if name == 'xgb_tab_growth':
        values = values - arrays['test_base']
    return values, {'architecture': name, 'seed': 42, 'seconds': time.perf_counter() - started}


def cache_run(key, signature, action):
    directory = OUT / 'runs';directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / f'{key}.json';values_path = directory / f'{key}.npy'
    if metadata_path.exists() and values_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('signature') == signature:
            return np.load(values_path), metadata
    values, metadata = action()
    if not np.isfinite(values).all():
        raise ValueError(f'{key}: nonfinite predictions')
    metadata.update({'signature': signature, 'key': key})
    np.save(values_path, values)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'completed': key, 'seconds': round(metadata['seconds'], 2),
                      'selected_epochs': metadata.get('selected_epochs')}) ,flush=True)
    return values, metadata


def prediction_rows(samples, residual, base, *, phase, method, seed, fold):
    growth = np.clip(base + residual, 0, 4)
    return [{'phase': phase, 'method': method, 'seed': seed, 'fold': fold,
             'event': s['curve'].event, 'rank': s['curve'].rank, 'stage': s['stage'],
             'event_type': s['curve'].kind, 'cutoff': s['cutoff'], 'observed_at': s['now'],
             'current': s['score'], 'actual': s['curve'].final,
             'predicted': float(s['score'] * np.exp(value))}
            for s, value in zip(samples, growth)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu','mps','auto'], default='mps')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,123,2026])
    parser.add_argument('--folds', type=int, default=4)
    args = parser.parse_args()
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    curves, excluded = load_curves();evaluation, unavailable = make_samples(curves)
    starts = {c.event: c.start for c in curves}
    pilot_start = starts[266]
    pilot_curves = [c for c in curves if c.end + 15 * 60000 < pilot_start]
    development = [s for s in evaluation if 266 <= s['curve'].event <= 291]
    assert all(s['curve'].start >= pilot_start and s['curve'].end < starts[292] for s in development)
    pilot_train = training_samples(pilot_curves)
    arrays = prepare_arrays(pilot_train, development)
    signature = hashlib.sha256((str(vars(args)) + ''.join(Path(p).read_text() for p in
        [__file__, ROOT/'training/analysis/advanced_models.py', ROOT/'training/analysis/backtest.py'])
        + hashlib.sha256((ROOT/'data/db.sqlite3').read_bytes()).hexdigest()).encode()).hexdigest()
    rows, records, selected_epochs, development_errors = [], [], {}, {}
    development_growth = np.array([s['target'] for s in development])
    print(json.dumps({'phase':'development','training_events':len({c.event for c in pilot_curves}),
                      'training_snapshots':len(pilot_train),'validation_snapshots':len(development),
                      'device':args.device}),flush=True)
    for name in TREE_METHODS + ARCHITECTURES:
        if name in TREE_METHODS:
            result, record = cache_run('development-'+name,signature,lambda n=name:tree_fit(n, arrays))
        else:
            result, record = cache_run('development-'+name, signature,
                lambda n=name:neural_fit(n,arrays,seed=42,device=args.device,max_epochs=args.epochs,
                    batch_size=args.batch_size,validation_growth=development_growth))
            selected_epochs[name] = record['selected_epochs']
        records.append(record)
        development_errors[name] = float(np.mean(abs(np.exp(np.clip(arrays['test_base']+result,0,4)-development_growth)-1))*100)
        rows.extend(prediction_rows(development,result,arrays['test_base'],phase='development',method=name,seed=42,fold=-1))
    ensemble_members = sorted(development_errors,key=development_errors.get)[:3]
    selection = {'epochs':selected_epochs,'development_mape':development_errors,'ensemble_members':ensemble_members}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    print('SELECTION '+json.dumps(selection),flush=True)
    test_ids = sorted({c.event for c in curves if 292 <= c.event <= 327},key=starts.get)
    audit=[]
    for fold, test_events in enumerate(np.array_split(test_ids,args.folds)):
        boundary=min(starts[int(event)] for event in test_events)
        train_curves=[c for c in curves if c.end+15*60000<boundary]
        assert set(test_events).isdisjoint(c.event for c in train_curves)
        train=training_samples(train_curves)
        test=[s for s in evaluation if s['curve'].event in set(test_events)]
        arrays=prepare_arrays(train,test)
        audit.append({'fold':fold,'training_events':sorted({c.event for c in train_curves}),
            'training_latest_end':max(c.end for c in train_curves),'test_earliest_start':boundary,
            'test_events':[int(e) for e in test_events],'training_snapshots':len(train),'test_snapshots':len(test)})
        fold_predictions={}
        for name in TREE_METHODS + ARCHITECTURES:
            predictions=[]
            for seed in ([42] if name in TREE_METHODS else args.seeds):
                key=f'test-fold{fold}-{name}-seed{seed}'
                if name in TREE_METHODS:
                    result,record=cache_run(key,signature,lambda n=name:tree_fit(n,arrays))
                else:
                    result,record=cache_run(key,signature,lambda n=name,z=seed:neural_fit(n,arrays,
                        seed=z,device=args.device,max_epochs=selected_epochs[n],batch_size=args.batch_size))
                predictions.append(result);records.append(record)
                rows.extend(prediction_rows(test,result,arrays['test_base'],phase='test',method=name,seed=seed,fold=fold))
            fold_predictions[name]=np.mean(predictions,axis=0)
            if name in ARCHITECTURES:
                rows.extend(prediction_rows(test,fold_predictions[name],arrays['test_base'],phase='test',
                    method=name+f'_ensemble{len(args.seeds)}',seed=-1,fold=fold))
        blend=np.mean([fold_predictions[name] for name in ensemble_members],axis=0)
        rows.extend(prediction_rows(test,blend,arrays['test_base'],phase='test',method='validation_top3_ensemble',seed=-1,fold=fold))
        for sample in test:
            for name,predicted in [('linear',sample['score']/sample['progress']),
                                   ('shape_analog',templates(sample,train_curves)[2])]:
                row=prediction_rows([sample],np.array([0.]),np.array([0.]),phase='test',method=name,seed=42,fold=fold)[0]
                row['predicted']=float(predicted);rows.append(row)
        pd.DataFrame(rows).to_csv(OUT/'predictions.csv',index=False)
        print(f'FOLD {fold} COMPLETE',flush=True)
    frame=pd.DataFrame(rows);frame['ape']=abs(frame.predicted/frame.actual-1)*100
    metrics=frame.groupby(['phase','method','seed','rank','stage']).ape.agg(['count','mean','median',lambda x:x.quantile(.9)]).reset_index()
    metrics.columns=['phase','method','seed','rank','stage','n','mape','median_ape','p90_ape']
    metrics.to_csv(OUT/'metrics.csv',index=False)
    report={'created_at':datetime.now(timezone.utc).isoformat(),'configuration':vars(args),
        'signature':signature,'database_sha256':hashlib.sha256((ROOT/'data/db.sqlite3').read_bytes()).hexdigest(),
        'selection':selection,'fold_audit':audit,'excluded':excluded,'unavailable_snapshots':unavailable,
        'training_runs':records,'sources':{
            'patchtst':'https://arxiv.org/abs/2211.14730','itransformer':'https://arxiv.org/abs/2310.06625',
            'tsmixer':'https://arxiv.org/abs/2303.06053','dlinear':'https://arxiv.org/abs/2205.13504'}}
    (OUT/'benchmark.json').write_text(json.dumps(report,indent=2))
    test_frame=frame[frame.phase=='test']
    print(test_frame.groupby(['method','seed']).ape.mean().unstack('seed').round(3).to_string())


if __name__=='__main__':
    main()
