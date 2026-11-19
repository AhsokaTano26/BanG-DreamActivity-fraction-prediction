"""Controlled Temporal V2 paired-feature, dual-TCN and phase-weight experiments."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.backtest import ROOT
from analysis.holdout_dataset import SPLITS, sha256, write_json
from analysis.multirank_dataset import identity
from analysis.multirank_model import event_weights, make_preprocessor, model_prediction
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_evaluate import aligned_frame, measured, verify_predictions
from analysis.temporal_rank_subset import DEFAULT
from analysis.temporal_paired_features import prepare_paired, phase_weights, sequence_arrays, SEQUENCE_COLUMNS
from analysis.temporal_paired_network import train_seed, infer, load_model

SEEDS = [42, 123, 2026]
CANDIDATES = ['hold_current', 'linear', 'reference_catboost', 'catboost_paired', 'catboost_phase', 'dual_tcn_mean']
METHODS = CANDIDATES + [f'dual_tcn_seed{seed}' for seed in SEEDS]
SOURCES = ['training/analysis/temporal_paired_benchmark.py', 'training/analysis/temporal_paired_features.py',
    'training/analysis/temporal_paired_network.py', 'training/analysis/temporal_dataset.py',
    'training/analysis/temporal_evaluate.py', 'training/analysis/temporal_rank_subset.py',
    'training/analysis/multirank_model.py', 'training/analysis/multirank_dataset.py', 'training/analysis/backtest.py',
    'training/analysis/event_features.py', 'training/analysis/holdout_dataset.py', 'training/analysis/holdout_evaluate.py']


def decode(arrays, residual):
    return arrays['current']*np.exp(np.clip(-np.log(arrays['progress'])+residual, 0, 4))


def neural_preprocessor(frame):
    columns = [k for k in frame if k not in SEQUENCE_COLUMNS]
    cats = [k for k in columns if not pd.api.types.is_numeric_dtype(frame[k]) and frame[k].notna().any()]
    nums = [k for k in columns if k not in cats]
    return ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), nums),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)])


def neural_values(arrays, preprocessor):
    return {'sequence': sequence_arrays(arrays['paired']),
            'tab': np.asarray(preprocessor.transform(arrays['paired']), dtype=np.float32), 'rank': arrays['rank']}


def prediction_frame(arrays, method, prediction):
    if len(prediction) != len(arrays['cases']) or not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError('Invalid final score prediction')
    return pd.DataFrame({'case_id': [c['case_id'] for c in arrays['cases']], 'method': method, 'predicted': prediction})


def reference_frame(reference, split, ids):
    frame = pd.read_csv(Path(reference)/f'{split}_predictions.csv')
    frame = frame[frame.method.isin(['hold_current', 'linear', 'catboost_rank'])].copy()
    for method in ['hold_current', 'linear', 'catboost_rank']:
        part = frame[frame.method == method]
        if set(part.case_id) != set(ids) or part.case_id.duplicated().any():
            raise ValueError('Reference case set differs')
    frame['method'] = frame.method.replace({'catboost_rank': 'reference_catboost'})
    return frame[['case_id', 'method', 'predicted']]


def save_features(run, split, arrays):
    ids = [c['case_id'] for c in arrays['cases']]
    for name in ['single', 'paired']:
        frame = arrays[name].copy(); frame.insert(0, 'case_id', ids)
        frame.to_csv(run/f'features/{split}_{name}.csv.gz', index=False)
    arrays['audit'].to_csv(run/f'features/{split}_pair_audit.csv', index=False)


def run_experiment(data, reference, run, *, iterations=600, max_epochs=120):
    run, reference = Path(run), Path(reference)
    if run.exists():
        raise FileExistsError(run)
    if iterations < 1 or max_epochs < 1 or data.manifest['protocol']['ranks'] != [1000, 2000]:
        raise ValueError('Invalid training configuration')
    previous = verify_predictions(reference)
    if previous['dataset_id'] != data.manifest['dataset_id']:
        raise ValueError('Reference dataset differs')
    config = {'dataset_id': data.manifest['dataset_id'], 'dataset_directory': str(data.root.resolve()),
        'reference_directory': str(reference.resolve()), 'reference_blind_manifest_sha256': sha256(reference/'blind_manifest.json'),
        'methods': METHODS, 'candidates': CANDIDATES, 'seeds': SEEDS,
        'tree': {'iterations': iterations, 'depth': 5, 'learning_rate': .035, 'l2_leaf_reg': 12, 'seed': 42},
        'neural': {'max_epochs': max_epochs, 'min_epochs': 30, 'patience': 20, 'batch': 128,
            'width': 24, 'dilations': [1, 2, 4], 'sequence_summary': 'last state plus full-prefix mean', 'tab_width': 32, 'dropout': .1,
            'optimizer': 'AdamW', 'lr': .001, 'weight_decay': .005, 'gradient_clip': 1, 'device': 'cpu', 'threads': 2},
        'target_and_loss': 'event-weighted squared linear-relative log-growth residual; existing clipped final-score decoder',
        'input': 'same joint feature frame for CatBoost and TCN; 33 own/peer curve values and explicit masks; other numeric/category features',
        'pairing': 'same split/event/requested cutoff only; peer truncated at primary observed time; invalid peer missing; no row removal',
        'phase_control': 'single-line features unchanged; within-event equal totals for available observed-progress bins [0,.25),[.25,.9),[.9,1]',
        'selection': 'validation event-macro MAPE only; no single seed selection; arithmetic mean of final scores; no refit',
        'python': platform.python_version(),
        'software': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'joblib', 'torch']},
        'source_hashes': {n: sha256(ROOT/n) for n in SOURCES}}
    config['run_id'] = identity(config)
    run.mkdir(parents=True)
    for folder in ['sources', 'models', 'features']:
        (run/folder).mkdir()
    write_json(run/'run_manifest.json', config)
    shutil.copyfile(reference/'blind_manifest.json', run/'reference_blind_manifest.json')
    for name in SOURCES:
        shutil.copyfile(ROOT/name, run/'sources'/Path(name).name)
    arrays = {s: prepare_paired(data.inputs(s)) for s in ['train', 'validation']}
    labels = {s: data.labels(s) for s in ['train', 'validation']}
    predictions = {}
    for split, a in arrays.items():
        save_features(run, split, a)
        predictions[split] = [reference_frame(reference, split, [c['case_id'] for c in a['cases']])]
    train, valid = arrays['train'], arrays['validation']
    events = [c['event'] for c in train['cases']]
    weights = event_weights(events); phase = phase_weights(events, train['progress'])
    pd.DataFrame({'case_id': [c['case_id'] for c in train['cases']], 'event': events,
        'progress': train['progress'], 'event_weight': weights, 'phase_weight': phase}).to_csv(run/'training_weights.csv', index=False)
    actual = np.array([labels['train'][c['case_id']] for c in train['cases']])
    target = np.log(actual/train['current'])+np.log(train['progress'])
    bundles = {}
    for method, features, fit_weights in [('catboost_paired', 'paired', weights), ('catboost_phase', 'single', phase)]:
        prep = make_preprocessor(train[features])
        x = prep.fit_transform(train[features])
        model = CatBoostRegressor(iterations=iterations, depth=5, learning_rate=.035, l2_leaf_reg=12,
            random_seed=42, thread_count=4, verbose=False, allow_writing_files=False)
        model.fit(x, target, sample_weight=fit_weights)
        bundle = {'model': model, 'preprocessor': prep, 'features': features}; bundles[method] = bundle
        joblib.dump(bundle, run/f'models/{method}.joblib')
        for split, a in arrays.items():
            predictions[split].append(prediction_frame(a, method, model_prediction(bundle, a)))
        print(f'Completed {method}', flush=True)
    prep = neural_preprocessor(train['paired']); prep.fit(train['paired'])
    joblib.dump(prep, run/'models/neural_preprocessor.joblib')
    nn_values = {s: neural_values(a, prep) for s, a in arrays.items()}
    validation_actual = np.array([labels['validation'][c['case_id']] for c in valid['cases']])
    _, validation_index, validation_counts = np.unique([c['event'] for c in valid['cases']], return_inverse=True, return_counts=True)
    def validation_score(residual):
        errors = np.abs(decode(valid, residual)/validation_actual-1)*100
        return float((np.bincount(validation_index, weights=errors)/validation_counts).mean())
    networks = {}; seed_predictions = {s: [] for s in arrays}
    for seed in SEEDS:
        method = f'dual_tcn_seed{seed}'
        model, trace = train_seed(nn_values['train'], nn_values['validation'], target, weights,
            validation_score, seed=seed, max_epochs=max_epochs, checkpoint=run/f'models/{method}.pt')
        networks[method] = model; write_json(run/f'models/{method}.json', trace)
        for split, a in arrays.items():
            p = decode(a, infer(model, nn_values[split])); seed_predictions[split].append(p)
            predictions[split].append(prediction_frame(a, method, p))
    for split, a in arrays.items():
        predictions[split].append(prediction_frame(a, 'dual_tcn_mean', np.mean(seed_predictions[split], axis=0)))
    validation = pd.concat(predictions['validation'], ignore_index=True)
    cases = data.case_index('validation')
    validation_metrics = {method: measured(aligned_frame(cases, labels['validation'], part))
                          for method, part in validation.groupby('method')}
    selected = min(CANDIDATES, key=lambda m: validation_metrics[m]['event_macro_mape_pct'])
    selection = {'selected_recipe': selected, 'candidates': CANDIDATES,
                 'criterion': 'validation event-macro MAPE', 'validation': validation_metrics,
                 'single_seeds_reported_only': True, 'refit': False, 'test_labels_read': False}
    write_json(run/'selection.json', selection)
    print('SELECTED '+selected+' '+str(validation_metrics[selected]['event_macro_mape_pct']), flush=True)
    # Training and selection are finished before opening test prefixes.
    test = prepare_paired(data.inputs('test')); save_features(run, 'test', test)
    predictions['test'] = [reference_frame(reference, 'test', [c['case_id'] for c in test['cases']])]
    for method, bundle in bundles.items():
        predictions['test'].append(prediction_frame(test, method, model_prediction(bundle, test)))
    values = neural_values(test, prep); test_seeds = []
    for method, model in networks.items():
        p = decode(test, infer(model, values)); test_seeds.append(p)
        predictions['test'].append(prediction_frame(test, method, p))
    predictions['test'].append(prediction_frame(test, 'dual_tcn_mean', np.mean(test_seeds, axis=0)))
    for split, frames in predictions.items():
        pd.concat(frames, ignore_index=True).to_csv(run/f'{split}_predictions.csv', index=False)
    seal = {k: config[k] for k in ['run_id', 'dataset_id', 'dataset_directory', 'methods']}
    seal['reference_method'] = selected
    seal['artifacts'] = {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*')) if p.is_file()}
    write_json(run/'blind_manifest.json', seal)
    return selection


def predict_saved(run, samples, method):
    run = Path(run); seal = verify_predictions(run)
    config = json.loads((run/'run_manifest.json').read_text())
    for name, digest in config['source_hashes'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Prediction source changed: '+name)
    if platform.python_version() != config['python'] or any(importlib.metadata.version(n) != v for n, v in config['software'].items()):
        raise ValueError('Prediction environment changed')
    if method not in seal['methods']:
        raise ValueError('Unknown method')
    arrays = prepare_paired(samples)
    if method in ['catboost_paired', 'catboost_phase']:
        p = model_prediction(joblib.load(run/f'models/{method}.joblib'), arrays)
    elif method.startswith('dual_tcn_'):
        prep = joblib.load(run/'models/neural_preprocessor.joblib'); values = neural_values(arrays, prep)
        seeds = SEEDS if method == 'dual_tcn_mean' else [int(method.removeprefix('dual_tcn_seed'))]
        p = np.mean([decode(arrays, infer(load_model(run/f'models/dual_tcn_seed{s}.pt'), values)) for s in seeds], axis=0)
    else:
        raise ValueError('Reference methods are copied, not fitted by this entry point')
    return prediction_frame(arrays, method, p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT)
    parser.add_argument('--reference', type=Path, default=ROOT/'analysis/temporal_results/v2_t1000_t2000_full_20261112')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=600)
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    run_experiment(TemporalDataset(args.dataset), args.reference, args.run, iterations=args.iterations, max_epochs=args.epochs)
