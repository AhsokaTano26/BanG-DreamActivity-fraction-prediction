"""Train fixed recipes on all Temporal V2 prefixes, select on validation only."""
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

from analysis.backtest import ROOT
from analysis.holdout_dataset import SPLITS, sha256, write_json
from analysis.multirank_dataset import identity
from analysis.multirank_model import (event_weights, make_preprocessor, metadata_for,
                                      model_prediction, prepare, records)
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_evaluate import score_temporal, verify_predictions
from analysis.temporal_rank_subset import DEFAULT

CANDIDATES = ['hold_current', 'linear', 'catboost_rank', 'catboost_context']
FROZEN_METHODS = ['frozen_catboost_rank', 'frozen_catboost_context']
SOURCES = ['training/analysis/temporal_train.py', 'training/analysis/temporal_rank_subset.py',
           'training/analysis/temporal_dataset.py', 'training/analysis/temporal_evaluate.py',
           'training/analysis/multirank_model.py', 'training/analysis/multirank_dataset.py',
           'training/analysis/backtest.py', 'training/analysis/event_features.py',
           'training/analysis/holdout_dataset.py', 'training/analysis/holdout_evaluate.py']


def reference_predictions(data, reference_run):
    """Copy existing blind predictions on the exact parent cases, never its scores."""
    reference_run = Path(reference_run)
    seal = verify_predictions(reference_run)
    if seal['dataset_id'] != data.manifest.get('parent_dataset_id'):
        raise ValueError('Reference predictions belong to a different parent dataset')
    if not set(FROZEN_METHODS).issubset(seal['methods']):
        raise ValueError('Missing frozen reference methods')
    result = {}
    for split in SPLITS:
        ids = set(data.case_index(split).case_id)
        frame = pd.read_csv(reference_run / f'{split}_predictions.csv')
        frame = frame[frame.case_id.isin(ids) & frame.method.isin(FROZEN_METHODS)]
        for method in FROZEN_METHODS:
            part = frame[frame.method == method]
            if (set(part.case_id) != ids or part.case_id.duplicated().any()
                    or not np.isfinite(part.predicted).all() or (part.predicted <= 0).any()):
                raise ValueError('Frozen predictions do not match the selected cases')
        result[split] = frame[['case_id', 'method', 'predicted']]
    return result


def run_training(data, run, *, iterations=600, reference_run=None):
    run = Path(run)
    if run.exists():
        raise FileExistsError(run)
    if iterations < 1:
        raise ValueError('iterations must be positive')
    frozen = reference_predictions(data, reference_run) if reference_run is not None else {}
    methods = CANDIDATES + (FROZEN_METHODS if frozen else [])
    config = {'dataset_directory': str(data.root.resolve()), 'dataset_id': data.manifest['dataset_id'],
        'protocol': 'temporal_holdout_v2', 'ranks': data.manifest['protocol']['ranks'],
        'methods': methods, 'candidates': CANDIDATES, 'iterations': iterations,
        'tree_depth': 5, 'learning_rate': .035, 'l2_leaf_reg': 12, 'seed': 42,
        'weights': 'equal total weight per training activity across all its available prefixes',
        'target': 'log(final/current) + log(observed progress)',
        'selection': 'minimum validation event-macro MAPE among candidates; no refit',
        'reference_run': str(Path(reference_run).resolve()) if reference_run else None,
        'reference_blind_manifest_sha256': sha256(Path(reference_run) / 'blind_manifest.json') if reference_run else None,
        'reference_policy': 'descriptive frozen 19-rank comparison; excluded from model selection',
        'python': platform.python_version(),
        'software': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'joblib']},
        'source_hashes': {n: sha256(ROOT / n) for n in SOURCES}}
    config['run_id'] = identity(config)
    run.mkdir(parents=True)
    (run / 'sources').mkdir()
    (run / 'models').mkdir()
    write_json(run / 'run_manifest.json', config)
    for name in SOURCES:
        shutil.copyfile(ROOT / name, run / 'sources' / Path(name).name)
    if reference_run:
        shutil.copyfile(Path(reference_run) / 'blind_manifest.json', run / 'reference_blind_manifest.json')
    metadata = metadata_for(data)
    joblib.dump(metadata, run / 'models/metadata.joblib')
    train = prepare(data.inputs('train'), metadata)
    valid = prepare(data.inputs('validation'), metadata)
    training_labels = data.labels('train')
    validation_labels = data.labels('validation')
    actual = np.array([training_labels[c['case_id']] for c in train['cases']])
    target = np.log(actual / train['current']) + np.log(train['progress'])
    weights = event_weights([c['event'] for c in train['cases']])
    pd.DataFrame({'case_id': [c['case_id'] for c in train['cases']],
                  'event': [c['event'] for c in train['cases']], 'weight': weights}).to_csv(run / 'training_weights.csv', index=False)
    bundles = {}
    predictions = {'train': [], 'validation': []}
    validation_results = {}
    cases, members = data.case_index('validation'), data.memberships('validation')
    for method in CANDIDATES:
        if method == 'hold_current':
            tp, vp = train['current'], valid['current']
        elif method == 'linear':
            tp, vp = train['current'] / train['progress'], valid['current'] / valid['progress']
        else:
            features = 'context' if method.endswith('context') else 'rank'
            prep = make_preprocessor(train[features])
            x = prep.fit_transform(train[features])
            model = CatBoostRegressor(iterations=iterations, depth=5, learning_rate=.035,
                l2_leaf_reg=12, random_seed=42, thread_count=4, verbose=False, allow_writing_files=False)
            model.fit(x, target, sample_weight=weights)
            bundle = {'model': model, 'preprocessor': prep, 'features': features}
            bundles[method] = bundle
            joblib.dump(bundle, run / f'models/{method}.joblib')
            tp, vp = model_prediction(bundle, train), model_prediction(bundle, valid)
        tr, vr = records(train, method, tp), records(valid, method, vp)
        predictions['train'].extend(tr)
        predictions['validation'].extend(vr)
        validation_results[method] = score_temporal(cases, members, validation_labels, pd.DataFrame(vr))
        print(json.dumps({'method': method, 'validation': validation_results[method]['overall']}, ensure_ascii=False), flush=True)
    selected = min(CANDIDATES, key=lambda m: validation_results[m]['overall']['event_macro_mape_pct'])
    selection = {'selected_recipe': selected, 'candidates': CANDIDATES,
        'criterion': 'validation event-macro MAPE', 'validation': validation_results,
        'test_labels_read': False, 'refit_after_selection': False}
    write_json(run / 'selection.json', selection)
    # No fits or model selection occur beyond this point. Test supplies prefixes only.
    test = prepare(data.inputs('test'), metadata)
    predictions['test'] = []
    for method in CANDIDATES:
        if method == 'hold_current':
            prediction = test['current']
        elif method == 'linear':
            prediction = test['current'] / test['progress']
        else:
            prediction = model_prediction(bundles[method], test)
        predictions['test'].extend(records(test, method, prediction))
    for split in SPLITS:
        frame = pd.DataFrame(predictions[split])[['case_id', 'method', 'predicted']]
        if frozen:
            frame = pd.concat([frame, frozen[split]], ignore_index=True)
        frame.to_csv(run / f'{split}_predictions.csv', index=False)
    seal = {k: config[k] for k in ['run_id', 'dataset_id', 'dataset_directory', 'methods']}
    seal['reference_method'] = selected
    seal['artifacts'] = {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*')) if p.is_file()}
    write_json(run / 'blind_manifest.json', seal)
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=600)
    parser.add_argument('--reference-run', type=Path)
    args = parser.parse_args()
    run_training(TemporalDataset(args.dataset), args.run, iterations=args.iterations, reference_run=args.reference_run)
