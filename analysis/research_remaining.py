"""Fixed remaining-growth recipes; development reads train and validation only.

The rate target is log((final/current - 1) * p/(1-p)): the future
average score rate relative to the observed average rate. Its decoder has
an explicit zero-time-to-end boundary. The direct final/current target
with CatBoost MAPE optimizes the requested case-relative error exactly
before the monotone-score projection, using equal total activity weights.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256, write_json
from analysis.multirank_model import event_weights, make_preprocessor
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_paired_features import prepare_paired
from analysis.temporal_rank_subset import DEFAULT


SOURCES = ['analysis/research_remaining.py', 'analysis/temporal_paired_features.py',
           'analysis/multirank_model.py', 'analysis/backtest.py',
           'analysis/holdout_dataset.py', 'analysis/multirank_dataset.py',
           'analysis/temporal_dataset.py', 'analysis/temporal_rank_subset.py']
LIMITATIONS = [
    'The test cohort appeared in earlier exploratory experiments; it is not a wholly unseen blind test.',
    'Metadata is a current retrospective snapshot without historical point-in-time archives.',
    'This development run reads no test answers or test predictions and performs no test inference.',
]


def recipes():
    """Six predeclared recipes; no validation-dependent fitting or early stopping."""
    common = {'features': 'paired', 'iterations': 650, 'depth': 5,
              'learning_rate': .035, 'l2_leaf_reg': 12, 'seed': 42,
              'thread_count': 2, 'separate_ranks': False}
    variants = [
        ('remaining_baseline_log_rmse', 'linear_log_residual', 'RMSE', False),
        ('remaining_rate_log_rmse', 'remaining_log_rate', 'RMSE', False),
        ('remaining_rate_log_mae', 'remaining_log_rate', 'MAE', False),
        ('remaining_ratio_mape', 'final_ratio', 'MAPE', False),
        ('remaining_ratio_mape_by_rank', 'final_ratio', 'MAPE', True),
        ('remaining_rate_log_rmse_by_rank', 'remaining_log_rate', 'RMSE', True),
    ]
    return [common | {'name': name, 'target': target, 'loss': loss, 'separate_ranks': separate}
            for name, target, loss, separate in variants]


def _coordinates(current, progress):
    current, progress = np.asarray(current, dtype=float), np.asarray(progress, dtype=float)
    if (current.ndim != 1 or current.shape != progress.shape or not len(current)
            or not np.isfinite(current).all() or not np.isfinite(progress).all()
            or np.any(current <= 0) or np.any((progress <= 0) | (progress > 1))):
        raise ValueError('Expected positive finite scores and progress in (0, 1]')
    return current, progress


def encode_target(target, actual, current, progress):
    current, progress = _coordinates(current, progress)
    actual = np.asarray(actual, dtype=float)
    if actual.shape != current.shape or not np.isfinite(actual).all() or np.any(actual <= 0):
        raise ValueError('Expected aligned positive training answers')
    if target == 'linear_log_residual':
        return np.log(actual / current) + np.log(progress)
    if target == 'final_ratio':
        # CatBoost MAPE divides by max(1, |target|). Frozen monotone curves
        # have final/current >= 1, giving exactly |predicted-final|/final.
        if np.any(actual < current - np.maximum(current, 1) * 1.e-10):
            raise ValueError('A final score precedes its observed prefix score')
        return actual / current
    if target == 'remaining_log_rate':
        if np.any(progress >= 1):
            raise ValueError('Cannot fit a remaining rate at/after event end')
        growth = np.maximum(actual / current - 1, 1.e-8)
        return np.log(growth) + np.log(progress) - np.log1p(-progress)
    raise ValueError('Unknown target: ' + str(target))


def decode_target(target, prediction, current, progress):
    current, progress = _coordinates(current, progress)
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != current.shape or not np.isfinite(prediction).all():
        raise ValueError('Expected aligned finite target predictions')
    if target == 'remaining_log_rate':
        growth = np.exp(np.clip(prediction, -20, 20)) * ((1 - progress) / progress)
        ratio = 1 + growth
    elif target == 'linear_log_residual':
        ratio = np.exp(np.clip(prediction - np.log(progress), 0, 20))
    elif target == 'final_ratio':
        ratio = np.maximum(prediction, 1)
    else:
        raise ValueError('Unknown target: ' + str(target))
    result = current * ratio
    if not np.isfinite(result).all():
        raise ValueError('Decoded final score overflow')
    return np.maximum(result, current)


def _check_arrays(arrays):
    current, progress = _coordinates(arrays['current'], arrays['progress'])
    cases, rank = arrays['cases'], np.asarray(arrays['rank'])
    if (len(cases) != len(current) or rank.shape != current.shape
            or not np.isin(rank, [0, 1]).all()
            or len({c['case_id'] for c in cases}) != len(cases)):
        raise ValueError('Invalid, repeated, or misaligned case arrays')
    return current, progress, rank


def _frame(arrays, feature_key, columns=None, categoricals=None):
    frame = arrays[feature_key].copy().reset_index(drop=True)
    if len(frame) != len(arrays['cases']) or frame.columns.duplicated().any():
        raise ValueError('Features do not preserve all cases')
    if columns is not None:
        if set(columns) - set(frame.columns):
            raise ValueError('A fitted feature is missing')
        frame = frame[list(columns)]
    if categoricals is None:
        categoricals = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])
                        and frame[c].notna().any()]
    for column in frame:
        if column in categoricals:
            frame[column] = frame[column].fillna('<missing>').astype(str)
        else:
            frame[column] = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return frame, list(categoricals)


def _actual(labels, cases):
    if isinstance(labels, Mapping):
        missing = {c['case_id'] for c in cases} - set(labels)
        if missing:
            raise ValueError('Training labels missing cases')
        return np.asarray([labels[c['case_id']] for c in cases], dtype=float)
    actual = np.asarray(labels, dtype=float)
    if actual.shape != (len(cases),):
        raise ValueError('Training label shape mismatch')
    return actual


def fit_recipe(config, train_arrays, train_labels):
    """Fit only on the supplied training rows; no validation data is accepted."""
    config = dict(config)
    current, progress, rank = _check_arrays(train_arrays)
    actual = _actual(train_labels, train_arrays['cases'])
    target = encode_target(config['target'], actual, current, progress)
    frame, categoricals = _frame(train_arrays, config['features'])
    preprocessor = make_preprocessor(frame)
    x = preprocessor.fit_transform(frame)
    weights = event_weights([c['event'] for c in train_arrays['cases']])
    if (int(config['iterations']) < 1 or int(config['thread_count']) not in (1, 2)):
        raise ValueError('Positive iterations and one or two CPU threads are required')
    models = {}
    groups = (0, 1) if config['separate_ranks'] else ('shared',)
    for group in groups:
        mask = rank == group if group != 'shared' else np.ones(len(rank), dtype=bool)
        if not mask.any():
            raise ValueError('A fitted rank has no training cases')
        model = CatBoostRegressor(
            iterations=int(config['iterations']), depth=int(config['depth']),
            learning_rate=float(config['learning_rate']), l2_leaf_reg=float(config['l2_leaf_reg']),
            loss_function=config['loss'], random_seed=int(config['seed']),
            thread_count=int(config['thread_count']), verbose=False, allow_writing_files=False)
        model.fit(x[mask], target[mask], sample_weight=weights[mask])
        models[group] = model
    return {'config': config, 'preprocessor': preprocessor, 'models': models,
            'feature_columns': list(frame), 'categoricals': categoricals,
            'training_case_ids': [c['case_id'] for c in train_arrays['cases']],
            'training_weights': weights,
            'target_training_range': [float(target.min()), float(target.max())]}


def predict_recipe(bundle, arrays):
    """Return one positive final score in the caller's unchanged case order."""
    current, progress, rank = _check_arrays(arrays)
    config = bundle['config']
    frame, _ = _frame(arrays, config['features'], bundle['feature_columns'], bundle['categoricals'])
    x = bundle['preprocessor'].transform(frame)
    prediction = np.empty(len(current), dtype=float)
    for group, model in bundle['models'].items():
        mask = rank == group if group != 'shared' else np.ones(len(rank), dtype=bool)
        if mask.any():
            prediction[mask] = model.predict(x[mask])
    return decode_target(config['target'], prediction, current, progress)


def validation_metrics(arrays, labels, prediction):
    actual = _actual(labels, arrays['cases'])
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != actual.shape or not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError('Invalid predictions')
    error = np.abs(prediction - actual)
    ape = error / actual * 100
    table = pd.DataFrame({'event': [c['event'] for c in arrays['cases']], 'ape': ape})
    return {'event_macro_mape_pct': float(table.groupby('event').ape.mean().mean()),
            'case_mape_pct': float(ape.mean()), 'mae': float(error.mean()),
            'rmse': float(np.sqrt(np.mean(error ** 2))), 'p90_ape_pct': float(np.percentile(ape, 90)),
            'events': int(table.event.nunique()), 'cases': len(table)}


def run_development(data, output):
    """Execute a bounded train/validation search, saving all recipes and fits."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    configs = recipes()
    train = prepare_paired(data.inputs('train'))
    valid = prepare_paired(data.inputs('validation'))
    train_labels, valid_labels = data.labels('train'), data.labels('validation')
    output.mkdir(parents=True)
    (output / 'models').mkdir()
    (output / 'sources').mkdir()
    manifest = {'dataset_id': data.manifest['dataset_id'], 'dataset_directory': str(data.root.resolve()),
                'recipes': configs, 'selection': 'validation event-macro MAPE; no refit',
                'weights': 'equal total weight per training event across all prefixes and ranks',
                'splits_read_for_modeling': ['train', 'validation'], 'test_inference': False,
                'limitations': LIMITATIONS, 'python': platform.python_version(),
                'software': {n: importlib.metadata.version(n) for n in
                             ['numpy', 'pandas', 'scikit-learn', 'catboost', 'joblib']},
                'source_hashes': {source: sha256(ROOT / source) for source in SOURCES}}
    write_json(output / 'run_manifest.json', manifest)
    for source in SOURCES:
        shutil.copyfile(ROOT / source, output / 'sources' / Path(source).name)
    pd.DataFrame(train['cases']).assign(weight=event_weights([c['event'] for c in train['cases']])).to_csv(
        output / 'training_weights.csv', index=False)
    train['audit'].to_csv(output / 'train_pairing_audit.csv', index=False)
    valid['audit'].to_csv(output / 'validation_pairing_audit.csv', index=False)
    results, predictions, model_paths = {}, [], {}
    for config in configs:
        name = config['name']
        bundle = fit_recipe(config, train, train_labels)
        model_path = output / 'models' / (name + '.joblib')
        joblib.dump(bundle, model_path)
        prediction = predict_recipe(bundle, valid)
        replay = predict_recipe(joblib.load(model_path), valid)
        if not np.array_equal(prediction, replay):
            raise ValueError('Persisted model replay changed predictions')
        results[name] = validation_metrics(valid, valid_labels, prediction)
        model_paths[name] = str(model_path.resolve())
        predictions.extend(c | {'method': name, 'predicted': float(p)}
                           for c, p in zip(valid['cases'], prediction))
        print(json.dumps({'method': name, 'validation': results[name]}, ensure_ascii=False), flush=True)
    pd.DataFrame(predictions).to_csv(output / 'validation_predictions.csv', index=False)
    pd.DataFrame([{'method': k, **v} for k, v in results.items()]).to_csv(output / 'validation_metrics.csv', index=False)
    selected = min(results, key=lambda k: results[k]['event_macro_mape_pct'])
    record = {'selected_recipe': selected, 'validation': results, 'model_paths': model_paths,
              'test_answers_read': False, 'test_inference': False, 'refit_after_selection': False,
              'limitations': LIMITATIONS}
    write_json(output / 'selection.json', record)
    write_json(output / 'development_manifest.json', {
        'dataset_id': data.manifest['dataset_id'], 'completed': True,
        'artifacts': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run_development(TemporalDataset(args.dataset), args.run)
