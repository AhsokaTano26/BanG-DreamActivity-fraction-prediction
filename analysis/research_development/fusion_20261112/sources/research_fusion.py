"""Six predeclared train-only single/peer fusion recipes for Temporal V2.

The library API does not open a dataset or read any labels. The optional CLI is
development-only: it fits train prefixes and scores every recipe on validation.
No validation target fits a gate, calibrator, preprocessor, or model.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from analysis.multirank_model import event_weights


def recipes():
    """Return independent copies of the complete grid, fixed before validation."""
    base = dict(family='research_fusion', algorithm='catboost', loss='RMSE',
                iterations=800, depth=5, learning_rate=.035, l2_leaf_reg=12.,
                random_seed=42, threads=2, rank_heads=False, gate='progress',
                paired_weight=.5, progress_edges=[.25, .9],
                progress_weights=[.65, .45, .25], peer_age_scale_hours=3.,
                correction_limit_log=float(np.log(1.6)), missing_replica_weight=.25,
                n_estimators=500, min_samples_leaf=5, max_features=.75)
    changes = [
        dict(name='fusion_cat_rmse_constant', gate='constant'),
        dict(name='fusion_cat_rmse_progress'),
        dict(name='fusion_cat_mae_progress', loss='MAE'),
        dict(name='fusion_cat_huber_progress', loss='Huber:delta=0.15'),
        dict(name='fusion_cat_rank_progress', rank_heads=True),
        dict(name='fusion_extra_progress', algorithm='extra_trees'),
    ]
    return [deepcopy(base | change) for change in changes]


def _check_arrays(arrays, *, training=False):
    cases = arrays['cases']
    n = len(cases)
    if not n or len({c['case_id'] for c in cases}) != n:
        raise ValueError('Empty or duplicate case rows')
    if training and any(not c['case_id'].startswith('train:') for c in cases):
        raise ValueError('Fitting requires train prefixes only')
    for name in ['single', 'paired', 'current', 'progress', 'rank']:
        if len(arrays[name]) != n:
            raise ValueError('Unaligned case rows: ' + name)
    current, progress = np.asarray(arrays['current']), np.asarray(arrays['progress'])
    if (not np.isfinite(current).all() or np.any(current <= 0)
            or not np.isfinite(progress).all() or np.any((progress <= 0) | (progress > 1))
            or not np.isin(arrays['rank'], [0, 1]).all()):
        raise ValueError('Invalid current score, progress, or rank')


def _canonical_frame(frame, *, paired=False, mask_peer=False):
    frame = frame.copy().reset_index(drop=True)
    if paired:
        available = pd.to_numeric(frame.get('peer_available', pd.Series(0, index=frame.index)),
                                  errors='coerce').fillna(0).to_numpy() > 0
        if mask_peer:
            available[:] = False
        for name in frame:
            if name.startswith('peer__') or name.startswith('pair_') or name == 'peer_age_hours':
                frame.loc[~available, name] = 0. if 'present_' in name else np.nan
        frame['peer_available'] = available.astype(float)
    return frame


def _feature_frame(frame, columns, numeric):
    """Use the training schema, including explicit sample-wise missing flags."""
    frame = frame.reindex(columns=columns).copy()
    numeric = set(numeric)
    for name in columns:
        if name in numeric:
            frame[name] = pd.to_numeric(frame[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
            frame[name + '__missing'] = frame[name].isna().astype(float)
        else:
            frame[name] = frame[name].fillna('<missing>').astype(str)
    return frame


def _fit_branch(config, frame, target, weights, *, paired):
    frame = _canonical_frame(frame, paired=paired)
    columns = list(frame.columns)
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c]) or frame[c].isna().all()]
    features = _feature_frame(frame, columns, numeric)
    original_sum = float(weights.sum())
    dropout = config['missing_replica_weight'] if paired else 0.
    if dropout:
        masked = _feature_frame(_canonical_frame(frame, paired=True, mask_peer=True), columns, numeric)
        features = pd.concat([features, masked], ignore_index=True)
        target = np.concatenate([target, target])
        weights = np.concatenate([weights * (1 - dropout), weights * dropout])
    nums = [c for c in features if pd.api.types.is_numeric_dtype(features[c])]
    cats = [c for c in features if c not in nums]
    prep = ColumnTransformer([
        ('num', SimpleImputer(strategy='median', keep_empty_features=True), nums),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats),
    ])
    x = prep.fit_transform(features)
    if config['algorithm'] == 'catboost':
        model = CatBoostRegressor(iterations=config['iterations'], depth=config['depth'],
            learning_rate=config['learning_rate'], l2_leaf_reg=config['l2_leaf_reg'],
            loss_function=config['loss'], random_seed=config['random_seed'],
            thread_count=config['threads'], verbose=False, allow_writing_files=False)
    elif config['algorithm'] == 'extra_trees':
        model = ExtraTreesRegressor(n_estimators=config['n_estimators'],
            min_samples_leaf=config['min_samples_leaf'], max_features=config['max_features'],
            random_state=config['random_seed'], n_jobs=config['threads'])
    else:
        raise ValueError('Unknown fusion algorithm')
    model.fit(x, target, sample_weight=weights)
    return dict(model=model, preprocessor=prep, columns=columns, numeric=numeric,
                paired=paired, training_weight_sum=original_sum,
                effective_weight_sum=float(weights.sum()))


def fit_recipe(config, train_arrays, train_labels):
    """Fit independent branches; all preprocessing and labels are train-only.

    Rank heads slice one globally event-balanced weight vector. Each paired
    prefix has observed/masked replicas with weights .75/.25, so its total and
    its event's total stay unchanged. No rows are removed for missing peers.
    """
    _check_arrays(train_arrays, training=True)
    config = deepcopy(config)
    if (config['threads'] not in [1, 2] or not 0 <= config['missing_replica_weight'] < 1
            or config['gate'] not in ['constant', 'progress']):
        raise ValueError('Invalid fusion configuration')
    cases = train_arrays['cases']
    actual = np.asarray([train_labels[c['case_id']] for c in cases]
                        if isinstance(train_labels, Mapping) else train_labels, dtype=float)
    if actual.shape != (len(cases),) or not np.isfinite(actual).all() or np.any(actual <= 0):
        raise ValueError('Invalid training labels')
    weights = event_weights([c['event'] for c in cases])
    target = np.log(actual / train_arrays['current']) + np.log(train_arrays['progress'])
    heads = {}
    for rank in ([0, 1] if config['rank_heads'] else ['shared']):
        indices = np.flatnonzero(np.asarray(train_arrays['rank']) == rank) if rank != 'shared' else np.arange(len(cases))
        if not len(indices):
            raise ValueError('A requested rank head has no training rows')
        heads[rank] = {'training_indices': indices}
        for branch in ['single', 'paired']:
            heads[rank][branch] = _fit_branch(config, train_arrays[branch].iloc[indices],
                target[indices], weights[indices], paired=branch == 'paired')
    return dict(format_version=1, config=config, heads=heads, training_weights=weights,
                training_case_ids=[c['case_id'] for c in cases])


def _branch_predict(branch, frame):
    frame = _canonical_frame(frame, paired=branch['paired'])
    values = _feature_frame(frame, branch['columns'], branch['numeric'])
    return np.asarray(branch['model'].predict(branch['preprocessor'].transform(values)))


def predict_recipe(bundle, arrays):
    """Return final scores in input order, with exact single fallback if absent."""
    _check_arrays(arrays)
    config = bundle['config']
    residuals = {name: np.empty(len(arrays['cases'])) for name in ['single', 'paired']}
    for rank, head in bundle['heads'].items():
        indices = np.flatnonzero(np.asarray(arrays['rank']) == rank) if rank != 'shared' else np.arange(len(arrays['cases']))
        if not len(indices):
            continue
        for branch in residuals:
            residuals[branch][indices] = _branch_predict(head[branch], arrays[branch].iloc[indices])
    frame = _canonical_frame(arrays['paired'], paired=True)
    gate = np.full(len(frame), config['paired_weight'], dtype=float)
    if config['gate'] == 'progress':
        gate = np.asarray(config['progress_weights'])[np.searchsorted(
            config['progress_edges'], arrays['progress'], side='right')].astype(float)
    age = pd.to_numeric(frame.get('peer_age_hours', pd.Series(0., index=frame.index)), errors='coerce').fillna(0).to_numpy()
    gate *= frame['peer_available'].to_numpy() * np.exp(-np.maximum(age, 0) / config['peer_age_scale_hours'])
    delta = np.clip(residuals['paired'] - residuals['single'],
                    -config['correction_limit_log'], config['correction_limit_log'])
    residual = residuals['single'] + gate * delta
    prediction = np.asarray(arrays['current']) * np.exp(np.clip(-np.log(arrays['progress']) + residual, 0, 4))
    if not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError('Nonfinite fusion final score')
    return prediction


def run_development(dataset, output):
    """Save the full predeclared grid before accessing validation answers."""
    from analysis.backtest import ROOT
    from analysis.holdout_dataset import sha256, write_json
    from analysis.multirank_dataset import identity
    from analysis.temporal_dataset import TemporalDataset
    from analysis.temporal_evaluate import score_temporal
    from analysis.temporal_paired_features import prepare_paired

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = TemporalDataset(dataset)
    configs = recipes()
    sources = ['analysis/research_fusion.py', 'analysis/temporal_paired_features.py',
               'analysis/multirank_model.py', 'analysis/temporal_dataset.py',
               'analysis/multirank_dataset.py', 'analysis/backtest.py',
               'analysis/holdout_dataset.py', 'analysis/event_features.py',
               'analysis/temporal_evaluate.py', 'analysis/holdout_evaluate.py']
    manifest = dict(dataset_id=data.manifest['dataset_id'], dataset_directory=str(data.root.resolve()),
        recipes=configs, test_labels_read=False, test_prefixes_read=False,
        training='global per-event weights; original/masked pair replicas split .75/.25',
        selection='minimum validation event-macro MAPE among all six; no validation fitting/refit',
        python=platform.python_version(), software={n: importlib.metadata.version(n)
            for n in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'joblib']},
        source_hashes={name: sha256(ROOT / name) for name in sources},
        limitations=['Test cohort appeared in earlier exploratory work and is not completely unseen.',
                     'Metadata is a current retrospective snapshot without historical-time archives.'])
    manifest['run_id'] = identity(manifest)
    output.mkdir(parents=True)
    (output / 'models').mkdir()
    (output / 'sources').mkdir()
    write_json(output / 'predeclared_manifest.json', manifest)
    for name in sources:
        shutil.copyfile(ROOT / name, output / 'sources' / Path(name).name)
    arrays = {split: prepare_paired(data.inputs(split)) for split in ['train', 'validation']}
    labels = {split: data.labels(split) for split in ['train', 'validation']}
    scores = {}
    prediction_frames = {'train': [], 'validation': []}
    for config in configs:
        name = config['name']
        bundle = fit_recipe(config, arrays['train'], labels['train'])
        joblib.dump(bundle, output / 'models' / f'{name}.joblib')
        scores[name] = {}
        for split in arrays:
            prediction = predict_recipe(bundle, arrays[split])
            frame = pd.DataFrame({'case_id': [c['case_id'] for c in arrays[split]['cases']],
                                  'method': name, 'predicted': prediction})
            prediction_frames[split].append(frame)
            scores[name][split] = score_temporal(data.case_index(split), data.memberships(split), labels[split], frame)
        print(json.dumps({'method': name, 'train_fit': scores[name]['train']['overall'],
                          'validation': scores[name]['validation']['overall']}), flush=True)
    for split, frames in prediction_frames.items():
        pd.concat(frames, ignore_index=True).to_csv(output / f'{split}_predictions.csv', index=False)
    selection = dict(selected_recipe=min(scores, key=lambda name: scores[name]['validation']['overall']['event_macro_mape_pct']),
                     scores=scores, test_labels_read=False, test_prefixes_read=False,
                     refit_after_selection=False)
    write_json(output / 'selection.json', selection)
    seal = dict(run_id=manifest['run_id'], artifacts={str(p.relative_to(output)): sha256(p)
                for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output / 'development_manifest.json', seal)
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/temporal_holdout_v2_t1000_t2000'))
    parser.add_argument('--output', type=Path, default=Path('analysis/research_development/fusion_20261112'))
    args = parser.parse_args()
    run_development(args.dataset, args.output)
