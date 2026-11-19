"""Train-only era/calendar models and chronological out-of-fold calibration.

This module consumes prepared blind-prefix arrays. It never opens a dataset,
metadata archive, previous predictions, or labels. Recipe selection belongs to
the caller and may use validation; every fit here uses its supplied training
arrays only. Rank-specific models slice one globally equal-event weight vector.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.multirank_model import event_weights
from analysis.temporal_paired_features import phase_weights

HOUR = 3_600_000
DAY = 24*HOUR
ERA_ORIGIN = 1_514_764_800_000  # 2018-01-01 UTC, a fixed calendar coordinate.


def recipes():
    """Six predeclared recipes; no validation-dependent parameter search."""
    base = {'iterations': 650, 'depth': 5, 'learning_rate': .035,
        'l2_leaf_reg': 20., 'loss': 'RMSE', 'rank_specific': False,
        'phase_balance': False, 'model': 'catboost', 'ridge_alpha': 80.,
        'calibrate': False, 'oof_folds': 3, 'oof_warm_fraction': .4,
        'calibration_shrink_events': 30., 'calibration_max_log': .18,
        'seed': 42, 'thread_count': 2, 'features': 'paired'}
    changes = [
        {'name': 'shift_calendar_pool'},
        {'name': 'shift_calendar_rank', 'rank_specific': True, 'depth': 4},
        {'name': 'shift_calendar_phase', 'loss': 'MAE', 'phase_balance': True},
        {'name': 'shift_era_ridge', 'model': 'ridge'},
        {'name': 'shift_pool_calibrated', 'calibrate': True},
        {'name': 'shift_rank_calibrated', 'calibrate': True,
            'rank_specific': True, 'depth': 4},
    ]
    return [base | change for change in changes]


def _validate(arrays):
    n = len(arrays['cases'])
    if not n:
        raise ValueError('Empty prefix arrays')
    for key in ('single', 'paired', 'audit', 'current', 'progress', 'rank'):
        if len(arrays[key]) != n:
            raise ValueError('Prefix array lengths differ')
    rank = np.asarray(arrays['rank'])
    expected = np.array([0 if c['rank'] == 1000 else 1 if c['rank'] == 2000 else -1
                         for c in arrays['cases']])
    if not np.array_equal(rank, expected) or np.any(expected < 0):
        raise ValueError('Rank encoding differs from blind cases')
    ids = [c['case_id'] for c in arrays['cases']]
    if len(set(ids)) != n or list(arrays['audit']['case_id']) != ids:
        raise ValueError('Duplicate or misaligned blind cases')
    current, progress = np.asarray(arrays['current']), np.asarray(arrays['progress'])
    if (not np.isfinite(current).all() or np.any(current <= 0)
            or not np.isfinite(progress).all() or np.any((progress <= 0) | (progress > 1))):
        raise ValueError('Invalid blind current/progress')
    audit = arrays['audit']
    if (audit['observed_at'] > audit['cutoff']).any():
        raise ValueError('Future observation in prefix arrays')


def subset_arrays(arrays, indices):
    """Keep complete case alignment while subsetting a training OOF fold."""
    indices = np.asarray(indices, dtype=int)
    result = {}
    for key, value in arrays.items():
        if isinstance(value, pd.DataFrame):
            result[key] = value.iloc[indices].reset_index(drop=True)
        elif isinstance(value, np.ndarray):
            result[key] = value[indices]
        elif key == 'cases':
            result[key] = [value[i] for i in indices]
        else:
            result[key] = value
    return result


def _timing(arrays):
    source = arrays['single']
    now = arrays['audit']['observed_at'].to_numpy(dtype=float)
    start = now-source['elapsed_h'].to_numpy(dtype=float)*HOUR
    end = start+source['duration_h'].to_numpy(dtype=float)*HOUR
    if not np.isfinite(start).all() or not np.isfinite(end).all() or np.any(end <= start):
        raise ValueError('Invalid scheduled activity times')
    return np.rint(start).astype(np.int64), np.rint(end).astype(np.int64), now


def _weekend_hours(start, end):
    # Calendar arithmetic uses the public event schedule, never future scores.
    total = 0.
    cursor = float(start)
    while cursor < end:
        day_index = int(np.floor((cursor+8*HOUR)/DAY))
        next_day = (day_index+1)*DAY-8*HOUR
        stop = min(float(end), next_day)
        if (day_index+3)%7 >= 5:  # 1970-01-01 was Thursday.
            total += (stop-cursor)/HOUR
        cursor = stop
    return total


def feature_frame(arrays, config):
    """Only deterministic transformations of contemporaneous prefix features."""
    _validate(arrays)
    frame = arrays[config.get('features', 'paired')].reset_index(drop=True).copy()
    start, end, now = _timing(arrays)
    date = pd.to_datetime(start, unit='ms', utc=True).tz_convert('Asia/Shanghai')
    finish = pd.to_datetime(end, unit='ms', utc=True).tz_convert('Asia/Shanghai')
    progress = np.asarray(arrays['progress'])
    rank = np.asarray(arrays['rank'])
    era = (start-ERA_ORIGIN)/(365.25*DAY)
    frame['calendar_era_years'] = era
    frame['calendar_year_sin'] = np.sin(2*np.pi*date.day_of_year/365.25)
    frame['calendar_year_cos'] = np.cos(2*np.pi*date.day_of_year/365.25)
    frame['calendar_halfyear_sin'] = np.sin(4*np.pi*date.day_of_year/365.25)
    frame['calendar_halfyear_cos'] = np.cos(4*np.pi*date.day_of_year/365.25)
    for prefix, times in (('start', date), ('end', finish)):
        frame[f'calendar_{prefix}_weekday_sin'] = np.sin(2*np.pi*times.day_of_week/7)
        frame[f'calendar_{prefix}_weekday_cos'] = np.cos(2*np.pi*times.day_of_week/7)
        frame[f'calendar_{prefix}_hour_sin'] = np.sin(2*np.pi*times.hour/24)
        frame[f'calendar_{prefix}_hour_cos'] = np.cos(2*np.pi*times.hour/24)
    frame['calendar_weekend_h_total'] = [_weekend_hours(a, b) for a, b in zip(start, end)]
    frame['calendar_weekend_h_remaining'] = [_weekend_hours(a, b) for a, b in zip(now, end)]
    frame['calendar_observation_lag_h'] = (arrays['audit']['cutoff'].to_numpy()-now)/HOUR
    frame['shift_rank'] = rank
    frame['shift_log_progress'] = np.log(progress)
    frame['shift_remaining_fraction'] = 1-progress
    frame['shift_log_remaining_h'] = np.log1p(frame['remaining_h'])
    frame['shift_log_elapsed_h'] = np.log1p(frame['elapsed_h'])
    frame['shift_era_x_remaining'] = era*(1-progress)
    frame['shift_era_x_log_progress'] = era*np.log(progress)
    frame['shift_rank_x_remaining'] = rank*(1-progress)
    frame['shift_rank_x_log_progress'] = rank*np.log(progress)
    frame['shift_era_x_rank'] = era*rank
    # A fixed linear interaction can extrapolate a changing popularity level.
    frame['shift_log_pace'] = np.log(np.asarray(arrays['current'])/progress)
    frame['shift_era_x_log_pace'] = era*frame['shift_log_pace']
    for name in ('rate_1h_relative', 'rate_3h_relative', 'rate_6h_relative', 'rate_12h_relative'):
        if name in frame:
            value = np.log1p(pd.to_numeric(frame[name], errors='coerce').clip(lower=0))
            frame[f'shift_log_{name}'] = value
            frame[f'shift_rank_x_{name}'] = rank*value
    return frame.replace([np.inf, -np.inf], np.nan)


def chronological_folds(arrays, *, folds=3, warm_fraction=.4):
    """Forward-only event blocks; all fit activities ended before the block."""
    _validate(arrays)
    if folds < 1 or not 0 < warm_fraction < 1:
        raise ValueError('Invalid OOF configuration')
    starts, ends, _ = _timing(arrays)
    events = np.array([c['event'] for c in arrays['cases']])
    intervals = {e: (int(starts[events == e].min()), int(ends[events == e].max()))
                 for e in np.unique(events)}
    ordered = sorted(intervals, key=lambda e: (intervals[e][0], e))
    warm = max(2, int(np.ceil(len(ordered)*warm_fraction)))
    blocks = np.array_split(np.array(ordered[warm:]), min(folds, max(len(ordered)-warm, 1)))
    result = []
    for block in blocks:
        if not len(block):
            continue
        boundary = min(intervals[e][0] for e in block)
        train_events = [e for e in ordered if intervals[e][1] < boundary]
        if len(train_events) < 2:
            continue
        train_indices = np.flatnonzero(np.isin(events, train_events))
        valid_indices = np.flatnonzero(np.isin(events, block))
        audit = {'train_events': [int(e) for e in train_events],
            'validation_events': [int(e) for e in block],
            'train_latest_end_ms': max(intervals[e][1] for e in train_events),
            'validation_earliest_start_ms': int(boundary),
            'train_cases': len(train_indices), 'validation_cases': len(valid_indices)}
        result.append((train_indices, valid_indices, audit))
    return result


def _preprocessor(frame):
    categorical = [name for name in frame if not pd.api.types.is_numeric_dtype(frame[name])
                   and frame[name].notna().any()]
    numeric = [name for name in frame if name not in categorical]
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
        ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categorical)])


def _fit_base(config, arrays, actual, weights):
    frame = feature_frame(arrays, config)
    target = np.log(actual/arrays['current'])+np.log(arrays['progress'])
    bundle = {'config': deepcopy(config), 'models': {},
        'fallback_correction': float(np.average(target, weights=weights))}
    ranks = (0, 1) if config['rank_specific'] else (-1,)
    for rank in ranks:
        indices = np.flatnonzero(arrays['rank'] == rank) if rank >= 0 else np.arange(len(frame))
        if not len(indices):
            continue
        part = frame.iloc[indices]
        prep = _preprocessor(part)
        x = prep.fit_transform(part)
        scaler = None
        if config['model'] == 'ridge':
            scaler = StandardScaler().fit(x, sample_weight=weights[indices])
            x = scaler.transform(x)
            model = Ridge(alpha=config['ridge_alpha'])
        else:
            model = CatBoostRegressor(iterations=int(config['iterations']),
                depth=int(config['depth']), learning_rate=config['learning_rate'],
                l2_leaf_reg=config['l2_leaf_reg'], loss_function=config['loss'],
                random_seed=config['seed'], thread_count=min(2, int(config['thread_count'])),
                verbose=False, allow_writing_files=False)
        model.fit(x, target[indices], sample_weight=weights[indices])
        bundle['models'][rank] = {'model': model, 'preprocessor': prep, 'scaler': scaler}
    return bundle


def _predict_base(bundle, arrays):
    frame = feature_frame(arrays, bundle['config'])
    correction = np.full(len(frame), bundle['fallback_correction'])
    for rank, part in bundle['models'].items():
        indices = np.flatnonzero(arrays['rank'] == rank) if rank >= 0 else np.arange(len(frame))
        if not len(indices):
            continue
        x = part['preprocessor'].transform(frame.iloc[indices])
        if part['scaler'] is not None:
            x = part['scaler'].transform(x)
        correction[indices] = part['model'].predict(x)
    return arrays['current']*np.exp(np.clip(-np.log(arrays['progress'])+correction, 0, 4))


def mape_factor(actual, predicted, weights):
    """Exact scalar minimizer of weighted MAPE via its weighted median."""
    actual, predicted, weights = map(lambda x: np.asarray(x, dtype=float), (actual, predicted, weights))
    if (not len(actual) or actual.shape != predicted.shape or actual.shape != weights.shape
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or not np.isfinite(weights).all() or np.any(actual <= 0)
            or np.any(predicted <= 0) or np.any(weights <= 0)):
        raise ValueError('Invalid calibration observations')
    ratio = actual/predicted
    objective_weight = weights/ratio
    order = np.argsort(ratio, kind='stable')
    median = np.searchsorted(np.cumsum(objective_weight[order]), objective_weight.sum()/2)
    return float(ratio[order[min(median, len(order)-1)]])


def _phase(progress):
    return np.searchsorted(np.array([.1, .5, .9]), progress, side='right')


def _fit_calibration(config, arrays, actual, predicted, weights):
    # The transformed median weights express MAPE, not recency reweighting.
    global_log = np.log(mape_factor(actual, predicted, weights))
    rank, phase = np.asarray(arrays['rank']), _phase(arrays['progress'])
    events = np.array([c['event'] for c in arrays['cases']])
    cells, counts = {}, {}
    for r in (0, 1):
        for p in range(4):
            mask = (rank == r) & (phase == p)
            count = len(np.unique(events[mask]))
            counts[f'{r}:{p}'] = count
            local = np.log(mape_factor(actual[mask], predicted[mask], weights[mask])) if count else global_log
            fraction = count/(count+config['calibration_shrink_events'])
            value = fraction*local+(1-fraction)*global_log
            cells[f'{r}:{p}'] = float(np.clip(value, -config['calibration_max_log'], config['calibration_max_log']))
    return {'ranks': [0, 1], 'log_factors': cells, 'event_counts': counts,
        'global_log_factor': float(global_log),
        'fit_policy': 'forward OOF training labels only; event-equal MAPE scalar factors, event-count shrinkage'}


def fit_recipe(config, train_arrays, train_labels):
    """Fit one recipe on training arrays/answers; return a joblib bundle."""
    _validate(train_arrays)
    config = deepcopy(config)
    if config.get('model') not in ('catboost', 'ridge') or config.get('thread_count', 2) < 1:
        raise ValueError('Unsupported recipe')
    if isinstance(train_labels, dict):
        actual = np.array([train_labels[c['case_id']] for c in train_arrays['cases']], dtype=float)
    else:
        actual = np.asarray(train_labels, dtype=float)
    if (actual.shape != train_arrays['current'].shape or not np.isfinite(actual).all()
            or np.any(actual <= 0)):
        raise ValueError('Invalid aligned training labels')
    events = [c['event'] for c in train_arrays['cases']]
    weights = (phase_weights(events, train_arrays['progress']) if config['phase_balance']
               else event_weights(events))
    calibration, oof_audit = None, []
    if config['calibrate']:
        oof_prediction = np.full(len(actual), np.nan)
        for training, heldout, audit in chronological_folds(train_arrays,
                folds=config['oof_folds'], warm_fraction=config['oof_warm_fraction']):
            # Each fold refits preprocessing using only earlier completed events.
            earlier = subset_arrays(train_arrays, training)
            model = _fit_base(config, earlier, actual[training], weights[training])
            oof_prediction[heldout] = _predict_base(model, subset_arrays(train_arrays, heldout))
            oof_audit.append(audit)
        available = np.flatnonzero(np.isfinite(oof_prediction))
        if len(available):
            oof_arrays = subset_arrays(train_arrays, available)
            # No event is favored due to the number of available OOF prefixes.
            oof_weights = event_weights([c['event'] for c in oof_arrays['cases']])
            calibration = _fit_calibration(config, oof_arrays, actual[available],
                oof_prediction[available], oof_weights)
    bundle = _fit_base(config, train_arrays, actual, weights)
    bundle['calibration'] = calibration
    bundle['oof_audit'] = oof_audit
    bundle['training_weights'] = [
        {'case_id': c['case_id'], 'event': c['event'], 'rank': c['rank'], 'weight': float(w)}
        for c, w in zip(train_arrays['cases'], weights)]
    bundle['training_event_ids'] = sorted(set(events))
    return bundle


def predict_recipe(bundle, arrays):
    """Predict every supplied case, never updating preprocessing/calibration."""
    prediction = _predict_base(bundle, arrays)
    calibration = bundle['calibration']
    if calibration is not None:
        factors = np.array([calibration['log_factors'][f'{r}:{p}']
            for r, p in zip(arrays['rank'], _phase(arrays['progress']))])
        prediction = np.maximum(arrays['current'], prediction*np.exp(factors))
    if len(prediction) != len(arrays['cases']) or not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError('Invalid final-score predictions')
    return prediction
