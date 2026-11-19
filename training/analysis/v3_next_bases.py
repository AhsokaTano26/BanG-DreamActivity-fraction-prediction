"""Four frozen rolling bases; pure fitting over explicitly released histories.

This module never receives a dataset, vault path, or held-out answer reader.
Paired inputs preserve their physical case IDs and only use as-of peer prefixes.
"""
from __future__ import annotations

from collections import defaultdict

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from analysis.data_policy import MIN_EVENT_START_MS
from analysis.innovation_common import prepare, event_weights
from analysis.innovation_hierarchy import ContextDesign, FramePreprocessor, _check_arrays
from analysis.temporal_paired_features import prepare_paired


_COMMON = {'iterations': 400, 'depth': 4, 'learning_rate': .04,
    'l2_leaf_reg': 12., 'seed': 42, 'thread_count': 2,
    'weights': 'equal total per completed event in each fitted rank scope',
    'context': 'training-only event-type vocabulary and fractional band shares',
    'prediction_log_growth_bounds': [0., 4.]}
CONFIGS = {
    'rank_log400': _COMMON | {'features': 'own+type+band', 'separate_ranks': True,
        'loss_function': 'RMSE', 'target': 'log(final/current)+log(progress)'},
    'paired_mae400': _COMMON | {'features': 'same-origin as-of pair+type+band',
        'separate_ranks': False, 'loss_function': 'MAE',
        'target': 'log(final/current)+log(progress)'},
    'paired_mape400': _COMMON | {'features': 'same-origin as-of pair+type+band',
        'separate_ranks': False, 'loss_function': 'MAPE', 'target': 'final/current'},
    'paired_rate400': _COMMON | {'features': 'same-origin as-of pair+type+band',
        'separate_ranks': False, 'loss_function': 'RMSE',
        'target': 'log1p((final-current)/remaining_hours)'},
}


def prepare_arrays(samples, metadata):
    """Group only for prepare_paired's origin contract; return original order."""
    samples = list(samples)
    arrays = prepare(samples, metadata)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample['case_id'].split(':')[0]].append(sample)
    paired, audits = [], []
    for group in groups.values():
        part = prepare_paired(group)
        frame = part['paired'].copy()
        frame.index = [c['case_id'] for c in part['cases']]
        paired.append(frame)
        audits.append(part['audit'].set_index('case_id', drop=False))
    ids = [c['case_id'] for c in arrays['cases']]
    arrays['paired'] = pd.concat(paired).loc[ids].reset_index(drop=True).drop(
        columns=['event_type'], errors='ignore')
    arrays['pair_audit'] = pd.concat(audits).loc[ids].reset_index(drop=True)
    return arrays


def _frame(arrays, design, config):
    own = arrays['base'] if config['separate_ranks'] else arrays['paired']
    context = pd.DataFrame(design.transform(arrays)[:, 4:],
        columns=['next_context_'+str(i) for i in range(len(design.types)+len(design.bands))])
    return pd.concat([own.reset_index(drop=True), context], axis=1)


def _targets(actual, arrays, config):
    current = np.asarray(arrays['current'], dtype=float)
    if config['target'] == 'final/current':
        return actual/current
    if config['target'] == 'log1p((final-current)/remaining_hours)':
        remaining = np.asarray(arrays['remaining_h'], dtype=float)
        if (remaining <= 0).any():
            raise ValueError('Rate training requires positive remaining hours')
        return np.log1p((actual-current)/remaining)
    return np.log(actual/current)+np.log(arrays['progress'])


def fit_base(name, arrays, labels, *, fit_before_ms, iterations=None):
    """Train only on completed injected cases; clock is frozen at event start."""
    if name not in CONFIGS:
        raise ValueError('Unknown preregistered base recipe')
    _check_arrays(arrays, training=True)
    if not np.isfinite(fit_before_ms) or int(fit_before_ms) != fit_before_ms:
        raise ValueError('Fit clock must be a finite integer')
    if any(c['end_ms'] >= fit_before_ms or c['start_ms'] < MIN_EVENT_START_MS
           for c in arrays['cases']):
        raise ValueError('Training activity not completed before fit clock or below date policy')
    ids = [c['case_id'] for c in arrays['cases']]
    if set(labels) != set(ids):
        raise ValueError('Training labels must match the exact injected cases')
    actual = np.array([labels[c] for c in ids], dtype=float)
    if not np.isfinite(actual).all() or (actual < arrays['current']).any():
        raise ValueError('Training labels must be finite and at least current')
    config = CONFIGS[name].copy()
    if iterations is not None:
        if int(iterations) != iterations or iterations < 1:
            raise ValueError('Iterations must be a positive integer')
        config['iterations'] = int(iterations)
    design = ContextDesign.fit(arrays)
    frame = _frame(arrays, design, config)
    target = _targets(actual, arrays, config)
    models, weight_rows = {}, []
    for rank in ([1000, 2000] if config['separate_ranks'] else ['shared']):
        indices = np.array([i for i, c in enumerate(arrays['cases'])
            if rank == 'shared' or c['rank'] == rank], dtype=int)
        if not len(indices):
            raise ValueError('Every rank model requires training prefixes')
        cases = [arrays['cases'][i] for i in indices]
        weights = event_weights(cases)
        preprocessor = FramePreprocessor.fit(frame.iloc[indices])
        model = CatBoostRegressor(iterations=config['iterations'], depth=config['depth'],
            learning_rate=config['learning_rate'], l2_leaf_reg=config['l2_leaf_reg'],
            loss_function=config['loss_function'], random_seed=config['seed'],
            thread_count=config['thread_count'], verbose=False,
            allow_writing_files=False, allow_const_label=True)
        model.fit(preprocessor.transform(frame.iloc[indices]), target[indices], sample_weight=weights)
        models[rank] = {'preprocessor': preprocessor, 'model': model}
        weight_rows.extend({'case_id': c['case_id'], 'event': int(c['event']),
            'rank': int(c['rank']), 'model_rank': str(rank), 'weight': float(w)}
            for c, w in zip(cases, weights))
    return {'name': name, 'config': config, 'context_design': design, 'models': models,
        'weight_rows': weight_rows, 'fit_before_ms': int(fit_before_ms),
        'training_case_ids': ids, 'training_events': sorted({int(c['event']) for c in arrays['cases']}),
        'diagnostics': {'training_cases': len(ids),
            'training_events': len({c['event'] for c in arrays['cases']}),
            'target_min': float(target.min()), 'target_max': float(target.max()),
            'context_types': list(design.types), 'context_bands': list(design.bands)}}


def predict_base(bundle, arrays):
    _check_arrays(arrays)
    if any(c['start_ms'] < bundle['fit_before_ms'] for c in arrays['cases']):
        raise ValueError('Prediction precedes the fitted event-start clock')
    config = bundle['config']; frame = _frame(arrays, bundle['context_design'], config)
    values = np.full(len(frame), np.nan)
    for rank, model in bundle['models'].items():
        indices = np.array([i for i, c in enumerate(arrays['cases'])
            if rank == 'shared' or c['rank'] == rank], dtype=int)
        if len(indices):
            values[indices] = model['model'].predict(model['preprocessor'].transform(frame.iloc[indices]))
    current = np.asarray(arrays['current'], dtype=float)
    if config['target'] == 'final/current':
        growth = np.log(np.clip(values, 1., np.exp(4)))
    elif config['target'] == 'log1p((final-current)/remaining_hours)':
        rate = np.expm1(np.clip(values, 0., 700.))
        # logaddexp avoids overflow for an unusually high predicted rate.
        remaining = np.maximum(np.asarray(arrays['remaining_h'], float), 0.)
        growth = np.logaddexp(0., np.log(np.maximum(rate, np.finfo(float).tiny))
            + np.log(np.maximum(remaining, np.finfo(float).tiny))-np.log(current))
    else:
        growth = values-np.log(arrays['progress'])
    prediction = current*np.exp(np.clip(growth, 0., 4.))
    if not np.isfinite(prediction).all():
        raise ValueError('Base predictions must be finite')
    return prediction


class OnlineEwma:
    """Cold per-rank/phase state updated only from own sealed raw forecasts."""
    def __init__(self, alpha):
        if alpha not in (.3, .6):
            raise ValueError('Only preregistered cold EWMA alphas are allowed')
        self.alpha = float(alpha); self.groups = {}; self.events = set()
        self.records = []; self.latest_end = None

    def observe(self, rows, labels, *, event, end_ms):
        if event in self.events or (self.latest_end is not None and end_ms < self.latest_end):
            raise ValueError('Feedback must be distinct and ordered by completion')
        if rows.case_id.duplicated().any() or set(rows.event)-{event} or set(labels) != set(rows.case_id):
            raise ValueError('Feedback must exactly match one sealed forecast event')
        rows = rows.copy()
        actual = np.array([labels[c] for c in rows.case_id], float)
        predicted = rows.predicted.to_numpy(float)
        if (not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or (actual <= 0).any() or (predicted <= 0).any()):
            raise ValueError('Feedback scores must be positive and finite')
        rows['error'] = np.log(predicted/actual)
        for (rank, phase), part in rows.groupby(['rank', 'phase']):
            key = int(rank), int(phase); error = float(part.error.mean())
            previous, n = self.groups.get(key, (0., 0))
            value = (1-self.alpha)*previous+self.alpha*error if n else error
            self.groups[key] = value, n+1
            self.records.append({'event': int(event), 'end_ms': int(end_ms),
                'rank': key[0], 'phase': key[1], 'mean_log_pred_over_actual': error,
                'cases': len(part), 'event_group_weight': 1.})
        self.events.add(event); self.latest_end = int(end_ms)

    def bias(self, rank, phase):
        value, n = self.groups.get((int(rank), int(phase)), (0., 0))
        return float(np.clip(value*n/(n+10), -.3, .3))
