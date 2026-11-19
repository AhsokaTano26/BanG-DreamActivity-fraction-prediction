"""Pure, preregistered residual filters over detached released forecast frames.

The coordinator owns release/seal checks. This additional boundary fails closed
on an illegal clock; it never opens datasets, labels, models, or file paths.
Positive bias means an overprediction: the caller applies base * exp(-bias).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


MIN_START_MS = 1706504400000
_COMMON = {'clip': [-.3, .3], 'phase_edges': [.25, .9],
           'phase_boundary': 'left closed, right open',
           'error': 'log(uncorrected bounded base prediction / actual)',
           'clock': 'start >= 1706504400000 and end < query_start_ms',
           'ordering': 'event end, event start, event ID',
           'initial_residuals': 'none; completed sealed forecast events only'}
CONFIGS = {
    'ewma_reference': _COMMON | {'alpha': .3, 'shrinkage': 10.,
        'initial': 'first event mean', 'aggregation': 'one mean per event/rank/phase'},
    'ewma_fast': _COMMON | {'alpha': .6, 'shrinkage': 10.,
        'initial': 'first event mean', 'aggregation': 'one mean per event/rank/phase'},
    'window5': _COMMON | {'window': 5, 'shrinkage': 10.,
        'n': 'number of events retained in window',
        'aggregation': 'one mean per event/rank/phase'},
    'kalman_level': _COMMON | {'q': .001, 'r': .01,
        'initial_mean': 0., 'initial_variance': .1,
        'time_step': 'one completed observation event in this rank/phase',
        'prior': 'add q before each observation', 'shrinkage': None,
        'forecast': 'filtered level', 'aggregation': 'one mean per event/rank/phase'},
    'kalman_trend': _COMMON | {'level_q': .001, 'trend_q': .0001, 'r': .01,
        'initial_state': [0., 0.], 'initial_covariance_diagonal': [.1, .01],
        'transition': [[1., 1.], [0., 1.]], 'observation': [1., 0.],
        'time_step': 'one completed observation event in this rank/phase',
        'prior': 'transition then add diagonal process noise before observation',
        'shrinkage': None, 'forecast': 'filtered level plus filtered trend',
        'aggregation': 'one mean per event/rank/phase'},
    'kernel_progress': _COMMON | {'bandwidth': .15, 'temporal_decay': .7,
        'shrinkage': 10., 'n': 'completed same-rank events',
        'aggregation': 'normalize Gaussian prefix weights within each same-rank event; '
                       'then average event means with decay^age',
        'age': 'completed same-rank event index, most recent age zero'},
    'context_partial_pool': _COMMON | {'alpha': .3, 'shrinkage': 10.,
        'type_coefficient': .25, 'type_shrinkage': 10.,
        'band_coefficient': .15, 'band_shrinkage': 15.,
        'aggregation': 'one mean per event/rank/phase',
        'deviation': 'same-context raw EWMA minus all-context raw EWMA in rank/phase',
        'formula': 'reference + .25*n_type/(n_type+10)*type_deviation '
                   '+ .15*n_band/(n_band+15)*band_deviation',
        'unseen_context': 'zero deviation contribution'},
}

_CORE = ['case_id', 'event', 'rank', 'phase', 'progress', 'start_ms', 'end_ms']
_TARGET_ANSWERS = {'actual', 'target', 'log_error', 'final', 'final_score',
                   'actual_score', 'actual_final', 'label', 'labels'}


def _validate_frame(frame, *, historical):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('Correction inputs must be pandas DataFrames')
    required = _CORE + (['log_error'] if historical else [])
    if set(required) - set(frame.columns):
        raise ValueError('Missing correction input columns: ' +
                         ', '.join(sorted(set(required) - set(frame.columns))))
    if not historical and _TARGET_ANSWERS.intersection(frame.columns):
        raise ValueError('Current target frame must contain no outcomes')
    result = frame.copy(deep=True)
    if result.case_id.isna().any() or result.case_id.duplicated().any():
        raise ValueError('Correction cases must have unique nonmissing case IDs')
    for name in ['event', 'rank', 'phase', 'progress', 'start_ms', 'end_ms'] + (
            ['log_error'] if historical else []):
        try:
            values = pd.to_numeric(result[name], errors='raise').to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('Correction numeric columns must be finite') from exc
        if not np.isfinite(values).all():
            raise ValueError('Correction numeric columns must be finite')
        if name in ('event', 'rank', 'phase', 'start_ms', 'end_ms'):
            if np.any(values != np.floor(values)):
                raise ValueError('Correction IDs, phases and clocks must be integers')
            result[name] = values.astype(np.int64)
        else:
            result[name] = values
    if ((result.event <= 0).any() or (result['rank'] <= 0).any()
            or (result.start_ms < MIN_START_MS).any()
            or (result.end_ms <= result.start_ms).any()):
        raise ValueError('Invalid event identifiers or dense-period event clock')
    progress = result.progress.to_numpy(float)
    if (np.any(progress < 0.) or np.any(progress > 1.) or
            np.any(result.phase.to_numpy() != np.searchsorted([.25, .9], progress, side='right'))):
        raise ValueError('Phase must match progress using .25/.9 right boundaries')
    for name in ('start_ms', 'end_ms', 'event_type', 'band_group'):
        if name in result and (result.groupby('event')[name].nunique(dropna=False) > 1).any():
            raise ValueError('An event must have one clock and context')
    return result


def _frames(history, target, query_start_ms, *, context):
    try:
        query = float(query_start_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError('Query start must be a dense-period integer clock') from exc
    if not np.isfinite(query) or query != np.floor(query) or query < MIN_START_MS:
        raise ValueError('Query start must be a dense-period integer clock')
    past = _validate_frame(history, historical=True)
    current = _validate_frame(target, historical=False)
    if ((past.end_ms >= query).any() or (current.start_ms != query).any()
            or current.event.nunique() > 1
            or set(past.event).intersection(current.event)
            or set(past.case_id).intersection(current.case_id)):
        raise ValueError('Only earlier completed events may inform one query event start')
    if context:
        for frame in (past, current):
            if {'event_type', 'band_group'} - set(frame):
                raise ValueError('Context correction requires event_type and band_group')
            # A missing category may match other missing history, never a future fit.
            for name in ('event_type', 'band_group'):
                frame[name] = frame[name].map(lambda value: None if pd.isna(value) else str(value))
    return past, current, int(query)


def _event_means(past):
    means = past.groupby(['event', 'rank', 'phase'], as_index=False, sort=True).agg(
        error=('log_error', 'mean'), start_ms=('start_ms', 'first'),
        end_ms=('end_ms', 'first'), cases=('case_id', 'size'))
    return means.sort_values(['end_ms', 'start_ms', 'event', 'rank', 'phase'])


def _ewma(errors, alpha):
    value = 0.
    for n, error in enumerate(errors):
        value = (1. - alpha) * value + alpha * float(error) if n else float(error)
    return float(value), len(errors)


def _kalman(errors, config, *, trend):
    if not trend:
        value, variance = config['initial_mean'], config['initial_variance']
        for error in errors:
            variance += config['q']
            gain = variance / (variance + config['r'])
            value += gain * (float(error) - value)
            variance *= 1. - gain
        return float(value), {'level': float(value), 'variance': float(variance)}
    transition = np.asarray(config['transition'])
    state = np.asarray(config['initial_state'], dtype=float)
    covariance = np.diag(config['initial_covariance_diagonal'])
    process = np.diag([config['level_q'], config['trend_q']])
    for error in errors:
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process
        gain = covariance[:, 0] / (covariance[0, 0] + config['r'])
        state = state + gain * (float(error) - state[0])
        covariance = covariance - np.outer(gain, covariance[0, :])
    return float(state.sum()), {'level': float(state[0]), 'trend': float(state[1]),
                                'covariance': covariance.tolist()}


def predict_bias(name, history, target, *, query_start_ms):
    """Return total log correction in target row order, plus JSON-safe state.

    Statistical group observations are event means, not individual prefixes.
    Kernel progress instead forms one normalized kernel mean per event so a
    dense prefix panel still has one event's total temporal weight. No mutable
    state or references to caller-owned frames survive this call.
    """
    if name not in CONFIGS:
        raise ValueError('Unknown statistical residual method: ' + str(name))
    config = CONFIGS[name]
    past, current, query = _frames(history, target, query_start_ms,
                                   context=name == 'context_partial_pool')
    means = _event_means(past)
    groups = {(int(rank), int(phase)): part for (rank, phase), part in
              means.groupby(['rank', 'phase'], sort=True)}
    group_states = []
    values = {}
    for key, part in groups.items():
        errors = part.error.to_numpy(float)
        n = len(errors)
        state = {}
        if name in ('ewma_reference', 'ewma_fast', 'context_partial_pool'):
            raw, n = _ewma(errors, config['alpha'])
            value = raw * n / (n + config['shrinkage'])
            state['raw_ewma'] = raw
        elif name == 'window5':
            errors = errors[-config['window']:]
            n = len(errors)
            value = float(errors.mean()) * n / (n + config['shrinkage'])
        elif name in ('kalman_level', 'kalman_trend'):
            value, state = _kalman(errors, config, trend=name == 'kalman_trend')
        else:
            continue
        values[key] = (float(value), state)
        group_states.append({'rank': key[0], 'phase': key[1], 'n_events': n,
                             'unclipped_bias': float(value), **state})

    metadata = past.drop_duplicates('event').set_index('event')
    results, row_states = [], []
    for row in current.itertuples(index=False):
        key = (int(row.rank), int(row.phase))
        value, state = values.get(key, (0., {'raw_ewma': 0.}))
        detail = {'case_id': str(row.case_id)}
        if name == 'context_partial_pool' and key in groups:
            part = groups[key]
            for column, coefficient, shrinkage in [
                    ('event_type', config['type_coefficient'], config['type_shrinkage']),
                    ('band_group', config['band_coefficient'], config['band_shrinkage'])]:
                category = getattr(row, column)
                matching = part.event.map(metadata[column]).map(
                    lambda item: item == category)
                context_raw, n = _ewma(part.loc[matching, 'error'].to_numpy(float),
                                       config['alpha'])
                delta = coefficient * n / (n + shrinkage) * (
                    context_raw - state['raw_ewma']) if n else 0.
                value += delta
                detail[column + '_events'] = n
                detail[column + '_deviation_contribution'] = float(delta)
        elif name == 'kernel_progress':
            rank_past = past[past['rank'] == row.rank]
            event_rows = []
            for event, panel in rank_past.groupby('event', sort=False):
                distance = (panel.progress.to_numpy(float) - row.progress) / config['bandwidth']
                log_weight = -.5 * distance ** 2
                # Subtracting the maximum preserves within-event kernel weights.
                weights = np.exp(log_weight - log_weight.max())
                error = float(np.dot(weights, panel.log_error.to_numpy(float)) / weights.sum())
                event_rows.append((int(panel.end_ms.iloc[0]), int(panel.start_ms.iloc[0]),
                                   int(event), error))
            event_rows.sort()
            n = len(event_rows)
            if n:
                weights = config['temporal_decay'] ** np.arange(n - 1, -1, -1)
                errors = np.asarray([item[3] for item in event_rows])
                value = float(np.dot(weights, errors) / weights.sum()) * n / (n + config['shrinkage'])
            detail['n_events'] = n
            detail['event_kernel_means'] = [
                {'event': item[2], 'mean_log_error': item[3], 'age': n - index - 1}
                for index, item in enumerate(event_rows)]
        bias = float(np.clip(value, *config['clip']))
        results.append(bias)
        row_states.append(detail | {'unclipped_bias': float(value), 'bias': bias})
    diagnostics = {'method': name, 'query_start_ms': query,
        'n_history_events': int(past.event.nunique()), 'n_history_cases': len(past),
        'latest_history_end_ms': int(past.end_ms.max()) if len(past) else None,
        'group_observations': [
            {'event': int(row.event), 'rank': int(row.rank), 'phase': int(row.phase),
             'start_ms': int(row.start_ms), 'end_ms': int(row.end_ms),
             'mean_log_error': float(row.error), 'cases': int(row.cases),
             'event_group_weight': 1.} for row in means.itertuples(index=False)],
        'group_states': group_states, 'target_states': row_states}
    return np.asarray(results, dtype=float), diagnostics
