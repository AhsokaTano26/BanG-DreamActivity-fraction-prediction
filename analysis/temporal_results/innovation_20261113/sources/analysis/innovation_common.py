"""Blind-prefix adapters and activity-grouped utilities for innovation experiments."""
from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.backtest import ROOT
from analysis.multirank_model import feature_row


def prepare(samples, metadata):
    samples = list(samples)
    if not samples:
        raise ValueError('Expected nonempty blind prefix samples')
    cases, bands, rows = [], [], []
    for sample in samples:
        c = sample['curve']
        if c.rank not in (1000, 2000) or c.times[-1] > sample['cutoff']:
            raise ValueError('Unexpected rank or future observation')
        if c.times[-1] != sample['now']:
            raise ValueError('Blind curve must end at its observed time')
        cases.append({'case_id': sample['case_id'], 'event': c.event, 'rank': c.rank,
                      'start_ms': c.start, 'end_ms': c.end})
        party = metadata.parties.get(c.event, {'bands': {}})
        bands.append({str(k): float(v) for k,v in party['bands'].items()})
        row=feature_row(sample)
        row.pop('event_type', None)  # Context is introduced only in named ablations.
        rows.append(row)
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate physical input case')
    return {'samples': samples, 'cases': cases, 'base': pd.DataFrame(rows),
            'band_shares': bands, 'event_type': np.array([s['curve'].kind for s in samples]),
            'current': np.array([s['score'] for s in samples], dtype=float),
            'progress': np.array([s['progress'] for s in samples], dtype=float),
            'duration_h': np.array([s['curve'].duration/3600000 for s in samples]),
            'remaining_h': np.array([(s['curve'].end-s['now'])/3600000 for s in samples]),
            'rank': np.array([0 if s['curve'].rank == 1000 else 1 for s in samples]),
            'history': []}


def subset_arrays(arrays, indices, *, history=None):
    indices = np.asarray(indices, dtype=int)
    result = {}
    for name, value in arrays.items():
        if name == 'history':
            result[name] = list(value if history is None else history)
        elif isinstance(value, pd.DataFrame):
            result[name] = value.iloc[indices].reset_index(drop=True)
        elif isinstance(value, np.ndarray):
            result[name] = value[indices]
        elif isinstance(value, list):
            result[name] = [value[i] for i in indices]
        else:
            raise TypeError('Unsupported array field '+name)
    return result


def chronological_folds(event_rows):
    index = {}
    for row in event_rows:
        row = {k: int(row[k]) for k in ['event', 'start_ms', 'end_ms']}
        if row['end_ms'] <= row['start_ms']:
            raise ValueError('Invalid activity duration')
        if row['event'] in index and row != index[row['event']]:
            raise ValueError('Inconsistent event dates')
        index[row['event']] = row
    ordered = sorted(index, key=lambda e: (index[e]['start_ms'], e))
    if len(ordered) < 6:
        raise ValueError('Too few activities for three temporal blocks')
    warmup = len(ordered)//2
    result = []
    for number, block in enumerate(np.array_split(ordered[warmup:], 3), 1):
        heldout = [int(e) for e in block]
        start = min(index[e]['start_ms'] for e in heldout)
        train = [e for e in ordered if index[e]['end_ms'] < start]
        result.append({'fold': number, 'training_events': train, 'heldout_events': heldout,
                       'training_latest_end': max(index[e]['end_ms'] for e in train),
                       'heldout_earliest_start': start})
    return result


def event_weights(cases):
    events = np.array([c['event'] for c in cases])
    _, inverse, counts = np.unique(events, return_inverse=True, return_counts=True)
    weights = 1/counts[inverse]
    return weights/weights.mean()


def event_mape(arrays, labels, prediction):
    actual = np.array([labels[c['case_id']] for c in arrays['cases']])
    prediction = np.asarray(prediction)
    if (prediction.shape != actual.shape or not np.isfinite(prediction).all()
            or np.any(prediction <= 0) or not np.isfinite(actual).all() or np.any(actual <= 0)):
        raise ValueError('Aligned positive final scores required')
    events = np.array([c['event'] for c in arrays['cases']])
    _, inverse, counts = np.unique(events, return_inverse=True, return_counts=True)
    return float(np.mean(np.bincount(inverse, weights=np.abs(prediction/actual-1)*100)/counts))


def prediction_rows(arrays, method, prediction):
    prediction = np.asarray(prediction)
    if (prediction.shape != np.asarray(arrays['current']).shape or not np.isfinite(prediction).all()
            or (prediction < np.asarray(arrays['current'])-1e-7).any()):
        raise ValueError('Invalid or below-current final prediction')
    return pd.DataFrame({'case_id': [c['case_id'] for c in arrays['cases']],
                         'method': method, 'predicted': prediction})


def local_sources(paths):
    """Archive all statically imported local dependencies, including lazy imports."""
    pending = [Path(p).resolve() for p in paths]
    found = set()
    while pending:
        path = pending.pop()
        if path in found or not path.is_file():
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
                if node.module == 'analysis':
                    names += ['analysis.'+x.name for x in node.names]
            pending.extend(ROOT/(name.replace('.', '/')+'.py') for name in names)
    return sorted(str(p.relative_to(ROOT)) for p in found)
