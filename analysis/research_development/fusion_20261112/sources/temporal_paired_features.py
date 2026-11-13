"""Pair only contemporaneous blind prefixes; never inspect a future peer suffix."""
from collections import Counter

import numpy as np
import pandas as pd

from analysis.backtest import snapshot
from analysis.holdout_dataset import prefix_curve
from analysis.multirank_model import feature_row

LENGTH = 33
SEQUENCE_COLUMNS = [f'{prefix}{kind}_{i:02}' for prefix in ('', 'peer__')
                    for kind in ('prefix', 'present') for i in range(LENGTH)]


def prepare_paired(samples):
    samples = list(samples)
    if not samples or len({s['case_id'].split(':')[0] for s in samples}) != 1:
        raise ValueError('Provide one nonempty split at a time')
    index = {}
    for s in samples:
        c = s['curve']
        key = (c.event, c.rank, s['cutoff'])
        if c.rank not in (1000, 2000) or key in index:
            raise ValueError('Unexpected rank or duplicate event/rank/origin')
        if s['times'][-1] > s['cutoff']:
            raise ValueError('Future primary observation')
        index[key] = s
    single, paired, audits, cases = [], [], [], []
    for s in samples:
        c = s['curve']; base = feature_row(s); row = dict(base)
        peer_source = index.get((c.event, 3000-c.rank, s['cutoff']))
        peer = None; reason = 'no same-origin peer case'
        if peer_source is not None:
            pc = peer_source['curve']
            if pc.start != c.start or pc.end != c.end:
                raise ValueError('Peer activity timing differs')
            truncated = prefix_curve(pc, s['now'])
            peer = snapshot(truncated, s['now']) if truncated is not None else None
            reason = 'available' if peer else 'peer invalid after as-of truncation'
        peer_row = feature_row(peer) if peer else {k: None for k in base}
        row.update({'peer__'+k: v for k, v in peer_row.items() if k != 'event_type'})
        row['peer_available'] = int(peer is not None)
        row['peer_age_hours'] = (s['now']-peer['now'])/3600000 if peer else None
        grid = np.linspace(c.start, s['now'], LENGTH)
        for prefix, part in [('', s), ('peer__', peer)]:
            for i, t in enumerate(grid):
                available = part is not None and part['times'][0] <= t <= part['now']
                row[f'{prefix}present_{i:02}'] = int(available)
                row[f'{prefix}prefix_{i:02}'] = float(np.interp(t, part['times'], part['scores']) / part['score']) if available else 0.
        for key in ['pair_log_score_ratio', 'pair_gap_relative'] + [f'pair_{kind}_{h}h' for h in (1, 3, 6) for kind in ('log_rate_ratio', 'log_ratio_change')]:
            row[key] = None
        if peer is not None and peer['now'] >= s['times'][0]:
            t = peer['now']
            own_score = float(np.interp(t, s['times'], s['scores']))
            peer_score = peer['score']
            ratio = np.log((own_score+1)/(peer_score+1))
            row['pair_log_score_ratio'] = float(np.log(own_score/peer_score)) if own_score > 0 else None
            row['pair_gap_relative'] = (own_score-peer_score)/max(own_score, peer_score, 1)
            for h in (1, 3, 6):
                before = t-h*3600000
                if before < max(s['times'][0], peer['times'][0]):
                    continue
                a = float(np.interp(before, s['times'], s['scores']))
                b = float(np.interp(before, peer['times'], peer['scores']))
                ar, br = max(0., own_score-a)/h, max(0., peer_score-b)/h
                row[f'pair_log_rate_ratio_{h}h'] = np.log1p(ar)-np.log1p(br)
                row[f'pair_log_ratio_change_{h}h'] = ratio-np.log((a+1)/(b+1))
        single.append(base); paired.append(row)
        cases.append({'case_id': s['case_id'], 'event': c.event, 'rank': c.rank})
        audits.append(cases[-1] | {'cutoff': s['cutoff'], 'observed_at': s['now'],
            'peer_case_id': peer_source['case_id'] if peer_source else None,
            'peer_observed_at': peer['now'] if peer else None,
            'peer_points': len(peer['times']) if peer else 0, 'peer_available': int(peer is not None), 'reason': reason})
    return {'single': pd.DataFrame(single), 'paired': pd.DataFrame(paired),
        'audit': pd.DataFrame(audits), 'cases': cases,
        'current': np.array([s['score'] for s in samples]),
        'progress': np.array([s['progress'] for s in samples]),
        'rank': np.array([0 if s['curve'].rank == 1000 else 1 for s in samples], dtype=np.int64)}


def sequence_arrays(frame):
    branches = []
    for prefix in ('', 'peer__'):
        values = frame[[f'{prefix}prefix_{i:02}' for i in range(LENGTH)]].fillna(0).to_numpy()
        mask = frame[[f'{prefix}present_{i:02}' for i in range(LENGTH)]].to_numpy()
        branches.append(np.stack([values, mask], axis=-1))
    return np.stack(branches, axis=1).astype(np.float32)


def phase_weights(events, progress):
    events, progress = list(events), np.asarray(progress)
    if len(events) != len(progress) or not events or not np.isfinite(progress).all() or np.any((progress <= 0) | (progress > 1)):
        raise ValueError('Invalid event/progress arrays')
    phase = np.where(progress < .25, 0, np.where(progress < .9, 1, 2))
    counts = Counter(zip(events, phase))
    available = Counter(e for e, p in counts)
    weight = np.array([1/(available[e]*counts[e, p]) for e, p in zip(events, phase)])
    return weight/weight.mean()
