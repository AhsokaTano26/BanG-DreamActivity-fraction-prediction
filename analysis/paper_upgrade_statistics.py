"""Independent, append-only scoring and fixed-issued-loss sensitivity analyses.

This module is an outcome-reading scorer. Predictors must never import it.
Resampling acts on sealed event losses; it does not rerun online histories.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/rolling_holdout_v3_dense_19rank_20261117'
RECORDS = ROOT / 'analysis/paper_upgrade_records/20261118'
OLD_RUN = ROOT / 'analysis/rolling_results/v3_task_learning_20261118'
OBSERVATIONS = ROOT / 'analysis/cross_rank_records/20261117/results/observations'
CONTROLS = ('uncorrected', 'ewma_reference', 'legacy_pair')
BLOCKS = (1, 3, 5, 7)
DENSE_START = 1706504400000
LIMITATIONS = [
    'Previously explored test cohort; repeatedly reused validation; retrospective metadata.',
    'Exploratory intervals conditional on fixed issued predictions and methods; no search adjustment.',
    'Event resampling does not regenerate chronological online training or feedback histories.',
    'Only 16 test events; circular blocks of length 5/7 contain few effective long segments.',
    'Stage memberships overlap; unique physical case IDs determine overall metrics.',
]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score_rows(predictions, cases, labels):
    """Exact frozen-panel join; no method-specific deletion or duplicate cases."""
    p, c, labels = predictions.copy(), cases.copy(), labels.copy()
    for frame, columns in [(p, ['case_id', 'method', 'predicted']),
                           (c, ['case_id', 'event', 'rank']), (labels, ['case_id', 'actual'])]:
        if not set(columns) <= set(frame):
            raise ValueError(f'Missing required columns: {columns}')
    if c.empty or p.empty or labels.empty:
        raise ValueError('Empty score panel')
    if c.case_id.duplicated().any() or labels.case_id.duplicated().any() or p.duplicated(['method', 'case_id']).any():
        raise ValueError('duplicate frozen cases/predictions/labels')
    expected = set(c.case_id)
    if set(labels.case_id) != expected:
        raise ValueError('Label coverage differs from frozen cases')
    if p[['case_id', 'method']].isna().any().any():
        raise ValueError('Missing prediction identity')
    for method, part in p.groupby('method'):
        if set(part.case_id) != expected:
            raise ValueError(f'{method}: exact frozen coverage required')
    if 'actual' in p:
        raise ValueError('Prediction artifact contains outcome actual')
    # Validate rather than silently overwrite identities supplied by a predictor.
    overlap = [name for name in c if name != 'case_id' and name in p]
    indexed = c.set_index('case_id')
    for name in overlap:
        expected_values = p.case_id.map(indexed[name])
        if pd.api.types.is_numeric_dtype(expected_values):
            same = np.isclose(p[name].to_numpy(float), expected_values.to_numpy(float),
                              rtol=1e-12 if name == 'current' else 0,
                              atol=1e-6 if name == 'current' else 0, equal_nan=False)
        else:
            same = (p[name].to_numpy() == expected_values.to_numpy())
        if not np.all(same):
            raise ValueError(f'Prediction {name} differs from frozen case metadata')
    result = p.drop(columns=overlap).merge(c, on='case_id', validate='many_to_one')
    result = result.merge(labels[['case_id', 'actual']], on='case_id', validate='many_to_one')
    values = result[['predicted', 'actual']].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Positive finite predictions and actuals required')
    result['error'] = result.predicted - result.actual
    result['abs_error'] = result.error.abs()
    result['squared_error'] = result.error ** 2
    result['ape_pct'] = result.abs_error / result.actual * 100.
    return result


def event_losses(frame):
    columns = ['method', 'event']
    result = frame.groupby(columns, sort=False).agg(
        mape_pct=('ape_pct', 'mean'), mae_ep=('abs_error', 'mean'),
        mse_ep2=('squared_error', 'mean'), p90_ape_pct=('ape_pct', lambda x: x.quantile(.9)),
        cases=('case_id', 'size'), ranks=('rank', 'nunique')).reset_index()
    if 'start_ms' in frame:
        clocks = frame.groupby('event').start_ms.agg(['min', 'max'])
        if (clocks['min'] != clocks['max']).any():
            raise ValueError('Event has inconsistent actual starts')
        result['start_ms'] = result.event.map(clocks['min'])
    rank_means = frame.groupby(['method', 'event', 'rank']).ape_pct.mean()
    rank_balanced = rank_means.groupby(['method', 'event']).mean()
    result['rank_balanced_mape_pct'] = [rank_balanced.loc[m, e] for m, e in zip(result.method, result.event)]
    return result


def metrics(frame):
    if frame.empty:
        return {'events': 0, 'cases': 0}
    if frame.method.nunique() != 1 or frame.case_id.duplicated().any():
        raise ValueError('Metrics require one method and unique physical cases')
    events = event_losses(frame)
    return {
        'event_macro_mape_pct': float(events.mape_pct.mean()),
        'event_rank_balanced_mape_pct': float(events.rank_balanced_mape_pct.mean()),
        'case_mape_pct': float(frame.ape_pct.mean()),
        'mae_ep': float(frame.abs_error.mean()),
        'rmse_ep': float(np.sqrt(frame.squared_error.mean())),
        'p90_ape_pct': float(frame.ape_pct.quantile(.9)),
        'event_macro_mae_ep': float(events.mae_ep.mean()),
        'event_equal_rmse_ep': float(np.sqrt(events.mse_ep2.mean())),
        'event_mean_p90_ape_pct': float(events.p90_ape_pct.mean()),
        'events': int(frame.event.nunique()), 'cases': len(frame),
        'ranks': int(frame['rank'].nunique()),
    }


def aggregate_metrics(frame, keys):
    """Vectorized equivalent of metrics for many small subgroup panels."""
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(keys, dropna=False, sort=False)
    result = grouped.agg(case_mape_pct=('ape_pct', 'mean'), mae_ep=('abs_error', 'mean'),
                         mse=('squared_error', 'mean'), events=('event', 'nunique'),
                         cases=('case_id', 'size'), ranks=('rank', 'nunique'))
    result['rmse_ep'] = np.sqrt(result.pop('mse'))
    result['p90_ape_pct'] = grouped.ape_pct.quantile(.9)
    event_keys = list(dict.fromkeys([*keys, 'event']))
    eg = frame.groupby(event_keys, dropna=False, sort=False)
    ev = eg.agg(event_macro_mape_pct=('ape_pct', 'mean'),
                event_macro_mae_ep=('abs_error', 'mean'), mse=('squared_error', 'mean'))
    ev['event_mean_p90_ape_pct'] = eg.ape_pct.quantile(.9)
    ev = ev.groupby(keys, dropna=False, sort=False).mean()
    ev['event_equal_rmse_ep'] = np.sqrt(ev.pop('mse'))
    rank_keys = list(dict.fromkeys([*event_keys, 'rank']))
    rank_mean = frame.groupby(rank_keys, dropna=False, sort=False).ape_pct.mean()
    rank_mean = rank_mean.groupby(event_keys, dropna=False, sort=False).mean()
    ev['event_rank_balanced_mape_pct'] = rank_mean.groupby(keys, dropna=False, sort=False).mean()
    return result.join(ev).reset_index()


def group_metrics(frame, *, members=None):
    rows = []
    for name in ['rank', 'phase', 'event_type', 'band_group']:
        if name in frame:
            value = aggregate_metrics(frame, ['method', name]).rename(columns={name: 'value'})
            value['group'], value['value'] = name, value.value.astype(str)
            rows.append(value)
    if members is not None:
        expanded = _expand_memberships(frame, members)
        value = aggregate_metrics(expanded, ['method', 'stage_id']).rename(columns={'stage_id': 'value'})
        value['group'], value['value'] = 'stage_id', value.value.astype(str)
        rows.append(value)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _expand_memberships(frame, members):
    if members.duplicated(['case_id', 'stage_id']).any() or not set(members.case_id) <= set(frame.case_id):
        raise ValueError('Invalid stage memberships')
    return frame.merge(members[['case_id', 'stage_id']], on='case_id', validate='many_to_many')


def stage_rank_metrics(frame, members):
    expanded = _expand_memberships(frame, members)
    return aggregate_metrics(expanded, ['method', 'stage_id', 'rank']).rename(columns={'stage_id': 'stage'})


def interval_reliability(frame):
    """All-case and finite-supported coverage are separate estimands."""
    required = {'lower', 'upper', 'nominal_coverage'}
    if not required <= set(frame):
        return {'status': 'not_provided', 'events': int(frame.event.nunique()), 'cases': len(frame)}
    lo, hi = frame.lower.to_numpy(float), frame.upper.to_numpy(float)
    nominal = frame.nominal_coverage.to_numpy(float)
    if (np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo > hi)
            or not np.isfinite(nominal).all() or np.any((nominal <= 0) | (nominal >= 1))):
        raise ValueError('Invalid interval bounds/nominal coverage')
    if len(np.unique(nominal)) != 1:
        raise ValueError('Report one nominal interval level at a time')
    tmp = frame.copy()
    tmp['covered'] = (tmp.actual >= lo) & (tmp.actual <= hi)
    tmp['width'] = hi - lo
    finite = np.isfinite(lo) & np.isfinite(hi)
    available = tmp.available.to_numpy(bool) if 'available' in tmp else finite
    supported = available & finite
    if np.any(available & ~finite):
        raise ValueError('Available interval must have finite bounds')
    result = {'status': 'available', 'nominal_coverage': float(nominal[0]),
              'events': int(tmp.event.nunique()), 'cases': len(tmp),
              'supported_cases': int(supported.sum()), 'unsupported_cases': int((~supported).sum()),
              'available_rate': float(supported.mean()),
              'case_coverage': float(tmp.covered.mean()),
              'event_macro_coverage': float(tmp.groupby('event').covered.mean().mean()),
              'unbounded_cases': int((~finite).sum()),
              'finite_rate': float(finite.mean()),
              'case_undercoverage': float((tmp.actual < lo).mean()),
              'case_overcoverage': float((tmp.actual > hi).mean()),
              'mean_width_ep': float(tmp.width.mean()) if finite.all() else None}
    subset = tmp[supported]
    result.update(supported_events=int(subset.event.nunique()),
                  supported_case_coverage=float(subset.covered.mean()) if len(subset) else None,
                  supported_event_macro_coverage=float(subset.groupby('event').covered.mean().mean()) if len(subset) else None,
                  supported_mean_width_ep=float(subset.width.mean()) if len(subset) else None)
    for column in ['support', 'own_support']:
        if column in tmp:
            result[column + '_min'] = float(tmp[column].min())
            result[column + '_median'] = float(tmp[column].median())
    if len(subset):
        result['supported_mean_relative_width_pct'] = float((subset.width / subset.actual * 100).mean())
        result['supported_event_mean_width_ep'] = float(subset.groupby('event').width.mean().mean())
    else:
        result['supported_mean_relative_width_pct'] = result['supported_event_mean_width_ep'] = None
    return result


def joint_interval_coverage(frame):
    """Match the historical event/rank/phase maximum-error calibration unit."""
    if frame.empty or not {'event', 'rank', 'phase', 'actual', 'lower', 'upper'} <= set(frame):
        raise ValueError('Joint interval coverage requires nonempty event/rank/phase rows')
    part = frame.copy()
    part['covered'] = (part.actual >= part.lower) & (part.actual <= part.upper)
    groups = part.groupby(['event', 'rank', 'phase'], sort=False).agg(
        jointly_covered=('covered', 'all'), prefixes=('covered', 'size')).reset_index()
    events = groups.groupby('event', sort=False).agg(
        group_joint_coverage=('jointly_covered', 'mean'),
        entire_event_joint_coverage=('jointly_covered', 'all'),
        groups=('jointly_covered', 'size'), prefixes=('prefixes', 'sum')).reset_index()
    events['pointwise_coverage'] = events.event.map(part.groupby('event').covered.mean())
    if 'start_ms' in part:
        events['start_ms'] = events.event.map(part.groupby('event').start_ms.min())
        events = events.sort_values(['start_ms', 'event'])
    summary = {'cases': len(part), 'events': len(events), 'groups': len(groups),
               'case_pointwise_coverage': float(part.covered.mean()),
               'event_equal_pointwise_coverage': float(events.pointwise_coverage.mean()),
               'group_equal_joint_coverage': float(groups.jointly_covered.mean()),
               'event_equal_group_joint_coverage': float(events.group_joint_coverage.mean()),
               'entire_event_joint_coverage': float(events.entire_event_joint_coverage.mean())}
    return groups, events, summary


def resample_indices(n, draws, block, *, seed=42):
    if n < 1 or draws < 1 or block < 1 or block > n:
        raise ValueError('Require n>=block>=1 and draws>=1')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, math.ceil(n / block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def paired_effect(events, *, draws=5000, seed=42, blocks=BLOCKS):
    if events.event.duplicated().any() or len(events) < 2:
        raise ValueError('Unique paired event losses required')
    sort = ['start_ms', 'event'] if 'start_ms' in events else ['event']
    ordered = events.sort_values(sort)
    differences = (ordered.method_loss - ordered.reference_loss).to_numpy(float)
    if not np.isfinite(differences).all():
        raise ValueError('Finite paired differences required')
    n = len(differences)
    loo = (differences.sum() - differences) / (n - 1)
    absolute = np.abs(differences)
    gains = np.maximum(0., -differences)
    result = {
        'events': n, 'event_order': ordered.event.astype(int).tolist(),
        'event_differences': differences.tolist(), 'mean_difference_points': float(differences.mean()),
        'improved_events': int((differences < -1e-12).sum()),
        'worse_events': int((differences > 1e-12).sum()),
        'tied_events': int((np.abs(differences) <= 1e-12).sum()),
        'median_difference_points': float(np.median(differences)),
        'leave_one_out_differences': loo.tolist(), 'leave_one_out_min': float(loo.min()),
        'leave_one_out_max': float(loo.max()),
        'top1_absolute_share': float(absolute.max() / absolute.sum()) if absolute.sum() else 0.,
        'top3_absolute_share': float(np.sort(absolute)[-3:].sum() / absolute.sum()) if absolute.sum() else 0.,
        'top1_gain_share': float(gains.max() / gains.sum()) if gains.sum() else 0.,
        'bootstrap': {}, 'multiple_comparisons_adjusted': False,
        'interval_scope': 'exploratory fixed issued event losses; online histories not rerun',
    }
    if draws:
        for block in blocks:
            if block > n:
                result['bootstrap'][str(block)] = {'status': 'insufficient_events'}
                continue
            means = differences[resample_indices(n, draws, block, seed=seed)].mean(axis=1)
            result['bootstrap'][str(block)] = {
                'status': 'available', 'draws': draws, 'seed': seed,
                'scheme': 'paired_iid' if block == 1 else 'paired_circular_moving_block',
                'ci95': np.quantile(means, [.025, .975]).tolist(),
                'resample_mass_below_zero': float((means < 0).mean()),
                'effective_blocks_approx': n / block,
            }
    return result


def matched_panels(frames, clock):
    """One jointly complete event/rank support, sorted by actual CN start."""
    if not frames:
        raise ValueError('At least one panel required')
    for frame in frames.values():
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError('Duplicate event or rank')
    common_events = set.intersection(*(set(f.index) for f in frames.values()))
    ranks = sorted(set.intersection(*(set(f.columns) for f in frames.values())))
    if not ranks:
        raise ValueError('No common ranks')
    ordered = sorted(common_events, key=lambda e: (clock[int(e)]['start_ms'], int(e)))
    valid = [e for e in ordered if all(np.isfinite(f.loc[e, ranks].to_numpy(float)).all() for f in frames.values())]
    if len(valid) < 3:
        raise ValueError('Fewer than three jointly supported events')
    union = set.union(*(set(f.index) for f in frames.values()))
    result = {name: frame.loc[valid, ranks].copy() for name, frame in frames.items()}
    return result, {'event_ids': [int(e) for e in valid], 'ranks': [int(r) for r in ranks],
                    'events': len(valid), 'rank_count': len(ranks),
                    'excluded_events': sorted(int(e) for e in union - set(valid)),
                    'input_events': {name: len(frame) for name, frame in frames.items()},
                    'order': 'actual CN start_ms then event ID'}


def correlation_rows(panels):
    rows = []
    for name, frame in panels.items():
        for i, rank_a in enumerate(frame.columns):
            for rank_b in frame.columns[i + 1:]:
                x, y = frame[rank_a].to_numpy(float), frame[rank_b].to_numpy(float)
                # Constant columns have undefined correlation and remain explicit.
                varying = np.ptp(x) > 0 and np.ptp(y) > 0
                rows.append({'panel': name, 'rank_a': int(rank_a), 'rank_b': int(rank_b), 'n': len(frame),
                             'spearman': float(stats.spearmanr(x, y).statistic) if varying else np.nan,
                             'pearson_log_scale': float(np.corrcoef(x, y)[0, 1]) if varying else np.nan,
                             'base_family_pair': ('context_context' if rank_a in [1000, 2000] and rank_b in [1000, 2000]
                                                  else 'mixed' if rank_a in [1000, 2000] or rank_b in [1000, 2000]
                                                  else 'shared_shared')})
    return pd.DataFrame(rows)


def correlation_bootstrap(panels, *, draws=5000, seed=42, blocks=BLOCKS):
    """Paired activity resampling for correlations; rank pairs are not sample units."""
    names = list(panels)
    first = panels[names[0]]
    if not all(f.index.equals(first.index) and f.columns.equals(first.columns) for f in panels.values()):
        raise ValueError('Correlation bootstrap requires exact jointly matched support')
    n, nranks = first.shape
    if n < 3 or nranks < 2:
        raise ValueError('At least three events and two ranks required')
    pair_index = np.triu_indices(nranks, 1)
    points = correlation_rows(panels)
    first_pairs = points[points.panel == names[0]].reset_index(drop=True)
    output, contrasts = [], []
    for block in blocks:
        if block > n:
            continue
        indices = resample_indices(n, draws, block, seed=seed)
        samples = {}
        for name, frame in panels.items():
            values = frame.to_numpy(float)
            batches = []
            for idx in np.array_split(indices, max(1, math.ceil(draws / 128))):
                ranks = stats.rankdata(values[idx], axis=1)
                centered = ranks - ranks.mean(axis=1, keepdims=True)
                norms = np.sqrt(np.sum(centered ** 2, axis=1, keepdims=True))
                normalized = np.divide(centered, norms, out=np.full_like(centered, np.nan), where=norms > 0)
                corr = np.einsum('bni,bnj->bij', normalized, normalized, optimize=True)
                batches.append(np.clip(corr[:, pair_index[0], pair_index[1]], -1., 1.))
            samples[name] = np.concatenate(batches)
            for j, row in first_pairs.iterrows():
                finite = samples[name][:, j]
                finite = finite[np.isfinite(finite)]
                ci = np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]
                output.append({'panel': name, 'rank_a': int(row.rank_a), 'rank_b': int(row.rank_b),
                               'n': n, 'block': block, 'draws': draws, 'valid_draws': len(finite),
                               'low95': ci[0], 'high95': ci[1]})
        # Median across fixed rank pairs, with uncertainty resampled by activities.
        for name in names[1:]:
            for family in ['all', *sorted(first_pairs.base_family_pair.unique())]:
                mask = np.ones(len(first_pairs), bool) if family == 'all' else first_pairs.base_family_pair.to_numpy() == family
                # Omit degenerate all-constant bootstrap draws, and record their count.
                a, b = samples[name][:, mask], samples[names[0]][:, mask]
                valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
                delta = np.median(a[valid], axis=1) - np.median(b[valid], axis=1)
                ci = np.quantile(delta, [.025, .975]).tolist() if len(delta) else [None, None]
                selected = points[points.panel == name]
                reference = points[points.panel == names[0]]
                if family != 'all':
                    selected = selected[selected.base_family_pair == family]
                    reference = reference[reference.base_family_pair == family]
                contrasts.append({'panel': name, 'reference': names[0], 'base_family_pair': family,
                                  'contrast': 'median rank-pair Spearman difference', 'n': n,
                                  'rank_pairs': int(mask.sum()), 'block': block, 'draws': draws,
                                  'valid_draws': int(valid.sum()),
                                  'difference': float(selected.spearman.median() - reference.spearman.median()),
                                  'low95': ci[0], 'high95': ci[1], 'multiple_comparisons_adjusted': False})
    return pd.DataFrame(output), pd.DataFrame(contrasts)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_bundle(output, tables, summary, *, sources):
    """Never overwrite even an interrupted prior destination."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Preserve existing statistics: {output}')
    output.mkdir(parents=True)
    (output / 'scorer_source.py').write_bytes(Path(__file__).read_bytes())
    for name, frame in tables.items():
        if Path(name).name != name:
            raise ValueError('Flat output table name required')
        frame.to_csv(output / name, index=False)
    (output / 'summary.json').write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy']}
    manifest = {'schema_version': 1, 'scorer_sha256': sha256(Path(__file__)),
                'python': platform.python_version(), 'dependencies': versions,
                'inputs': {str(Path(p)): sha256(p) for p in sorted(set(map(Path, sources)))},
                'artifacts': {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
                'limitations': LIMITATIONS}
    manifest['analysis_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return summary


def _sealed_json(path):
    obj = json.loads(Path(path).read_text())
    if 'seal_id' in obj:
        content = {k: v for k, v in obj.items() if k != 'seal_id'}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if digest != obj['seal_id']:
            raise ValueError(f'Manifest seal changed: {path}')
    return obj


def _read_bound_csv(path, manifest, root, sources):
    path, root = Path(path), Path(root)
    relative = str(path.relative_to(root))
    if manifest['artifacts'].get(relative) != sha256(path):
        raise ValueError(f'Artifact hash differs: {path}')
    sources.append(path)
    return pd.read_csv(path, float_precision='round_trip', low_memory=False,
                       dtype={'event_type': 'string', 'band_group': 'string'})


def verify_prediction_run(run):
    """Check full root/configuration seals and frozen model sources before labels."""
    run = Path(run)
    blind = _sealed_json(run / 'blind_manifest.json')
    checked = [run / 'blind_manifest.json']
    manifests = [blind]
    if (run / 'configuration_seal.json').exists():
        manifests.append(_sealed_json(run / 'configuration_seal.json'))
    for manifest in manifests:
        for name, expected in manifest['artifacts'].items():
            path = run / name
            if not path.resolve().is_relative_to(run.resolve()) or sha256(path) != expected:
                raise ValueError(f'Prediction artifact hash differs: {path}')
            checked.append(path)
    if (run / 'run_manifest.json').exists():
        configuration = json.loads((run / 'run_manifest.json').read_text())
        if configuration.get('run_id') != blind.get('run_id'):
            raise ValueError('Run/configuration identity differs')
        for name, expected in configuration.get('source_hashes', {}).items():
            archive = run / 'sources' / name
            if sha256(archive) != expected:
                raise ValueError(f'Frozen model source hash differs: {archive}')
            checked.append(archive)
    return blind, sorted(set(checked))


def planned_comparisons(methods, metadata=None):
    """Declare contrasts by identifiers and design, never by observed losses."""
    methods = set(methods)
    pairs = [(m, 'ewma_reference', 'fixed_primary') for m in sorted(methods) if 'ewma_reference' in methods]
    extra = [('legacy_pair', 'pair_anchor_only'), ('legacy_pair', 'pair_difference_only'),
             ('legacy_pair', 'pair_swapped_anchor'), ('same_filter_pair', 'same_panel_independent'),
             ('history_growth', 'time_proportion')]
    pairs.extend((a, b, 'structure_diagnostic') for a, b in extra if a in methods and b in methods)
    pairs.extend((m, m.split('__')[0], 'feedback_sensitivity') for m in sorted(methods)
                 if '__' in m and m.split('__')[0] in methods)
    if metadata is not None and not metadata.empty:
        records = metadata.to_dict('records')
        for row in records:
            for ref in records:
                same_seed = str(row['seed']) == str(ref['seed'])
                if row['family'] == ref['family'] and same_seed:
                    if row['training_events'] == ref['training_events'] and row['structure'] != 'shared' and ref['structure'] == 'shared':
                        pairs.append((row['method'], ref['method'], 'learning_structure'))
                    if row['structure'] == ref['structure'] and row['training_events'] < ref['training_events'] == 67:
                        pairs.append((row['method'], ref['method'], 'learning_size'))
    return list(dict.fromkeys(pairs))


def score_run(run, output, *, data=DATA, methods=None, draws=5000, reference='ewma_reference', splits=None):
    """Trusted scorer alone opens held-out labels, after blind-manifest binding."""
    run, data = Path(run), Path(data)
    blind_path, data_path = run / 'blind_manifest.json', data / 'manifest.json'
    blind, checked_sources = verify_prediction_run(run)
    dataset = _sealed_json(data_path)
    if blind['dataset_id'] != dataset['dataset_id']:
        raise ValueError('Frozen dataset identity differs')
    sources = [*checked_sources, data_path]
    tables, summary = {}, {'dataset_id': dataset['dataset_id'], 'run': str(run), 'draws': draws,
                           'blocks': list(BLOCKS), 'seed': 42, 'reference': reference,
                           'limitations': LIMITATIONS, 'splits': {}}
    splits = list(splits or blind.get('scored_splits', ['validation', 'test']))
    allowed = set(blind.get('scored_splits', ['validation', 'test']))
    if not splits or not set(splits) <= allowed:
        raise ValueError('Scoring split not allowed by blind seal')
    for split in splits:
        predictions = _read_bound_csv(run / f'{split}_predictions.csv', blind, run, sources)
        if 'methods' in blind and set(predictions.method) != set(blind['methods']):
            raise ValueError('Incomplete sealed method menu')
        if methods:
            if not set(methods) <= set(predictions.method):
                raise ValueError('Requested methods absent')
            predictions = predictions[predictions.method.isin(methods)]
        cases = _read_bound_csv(data / f'{split}_cases.csv', dataset, data, sources)
        if (cases.start_ms < DENSE_START).any():
            raise ValueError('Dense period policy violated')
        members = _read_bound_csv(data / f'{split}_memberships.csv', dataset, data, sources)
        labels = _read_bound_csv(data / f'{split}_labels.csv', dataset, data, sources)
        scored = score_rows(predictions, cases, labels)
        interval_path = run / f'{split}_intervals.csv'
        interval_summary, interval_groups = [], []
        if interval_path.exists():
            intervals = _read_bound_csv(interval_path, blind, run, sources)
            for (method, level), part in intervals.groupby(['method', 'nominal_coverage']):
                available_scores = scored[scored.method == method]
                if available_scores.empty:
                    continue
                if part.case_id.duplicated().any() or set(part.case_id) != set(cases.case_id):
                    raise ValueError('Interval exact frozen coverage required')
                columns = ['case_id', 'lower', 'upper', 'nominal_coverage'] + [c for c in ['available', 'support', 'own_support', 'order_index', 'source'] if c in part]
                joined = available_scores.merge(part[columns], on='case_id', validate='one_to_one')
                interval_summary.append({'method': method, **interval_reliability(joined)})
                for name in [n for n in ['rank', 'phase', 'event_type', 'band_group'] if n in joined]:
                    for value, group in joined.groupby(name, sort=False):
                        interval_groups.append({'method': method, 'group': name, 'value': str(value), **interval_reliability(group)})
                expanded = members[['case_id', 'stage_id']].merge(joined, on='case_id', validate='many_to_one')
                for value, group in expanded.groupby('stage_id', sort=False):
                    interval_groups.append({'method': method, 'group': 'stage_id', 'value': str(value), **interval_reliability(group)})
        events = event_losses(scored)
        metrics_table = aggregate_metrics(scored, ['method'])
        metadata_columns = [c for c in ['training_events', 'family', 'structure', 'seed'] if c in predictions]
        if metadata_columns:
            metadata = predictions[['method', *metadata_columns]].drop_duplicates()
            if metadata.method.duplicated().any():
                raise ValueError('Method has inconsistent learning metadata')
            metrics_table = metrics_table.merge(metadata, on='method', validate='one_to_one')
        method_summary = metrics_table.to_dict('records')
        effects = []
        metadata = predictions[['method', *metadata_columns]].drop_duplicates() if metadata_columns else None
        for method, comparator, scope in planned_comparisons(events.method.unique(), metadata):
            ref = events[events.method == comparator].set_index('event')
            part = events[events.method == method]
            if set(part.event) != set(ref.index):
                raise ValueError('Contrast event coverage differs')
            for loss, label in [('mape_pct', 'activity_case_weighting'), ('rank_balanced_mape_pct', 'within_activity_equal_rank_sensitivity')]:
                paired = part[['event', 'start_ms', loss]].rename(columns={loss: 'method_loss'})
                paired['reference_loss'] = paired.event.map(ref[loss])
                effects.append({'method': method, 'reference': comparator, 'scope': scope, 'weighting': label,
                                **paired_effect(paired, draws=draws)})
        summary['splits'][split] = {'metrics': method_summary, 'paired_effects': effects,
                                    'intervals': interval_summary or [{'status': 'not_provided'}]}
        tables.update({f'{split}_case_losses.csv': scored, f'{split}_event_losses.csv': events,
                       f'{split}_metrics.csv': metrics_table,
                       f'{split}_groups.csv': group_metrics(scored, members=members),
                       f'{split}_stage_rank.csv': stage_rank_metrics(scored, members),
                       f'{split}_interval_reliability.csv': pd.DataFrame(interval_summary),
                       f'{split}_interval_groups.csv': pd.DataFrame(interval_groups)})
        if metadata_columns:
            tables['learning_metrics.csv'] = metrics_table.assign(split=split)
        print(json.dumps({'scored_split': split, 'methods': scored.method.nunique(),
                          'cases_per_method': len(cases), 'events': cases.event.nunique()}, default=int), flush=True)
    return write_bundle(output, tables, summary, sources=sources)


def correlation_audit(output, *, observations=OBSERVATIONS, data=DATA, draws=0):
    """Train-only matched35 diagnostic; no new labels or held-out outcomes."""
    observations, data = Path(observations), Path(data)
    sources = [data / 'manifest.json']
    manifest = _sealed_json(sources[0])
    cases = _read_bound_csv(data / 'train_cases.csv', manifest, data, sources)
    clock = cases.drop_duplicates('event').set_index('event')[['start_ms', 'end_ms']].to_dict('index')
    frames = {}
    for label, name in [('log_final', 'train_log_final'), ('remaining_growth_p50', 'train_remaining_growth_p50'),
                        ('raw_error_phase1', 'train_error_raw_phase1'),
                        ('independent_error_phase1', 'train_error_independent_phase1')]:
        path = observations / f'{name}.csv'
        sources.append(path)
        frame = pd.read_csv(path, index_col='event', float_precision='round_trip')
        frame.columns = frame.columns.astype(int)
        frames[label] = frame
    matched, support = matched_panels(frames, clock)
    if not set(support['event_ids']) <= set(manifest['splits']['train']['event_ids']):
        raise ValueError('Matched diagnostic contains nontraining event')
    if any(clock[e]['start_ms'] < DENSE_START for e in support['event_ids']):
        raise ValueError('Matched diagnostic violates dense-period policy')
    rows = correlation_rows(matched)
    summaries = rows.groupby(['panel', 'base_family_pair']).agg(
        median_spearman=('spearman', 'median'), min_spearman=('spearman', 'min'),
        max_spearman=('spearman', 'max'), rank_pairs=('spearman', 'count')).reset_index()
    tables = {'matched_correlation_pairs.csv': rows, 'matched_correlation_summary.csv': summaries}
    if draws:
        intervals, contrasts = correlation_bootstrap(matched, draws=draws)
        tables['matched_correlation_intervals.csv'] = intervals
        tables['matched_correlation_contrasts.csv'] = contrasts
    for name, frame in matched.items():
        tables[f'matched_{name}.csv'] = frame.rename_axis('event').reset_index()
    support['scope'] = 'same activities and ranks across final/growth p50/raw phase1/independent phase1; training only'
    support['diagnostic_limit'] = 'Target transformation and base model families differ; correlations are descriptive.'
    support['resampling'] = {'draws': draws, 'blocks': list(BLOCKS), 'seed': 42} if draws else 'not run; heavy numeric permission pending'
    return write_bundle(output, tables, {'support': support, 'limitations': LIMITATIONS}, sources=sources)


def joint_and_learning_supplement(output, *, root=RECORDS, draws=5000):
    """Additional prespecified joint-cover and same-N ensemble contrasts."""
    root = Path(root)
    fixed, learning = root / 'results/statistics/fixed_replay_v2', root / 'results/statistics/learning_curve_v2'
    run = root / 'runs/fixed_replay_v2'
    bound = _sealed_json(run / 'blind_manifest.json')
    sources = [run / 'blind_manifest.json', fixed / 'manifest.json', learning / 'manifest.json']
    summaries, event_tables, group_tables, coverage_intervals = [], [], [], []
    for split in ['validation', 'test']:
        path = fixed / f'{split}_case_losses.csv'
        fixed_manifest = json.loads((fixed / 'manifest.json').read_text())
        if sha256(path) != fixed_manifest['artifacts'][path.name]:
            raise ValueError('Independent score artifact changed')
        sources.append(path)
        columns = ['case_id', 'method', 'event', 'rank', 'phase', 'start_ms', 'actual']
        actual = pd.read_csv(path, usecols=columns, low_memory=False)
        actual = actual[actual.method == 'ewma_reference']
        intervals = _read_bound_csv(run / f'{split}_intervals.csv', bound, run, sources)
        for level, values in intervals.groupby('nominal_coverage', sort=False):
            if values.case_id.duplicated().any() or set(values.case_id) != set(actual.case_id):
                raise ValueError('Interval joint exact coverage differs')
            joined = actual.merge(values[['case_id', 'lower', 'upper']], on='case_id', validate='one_to_one')
            groups, events, summary = joint_interval_coverage(joined)
            tags = {'split': split, 'method': 'ewma_reference', 'nominal_coverage': float(level)}
            summaries.append({**tags, **summary})
            group_tables.append(groups.assign(**tags))
            event_tables.append(events.assign(**tags))
            for target in ['pointwise_coverage', 'group_joint_coverage', 'entire_event_joint_coverage']:
                losses = events[target].to_numpy(float)
                for block in BLOCKS:
                    samples = losses[resample_indices(len(events), draws, block, seed=42)].mean(axis=1)
                    low, high = np.quantile(samples, [.025, .975])
                    coverage_intervals.append({**tags, 'target': target, 'block': block, 'draws': draws,
                                               'events': len(events), 'coverage': float(losses.mean()),
                                               'low95': float(low), 'high95': float(high),
                                               'scope': 'fixed issued interval outcomes; exploratory activity resampling'})
    metrics_path, events_path = learning / 'learning_metrics.csv', learning / 'validation_event_losses.csv'
    learning_manifest = json.loads((learning / 'manifest.json').read_text())
    for path in [metrics_path, events_path]:
        if sha256(path) != learning_manifest['artifacts'][path.name]:
            raise ValueError('Learning score artifact changed')
        sources.append(path)
    meta, event_losses_frame = pd.read_csv(metrics_path), pd.read_csv(events_path)
    effects, flat = [], []
    for n in [16, 32, 48, 67]:
        for method_structure, ref_structure in [('independent', 'shared'), ('rank_heads', 'shared'), ('independent', 'rank_heads')]:
            method, reference = f'neural_{method_structure}_n{n}_mean', f'neural_{ref_structure}_n{n}_mean'
            if not {method, reference} <= set(meta.method):
                raise ValueError('Missing predefined ensemble learning contrast')
            part = event_losses_frame[event_losses_frame.method == method]
            ref = event_losses_frame[event_losses_frame.method == reference].set_index('event')
            paired = part[['event', 'start_ms', 'mape_pct']].rename(columns={'mape_pct': 'method_loss'})
            paired['reference_loss'] = paired.event.map(ref.mape_pct)
            effect = {'method': method, 'reference': reference, 'training_events': n,
                      'scope': 'same-N validation; arithmetic prediction ensemble; common14events across all N',
                      **paired_effect(paired, draws=draws)}
            effects.append(effect)
            for block, value in effect['bootstrap'].items():
                flat.append({k: effect[k] for k in ['method', 'reference', 'training_events', 'events',
                                                  'mean_difference_points', 'improved_events', 'leave_one_out_min', 'leave_one_out_max']}
                            | {'block': int(block), 'low95': value['ci95'][0], 'high95': value['ci95'][1], 'draws': draws})
    tables = {'interval_joint_summary.csv': pd.DataFrame(summaries),
              'interval_joint_event.csv': pd.concat(event_tables, ignore_index=True),
              'interval_joint_groups.csv': pd.concat(group_tables, ignore_index=True),
              'interval_coverage_intervals.csv': pd.DataFrame(coverage_intervals),
              'learning_ensemble_contrasts.csv': pd.DataFrame(flat)}
    return write_bundle(output, tables, {'draws': draws, 'seed': 42, 'blocks': list(BLOCKS),
                        'learning_contrasts': effects, 'interval_joint_summary': summaries,
                        'joint_scope': 'event/rank/phase all frozen prefixes is calibration target; entire-event cover is separate stricter diagnostic',
                        'learning_scope': '14 reused validation events; N levels share evaluation activities; independent models differ in parameter count/total budget',
                        'limitations': LIMITATIONS}, sources=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    scoring = sub.add_parser('score')
    scoring.add_argument('--run', type=Path, default=OLD_RUN)
    scoring.add_argument('--data', type=Path, default=DATA)
    scoring.add_argument('--output', type=Path, required=True)
    scoring.add_argument('--methods', nargs='+')
    scoring.add_argument('--draws', type=int, default=5000)
    scoring.add_argument('--splits', nargs='+', choices=['validation', 'test'])
    corr = sub.add_parser('correlations')
    corr.add_argument('--output', type=Path, required=True)
    corr.add_argument('--draws', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'score':
        result = score_run(args.run, args.output, data=args.data, methods=args.methods, draws=args.draws, splits=args.splits)
        print(json.dumps({'output': str(args.output), 'methods_by_split': {s:len(v['metrics']) for s,v in result['splits'].items()}}, ensure_ascii=False))
    else:
        result = correlation_audit(args.output, draws=args.draws)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
