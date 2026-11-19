"""Immutable production model packs over explicitly supplied detached histories.

This module never opens a database, benchmark labels or a history vault.  The
coordinator owns collection, promotion, event pins and issued-forecast feedback.
Technical smoke replay is a serialization check, never an accuracy estimate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import tempfile
import time

import joblib
import numpy as np

from analysis.backtest import Curve, ROOT, snapshot
from analysis.data_policy import MIN_EVENT_START_MS, POLICY_ID
from analysis.event_features import EventFeatures
from analysis.holdout_dataset import curve_record, from_record, prefix_curve, sha256
from analysis.innovation_common import local_sources, prepare
from analysis.multirank_model import model_prediction, prepare as prepare_shared
from analysis import v2_next_adaptation as adaptation
from download_ranks import RANKS
from production.source_compat import sources_compatible

SCHEMA_VERSION = 1
METADATA_NAMES = ('events', 'characters', 'bands', 'cards')
DEPENDENCIES = ('numpy', 'pandas', 'scikit-learn', 'catboost', 'joblib', 'scipy', 'xgboost')


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _software():
    return {'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name) for name in DEPENDENCIES}}


def _sources():
    paths = [Path(__file__), ROOT / 'production/__init__.py', ROOT / 'training/__init__.py']
    return {name: sha256(ROOT / name) for name in local_sources(paths)}


def _base_predictions(directory, samples, metadata):
    """Replay fitted models; public causal checks occur before entering here."""
    predictions = {}
    context = [s for s in samples if s['curve'].rank in (1000, 2000)]
    shared = [s for s in samples if s['curve'].rank not in (1000, 2000)]
    if context and (directory / 'context.joblib').exists():
        bundle = joblib.load(directory / 'context.joblib')
        values = adaptation.predict_recipe(bundle, prepare(context, metadata))
        predictions.update((s['case_id'], float(v)) for s, v in zip(context, values))
    if shared and (directory / 'shared.joblib').exists():
        bundle = joblib.load(directory / 'shared.joblib')
        values = model_prediction(bundle, prepare_shared(shared, metadata))
        predictions.update((s['case_id'], float(v)) for s, v in zip(shared, values))
    for s in samples:
        value = predictions.get(s['case_id'])
        if value is not None and (not math.isfinite(value) or value < s['score']):
            raise ValueError('Model returned invalid or below-current prediction')
    return predictions


def _sample_record(sample):
    return curve_record(sample['curve']) | {k: sample[k] for k in ('case_id', 'cutoff', 'stage')}


def _record_sample(record):
    sample = snapshot(from_record(record), record['cutoff'])
    if sample is None:
        raise ValueError('Invalid smoke prefix')
    sample.update(case_id=record['case_id'], stage=record['stage'])
    return sample


def train_pack(curves, metadata_directory, versions_directory, *, before_ms, ranks=None, iterations=None):
    """Compatibility API; actual fitting lives in training/model_fit.py."""
    from training.model_fit import train_pack as fit
    return fit(curves, metadata_directory, versions_directory,
               before_ms=before_ms, ranks=ranks, iterations=iterations)


def technical_replay_matches(actual, expected):
    """Allow CPU floating-point roundoff in replay without changing predictions."""
    if not isinstance(actual, dict) or not isinstance(expected, dict) or actual.keys() != expected.keys():
        return False
    try:
        return all(math.isfinite(actual[key]) and math.isfinite(expected[key])
                   and math.isclose(actual[key], expected[key], rel_tol=1e-12, abs_tol=1e-8)
                   for key in actual)
    except (TypeError, ValueError, OverflowError):
        return False


def verify_pack(path):
    """Verify every artifact before deserialization; reject old-policy packs."""
    path = Path(path)
    try:
        manifest = json.loads((path / 'manifest.json').read_text())
        seal = {k: v for k, v in manifest.items() if k != 'seal_id'}
        if _identity(seal) != manifest.get('seal_id'):
            raise ValueError('Model manifest changed or corrupt')
        if manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('policy_id') != POLICY_ID:
            raise ValueError('Unsafe model pack policy or schema')
        identity_keys = ('schema_version', 'policy_id', 'history_digest', 'metadata_digest',
                         'config', 'source_hashes', 'software')
        if _identity({k: manifest[k] for k in identity_keys}) != manifest['model_id']:
            raise ValueError('Model content identity differs')
        if (manifest['history_max_end_ms'] >= manifest['training_before_ms']
                or manifest['training_events'] != len(set(manifest['history_event_ids']))):
            raise ValueError('Unsafe training history clock')
        artifacts = manifest['artifacts']
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p != path / 'manifest.json'}
        if actual != set(artifacts):
            raise ValueError('Model artifact set changed')
        for name, digest in artifacts.items():
            artifact = path / name
            if (Path(name).is_absolute() or '..' in Path(name).parts or artifact.is_symlink()
                    or not artifact.is_file() or sha256(artifact) != digest):
                raise ValueError('Model artifact changed or corrupt: ' + name)
        if manifest['software'] != _software() or not sources_compatible(manifest['source_hashes'], _sources()):
            raise ValueError('Model code/dependency fingerprint differs; retraining required')
        support = json.loads((path / 'curve_support.json').read_text())
        if (not support or any(r['start_ms'] < MIN_EVENT_START_MS
                              or not r['start_ms'] < r['end_ms'] < manifest['training_before_ms']
                              for r in support)):
            raise ValueError('Unsafe model training support')
        smoke = json.loads((path / 'smoke.json').read_text())
        samples = [_record_sample(r) for r in smoke['samples']]
        values = _base_predictions(path, samples, joblib.load(path / 'metadata.joblib'))
        if (not samples or len(values) != len(samples)
                or not technical_replay_matches(values, smoke['predictions'])):
            preview = [(key, values.get(key), smoke['predictions'].get(key))
                       for key in sorted(values.keys() | smoke['predictions'].keys())
                       if values.get(key) != smoke['predictions'].get(key)][:3]
            raise ValueError('Model smoke replay changed; (case, actual, expected): ' + repr(preview))
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid or corrupt model pack: ' + str(path)) from error


def _blind_samples(samples, manifest):
    result, keys = [], set()
    for supplied in samples:
        curve, cutoff = supplied['curve'], int(supplied['cutoff'])
        if (curve.start < MIN_EVENT_START_MS or curve.start <= manifest['history_max_end_ms']
                or curve.event in manifest['history_event_ids']):
            raise ValueError('Query must exclude own training history and start strictly after all history ends')
        times, scores = np.asarray(curve.times), np.asarray(curve.scores, dtype=float)
        if (curve.end <= curve.start or not curve.start < cutoff < curve.end
                or times.ndim != 1 or scores.ndim != 1 or len(times) != len(scores) or len(times) < 4
                or not np.isfinite(times).all() or not np.isfinite(scores).all()
                or np.any(np.diff(times) <= 0) or np.any(scores < 0) or times[-1] > cutoff):
            raise ValueError('Expected finite ordered blind prefix without future observations')
        prefix = Curve(int(curve.event), int(curve.rank), str(curve.kind), int(curve.start), int(curve.end),
                       times.astype(np.int64).copy(), scores.copy())
        sample = snapshot(prefix, cutoff)
        if sample is None:
            raise ValueError('Unusable or stale blind prefix')
        sample.update(case_id=str(supplied.get('case_id', f'live:{curve.event}:{curve.rank}:{cutoff}')),
                      stage=str(supplied.get('stage', 'live')))
        if sample['case_id'] in keys:
            raise ValueError('Duplicate blind prefix case')
        keys.add(sample['case_id'])
        result.append(sample)
    return result


def predict_pack(path, samples, *, metadata_directory=None, calibration=None):
    """Return bounded raw and issued scores; unknown ranks have no forecast."""
    path = Path(path)
    manifest = verify_pack(path)
    samples = _blind_samples(samples, manifest)
    if not samples:
        return []
    fitted_metadata = joblib.load(path / 'metadata.joblib')
    metadata = EventFeatures.load(metadata_directory) if metadata_directory is not None else fitted_metadata
    metadata.character_ids = list(fitted_metadata.character_ids)
    metadata.band_ids = list(fitted_metadata.band_ids)
    metadata._calendar_cache = {}
    if any(s['curve'].event not in metadata.parties for s in samples):
        raise ValueError('Current query event is missing metadata; provide current metadata_directory')
    eligible = [s for s in samples if manifest['rank_support'].get(str(s['curve'].rank), {}).get('cases', 0)]
    predictions = _base_predictions(path, eligible, metadata)
    result = []
    for sample in samples:
        curve = sample['curve']
        phase = int(np.searchsorted([.25, .9], sample['progress'], side='right'))
        raw = predictions.get(sample['case_id'])
        bias = bias_for(calibration or {}, curve.rank, phase)
        # The coordinator pins a calibration rebuilt strictly before this event.
        if calibration and calibration.get('latest_end_ms') is not None:
            if calibration['latest_end_ms'] >= curve.start:
                raise ValueError('Calibration includes current or future event outcomes')
        predicted = raw
        if raw is not None and bias:
            log_score = min(math.log(sample['score']) + 4, math.log(raw) - bias,
                            math.log(np.finfo(float).max))
            predicted = max(sample['score'], math.exp(log_score))
        support = manifest['rank_support'].get(str(curve.rank), {}).get('events', 0)
        low_support = support < manifest['config']['minimal_support_for_confident']
        result.append({'case_id': sample['case_id'], 'event': curve.event, 'rank': curve.rank,
            'start_ms': curve.start, 'end_ms': curve.end, 'cutoff': sample['cutoff'],
            'observed_at': sample['now'], 'current_ep': sample['score'], 'phase': phase,
            'progress': sample['progress'], 'raw_predicted_final_ep': raw,
            'predicted_final_ep': predicted, 'log_bias': bias if raw is not None else 0.,
            'model_id': manifest['model_id'], 'training_events': manifest['training_events'],
            'rank_training_events': support,
            'provisional': not manifest['production_ready'] or low_support, 'available': raw is not None,
            'reason': ('No eligible training prefixes for configured rank' if raw is None else
                       'insufficient historical activities for this rank' if low_support else None)})
    return result


def build_bias(feedback_rows, *, before_ms, exclude_event_id=None):
    """Rebuild event-group EWMA in completion order, including late arrivals.

    Inputs are coordinator-verified event/rank/phase means of log(base/actual),
    never training residuals. Repeated identical groups are idempotent;
    conflicting duplicates require the coordinator to resolve its ledger.
    """
    unique = {}
    for supplied in feedback_rows:
        row = dict(supplied)
        event, rank, phase = int(row['event']), int(row['rank']), int(row['phase'])
        start, end, error = int(row['start_ms']), int(row['end_ms']), float(row['mean_log_error'])
        if start < MIN_EVENT_START_MS or end >= int(before_ms) or event == exclude_event_id:
            continue
        if start >= end or not math.isfinite(error) or rank <= 0 or phase not in (0, 1, 2):
            raise ValueError('Invalid completed event feedback group')
        record = {'event': event, 'rank': rank, 'phase': phase, 'start_ms': start, 'end_ms': end,
                  'mean_log_error': error, 'cases': int(row.get('cases', row.get('n_forecasts', 1))),
                  'origin': row.get('origin', 'issued')}
        if record['cases'] < 1:
            raise ValueError('Feedback requires at least one issued forecast')
        key = (event, rank, phase)
        if key in unique and unique[key] != record:
            raise ValueError('Conflicting duplicate event/rank/phase feedback')
        unique[key] = record
    records = sorted(unique.values(), key=lambda r: (r['end_ms'], r['event'], r['rank'], r['phase']))
    groups = {}
    for row in records:
        key = f"{row['rank']}:{row['phase']}"
        previous = groups.get(key)
        value = row['mean_log_error'] if previous is None else .7 * previous['mean'] + .3 * row['mean_log_error']
        n = 1 if previous is None else previous['count'] + 1
        groups[key] = {'mean': value, 'count': n, 'bias': float(np.clip(value * n / (n + 10), -.3, .3))}
    return {'schema_version': 1, 'policy_id': POLICY_ID, 'before_ms': int(before_ms),
            'groups': groups, 'records': records, 'latest_end_ms': max((r['end_ms'] for r in records), default=None),
            'source': 'verified issued uncorrected forecast event/rank/phase means',
            'calibration_id': _identity(records)}


def bias_for(calibration, rank, phase):
    """Return bounded shrinking EWMA correction for a JSON calibration state."""
    if calibration and (calibration.get('schema_version') != 1 or calibration.get('policy_id') != POLICY_ID):
        raise ValueError('Uncertified calibration policy or schema')
    group = calibration.get('groups', {}).get(f'{int(rank)}:{int(phase)}')
    if group is None:
        return 0.
    count, mean = int(group['count']), float(group['mean'])
    if count < 1 or not math.isfinite(mean):
        raise ValueError('Invalid calibration group')
    return float(np.clip(mean * count / (count + 10), -.3, .3))
