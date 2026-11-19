"""Fixed production recipe fitting, invoked only by an explicit training command."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
import joblib
import numpy as np
from catboost import CatBoostRegressor
from analysis.backtest import Curve, snapshot
from analysis.data_policy import MIN_EVENT_START_MS
from analysis.event_features import EventFeatures
from analysis.temporal_dataset import origin_groups
from analysis.holdout_dataset import curve_record, sha256, prefix_curve
from analysis.innovation_common import prepare, event_weights
from analysis.multirank_model import (event_weights as shared_weights, make_preprocessor,
                                    model_prediction, prepare as prepare_shared)
from analysis.rolling_benchmark import fit_context
from analysis import v2_next_adaptation as adaptation
from download_ranks import RANKS
from production.model import (ROOT, SCHEMA_VERSION, POLICY_ID, METADATA_NAMES,
    _sources, _software, _identity, _write_json, _sample_record, _base_predictions, verify_pack)


def _valid_curve(curve, before_ms):
    """The dense production quality gate uses actual dates, never event IDs."""
    times = np.asarray(curve.times)
    scores = np.asarray(curve.scores, dtype=float)
    return bool(curve.start >= MIN_EVENT_START_MS and curve.start < curve.end < before_ms
                and times.ndim == scores.ndim == 1 and len(times) == len(scores) >= 5
                and np.isfinite(times).all() and np.isfinite(scores).all()
                and np.all(np.diff(times) > 0) and np.all(scores >= 0) and scores[-1] > 0
                and curve.end - 60_000 <= times[-1] <= curve.end + 300_000
                and times[-1] <= before_ms)

def _history(curves, before_ms, ranks):
    result, seen, event_times = [], set(), {}
    for curve in curves:
        if curve.rank not in ranks or not _valid_curve(curve, before_ms):
            continue
        key = (int(curve.event), int(curve.rank))
        if key in seen:
            raise ValueError('Duplicate eligible history event/rank')
        seen.add(key)
        dates = (int(curve.start), int(curve.end))
        if event_times.setdefault(int(curve.event), dates) != dates:
            raise ValueError('History ranks disagree on event dates')
        # Never retain mutable arrays owned by the data store or caller.
        result.append(Curve(int(curve.event), int(curve.rank), str(curve.kind), *dates,
                            np.asarray(curve.times, dtype=np.int64).copy(),
                            np.asarray(curve.scores, dtype=float).copy()))
    return sorted(result, key=lambda c: (c.start, c.event, c.rank))

def _samples(curves, *, unsupported_labels=None):
    result = []
    for curve in curves:
        for group in origin_groups(curve):
            cutoff = group['cutoff']
            if not curve.start < cutoff < curve.end:
                continue
            prefix = prefix_curve(curve, cutoff)
            if prefix is None:
                continue
            sample = snapshot(prefix, cutoff)
            if sample is None:
                continue
            # Verified public scores may decrease after cheating removals or
            # corrections. The fixed nonnegative-growth decoder cannot fit a
            # prefix already above its actual final. Omit only that supervised
            # case; never inflate the final or rewrite the complete history.
            if sample['score'] > curve.final:
                if unsupported_labels is not None:
                    unsupported_labels.append({'event': curve.event, 'rank': curve.rank,
                        'cutoff': cutoff, 'observed_at': sample['now'],
                        'current_ep': sample['score'], 'verified_final_ep': curve.final})
                continue
            sample.update(case_id=f'production:{curve.event}:{curve.rank}:{cutoff}',
                          stage=group['stages'][0]['stage_id'])
            result.append(sample)
    return result

def _frozen_metadata(directory, events):
    metadata = EventFeatures.load(directory)
    missing = set(events) - metadata.parties.keys()
    if missing:
        raise ValueError('Missing training event metadata: ' + str(sorted(missing)))
    metadata.character_ids = sorted({c for e in events for c in metadata.parties[e]['characters']})
    metadata.band_ids = sorted({b for e in events for b in metadata.parties[e]['bands']})
    return metadata

def _labels(samples, curves):
    finals = {(c.event, c.rank): c.final for c in curves}
    return {s['case_id']: finals[(s['curve'].event, s['curve'].rank)] for s in samples}

def _weights(cases, weights):
    return [{'case_id': c['case_id'], 'event': int(c['event']), 'rank': int(c['rank']),
             'weight': float(w)} for c, w in zip(cases, weights)]

def _config(ranks, iterations):
    if iterations is not None and (isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1):
        raise ValueError('Fixture iterations must be a positive integer')
    return {'ranks': ranks, 'context': adaptation.recipes()[0] |
            {'iterations': 400 if iterations is None else int(iterations)},
            'shared': {'features': 'rank', 'iterations': 600 if iterations is None else int(iterations),
                       'depth': 5, 'learning_rate': .035, 'l2_leaf_reg': 12, 'seed': 42, 'thread_count': 2},
            'sampling': 'temporal_dataset.origin_groups; unique physical cutoff; snapshot >=4 points and <=60m age',
            'corrected_score_policy': 'production only: omit supervised prefixes above verified final; retain actual final and raw history',
            'weights': 'equal total event weight separately for each fitted family',
            'minimal_support_for_confident': 20,
            'fixture': iterations is not None,
            'calibration': {'alpha': .3, 'shrinkage_events': 10, 'clip': [-.3, .3],
                            'phase_edges': [.25, .9], 'source': 'verified issued base forecasts only'},
            'policy_id': POLICY_ID}


def train_pack(curves, metadata_directory, versions_directory, *, before_ms, ranks=None, iterations=None):
    """Fit fixed recipes and atomically publish one content-addressed directory.

    ``iterations`` is solely a fast fixture override: any supplied value marks
    the resulting pack ``production_ready=False``. No active pointer is changed.
    Identical eligible content/config/source/dependencies returns the verified
    existing directory without fitting, regardless of polling wall clock.
    """
    ranks = sorted(set(RANKS if ranks is None else map(int, ranks)))
    if not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError('Expected positive rank scope')
    before_ms = int(before_ms)
    history = _history(list(curves), before_ms, ranks)
    if not history:
        raise ValueError('No eligible completed dense training histories')
    unsupported_labels = []
    samples = _samples(history, unsupported_labels=unsupported_labels)
    if not samples:
        raise ValueError('No usable temporal training prefixes')
    metadata_directory, versions_directory = Path(metadata_directory), Path(versions_directory)
    config = _config(ranks, iterations)
    source_hashes, software = _sources(), _software()
    metadata_values = {name: json.loads((metadata_directory / (name + '.json')).read_text())
                       for name in METADATA_NAMES}
    # Query-event metadata comes from the coordinator at inference. Changes to
    # unrelated future events must not cause an otherwise identical refit.
    event_ids = {str(c.event) for c in history}
    metadata_values['events'] = {key: value for key, value in metadata_values['events'].items()
                                 if key in event_ids}
    identity = {'schema_version': SCHEMA_VERSION, 'policy_id': POLICY_ID,
                'history_digest': _identity([curve_record(c) for c in history]),
                'metadata_digest': _identity(metadata_values), 'config': config,
                'source_hashes': source_hashes, 'software': software}
    model_id = _identity(identity)
    versions_directory.mkdir(parents=True, exist_ok=True)
    destination = versions_directory / model_id
    if destination.exists():
        verify_pack(destination)
        return destination
    stage = Path(tempfile.mkdtemp(prefix='.training-', dir=versions_directory))
    try:
        events = sorted({c.event for c in history})
        (stage / 'metadata').mkdir()
        for name, value in metadata_values.items():
            _write_json(stage / 'metadata' / (name + '.json'), value)
        metadata = _frozen_metadata(stage / 'metadata', events)
        joblib.dump(metadata, stage / 'metadata.joblib', compress=3)
        for name in source_hashes:
            target = stage / 'sources' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        labels = _labels(samples, history)
        weight_rows = {'context': [], 'shared': []}
        fitted = {}
        context_samples = [s for s in samples if s['curve'].rank in (1000, 2000)]
        if context_samples:
            arrays = prepare(context_samples, metadata)
            arrays['history'] = [c for c in history if c.rank in (1000, 2000)]
            bundle = fit_context(arrays, labels, iterations=config['context']['iterations'])
            fitted['context'] = bundle
            joblib.dump({k: v for k, v in bundle.items() if k != 'history'}, stage / 'context.joblib', compress=3)
            weight_rows['context'] = _weights(arrays['cases'], event_weights(arrays['cases']))
        # The full shared rank recipe learns all available configured ranks.
        # The two dedicated ranks still route to Context when issuing forecasts.
        shared_samples = samples if any(s['curve'].rank not in (1000, 2000) for s in samples) else []
        if shared_samples:
            arrays = prepare_shared(shared_samples, metadata)
            actual = np.array([labels[c['case_id']] for c in arrays['cases']])
            if not np.isfinite(actual).all() or np.any(actual < arrays['current']):
                raise ValueError('Final training scores must be finite and at least current')
            target = np.log(actual / arrays['current']) + np.log(arrays['progress'])
            preprocessor = make_preprocessor(arrays['rank'])
            x = preprocessor.fit_transform(arrays['rank'])
            conf = config['shared']
            model = CatBoostRegressor(iterations=conf['iterations'], depth=conf['depth'],
                learning_rate=conf['learning_rate'], l2_leaf_reg=conf['l2_leaf_reg'],
                random_seed=conf['seed'], thread_count=conf['thread_count'], verbose=False,
                allow_writing_files=False)
            weights = shared_weights([c['event'] for c in arrays['cases']])
            model.fit(x, target, sample_weight=weights)
            fitted['shared'] = {'model': model, 'preprocessor': preprocessor, 'features': 'rank'}
            joblib.dump(fitted['shared'], stage / 'shared.joblib', compress=3)
            weight_rows['shared'] = _weights(arrays['cases'], weights)
        _write_json(stage / 'training_weights.json', weight_rows)
        omitted_counts = Counter(f"{row['event']}:{row['rank']}" for row in unsupported_labels)
        training_quality = {'scope': 'production training only; frozen benchmark cases are unchanged',
            'reason': 'observed prefix score exceeds verified final after public score correction; fixed decoder requires nonnegative growth',
            'policy': 'omit unsupported supervised prefix only; retain actual final and untouched complete history',
            'unsupported_label_cases': len(unsupported_labels),
            'unsupported_label_cases_by_event_rank': dict(sorted(omitted_counts.items())),
            'unsupported_label_case_details': unsupported_labels,
            'retained_supervised_cases': len(samples),
            'event_weights': 'recalculated over retained cases, equal total activity mass per fitted family'}
        _write_json(stage / 'training_quality.json', training_quality)
        counts, fitted_events = Counter(), defaultdict(set)
        for sample in samples:
            rank = sample['curve'].rank
            counts[rank] += 1
            fitted_events[rank].add(sample['curve'].event)
        support = {str(rank): {'events': len(fitted_events[rank]), 'cases': counts[rank]}
                   for rank in ranks}
        _write_json(stage / 'curve_support.json', [{'event': c.event, 'rank': c.rank,
            'start_ms': c.start, 'end_ms': c.end, 'observations': len(c.times),
            'final_observed_at': int(c.times[-1]), 'curve_digest': _identity(curve_record(c))}
            for c in history])
        # A bounded prefix per fitted rank, replayed after serialization. These
        # are technical fit samples and are never used for model selection or EWMA.
        smoke = [next(s for s in samples if s['curve'].rank == rank)
                 for rank in ranks if counts[rank]]
        before_reload = {}
        for family in fitted:
            subset = [s for s in smoke if (s['curve'].rank in (1000, 2000)) == (family == 'context')]
            if not subset:
                continue
            values = (adaptation.predict_recipe(fitted[family], prepare(subset, metadata))
                      if family == 'context' else
                      model_prediction(fitted[family], prepare_shared(subset, metadata)))
            before_reload.update((s['case_id'], float(v)) for s, v in zip(subset, values))
        _write_json(stage / 'smoke.json', {'scope': 'technical serialization replay; no accuracy claim',
                    'samples': [_sample_record(s) for s in smoke], 'predictions': before_reload})
        after_reload = _base_predictions(stage, smoke, joblib.load(stage / 'metadata.joblib'))
        if before_reload != after_reload or len(after_reload) != len(smoke):
            raise ValueError('Model serialization smoke replay differs')
        manifest = identity | {'model_id': model_id, 'created_ms': int(time.time() * 1000),
            'training_before_ms': before_ms, 'training_events': len(events),
            'history_event_ids': events, 'history_max_end_ms': max(c.end for c in history),
            'ranks': ranks, 'rank_support': support, 'production_ready': iterations is None,
            'training_quality': training_quality,
            'supervised_events': len({s['curve'].event for s in samples}),
            'limitations': ['Production fit is not a new accuracy evaluation.',
                'Historical cohort was previously explored; metadata is a retrospective snapshot.'],
            'artifacts': {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}}
        manifest['seal_id'] = _identity(manifest)
        _write_json(stage / 'manifest.json', manifest)
        verify_pack(stage)
        try:
            stage.rename(destination)
        except FileExistsError:
            verify_pack(destination)
        return destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)

