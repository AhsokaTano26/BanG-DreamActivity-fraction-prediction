"""Preregistered V3 static/expanding controls with sealed causal feedback.

Only the coordinator can open the history vault. Pure model functions receive
released training arrays/labels and the current blind prefixes. Initial training
activities are warmup, never resubstitution calibration or reported test cases.
Incomplete runs intentionally fail closed: preserve them and choose a new run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd

from analysis.backtest import ROOT
from analysis.event_features import EventFeatures
from analysis.holdout_dataset import sha256
from analysis.innovation_common import prepare, event_weights, event_mape, local_sources
from analysis.multirank_dataset import identity
from analysis.rolling_dataset import RollingDataset, DEFAULT
from analysis import v2_next_adaptation as adaptation

METHODS = ['static_context', 'rolling_context', 'static_context_ewma', 'rolling_context_ewma']
BASES = ['static_context', 'rolling_context']
PREDICTION_COLUMNS = ['case_id', 'event', 'rank', 'phase', 'method', 'current', 'predicted', 'log_bias']


def write_new_json(path, content):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def seal_directory(directory, filename, fields):
    directory = Path(directory)
    body = dict(fields) | {'artifacts': {str(p.relative_to(directory)): sha256(p)
        for p in sorted(directory.rglob('*')) if p.is_file()}}
    body['seal_id'] = identity(body)
    write_new_json(directory/filename, body)
    return body


def verify_seal(path):
    path = Path(path)
    body = json.loads(path.read_text())
    if identity({k: v for k, v in body.items() if k != 'seal_id'}) != body.get('seal_id'):
        raise ValueError('Sealed manifest changed: '+str(path))
    for name, digest in body['artifacts'].items():
        if sha256(path.parent/name) != digest:
            raise ValueError('Sealed artifact changed: '+str(path.parent/name))
    return body


class OnlineBias:
    """One log(predicted/actual) mean per completed event/rank/phase."""
    def __init__(self):
        self.groups = {}
        self.events = set()
        self.records = []
        self.latest_end = None

    def observe(self, rows, labels, *, event, end_ms):
        if event in self.events:
            raise ValueError('Duplicate event feedback')
        if self.latest_end is not None and end_ms < self.latest_end:
            raise ValueError('Feedback must follow completion chronology')
        if rows.case_id.duplicated().any() or set(rows.event) - {event}:
            raise ValueError('Feedback requires one event and unique case IDs')
        if set(labels) != set(rows.case_id):
            raise ValueError('Feedback labels must exactly match sealed event cases')
        rows = rows.copy()
        actual = np.array([labels[c] for c in rows.case_id], dtype=float)
        predicted = rows.predicted.to_numpy(float)
        if (not np.isfinite(actual).all() or not np.isfinite(predicted).all()
                or np.any(actual <= 0) or np.any(predicted <= 0)):
            raise ValueError('Feedback requires positive finite scores')
        rows['error'] = np.log(predicted)-np.log(actual)
        for (rank, phase), part in rows.groupby(['rank', 'phase']):
            key = (int(rank), int(phase)); error = float(part.error.mean())
            previous, n = self.groups.get(key, (0., 0))
            value = .7*previous+.3*error if n else error
            self.groups[key] = (value, n+1)
            self.records.append({'event': int(event), 'end_ms': int(end_ms),
                'rank': key[0], 'phase': key[1], 'mean_log_pred_over_actual': error,
                'cases': len(part), 'event_group_weight': 1.})
        self.events.add(event); self.latest_end = end_ms

    def bias(self, rank, phase):
        value, n = self.groups.get((int(rank), int(phase)), (0., 0))
        return float(np.clip(value*n/(n+10), -.3, .3))


def fit_context(arrays, labels, *, iterations=400):
    """Pure fixed CatBoost context recipe; no filesystem or dataset argument."""
    config = adaptation.recipes()[0] | {'iterations': iterations}
    return adaptation.fit_recipe(config, arrays, labels)


def _labels(arrays, history):
    finals = {(c.event, c.rank): c.final for c in history}
    return {c['case_id']: finals[(c['event'], c['rank'])] for c in arrays['cases']}


def _training(samples_by_event, history, metadata):
    allowed = {c.event for c in history}
    samples = [s for event, values in samples_by_event.items() if event in allowed for s in values]
    arrays = prepare(samples, metadata)
    arrays['history'] = list(history)
    return arrays, _labels(arrays, history)


def _weight_rows(arrays):
    return pd.DataFrame([{'case_id': c['case_id'], 'event': c['event'], 'weight': float(w)}
        for c, w in zip(arrays['cases'], event_weights(arrays['cases']))])


def _save_model(bundle, path):
    # The complete histories already live in the sealed data vault. Keep the
    # model/preprocessor/summaries/provenance; avoid duplicating raw curves.
    joblib.dump({k: v for k, v in bundle.items() if k != 'history'}, path, compress=3)


def _configuration(data, iterations, allow_fixture):
    if iterations < 1 or int(iterations) != iterations:
        raise ValueError('Iterations must be a positive integer')
    if not allow_fixture:
        if iterations != 400:
            raise ValueError('Formal V3 uses the preregistered 400 iterations')
        if [len(data.manifest['splits'][s]['event_ids']) for s in ('train', 'validation', 'test')] != [67, 14, 16]:
            raise ValueError('Formal V3 requires the frozen dense 67/14/16 cohort')
    sources = local_sources([Path(__file__)])
    config = {'dataset_directory': str(data.root.resolve()), 'dataset_id': data.manifest['dataset_id'],
        'methods': METHODS, 'scored_splits': ['validation', 'test'],
        'base_recipe': adaptation.recipes()[0] | {'iterations': int(iterations)},
        'seed': 42, 'allow_fixture': bool(allow_fixture),
        'static_scope': 'Initial completed train group only, frozen at first validation event start',
        'rolling_scope': 'All cohort events completed strictly before query event start',
        'fit_clock': 'All prefixes of one activity share its start-frozen fit/history',
        'supervision': 'Equal total weight per fitted event',
        'ewma': {'alpha': .3, 'shrinkage': 'n/(n+10)', 'clip': [-.3, .3],
            'phase_edges': [.25, .9], 'phase_boundary': 'left closed, right open',
            'error': 'log uncorrected bounded predicted / actual',
            'aggregation': 'one mean per completed event/rank/phase',
            'source': 'only already sealed validation/test forecasts; no train residuals'},
        'selection': 'Minimum validation event-macro MAPE; fixed order breaks exact ties',
        'resume_policy': 'Completed identical runs only; incomplete runs require a new directory',
        'python': platform.python_version(),
        'software': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost', 'joblib', 'httpx', 'scipy']},
        'source_hashes': {name: sha256(ROOT/name) for name in sources}}
    config['run_id'] = identity(config)
    return config


def run_experiment(data, run, *, iterations=400, resume=False, allow_fixture=False):
    run = Path(run); config = _configuration(data, iterations, allow_fixture)
    if run.exists():
        if not resume:
            raise FileExistsError('Run already exists; sealed predictions cannot be overwritten')
        saved = json.loads((run/'run_manifest.json').read_text())
        if saved != config:
            raise ValueError('Run source/dependency/configuration identity changed')
        if not (run/'blind_manifest.json').exists():
            # Never recompute a prefix after any outcome may have been released.
            raise FileExistsError('Cannot resume incomplete V3 run; preserve it and choose a new run directory')
        verify_seal(run/'blind_manifest.json')
        return json.loads((run/'selection.json').read_text())
    run.mkdir(parents=True); (run/'events').mkdir(); (run/'models').mkdir()
    write_new_json(run/'run_manifest.json', config)
    for source in config['source_hashes']:
        dest = run/'sources'/source; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/source, dest)
    metadata = EventFeatures.load(data.metadata_directory)
    samples_by_event = defaultdict(list)
    for sample in data.inputs('train'):
        samples_by_event[sample['curve'].event].append(sample)
    events = sorted(data.event_index.values(), key=lambda r: (r['start_ms'], r['event']))
    validation_events = [r for r in events if r['split'] == 'validation']
    test_events = [r for r in events if r['split'] == 'test']
    if not validation_events or not test_events:
        raise ValueError('Validation and test groups must be nonempty')
    first_start = validation_events[0]['start_ms']
    initial_ids = set(data.manifest['splits']['train']['event_ids'])
    initial_history = [c for r in events if r['event'] in initial_ids and r['end_ms'] < first_start
                       for c in data.reveal_event(r['event'], before_ms=first_start)]
    static_arrays, static_labels = _training(samples_by_event, initial_history, metadata)
    static_bundle = fit_context(static_arrays, static_labels, iterations=iterations)
    _save_model(static_bundle, run/'models/static.joblib')
    _weight_rows(static_arrays).to_csv(run/'models/static_training_weights.csv', index=False)
    write_new_json(run/'models/static_diagnostics.json', static_bundle['diagnostics'])
    seal_directory(run/'models', 'model_manifest.json', {'run_id': config['run_id'], 'fit_before_ms': first_start})
    static_history_events = [r['event'] for r in events if r['event'] in {c.event for c in initial_history}]
    released = {event: [c for c in initial_history if c.event == event] for event in static_history_events}
    states = {base: OnlineBias() for base in BASES}
    prediction_files = {}

    def release_before(start):
        for row in sorted(events, key=lambda r: (r['end_ms'], r['event'])):
            event = row['event']
            if row['end_ms'] >= start or event in released:
                continue
            if row['split'] != 'train':
                seal = verify_seal(run/'events'/str(event)/'event_manifest.json')
                if seal['event'] != event or seal['run_id'] != config['run_id']:
                    raise ValueError('Released event prediction identity differs')
            curves = data.reveal_event(event, before_ms=start)
            released[event] = curves
            if row['split'] != 'train':
                predictions = pd.read_csv(run/'events'/str(event)/'predictions.csv')
                finals = {c.rank: c.final for c in curves}
                for base in BASES:
                    rows = predictions[predictions.method == base]
                    actual = {r.case_id: finals[int(r.rank)] for r in rows.itertuples()}
                    states[base].observe(rows, actual, event=event, end_ms=row['end_ms'])
        # Event dates, not IDs or arrival order, define both the fit and audit.
        return [c for row in events if row['event'] in released for c in released[row['event']]]

    def predict_split(split, event_rows):
        for sample in data.inputs(split):
            samples_by_event[sample['curve'].event].append(sample)
        paths = []
        for row in event_rows:
            event, start = row['event'], row['start_ms']
            history = release_before(start)
            fitting, labels = _training(samples_by_event, history, metadata)
            target_samples = samples_by_event[event]
            directory = Path(tempfile.mkdtemp(prefix=f'.event-{event}-', dir=run/'events'))
            # No event labels are queried between here and the atomic seal.
            bundle = fit_context(fitting, labels, iterations=iterations)
            _save_model(bundle, directory/'rolling.joblib')
            _weight_rows(fitting).to_csv(directory/'training_weights.csv', index=False)
            write_new_json(directory/'fit_diagnostics.json', bundle['diagnostics'])
            records = []
            if target_samples:
                target = prepare(target_samples, metadata)
                for base, model in [('static_context', static_bundle), ('rolling_context', bundle)]:
                    values = adaptation.predict_recipe(model, target)
                    for i, case in enumerate(target['cases']):
                        phase = int(np.searchsorted([.25, .9], target['progress'][i], side='right'))
                        bias = states[base].bias(case['rank'], phase)
                        for method, value, applied in [(base, values[i], 0.),
                                (base+'_ewma', max(target['current'][i],
                                    min(target['current'][i]*np.exp(4), values[i]*np.exp(-bias))), bias)]:
                            if not np.isfinite(value) or value < target['current'][i]:
                                raise ValueError('Invalid bounded prediction')
                            records.append({'case_id': case['case_id'], 'event': event,
                                'rank': case['rank'], 'phase': phase, 'method': method,
                                'current': float(target['current'][i]), 'predicted': float(value),
                                'log_bias': float(applied)})
            pd.DataFrame(records, columns=PREDICTION_COLUMNS).to_csv(directory/'predictions.csv', index=False)
            history_ids = {c.event for c in history}
            training_ids = {c['event'] for c in fitting['cases']}
            write_new_json(directory/'audit.json', {'event': event, 'split': split, 'query_start_ms': start,
                'static_history_events': static_history_events,
                'rolling_training_events': [r['event'] for r in events if r['event'] in training_ids],
                'released_history': [{k: r[k] for k in ['event', 'split', 'start_ms', 'end_ms', 'sha256']}
                                     for r in events if r['event'] in history_ids],
                'static_model_sha256': sha256(run/'models/static.joblib'),
                'ewma_records': {base: states[base].records for base in BASES},
                'selection_sha256': sha256(run/'selection.json') if split == 'test' else None,
                'current_outcome_released': False})
            seal_directory(directory, 'event_manifest.json', {'run_id': config['run_id'], 'event': event,
                'split': split, 'query_start_ms': start, 'end_ms': row['end_ms'],
                'methods': METHODS, 'cases': len(target_samples),
                'outcome_access': 'none until this event seal and a later start strictly after its end'})
            final_directory = run/'events'/str(event)
            if final_directory.exists():
                raise FileExistsError('Event prediction seal cannot be overwritten')
            directory.rename(final_directory)
            paths.append(final_directory/'predictions.csv')
            print(json.dumps({'event': event, 'split': split, 'sealed_cases': len(target_samples),
                              'training_events': len(training_ids)}), flush=True)
        predictions = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        expected = set(data.case_index(split).case_id)
        for method in METHODS:
            rows = predictions[predictions.method == method]
            if rows.case_id.duplicated().any() or set(rows.case_id) != expected:
                raise ValueError('Event seals do not cover the exact frozen case set')
        predictions.to_csv(run/f'{split}_predictions.csv', index=False)
        prediction_files[split] = predictions

    predict_split('validation', validation_events)
    test_start = test_events[0]['start_ms']
    if any(r['end_ms'] >= test_start for r in validation_events):
        raise ValueError('Validation selection cannot access events unfinished at first test start')
    selection_history = release_before(test_start)
    validation_arrays = prepare([s for r in validation_events for s in samples_by_event[r['event']]], metadata)
    validation_labels = _labels(validation_arrays, selection_history)
    scores = {}
    for method in METHODS:
        rows = prediction_files['validation']; rows = rows[rows.method == method].set_index('case_id')
        values = rows.loc[[c['case_id'] for c in validation_arrays['cases']], 'predicted'].to_numpy()
        scores[method] = event_mape(validation_arrays, validation_labels, values)
    selected = {'selected_recipe': min(METHODS, key=lambda m: (scores[m], METHODS.index(m))),
        'validation_scores': scores, 'criterion': config['selection'], 'methods': METHODS,
        'test_labels_used_for_selection': False, 'sealed_before_test_start_ms': test_start,
        'validation_event_seals': {str(r['event']): sha256(run/'events'/str(r['event'])/'event_manifest.json')
                                   for r in validation_events}, 'run_id': config['run_id']}
    write_new_json(run/'selection.json', selected)
    predict_split('test', test_events)
    seal_directory(run, 'blind_manifest.json', {'run_id': config['run_id'],
        'dataset_id': data.manifest['dataset_id'], 'dataset_directory': str(data.root.resolve()),
        'methods': METHODS, 'scored_splits': ['validation', 'test'],
        'reference_method': selected['selected_recipe'], 'bootstrap_reference': 'static_context',
        'protocol': 'rolling_holdout_v3', 'final_outcome_policy': 'Only independent scorer opens answer files'})
    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_experiment(RollingDataset(args.dataset), args.run, resume=args.resume), ensure_ascii=False))
