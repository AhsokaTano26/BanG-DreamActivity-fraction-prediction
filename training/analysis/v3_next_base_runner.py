"""Preregistered rolling-base optimization with guarded releases and blind seals.

Only this coordinator may read completed-event vault curves. It never opens
answer files and leaves independent scoring to rolling_residual_evaluate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
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
from analysis.data_policy import MIN_EVENT_START_MS
from analysis.event_features import EventFeatures
from analysis.holdout_dataset import sha256
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import write_new_json, seal_directory, verify_seal
from analysis.rolling_dataset import RollingDataset, DEFAULT
from analysis import v3_next_bases as bases

BASE_RUN = ROOT/'analysis/rolling_results/v3_dense_full_retrain_20261115'
RUN = ROOT/'analysis/rolling_results/v3_next_bases_20261116'
DATASET_ID = '85f5924e0b90deb0e1a30f3aa27d2237640cca4d968d96612dda8fee1d9c72ec'
METHODS = ['uncorrected', 'ewma_reference'] + [method for name in bases.CONFIGS
    for method in (name, name+'_ewma', name+'_fast')]
PRED_COLUMNS = ['case_id', 'event', 'rank', 'phase', 'method', 'current', 'predicted', 'log_bias']


def _configuration(data, base_run, iterations, allow_fixture):
    if int(iterations) != iterations or iterations < 1:
        raise ValueError('Iterations must be positive integers')
    old_seal = verify_seal(base_run/'blind_manifest.json')
    old_config = json.loads((base_run/'run_manifest.json').read_text())
    if old_seal['dataset_id'] != data.manifest['dataset_id'] or old_config['run_id'] != old_seal['run_id']:
        raise ValueError('Different baseline cache dataset or run identity')
    if not allow_fixture:
        if (data.manifest['dataset_id'] != DATASET_ID or iterations != 400
            or old_config['allow_fixture']
            or [len(data.manifest['splits'][s]['event_ids']) for s in ('train', 'validation', 'test')] != [67, 14, 16]
            or [data.manifest['splits'][s]['cases'] for s in ('validation', 'test')] != [488, 560]):
            raise ValueError('Formal bases require unchanged dense V3 cohort/cases and 400 iterations')
    sources = local_sources([Path(__file__)])
    config = {'protocol': 'rolling_residual_v3', 'dataset_id': data.manifest['dataset_id'],
        'dataset_directory': str(data.root.resolve()), 'base_run': str(base_run.resolve()),
        'base_run_id': old_seal['run_id'], 'base_blind_sha256': sha256(base_run/'blind_manifest.json'),
        'methods': METHODS, 'candidates': METHODS, 'scored_splits': ['validation', 'test'],
        'recipes': {n: c | {'iterations': int(iterations)} for n, c in bases.CONFIGS.items()},
        'feedback': {'source': 'Only own raw sealed forecasts of strictly completed earlier validation/test events',
            'no_train_resubstitution': True, 'no_borrowing_other_bases': True,
            'alphas': [.3, .6], 'aggregation': 'One mean log(predicted/actual) per event/rank/phase',
            'shrinkage': 'n/(n+10)', 'clip': [-.3, .3], 'phase_edges': [.25, .9]},
        'history': 'Actual start >= minimum; all cohort curves strictly completed before event start',
        'event_clock': 'One fitted model, preprocessing, vocabulary and history pool for every prefix of an event',
        'peer': 'Same case origin and event, other rank; truncate to own last observed time',
        'minimum_start_ms': MIN_EVENT_START_MS,
        'selection': 'Minimum validation event-macro MAPE; fixed method order resolves exact ties',
        'test_input_policy': 'Do not load test inputs until selection.json is sealed',
        'resume_policy': 'Completed identical runs only; preserve interrupted directories',
        'exposure': 'Exploratory follow-up on previously exposed tests and repeatedly used validation',
        'metadata_caveat': data.manifest['protocol']['metadata_caveat'],
        'allow_fixture': bool(allow_fixture), 'python': platform.python_version(),
        'software': {n: importlib.metadata.version(n) for n in
            ['numpy', 'pandas', 'scikit-learn', 'catboost', 'joblib', 'scipy', 'httpx']},
        'source_hashes': {name: sha256(ROOT/name) for name in sources},
        'specification_sha256': sha256(ROOT/'docs/superpowers/specs/2026-11-16-v3-next-design.md')}
    config['run_id'] = identity(config)
    return config


def _aligned(frame, method, ids):
    part = frame[frame.method == method]
    if part.case_id.duplicated().any() or set(part.case_id) != set(ids):
        raise ValueError('Baseline frozen case coverage differs')
    return part.set_index('case_id').loc[ids]


def _cached_event(data, base_run, config, row, samples):
    event, start = row['event'], row['start_ms']
    directory = base_run/'events'/str(event)
    receipt = verify_seal(directory/'event_manifest.json')
    if any(receipt.get(k) != v for k, v in [('event', event), ('split', row['split']),
        ('query_start_ms', start), ('end_ms', row['end_ms']), ('run_id', config['base_run_id'])]):
        raise ValueError('Baseline event cache date/run identity differs')
    audit = json.loads((directory/'audit.json').read_text())
    if audit['event'] != event or audit['query_start_ms'] != start or audit['current_outcome_released']:
        raise ValueError('Baseline event audit identity differs')
    expected = {r['event'] for r in data.event_index.values() if r['end_ms'] < start}
    actual = set()
    for item in audit['released_history']:
        canonical = data.event_index[item['event']]
        if (item['end_ms'] >= start or item['start_ms'] < MIN_EVENT_START_MS
            or any(item[k] != canonical[k] for k in ['split', 'start_ms', 'end_ms', 'sha256'])):
            raise ValueError('Baseline cache includes an illegal history/date')
        actual.add(item['event'])
    if actual != expected:
        raise ValueError('Baseline cache completed history pool differs')
    old = pd.read_csv(directory/'predictions.csv')
    ids = [s['case_id'] for s in samples]
    current = np.array([s['score'] for s in samples], float)
    rows = []
    for method, source in [('uncorrected', 'rolling_context'), ('ewma_reference', 'rolling_context_ewma')]:
        part = _aligned(old, source, ids)
        if (not np.allclose(part.current.to_numpy(), current, rtol=1e-12, atol=1e-6)
            or set(part.event) != {event} or part['rank'].tolist() != [s['curve'].rank for s in samples]):
            raise ValueError('Baseline current-prefix/rank identity differs')
        if (not np.isfinite(part.predicted).all() or (part.predicted.to_numpy() < current-1e-7).any()):
            raise ValueError('Invalid baseline predictions')
        part = part.reset_index()[PRED_COLUMNS].copy(); part['method'] = method
        rows.append(part)
    return pd.concat(rows, ignore_index=True), sha256(directory/'event_manifest.json')


def _training(samples_by_event, history, metadata):
    allowed = {c.event for c in history}
    samples = [s for event, values in samples_by_event.items() if event in allowed for s in values]
    arrays = bases.prepare_arrays(samples, metadata)
    finals = {(c.event, c.rank): c.final for c in history}
    labels = {c['case_id']: finals[c['event'], c['rank']] for c in arrays['cases']}
    return arrays, labels


def run_experiment(data, run, *, base_run=BASE_RUN, iterations=400, allow_fixture=False, resume=False):
    run, base_run = Path(run), Path(base_run)
    config = _configuration(data, base_run, iterations, allow_fixture)
    if run.exists():
        if not resume:
            raise FileExistsError('Preserve old runs and choose a new directory')
        if json.loads((run/'run_manifest.json').read_text()) != config:
            raise ValueError('Changed source/config/dependency identity cannot resume')
        if not (run/'blind_manifest.json').exists():
            raise FileExistsError('Incomplete run preserved; choose a new directory')
        verify_seal(run/'blind_manifest.json')
        return json.loads((run/'selection.json').read_text())
    run.mkdir(parents=True); (run/'events').mkdir()
    write_new_json(run/'run_manifest.json', config)
    for name in config['source_hashes']:
        dest = run/'sources'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    for name in ['specs/2026-11-16-v3-next-design.md', 'plans/2026-11-16-v3-next.md']:
        shutil.copyfile(ROOT/'docs/superpowers'/name, run/Path(name).name)
    metadata = EventFeatures.load(data.metadata_directory)
    events = sorted(data.event_index.values(), key=lambda r: (r['start_ms'], r['event']))
    queries = [r for r in events if r['split'] != 'train']
    first_start = next(r['start_ms'] for r in queries if r['split'] == 'validation')
    samples_by_event = defaultdict(list)
    for sample in data.inputs('train'):
        samples_by_event[sample['curve'].event].append(sample)
    released = {r['event']: data.reveal_event(r['event'], before_ms=first_start)
        for r in events if r['split'] == 'train' and r['end_ms'] < first_start}
    states = {name: {suffix: bases.OnlineEwma(alpha) for suffix, alpha in [('ewma', .3), ('fast', .6)]}
        for name in bases.CONFIGS}
    reference_state = bases.OnlineEwma(.3)
    predictions = {}

    def release_before(start):
        for row in sorted(events, key=lambda r: (r['end_ms'], r['event'])):
            event = row['event']
            if row['end_ms'] >= start or event in released:
                continue
            if row['split'] != 'train':
                receipt = verify_seal(run/'events'/str(event)/'event_manifest.json')
                if receipt['event'] != event or receipt['run_id'] != config['run_id']:
                    raise ValueError('Feedback forecast seal identity differs')
            curves = data.reveal_event(event, before_ms=start)
            released[event] = curves
            if row['split'] != 'train':
                frame = pd.read_csv(run/'events'/str(event)/'predictions.csv')
                finals = {c.rank: c.final for c in curves}
                for name, pair in states.items():
                    raw = frame[frame.method == name]
                    labels = {r.case_id: finals[int(r.rank)] for r in raw.itertuples()}
                    for state in pair.values():
                        state.observe(raw, labels, event=event, end_ms=row['end_ms'])
                raw = frame[frame.method == 'uncorrected']
                labels = {r.case_id: finals[int(r.rank)] for r in raw.itertuples()}
                reference_state.observe(raw, labels, event=event, end_ms=row['end_ms'])
        return [c for r in events if r['event'] in released for c in released[r['event']]]

    def predict_event(row):
        event, start = row['event'], row['start_ms']
        history = release_before(start)
        fitting, labels = _training(samples_by_event, history, metadata)
        samples = samples_by_event[event]
        target = bases.prepare_arrays(samples, metadata)
        cached, cache_hash = _cached_event(data, base_run, config, row, samples)
        raw_reference = cached[cached.method == 'uncorrected'].predicted.to_numpy(float)
        expected_reference = cached[cached.method == 'ewma_reference'].predicted.to_numpy(float)
        phase = np.searchsorted([.25, .9], target['progress'], side='right')
        ref_bias = np.array([reference_state.bias(c['rank'], p) for c, p in zip(target['cases'], phase)])
        replay = np.maximum(target['current'], np.minimum(target['current']*np.exp(4), raw_reference*np.exp(-ref_bias)))
        if not np.allclose(replay, expected_reference, rtol=1e-12, atol=1e-6):
            raise ValueError('Reference EWMA replay differs from frozen baseline')
        directory = Path(tempfile.mkdtemp(prefix=f'.event-{event}-', dir=run/'events'))
        records = cached.to_dict('records')
        for name in bases.CONFIGS:
            bundle = bases.fit_base(name, fitting, labels, fit_before_ms=start, iterations=iterations)
            model_dir = directory/name; model_dir.mkdir()
            joblib.dump(bundle, model_dir/'model.joblib', compress=3)
            pd.DataFrame(bundle['weight_rows']).to_csv(model_dir/'training_weights.csv', index=False)
            write_new_json(model_dir/'config.json', bundle['config'])
            write_new_json(model_dir/'diagnostics.json', bundle['diagnostics'])
            write_new_json(model_dir/'preprocessing.json', {str(rank): asdict(model['preprocessor'])
                for rank, model in bundle['models'].items()})
            base = bases.predict_base(bundle, target)
            for suffix in ['', 'ewma', 'fast']:
                bias = (np.zeros(len(samples)) if not suffix else np.array([
                    states[name][suffix].bias(c['rank'], p) for c, p in zip(target['cases'], phase)]))
                values = np.maximum(target['current'], np.minimum(target['current']*np.exp(4), base*np.exp(-bias)))
                method = name+('_'+suffix if suffix else '')
                for i, case in enumerate(target['cases']):
                    records.append({'case_id': case['case_id'], 'event': event, 'rank': case['rank'],
                        'phase': int(phase[i]), 'method': method, 'current': float(target['current'][i]),
                        'predicted': float(values[i]), 'log_bias': float(bias[i])})
        result = pd.DataFrame(records, columns=PRED_COLUMNS)
        result.to_csv(directory/'predictions.csv', index=False)
        fitting['pair_audit'].to_csv(directory/'training_pair_audit.csv', index=False)
        target['pair_audit'].to_csv(directory/'target_pair_audit.csv', index=False)
        target['base'].to_csv(directory/'target_own_features.csv.gz', index=False)
        target['paired'].to_csv(directory/'target_paired_features.csv.gz', index=False)
        write_new_json(directory/'audit.json', {'event': event, 'split': row['split'], 'query_start_ms': start,
            'training_events': [r['event'] for r in events if r['event'] in {c['event'] for c in fitting['cases']}],
            'released_history': [{k: r[k] for k in ['event', 'split', 'start_ms', 'end_ms', 'sha256']}
                for r in events if r['event'] in released],
            'feedback_records': {name: pair['ewma'].records for name, pair in states.items()},
            'feedback_states': {name: {suffix: {str(k): [v, n] for k, (v, n) in state.groups.items()}
                for suffix, state in pair.items()} for name, pair in states.items()},
            'base_event_manifest_sha256': cache_hash,
            'reference_ewma_max_relative_error': float(np.max(np.abs(replay/expected_reference-1))),
            'selection_sha256': sha256(run/'selection.json') if row['split'] == 'test' else None,
            'current_outcome_released': False})
        seal_directory(directory, 'event_manifest.json', {'run_id': config['run_id'], 'event': event,
            'split': row['split'], 'query_start_ms': start, 'end_ms': row['end_ms'],
            'cases': len(samples), 'methods': METHODS,
            'outcome_access': 'None until own seal and a later start strictly after own end'})
        directory.rename(run/'events'/str(event))
        print(json.dumps({'sealed_event': event, 'split': row['split'], 'training_events': len(released),
            'cases': len(samples), 'methods': len(METHODS)}), flush=True)
        return result

    chosen = None
    for split in ['validation', 'test']:
        split_rows = [r for r in queries if r['split'] == split]
        if split == 'test':
            history = release_before(split_rows[0]['start_ms'])
            finals = {(c.event, c.rank): c.final for c in history}
            scores = {}
            for method in METHODS:
                part = predictions['validation'].query('method == @method').copy()
                if any((r.event, r.rank) not in finals for r in part.itertuples()):
                    raise ValueError('Validation outcome unavailable before first test start')
                actual = np.array([finals[r.event, r.rank] for r in part.itertuples()])
                part['ape'] = np.abs(part.predicted.to_numpy(float)/actual-1)*100
                scores[method] = float(part.groupby('event').ape.mean().mean())
            chosen = min(METHODS, key=lambda name: (scores[name], METHODS.index(name)))
            write_new_json(run/'selection.json', {'run_id': config['run_id'], 'selected_recipe': chosen,
                'methods': METHODS, 'candidates': METHODS, 'validation_scores': scores,
                'criterion': config['selection'], 'test_labels_used_for_selection': False,
                'test_inputs_loaded': False, 'sealed_before_test_start_ms': split_rows[0]['start_ms'],
                'validation_event_seals': {str(r['event']): sha256(run/'events'/str(r['event'])/'event_manifest.json')
                    for r in queries if r['split'] == 'validation'}})
        for sample in data.inputs(split):
            samples_by_event[sample['curve'].event].append(sample)
        predictions[split] = pd.concat([predict_event(row) for row in split_rows], ignore_index=True)
        expected = set(data.case_index(split).case_id)
        if set(predictions[split].method) != set(METHODS):
            raise ValueError('Reported method coverage differs')
        for method, part in predictions[split].groupby('method'):
            if part.case_id.duplicated().any() or set(part.case_id) != expected:
                raise ValueError('Frozen case coverage differs for '+method)
        predictions[split].to_csv(run/f'{split}_predictions.csv', index=False)
    for name, digest in config['source_hashes'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Source changed during experiment: '+name)
    if sha256(base_run/'blind_manifest.json') != config['base_blind_sha256']:
        raise ValueError('Baseline seal changed during experiment')
    seal_directory(run, 'blind_manifest.json', {'protocol': 'rolling_residual_v3',
        'run_id': config['run_id'], 'dataset_id': config['dataset_id'],
        'dataset_directory': config['dataset_directory'], 'methods': METHODS,
        'scored_splits': ['validation', 'test'], 'reference_method': 'ewma_reference',
        'selected_recipe': chosen, 'base_blind_sha256': config['base_blind_sha256']})
    return json.loads((run/'selection.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--base-run', type=Path, default=BASE_RUN)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_experiment(RollingDataset(DEFAULT), args.run,
        base_run=args.base_run, resume=args.resume), ensure_ascii=False))
