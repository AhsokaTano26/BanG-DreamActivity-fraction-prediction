"""Preregistered V2 follow-up experiments with resumable blind predictions."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from analysis.backtest import ROOT
from analysis.event_features import EventFeatures
from analysis.holdout_dataset import sha256, write_json
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import TemporalDataset
from analysis.innovation_common import (prepare, subset_arrays, chronological_folds,
    event_mape, event_weights, prediction_rows, local_sources)
from analysis.innovation_runner import diagnostic_rows

DATASET_ID = '1f76f39af8151bba185496411bbcee064b16a4c5aca5c515c1d957a5056fe854'
FAMILIES = {name: 'analysis.v2_next_'+name for name in ['adaptation', 'growth', 'foundation']}
REFERENCE = ROOT/'analysis/temporal_results/v2_multiagent_20261112'
DESIGN = ROOT/'docs/superpowers/specs/2026-11-14-v2-next-experiments-design.md'


def align_prediction(arrays, frame):
    ids = [c['case_id'] for c in arrays['cases']]
    if frame.case_id.duplicated().any() or len(set(ids)) != len(ids) or set(frame.case_id) != set(ids):
        raise ValueError('Prediction cases differ from the frozen input cases')
    values = frame.set_index('case_id').loc[ids, 'predicted'].to_numpy(float)
    prediction_rows(arrays, 'aligned', values)
    return values


def combine(spec, predictions):
    return sum(float(w)*predictions[n] for n, w in zip(spec['members'], spec['weights']))


def select_candidates(arrays, labels, predictions, recipes):
    scores = {name: event_mape(arrays, labels, predictions[name]) for name in recipes}
    family_best = {family: min((n for n, s in recipes.items() if s['family'] == family),
                              key=lambda n: (scores[n], n))
                   for family in dict.fromkeys(s['family'] for s in recipes.values())}
    best = min(scores, key=lambda n: (scores[n], n))
    members = list(family_best.values())
    specs = {'family_mean': {'members': members, 'weights': [1/len(members)]*len(members)}}
    for percent in [25, 50, 75]:
        specs[f'blend_new_{percent}'] = {'members': [best, 'old_selected'],
                                       'weights': [percent/100, 1-percent/100]}
    values = dict(predictions)
    values.update({n: combine(s, values) for n, s in specs.items()})
    candidates = [n for n in ['old_selected', 'hold_current', 'linear'] if n in values]
    candidates += list(recipes)+list(specs)
    validation = {n: event_mape(arrays, labels, values[n]) for n in candidates}
    return {'selected_recipe': min(candidates, key=lambda n: (validation[n], candidates.index(n))),
            'candidates': candidates, 'validation_scores': validation, 'family_best': family_best,
            'best_new': best, 'combination_specs': specs, 'reported_methods': list(values),
            'criterion': 'validation event-macro MAPE; individual seeds excluded',
            'test_labels_used_for_selection': False, 'refit_after_selection': False}


def verify_manifest(path):
    path = Path(path)
    record = json.loads(path.read_text())
    for name, digest in record['artifacts'].items():
        if sha256(path.parent/name) != digest:
            raise ValueError('Sealed artifact changed: '+str(path.parent/name))
    return record


def _reference_identity(reference, dataset_id):
    reference = Path(reference)
    seal = verify_manifest(reference/'blind_manifest.json')
    if seal['dataset_id'] != dataset_id:
        raise ValueError('Reference dataset fingerprint differs')
    return {'directory': str(reference.resolve()), 'method': seal['reference_method'],
            'blind_manifest_sha256': sha256(reference/'blind_manifest.json'),
            'predictions': {s: sha256(reference/f'{s}_predictions.csv')
                            for s in ['train', 'validation', 'test']}}


def _reference_prediction(descriptor, split, arrays):
    path = Path(descriptor['directory'])/f'{split}_predictions.csv'
    if sha256(path) != descriptor['predictions'][split]:
        raise ValueError('Reference predictions changed')
    frame = pd.read_csv(path)
    return align_prediction(arrays, frame[frame.method == descriptor['method']])


def _predict(module, bundle, arrays, name):
    if hasattr(module, 'predict_members'):
        members = module.predict_members(bundle, arrays)
        if set(members) != {'seed_42', 'seed_123', 'seed_2026'}:
            raise ValueError('Neural recipes must report all three fixed seeds')
        values = {name: np.mean(np.stack(list(members.values())), axis=0)}
        values.update({name+'__'+k: np.asarray(v) for k, v in members.items()})
    else:
        values = {name: np.asarray(module.predict_recipe(bundle, arrays))}
    for n, p in values.items():
        prediction_rows(arrays, n, p)
    return values


def _checkpoint(directory, run_id, module, name, config, fitting, labels, targets):
    directory = Path(directory)
    receipt = directory/'checkpoint.json'
    if receipt.exists():
        saved = verify_manifest(receipt)
        if saved['run_id'] != run_id or saved['method'] != name:
            raise ValueError('Checkpoint configuration changed')
        values = {}
        for split, arrays in targets.items():
            frame = pd.read_csv(directory/f'{split}_predictions.csv')
            values[split] = {n: align_prediction(arrays, g) for n, g in frame.groupby('method', sort=False)}
        return values, saved
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bundle = module.fit_recipe(config, fitting, labels)
    model_path = directory/'model.joblib'
    joblib.dump(bundle, model_path)
    values = {}
    for split, arrays in targets.items():
        values[split] = _predict(module, bundle, arrays, name)
        pd.concat([prediction_rows(arrays, n, p) for n, p in values[split].items()],
                  ignore_index=True).to_csv(directory/f'{split}_predictions.csv', index=False)
    split = next(reversed(targets)); a = targets[split]
    # Cached TabPFN uses fixed query chunks. Replay one identical chunk so an
    # MPS batch-shape rounding difference is not mistaken for corrupted state.
    count = min(int(config.get('query_batch_size', 8)), len(a['cases']))
    small = subset_arrays(a, np.arange(count), history=[])
    replay = np.asarray(module.predict_recipe(joblib.load(model_path), small))
    expected = values[split][name][:count]
    np.testing.assert_allclose(replay, expected, rtol=1e-5, atol=1e-5)
    if 'diagnostics' in bundle:
        write_json(directory/'diagnostics.json', bundle['diagnostics'])
    elif hasattr(module, 'diagnostics'):
        write_json(directory/'diagnostics.json', module.diagnostics(bundle))
    if hasattr(module, 'diagnostics'):
        for split, arrays in targets.items():
            write_json(directory/f'{split}_diagnostics.json', module.diagnostics(bundle, arrays))
    saved = {'run_id': run_id, 'method': name, 'seconds': time.perf_counter()-started,
             'training_cases': len(fitting['cases']),
             'training_events': len({c['event'] for c in fitting['cases']}),
             'reload_check_cases': count,
             'reload_max_relative_difference': float(np.max(abs(replay/np.maximum(expected, 1e-9)-1))),
             'artifacts': {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}}
    write_json(receipt, saved)
    return values, saved


def _seal(run, name, **extra):
    write_json(run/name, {**extra, 'artifacts': {
        str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*')) if p.is_file() and p != run/name}})


def run_experiment(data, run, *, reference=REFERENCE, families=None, phase='all', resume=False,
                   run_cv=True, allow_fixture=False, recipe_limit=None, iterations_override=None):
    run = Path(run)
    if not allow_fixture and data.manifest['dataset_id'] != DATASET_ID:
        raise ValueError('Formal dataset fingerprint differs from frozen V2')
    if phase not in ['develop', 'blind', 'all']:
        raise ValueError('Unknown phase')
    if run.exists() and not resume:
        raise FileExistsError(run)
    if not allow_fixture and (recipe_limit is not None or iterations_override is not None or not run_cv):
        raise ValueError('Formal runs use the full preregistered recipe and CV budget')
    modules = {f: importlib.import_module(FAMILIES[f]) for f in (families or FAMILIES)}
    recipes = {}
    for family, module in modules.items():
        configs = module.recipes()
        if recipe_limit is not None:
            configs = configs[:recipe_limit]
        for config in configs:
            config = dict(config)
            if iterations_override is not None:
                config['iterations'] = iterations_override
            recipes[family+'__'+config['name']] = {'family': family, 'config': config}
    if not recipes:
        raise ValueError('No runnable preregistered candidates')
    sources = local_sources([Path(__file__), *(Path(m.__file__) for m in modules.values()),
                             ROOT/'analysis/temporal_evaluate.py'])
    cfg = {'dataset_id': data.manifest['dataset_id'], 'dataset_directory': str(data.root.resolve()),
           'protocol': 'v2', 'synthetic_fixture': allow_fixture,
           'reference': _reference_identity(reference, data.manifest['dataset_id']),
           'recipes': recipes, 'internal_cv': run_cv,
           'cv': 'first half warmup; three contiguous later training blocks; no CV for zero-shot recipes',
           'weights': 'equal total supervised mass per activity; no recency loss reweighting',
           'selection': 'validation only; old selected model, all recipes, family mean, best-new blends25/50/75',
           'python': platform.python_version(),
           'software': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())),
           'source_hashes': {n: sha256(ROOT/n) for n in sources},
           'design_sha256': sha256(DESIGN),
           'family_provenance': {f: m.provenance() for f, m in modules.items() if hasattr(m, 'provenance')}}
    cfg['run_id'] = identity(cfg)
    if run.exists():
        if json.loads((run/'run_manifest.json').read_text()) != cfg:
            raise ValueError('Run source, dependencies, weights or configuration changed')
    else:
        if phase == 'blind':
            raise FileNotFoundError('Run development before blind prediction')
        run.mkdir(parents=True)
        write_json(run/'run_manifest.json', cfg)
        for name in sources:
            path = run/'sources'/name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, path)
        shutil.copyfile(DESIGN, run/'experiment_design.md')
    if (run/'blind_manifest.json').exists():
        verify_manifest(run/'blind_manifest.json')
        return json.loads((run/'selection.json').read_text())
    if (run/'development_manifest.json').exists():
        verify_manifest(run/'development_manifest.json')
        selection = json.loads((run/'selection.json').read_text())
    else:
        if phase == 'blind':
            raise ValueError('Development must finish and seal before blind prediction')
        metadata = EventFeatures.load(data.metadata_directory)
        (run/'models').mkdir(exist_ok=True)
        joblib.dump(metadata, run/'models/metadata.joblib')
        history = list(data.train_curves())
        if {c.event for c in history} != set(data.manifest['splits']['train']['event_ids']):
            raise ValueError('Training complete-history cohort differs')
        train = prepare(data.inputs('train'), metadata); train['history'] = history
        train_labels = data.labels('train')
        pd.DataFrame(train['cases']).assign(weight=event_weights(train['cases'])).to_csv(
            run/'training_weights.csv', index=False)
        event_rows = [{'event': c.event, 'start_ms': c.start, 'end_ms': c.end} for c in history]
        folds = chronological_folds(event_rows) if run_cv else []
        (run/'cv').mkdir(exist_ok=True); write_json(run/'cv/folds.json', folds)
        cv_metrics = []
        for fold in folds:
            fit_events, held_events = set(fold['training_events']), set(fold['heldout_events'])
            fit = subset_arrays(train, [i for i, c in enumerate(train['cases']) if c['event'] in fit_events],
                                history=[c for c in history if c.event in fit_events])
            held = subset_arrays(train, [i for i, c in enumerate(train['cases']) if c['event'] in held_events], history=[])
            labels = {c['case_id']: train_labels[c['case_id']] for c in fit['cases']}
            for name, spec in recipes.items():
                if spec['config'].get('zero_shot', False):
                    continue
                values, receipt = _checkpoint(run/f"cv/fold_{fold['fold']}"/name, cfg['run_id'],
                    modules[spec['family']], name, spec['config'], fit, labels, {'heldout': held})
                rows = diagnostic_rows(held, train_labels, values['heldout'], fold=fold['fold'])
                cv_metrics.extend(rows)
                print(json.dumps({'phase': 'cv', 'fold': fold['fold'], 'method': name,
                    'event_mape': event_mape(held, train_labels, values['heldout'][name]),
                    'seconds': receipt['seconds']}), flush=True)
                pd.DataFrame(cv_metrics).to_csv(run/'cv/metrics.csv', index=False)
        valid = prepare(data.inputs('validation'), metadata)
        valid_labels = data.labels('validation')
        arrays = {'train': train, 'validation': valid}
        values = {s: {'hold_current': a['current'], 'linear': a['current']/a['progress'],
                      'old_selected': _reference_prediction(cfg['reference'], s, a)} for s, a in arrays.items()}
        ledger = []
        for name, spec in recipes.items():
            predictions, receipt = _checkpoint(run/'models'/name, cfg['run_id'], modules[spec['family']],
                name, spec['config'], train, train_labels, arrays)
            for split in arrays:
                values[split].update(predictions[split])
            ledger.append({'method': name, 'train_fit_mape': event_mape(train, train_labels, values['train'][name]),
                'validation_mape': event_mape(valid, valid_labels, values['validation'][name]),
                'seconds': receipt['seconds']})
            write_json(run/'development_ledger.json', ledger)
            print(json.dumps(ledger[-1]), flush=True)
        selection = select_candidates(valid, valid_labels, values['validation'], recipes)
        selection.update({'run_id': cfg['run_id'], 'dataset_id': cfg['dataset_id']})
        write_json(run/'selection.json', selection)
        metrics = []
        for split, a in arrays.items():
            values[split].update({n: combine(s, values[split]) for n, s in selection['combination_specs'].items()})
            pd.concat([prediction_rows(a, n, p) for n, p in values[split].items()], ignore_index=True).to_csv(
                run/f'{split}_predictions.csv', index=False)
            metrics.extend(diagnostic_rows(a, train_labels if split == 'train' else valid_labels,
                                           values[split], split=split))
        pd.DataFrame(metrics).to_csv(run/'development_metrics.csv', index=False)
        _check_sources(cfg)
        _seal(run, 'development_manifest.json', run_id=cfg['run_id'], dataset_id=cfg['dataset_id'])
        print('SELECTED '+selection['selected_recipe'], flush=True)
    if phase == 'develop':
        return selection
    metadata = joblib.load(run/'models/metadata.joblib')
    test = prepare(data.inputs('test'), metadata)
    values = {'hold_current': test['current'], 'linear': test['current']/test['progress'],
              'old_selected': _reference_prediction(cfg['reference'], 'test', test)}
    for name, spec in recipes.items():
        path = run/'models'/name
        verify_manifest(path/'checkpoint.json')
        module = modules[spec['family']]
        bundle = joblib.load(path/'model.joblib')
        values.update(_predict(module, bundle, test, name))
        if hasattr(module, 'diagnostics'):
            directory = run/'test_diagnostics'; directory.mkdir(exist_ok=True)
            write_json(directory/(name+'.json'), module.diagnostics(bundle, test))
        print('BLIND '+name, flush=True)
    values.update({n: combine(s, values) for n, s in selection['combination_specs'].items()})
    pd.concat([prediction_rows(test, n, p) for n, p in values.items()], ignore_index=True).to_csv(
        run/'test_predictions.csv', index=False)
    _check_sources(cfg)
    verify_manifest(run/'development_manifest.json')
    _seal(run, 'blind_manifest.json', run_id=cfg['run_id'], dataset_id=cfg['dataset_id'],
          dataset_directory=cfg['dataset_directory'], methods=selection['reported_methods'],
          reference_method=selection['selected_recipe'], test_cases=len(test['cases']))
    return selection


def _check_sources(cfg):
    for name, digest in cfg['source_hashes'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Source changed during experiment: '+name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['develop', 'blind', 'all'], default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_experiment(TemporalDataset(ROOT/'data/temporal_holdout_v2_t1000_t2000'), args.run,
                   phase=args.phase, resume=args.resume)
