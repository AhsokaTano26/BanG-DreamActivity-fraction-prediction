"""Train/validation-only multi-family selection, then emit one sealed blind test."""
from __future__ import annotations
import argparse, ast, importlib, importlib.metadata, json, platform, shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from training.paths import module_source
from analysis.backtest import ROOT
from analysis.holdout_dataset import SPLITS, sha256, write_json
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_rank_subset import DEFAULT
from analysis.temporal_paired_features import prepare_paired
from analysis.temporal_evaluate import verify_predictions

FAMILIES = {'remaining': 'analysis.research_remaining', 'fusion': 'analysis.research_fusion', 'shift': 'analysis.research_shift'}
REFERENCE_METHODS = {'hold_current': 'hold_current', 'linear': 'linear', 'reference_single': 'reference_catboost',
                     'reference_paired': 'catboost_paired', 'reference_tcn': 'dual_tcn_mean'}
COMBINATIONS = ['family_mean', 'family_median', 'blend_25', 'blend_50', 'blend_75', 'rank_route', 'phase_route']


def event_mape(arrays, labels, prediction):
    actual = np.array([labels[c['case_id']] for c in arrays['cases']])
    prediction = np.asarray(prediction)
    if prediction.shape != actual.shape or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError('Predictions must be positive and aligned to every case')
    _, index, counts = np.unique([c['event'] for c in arrays['cases']], return_inverse=True, return_counts=True)
    return float((np.bincount(index, weights=np.abs(prediction/actual-1)*100)/counts).mean())


def combination_specs(arrays, labels, predictions, family_best):
    best = min(predictions, key=lambda name: event_mape(arrays, labels, predictions[name]))
    specs = {'family_mean': {'kind': 'mean', 'members': list(family_best.values())},
             'family_median': {'kind': 'median', 'members': list(family_best.values())}}
    for fraction in [.25, .5, .75]:
        specs[f'blend_{round(fraction*100)}'] = {'kind': 'weighted', 'members': [best, 'reference_paired'], 'fraction': fraction}
    routes = {}
    for rank in [0, 1]:
        mask = arrays['rank'] == rank
        if not mask.any():
            routes[str(rank)] = best
            continue
        subset = {'cases': [c for c, keep in zip(arrays['cases'], mask) if keep]}
        routes[str(rank)] = min(predictions, key=lambda name: event_mape(subset, labels, predictions[name][mask]))
    specs['rank_route'] = {'kind': 'rank', 'routes': routes}
    phases = np.searchsorted([.25, .9], arrays['progress'], side='right')
    routes = {}
    for phase in range(3):
        mask = phases == phase
        subset = {'cases': [c for c, keep in zip(arrays['cases'], mask) if keep]}
        routes[str(phase)] = min(predictions, key=lambda name: event_mape(subset, labels, predictions[name][mask])) if mask.any() else best
    specs['phase_route'] = {'kind': 'phase', 'routes': routes, 'edges': [.25, .9]}
    return specs


def combine(spec, predictions, arrays):
    if spec['kind'] == 'phase':
        phases = np.searchsorted(spec['edges'], arrays['progress'], side='right')
        return np.choose(phases, [predictions[spec['routes'][str(i)]] for i in range(3)])
    if spec['kind'] == 'rank':
        return np.where(arrays['rank'] == 0, predictions[spec['routes']['0']], predictions[spec['routes']['1']])
    values = np.stack([predictions[name] for name in spec['members']])
    if spec['kind'] == 'mean':
        return values.mean(axis=0)
    if spec['kind'] == 'median':
        return np.median(values, axis=0)
    if spec['kind'] == 'weighted':
        return spec['fraction']*values[0]+(1-spec['fraction'])*values[1]
    raise ValueError('Unknown combination kind')


def sources_for(modules):
    pending = [Path(__file__).resolve(), Path(__file__).with_name('temporal_evaluate.py')]
    pending += [Path(module.__file__).resolve() for module in modules.values()]
    found = set()
    while pending:
        source = pending.pop()
        if source in found or not source.is_file():
            continue
        found.add(source)
        for node in ast.walk(ast.parse(source.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
                if node.module == 'analysis':
                    names += ['analysis.'+a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            pending += [source for name in names if (source := module_source(name)) is not None]
    return sorted(str(p.relative_to(ROOT)) for p in found)


def reference_predictions(reference, split, arrays):
    frame = pd.read_csv(Path(reference)/f'{split}_predictions.csv')
    ids = [c['case_id'] for c in arrays['cases']]; predictions = {}
    for name, old in REFERENCE_METHODS.items():
        part = frame[frame.method == old].set_index('case_id')
        if not part.index.is_unique or set(part.index) != set(ids):
            raise ValueError('Reference cases differ')
        p = part.loc[ids, 'predicted'].to_numpy()
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError('Invalid reference prediction')
        predictions[name] = p
    return predictions


def rows(arrays, method, prediction):
    if len(prediction) != len(arrays['cases']) or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError('Invalid predictions')
    return pd.DataFrame({'case_id': [c['case_id'] for c in arrays['cases']], 'method': method, 'predicted': prediction})


def run_experiment(data, reference, run, *, limit_per_family=None, iterations_override=None):
    run, reference = Path(run), Path(reference)
    if run.exists():
        raise FileExistsError(run)
    seal = verify_predictions(reference)
    if seal['dataset_id'] != data.manifest['dataset_id']:
        raise ValueError('Reference data differs')
    modules = {name: importlib.import_module(path) for name, path in FAMILIES.items()}
    recipes = {}
    for family, module in modules.items():
        configs = module.recipes()
        if limit_per_family is not None:
            configs = configs[:limit_per_family]
        for config in configs:
            config = dict(config)
            if iterations_override is not None:
                config['iterations'] = iterations_override
                if 'n_estimators' in config:
                    config['n_estimators'] = iterations_override
            name = family+'__'+config['name']
            if name in recipes:
                raise ValueError('Duplicate recipe')
            recipes[name] = {'family': family, 'config': config}
    methods = list(REFERENCE_METHODS)+list(recipes)+COMBINATIONS
    sources = sources_for(modules)
    config = {'dataset_id': data.manifest['dataset_id'], 'dataset_directory': str(data.root.resolve()),
        'reference': str(reference.resolve()), 'reference_blind_manifest_sha256': sha256(reference/'blind_manifest.json'),
        'recipes': recipes, 'methods': methods, 'selection': 'minimum validation event-macro MAPE; seven frozen combinations; no test selection/refit',
        'combination_menu': COMBINATIONS, 'python': platform.python_version(),
        'software': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'joblib']},
        'source_hashes': {n: sha256(ROOT/n) for n in sources}}
    config['run_id'] = identity(config)
    run.mkdir(parents=True)
    for directory in ['sources', 'models', 'features']:
        (run/directory).mkdir()
    write_json(run/'run_manifest.json', config)
    for name in sources:
        path = run/'sources'/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, path)
    shutil.copyfile(reference/'blind_manifest.json', run/'reference_blind_manifest.json')
    arrays = {split: prepare_paired(data.inputs(split)) for split in ['train', 'validation']}
    labels = {split: data.labels(split) for split in ['train', 'validation']}
    predictions = {split: reference_predictions(reference, split, a) for split, a in arrays.items()}
    for split, a in arrays.items():
        a['audit'].to_csv(run/f'features/{split}_audit.csv', index=False)
        for kind in ['single', 'paired']:
            a[kind].to_csv(run/f'features/{split}_{kind}.csv.gz', index=False)
    bundles = {}; validation = {}; ledger = []
    for name, spec in recipes.items():
        module = modules[spec['family']]
        bundle = module.fit_recipe(spec['config'], arrays['train'], labels['train'])
        joblib.dump(bundle, run/f'models/{name}.joblib')
        bundles[name] = bundle
        for split, a in arrays.items():
            predictions[split][name] = np.asarray(module.predict_recipe(bundle, a))
            event_mape(a, labels[split], predictions[split][name])
        replay = module.predict_recipe(joblib.load(run/f'models/{name}.joblib'), arrays['validation'])
        np.testing.assert_allclose(replay, predictions['validation'][name], rtol=1e-10)
        metric = event_mape(arrays['validation'], labels['validation'], predictions['validation'][name])
        validation[name] = metric
        ledger.append({'method': name, 'family': spec['family'], 'validation_event_mape': metric,
                       'training_fit_event_mape': event_mape(arrays['train'], labels['train'], predictions['train'][name])})
        write_json(run/'development_ledger.json', ledger)
        print(json.dumps(ledger[-1]), flush=True)
    family_best = {family: min((name for name, s in recipes.items() if s['family'] == family), key=validation.get) for family in modules}
    specs = combination_specs(arrays['validation'], labels['validation'], predictions['validation'], family_best)
    for split, a in arrays.items():
        for name, spec in specs.items():
            predictions[split][name] = combine(spec, predictions[split], a)
    validation = {name: event_mape(arrays['validation'], labels['validation'], p) for name, p in predictions['validation'].items()}
    selected = min(validation, key=validation.get)
    selection = {'selected_recipe': selected, 'validation': validation, 'family_best': family_best,
                 'combination_specs': specs, 'test_labels_read': False, 'refit_after_selection': False}
    write_json(run/'selection.json', selection)
    print('SELECTED '+selected+' '+str(validation[selected]), flush=True)
    # The test input stream is opened only after every candidate and combination is fixed.
    test = prepare_paired(data.inputs('test'))
    test['audit'].to_csv(run/'features/test_audit.csv', index=False)
    predictions['test'] = reference_predictions(reference, 'test', test)
    for name, spec in recipes.items():
        predictions['test'][name] = np.asarray(modules[spec['family']].predict_recipe(bundles[name], test))
    for name, spec in specs.items():
        predictions['test'][name] = combine(spec, predictions['test'], test)
    arrays['test'] = test
    for split in SPLITS:
        pd.concat([rows(arrays[split], m, predictions[split][m]) for m in methods], ignore_index=True).to_csv(run/f'{split}_predictions.csv', index=False)
    # Fitting must not race source edits by any worker.
    for name, digest in config['source_hashes'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('A model source changed during the run: '+name)
    blind = {k: config[k] for k in ['run_id', 'dataset_id', 'dataset_directory', 'methods']}
    blind['reference_method'] = selected
    blind['artifacts'] = {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*')) if p.is_file()}
    write_json(run/'blind_manifest.json', blind)
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT)
    parser.add_argument('--reference', type=Path, default=ROOT/'analysis/temporal_results/v2_paired_tcn_20261112')
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run_experiment(TemporalDataset(args.dataset), args.reference, args.run)
