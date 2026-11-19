"""Independent event-paired audit of validation-selected candidates only."""
import argparse, importlib.metadata, json, platform, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256, write_json
from analysis.holdout_evaluate import paired_event_errors
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_evaluate import verify_predictions


def evaluate(run):
    run = Path(run); seal = verify_predictions(run)
    data = TemporalDataset(seal['dataset_directory'])
    if data.manifest['dataset_id'] != seal['dataset_id']:
        raise ValueError('Dataset differs')
    selection = json.loads((run/'selection.json').read_text())
    methods = list(dict.fromkeys([selection['selected_recipe']]+list(selection['family_best'].values())))
    sources = ['training/analysis/research_compare.py', 'training/analysis/holdout_evaluate.py', 'training/analysis/holdout_dataset.py',
               'training/analysis/temporal_dataset.py', 'training/analysis/temporal_evaluate.py', 'training/analysis/multirank_dataset.py', 'training/analysis/backtest.py']
    config = {'blind_manifest_sha256': sha256(run/'blind_manifest.json'), 'dataset_id': seal['dataset_id'],
        'methods': methods, 'references': ['reference_single', 'reference_paired'], 'seed': 42, 'draws': 5000,
        'sources': {n: sha256(ROOT/n) for n in sources}, 'python': platform.python_version(),
        'software': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas']},
        'interpretation': 'descriptive marginal percentile intervals; no multiple-comparison adjustment or selection'}
    evaluation_id = identity(config); output = run/'comparison'; receipt = output/'evaluation_manifest.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['evaluation_id'] != evaluation_id:
            raise FileExistsError('Changed comparison evaluator; preserve the previous audit')
        for n, digest in old['artifacts'].items():
            if sha256(output/n) != digest:
                raise ValueError('Comparison output changed')
        return json.loads((output/'comparison.json').read_text())
    if output.exists():
        raise FileExistsError(output)
    cases = data.case_index('test').to_dict('records')
    labels = data.labels('test', allow_test=True)  # final independent audit only
    predictions = pd.read_csv(run/'test_predictions.csv')
    errors = {m: paired_event_errors(cases, labels, predictions[predictions.method == m].to_dict('records'))
              for m in set(methods+config['references'])}
    events = errors[config['references'][0]].index
    indices = np.random.default_rng(42).integers(0, len(events), size=(5000, len(events)))
    results = []; deltas = []
    for method in methods:
        for reference in config['references']:
            if not errors[method].index.equals(errors[reference].index):
                raise ValueError('Event membership differs')
            delta = (errors[method]-errors[reference]).to_numpy(); boot = delta[indices].mean(axis=1)
            results.append({'method': method, 'reference': reference, 'events': len(events),
                'new_minus_reference_mape_points': float(delta.mean()), 'ci95_low': float(np.quantile(boot, .025)),
                'ci95_high': float(np.quantile(boot, .975)), 'events_improved': int((delta < 0).sum()),
                'events_worsened': int((delta > 0).sum())})
            deltas.extend({'method': method, 'reference': reference, 'event': int(event), 'delta': float(d)} for event, d in zip(events, delta))
    output.mkdir()
    result = {'evaluation_id': evaluation_id, 'identity': config, 'comparison': results}
    write_json(output/'comparison.json', result)
    pd.DataFrame(deltas).to_csv(output/'event_deltas.csv', index=False)
    for source in sources:
        path = output/'sources'/source; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/source, path)
    write_json(receipt, {'evaluation_id': evaluation_id, 'identity': config,
        'artifacts': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(evaluate(parser.parse_args().run), ensure_ascii=False, indent=2))
