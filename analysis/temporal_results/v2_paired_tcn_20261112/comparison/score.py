"""Independent descriptive comparison of frozen models; never used for selection."""
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.holdout_dataset import ROOT, sha256, write_json
from analysis.holdout_evaluate import paired_event_errors
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_evaluate import verify_predictions

OUTPUT = Path(__file__).resolve().parent
RUN = OUTPUT.parent
SOURCES = ['analysis/holdout_evaluate.py', 'analysis/holdout_dataset.py', 'analysis/backtest.py',
           'analysis/temporal_dataset.py', 'analysis/temporal_evaluate.py', 'analysis/multirank_dataset.py']


def score():
    seal = verify_predictions(RUN)
    data = TemporalDataset(seal['dataset_directory'])
    if data.manifest['dataset_id'] != seal['dataset_id']:
        raise ValueError('Dataset differs')
    config = {'blind_manifest_sha256': sha256(RUN/'blind_manifest.json'),
        'dataset_id': seal['dataset_id'], 'script_sha256': sha256(Path(__file__)),
        'sources': {n: sha256(ROOT/n) for n in SOURCES},
        'python': platform.python_version(), 'software': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas']},
        'seed': 42, 'draws': 5000, 'reference': 'reference_catboost',
        'methods': ['catboost_paired', 'catboost_phase', 'dual_tcn_mean'],
        'interpretation': 'descriptive marginal 95% percentile intervals; no multiple-comparison adjustment or selection'}
    evaluation_id = identity(config)
    receipt = OUTPUT/'evaluation_manifest.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['evaluation_id'] != evaluation_id:
            raise FileExistsError('Changed comparison scorer; use a separate audit directory')
        for n, d in old['artifacts'].items():
            if sha256(OUTPUT/n) != d:
                raise ValueError('Comparison artifact changed')
        return json.loads((OUTPUT/'comparison.json').read_text())
    if (OUTPUT/'comparison.json').exists() or (OUTPUT/'event_deltas.csv').exists():
        raise FileExistsError('Comparison outputs already exist')
    cases = data.case_index('test').to_dict('records')
    # Final scoring only. No training, feature engineering, or candidate changes.
    labels = data.labels('test', allow_test=True)
    predictions = pd.read_csv(RUN/'test_predictions.csv')
    errors = {m: paired_event_errors(cases, labels, predictions[predictions.method == m].to_dict('records'))
              for m in [config['reference']]+config['methods']}
    reference = errors[config['reference']]
    n = len(reference); rng = np.random.default_rng(config['seed'])
    indices = rng.integers(0, n, size=(config['draws'], n))
    results = []; delta_rows = []
    for method in config['methods']:
        if not errors[method].index.equals(reference.index):
            raise ValueError('Event membership differs')
        delta = (errors[method]-reference).to_numpy()
        boot = delta[indices].mean(axis=1)
        results.append({'method': method, 'events': n, 'new_minus_reference_mape_points': float(delta.mean()),
            'ci95_low': float(np.quantile(boot, .025)), 'ci95_high': float(np.quantile(boot, .975)),
            'events_improved': int((delta < 0).sum()), 'events_worsened': int((delta > 0).sum())})
        delta_rows.extend({'event': int(e), 'method': method, 'new_minus_reference_mape_points': float(v)}
                          for e, v in zip(reference.index, delta))
    result = {'evaluation_id': evaluation_id, 'identity': config, 'comparison': results}
    write_json(OUTPUT/'comparison.json', result)
    pd.DataFrame(delta_rows).to_csv(OUTPUT/'event_deltas.csv', index=False)
    write_json(receipt, {'evaluation_id': evaluation_id, 'identity': config,
        'artifacts': {n: sha256(OUTPUT/n) for n in ['score.py', 'comparison.json', 'event_deltas.csv']}})
    return result


if __name__ == '__main__':
    print(json.dumps(score(), ensure_ascii=False, indent=2))
