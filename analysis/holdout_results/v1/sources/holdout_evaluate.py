"""Separate final-label scoring for an already sealed blind-prediction run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.holdout_dataset import HoldoutDataset, sha256, write_json


def case_metadata(samples, metadata=None):
    rows = []
    for sample in samples:
        curve = sample['curve']
        row = {'case_id':sample['case_id'], 'event':curve.event, 'rank':curve.rank,
               'stage':sample['stage'], 'event_type':curve.kind}
        if metadata is not None:
            row['band_group'] = metadata.band_only(curve)['band_group']
        rows.append(row)
    return rows


def metrics(frame):
    actual = frame.actual.to_numpy(dtype=float)
    delta = frame.predicted.to_numpy(dtype=float) - actual
    ape = np.abs(delta / actual) * 100
    groups = pd.DataFrame({'event':frame.event.to_numpy(), 'ape':ape}).groupby('event').ape.mean()
    return {'event_macro_mape_pct':float(groups.mean()), 'case_mape_pct':float(ape.mean()),
            'median_ape_pct':float(np.median(ape)), 'p90_ape_pct':float(np.quantile(ape,.9)),
            'mae_ep':float(np.mean(abs(delta))), 'rmse_ep':float(np.sqrt(np.mean(delta**2))),
            'bias_pct':float(np.mean(delta/actual)*100), 'events':len(groups), 'cases':len(frame)}


def score_predictions(cases, labels, predictions):
    expected = [r['case_id'] for r in cases]
    ids = [r['case_id'] for r in predictions]
    if len(expected) != len(set(expected)) or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Prediction case IDs must match the fixed cases exactly, without duplicates')
    if set(labels) != set(expected):
        raise ValueError('Label case IDs must match the fixed cases exactly')
    lookup = {r['case_id']:float(r['predicted']) for r in predictions}
    frame = pd.DataFrame(cases)
    frame['actual'] = [labels[c] for c in expected]
    frame['predicted'] = [lookup[c] for c in expected]
    if not np.isfinite(frame[['actual','predicted']]).all().all() or (frame.actual <= 0).any() or (frame.predicted <= 0).any():
        raise ValueError('Final scores and predictions must be positive and finite')
    result = {'overall':metrics(frame)}
    for key in ['rank','stage','event_type','band_group']:
        if key in frame:
            result['by_'+key] = {str(k):metrics(g) for k,g in frame.groupby(key)}
    return result


def evaluate_run(run_directory, dataset_directory=None):
    run_directory = Path(run_directory)
    frozen = json.loads((run_directory/'blind_test_manifest.json').read_text())
    data = HoldoutDataset(dataset_directory or frozen['dataset_directory'])
    if data.manifest['dataset_id'] != frozen['dataset_id']:
        raise ValueError('Run uses a different dataset fingerprint')
    for name, digest in frozen['artifacts'].items():
        if sha256(run_directory/name) != digest:
            raise ValueError(f'Sealed run hash mismatch: {name}')
    from analysis.event_features import EventFeatures
    cases = case_metadata(data.inputs('test'), EventFeatures.load(data.metadata_directory))
    # This is the only test-label request in the benchmark pipeline.
    labels = data.labels('test', allow_test=True)
    predictions = pd.read_csv(run_directory/'test_predictions.csv')
    selection = json.loads((run_directory/'selection.json').read_text())
    if set(predictions.method) != set(selection['reported_methods']):
        raise ValueError('Reported methods differ from the validation-sealed selection')
    results = {}
    summary = []
    for method, group in predictions.groupby('method'):
        score = score_predictions(cases, labels, group.to_dict('records'))
        results[method] = score
        summary.append({'method':method, **score['overall']})
    summary = pd.DataFrame(summary).sort_values('event_macro_mape_pct')
    summary.to_csv(run_directory/'test_summary.csv', index=False)
    # Paired bootstrap resamples whole event groups, not correlated snapshots.
    selected = selection['selected_recipe']
    errors = {}
    for method, group in predictions.groupby('method'):
        g = group.copy(); g['actual'] = g.case_id.map(labels)
        g['ape'] = abs(g.predicted/g.actual-1)*100
        errors[method] = g.groupby('event').ape.mean().sort_index()
    rng = np.random.default_rng(42)
    event_ids = errors[selected].index
    draws = rng.integers(0,len(event_ids),(5000,len(event_ids)))
    comparisons = {}
    for reference in ['linear','lightgbm_sequence','catboost_peer']:
        if reference == selected or reference not in errors: continue
        difference = (errors[selected]-errors[reference]).to_numpy()
        boot = difference[draws].mean(axis=1)
        comparisons[reference] = {'selected_minus_reference_mape_points':float(difference.mean()),
            'paired_event_bootstrap_95pct_ci':np.quantile(boot,[.025,.975]).tolist()}
    result = {'dataset_id':data.manifest['dataset_id'], 'run_id':frozen['run_id'],
              'selection':selection, 'test':results, 'paired_comparisons':comparisons,
              'exposure':data.manifest['protocol']['exposure']}
    write_json(run_directory/'evaluation.json', result)
    predictions['actual'] = predictions.case_id.map(labels)
    predictions['ape_pct'] = abs(predictions.predicted/predictions.actual-1)*100
    predictions.to_csv(run_directory/'scored_test_predictions.csv', index=False)
    print(summary[['method','event_macro_mape_pct','case_mape_pct','events','cases']].to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--dataset', type=Path)
    args = parser.parse_args()
    evaluate_run(args.run,args.dataset)


if __name__ == '__main__':main()
