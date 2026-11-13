import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis.holdout_dataset import HoldoutDataset, sha256, write_json
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_rank_subset import DEFAULT
from analysis.temporal_evaluate import evaluate, verify_predictions
from analysis.temporal_paired_benchmark import predict_saved

run = Path('analysis/temporal_results/v2_paired_tcn_20261112')
old = Path('analysis/temporal_results/v2_t1000_t2000_full_20261112')
v1 = HoldoutDataset(); data = TemporalDataset(DEFAULT)
seal = verify_predictions(run); verify_predictions(old)
checks = {'dataset_id': data.manifest['dataset_id']}
for split in ('train', 'validation', 'test'):
    assert data.manifest['splits'][split]['event_ids'] == v1.manifest['splits'][split]['event_ids']
    audit = pd.read_csv(run/f'features/{split}_pair_audit.csv')
    cases = data.case_index(split)
    assert set(audit.case_id) == set(cases.case_id) and not audit.case_id.duplicated().any()
    available = audit.peer_available == 1
    assert (audit.loc[available, 'peer_observed_at'] <= audit.loc[available, 'observed_at']).all()
    predictions = pd.read_csv(run/f'{split}_predictions.csv')
    for method, group in predictions.groupby('method'):
        assert set(group.case_id) == set(cases.case_id) and len(group) == len(cases)
    prior = pd.read_csv(old/f'{split}_predictions.csv')
    for current, previous in [('reference_catboost', 'catboost_rank'), ('linear', 'linear'), ('hold_current', 'hold_current')]:
        a = predictions[predictions.method == current].set_index('case_id').predicted.sort_index()
        b = prior[prior.method == previous].set_index('case_id').predicted.sort_index()
        np.testing.assert_allclose(a, b, rtol=1e-14)
    pivot = predictions.pivot(index='case_id', columns='method', values='predicted')
    np.testing.assert_allclose(pivot.dual_tcn_mean, pivot[['dual_tcn_seed42', 'dual_tcn_seed123', 'dual_tcn_seed2026']].mean(axis=1), rtol=1e-14)
    checks[split] = {'cases': len(cases), 'available_peers': int(available.sum()), 'future_peer_observations': 0,
                     'all_nine_methods_same_cases': True, 'reference_predictions_unchanged': True, 'final_score_ensemble_verified': True}
weights = pd.read_csv(run/'training_weights.csv')
for name in ('event_weight', 'phase_weight'):
    totals = weights.groupby('event')[name].sum()
    assert len(totals) == 196 and np.allclose(totals, totals.iloc[0])
weights['phase'] = np.where(weights.progress < .25, 0, np.where(weights.progress < .9, 1, 2))
for event, group in weights.groupby('event'):
    phase_totals = group.groupby('phase').phase_weight.sum()
    assert np.allclose(phase_totals, phase_totals.iloc[0])
checks['weights'] = {'event_totals_equal': True, 'available_phase_totals_equal_within_event': True}
with patch.object(TemporalDataset, 'labels', side_effect=AssertionError('Cached scorers must not read answers')):
    result = evaluate(run)
    spec = importlib.util.spec_from_file_location('comparison', run/'comparison/score.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    comparison = module.score()
checks['both_cached_scorers_return_without_reading_answers'] = True
for row in comparison['comparison']:
    difference = result['splits']['test'][row['method']]['overall']['event_macro_mape_pct'] - result['splits']['test']['reference_catboost']['overall']['event_macro_mape_pct']
    assert np.isclose(difference, row['new_minus_reference_mape_points'])
samples = list(data.inputs('test'))
saved = pd.read_csv(run/'test_predictions.csv').pivot(index='case_id', columns='method', values='predicted')
for method in ['catboost_paired', 'catboost_phase', 'dual_tcn_seed42', 'dual_tcn_seed123', 'dual_tcn_seed2026', 'dual_tcn_mean']:
    replay = predict_saved(run, samples, method)
    np.testing.assert_allclose(replay.predicted, replay.case_id.map(saved[method]), rtol=1e-6)
    print('Checkpoint replay verified:', method, flush=True)
checks['all_fitted_models_and_ensemble_replayed_on_complete_test_prefixes'] = True
for receipt in [old/'delivery_manifest.json', run/'evaluation_manifest.json', run/'report/manifest.json', run/'comparison/evaluation_manifest.json']:
    for name, digest in json.loads(receipt.read_text())['artifacts'].items():
        assert sha256(receipt.parent/name) == digest, name
for name, digest in json.loads(Path('docs/superpowers/plans/2026-11-12-temporal-preservation.json').read_text()).items():
    assert sha256(Path(name)) == digest, name
checks['old_datasets_database_and_delivery_unchanged'] = True
checks['new_prediction_evaluation_report_comparison_seals_verified'] = True
write_json(run/'verification.json', checks)
print(json.dumps(checks, ensure_ascii=False, indent=2))
