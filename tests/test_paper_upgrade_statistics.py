"""Independent statistics: frozen support, activity units and paired resampling."""
import tempfile
import unittest
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import paper_upgrade_statistics as statistics


class IndependentScoringTests(unittest.TestCase):
    def setUp(self):
        # Unequal snapshots and rank support deliberately distinguish the estimands.
        self.cases = pd.DataFrame({
            'case_id': ['a', 'b', 'c', 'd'], 'event': [2, 2, 2, 1],
            'rank': [10, 10, 20, 10], 'start_ms': [20, 20, 20, 10],
            'event_type': ['x', 'x', 'x', 'y']})
        self.labels = pd.DataFrame({'case_id': ['a', 'b', 'c', 'd'], 'actual': [100.] * 4})
        self.predictions = pd.DataFrame({
            'case_id': ['d', 'c', 'b', 'a'], 'method': ['m'] * 4,
            'predicted': [140., 130., 110., 100.]})

    def test_activity_equal_and_rank_balanced_estimands_remain_distinct(self):
        frame = statistics.score_rows(self.predictions, self.cases, self.labels)
        result = statistics.metrics(frame)
        self.assertAlmostEqual(result['event_macro_mape_pct'], 26.666666666666668)
        self.assertAlmostEqual(result['case_mape_pct'], 20.)
        self.assertAlmostEqual(result['event_rank_balanced_mape_pct'], 28.75)
        self.assertAlmostEqual(result['mae_ep'], 20.)
        self.assertAlmostEqual(result['rmse_ep'], np.sqrt(650.))
        self.assertAlmostEqual(result['p90_ape_pct'], 37.)
        self.assertEqual((result['events'], result['cases']), (2, 4))

    def test_missing_case_in_any_method_is_rejected_before_comparison(self):
        p = pd.concat([self.predictions, self.predictions.iloc[:-1].assign(method='other')])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            statistics.score_rows(p, self.cases, self.labels)

    def test_duplicate_and_changed_case_metadata_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            statistics.score_rows(pd.concat([self.predictions, self.predictions.iloc[:1]]), self.cases, self.labels)
        with self.assertRaisesRegex(ValueError, 'event'):
            statistics.score_rows(self.predictions.assign(event=99), self.cases, self.labels)

    def test_zero_actual_and_nonfinite_prediction_are_rejected(self):
        with self.assertRaises(ValueError):
            statistics.score_rows(self.predictions, self.cases, self.labels.assign(actual=0.))
        with self.assertRaises(ValueError):
            statistics.score_rows(self.predictions.assign(predicted=np.inf), self.cases, self.labels)

    def test_overlapping_stage_memberships_do_not_multiply_overall(self):
        frame = statistics.score_rows(self.predictions, self.cases, self.labels)
        members = pd.DataFrame({'case_id': ['a', 'a', 'd'], 'stage_id': ['p50', 'h24', 'p50']})
        groups = statistics.group_metrics(frame, members=members)
        p50 = groups[(groups.group == 'stage_id') & (groups.value == 'p50')].iloc[0]
        self.assertAlmostEqual(p50.event_macro_mape_pct, 20.)
        self.assertEqual(p50.cases, 2)
        self.assertEqual(statistics.metrics(frame)['cases'], 4)

    def test_stage_rank_panel_preserves_frozen_case_counts(self):
        frame = statistics.score_rows(self.predictions, self.cases, self.labels)
        members = pd.DataFrame({'case_id': ['a', 'b', 'c', 'd'], 'stage_id': ['p50'] * 4})
        result = statistics.stage_rank_metrics(frame, members)
        rank10 = result[result['rank'] == 10].iloc[0]
        self.assertAlmostEqual(rank10.event_macro_mape_pct, 22.5)
        self.assertEqual((rank10.events, rank10.cases), (2, 3))

    def test_intervals_use_equal_event_mass_and_report_support(self):
        frame = statistics.score_rows(self.predictions, self.cases, self.labels)
        frame['lower'] = frame.case_id.map({'a': 90., 'b': 90., 'c': 90., 'd': 110.})
        frame['upper'] = frame.case_id.map({'a': 110., 'b': 110., 'c': 110., 'd': 150.})
        frame['nominal_coverage'] = .9
        value = statistics.interval_reliability(frame)
        self.assertEqual(value['status'], 'available')
        self.assertAlmostEqual(value['case_coverage'], .75)
        self.assertAlmostEqual(value['event_macro_coverage'], .5)
        self.assertEqual(value['events'], 2)
        with self.assertRaisesRegex(ValueError, 'interval'):
            statistics.interval_reliability(frame.assign(lower=200.))

    def test_unsupported_unbounded_intervals_are_reported_separately(self):
        frame = statistics.score_rows(self.predictions, self.cases, self.labels)
        frame['lower'], frame['upper'] = 0., np.inf
        frame['nominal_coverage'], frame['available'] = .9, False
        result = statistics.interval_reliability(frame)
        self.assertEqual(result['case_coverage'], 1.)
        self.assertEqual(result['supported_cases'], 0)
        self.assertEqual(result['available_rate'], 0.)
        self.assertIsNone(result['supported_case_coverage'])
        self.assertIsNone(result['mean_width_ep'])

    def test_vectorized_metrics_preserve_unequal_activity_and_rank_weights(self):
        frame = statistics.score_rows(self.predictions, self.cases, self.labels)
        result = statistics.aggregate_metrics(frame, ['method']).iloc[0]
        self.assertAlmostEqual(result.event_macro_mape_pct, 26.666666666666668)
        self.assertAlmostEqual(result.event_rank_balanced_mape_pct, 28.75)
        self.assertAlmostEqual(result.p90_ape_pct, 37.)
        self.assertAlmostEqual(result.rmse_ep, np.sqrt(650.))

    def test_learning_validation_only_never_opens_test_artifacts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data, run = root / 'data', root / 'run'
            data.mkdir(); run.mkdir()
            cases = self.cases.assign(start_ms=statistics.DENSE_START + self.cases.start_ms)
            cases.to_csv(data / 'validation_cases.csv', index=False)
            self.labels.to_csv(data / 'validation_labels.csv', index=False)
            pd.DataFrame({'case_id': ['a', 'b', 'c', 'd'], 'stage_id': ['p50'] * 4}).to_csv(data / 'validation_memberships.csv', index=False)
            p = self.predictions.assign(training_events=16, family='neural', structure='independent', seed='mean')
            p.to_csv(run / 'validation_predictions.csv', index=False)
            def manifest(path, values):
                values['artifacts'] = {q.name: statistics.sha256(q) for q in path.iterdir() if q.is_file()}
                values['seal_id'] = hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                (path / ('manifest.json' if path == data else 'blind_manifest.json')).write_text(json.dumps(values))
            manifest(data, {'dataset_id': 'fixture'})
            manifest(run, {'dataset_id': 'fixture', 'scored_splits': ['validation'], 'methods': ['m']})
            output = root / 'scored'
            result = statistics.score_run(run, output, data=data, draws=10)
            self.assertEqual(set(result['splits']), {'validation'})
            self.assertFalse((output / 'test_case_losses.csv').exists())
            learning = pd.read_csv(output / 'learning_metrics.csv')
            self.assertEqual(learning.iloc[0].training_events, 16)
            self.assertEqual(learning.iloc[0].structure, 'independent')
            self.assertAlmostEqual(learning.iloc[0].event_macro_mape_pct, 26.666666666666668)

    def test_learning_comparisons_match_size_seed_and_family(self):
        methods = ['neural_independent_n16_seed42', 'neural_shared_n16_seed42',
                   'neural_shared_n67_seed42', 'neural_rank_heads_n16_seed123']
        metadata = pd.DataFrame({'method': methods, 'family': ['neural'] * 4,
                                 'structure': ['independent', 'shared', 'shared', 'rank_heads'],
                                 'training_events': [16, 16, 67, 16], 'seed': ['42', '42', '42', '123']})
        contrasts = statistics.planned_comparisons(methods, metadata)
        self.assertIn(('neural_independent_n16_seed42', 'neural_shared_n16_seed42', 'learning_structure'), contrasts)
        self.assertIn(('neural_shared_n16_seed42', 'neural_shared_n67_seed42', 'learning_size'), contrasts)
        self.assertFalse(any(a == 'neural_rank_heads_n16_seed123' for a, b, scope in contrasts))

    def test_prediction_source_tamper_is_rejected_before_outcome_read(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'pred.csv').write_text('a\n1\n')
            values = {'artifacts': {'pred.csv': '0' * 64}}
            (root / 'blind_manifest.json').write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, 'hash'):
                statistics.verify_prediction_run(root)


class PairedInferenceTests(unittest.TestCase):
    def test_interval_joint_coverage_has_distinct_group_and_whole_event_targets(self):
        frame = pd.DataFrame({'event': [1, 1, 1, 1, 2], 'rank': [10, 10, 20, 20, 10],
                              'phase': [0] * 5, 'start_ms': [10, 10, 10, 10, 20],
                              'actual': [100.] * 5, 'lower': [90.] * 5,
                              'upper': [110., 90., 110., 110., 110.]})
        groups, events, summary = statistics.joint_interval_coverage(frame)
        self.assertEqual(len(groups), 3)
        self.assertEqual(events.group_joint_coverage.tolist(), [.5, 1.])
        self.assertEqual(events.entire_event_joint_coverage.tolist(), [False, True])
        self.assertEqual(summary['event_equal_group_joint_coverage'], .75)
        self.assertEqual(summary['entire_event_joint_coverage'], .5)
        self.assertAlmostEqual(summary['group_equal_joint_coverage'], 2 / 3)

    def test_circular_blocks_are_contiguous_with_wrap_and_fixed_length(self):
        indices = statistics.resample_indices(5, 20, 3, seed=42)
        self.assertEqual(indices.shape, (20, 5))
        self.assertTrue(np.all((indices[:, 1] - indices[:, 0]) % 5 == 1))
        self.assertTrue(np.all((indices[:, 2] - indices[:, 1]) % 5 == 1))
        self.assertTrue(np.all((indices[:, 4] - indices[:, 3]) % 5 == 1))
        np.testing.assert_array_equal(indices, statistics.resample_indices(5, 20, 3, seed=42))

    def test_pairing_and_sign_are_preserved_and_leave_one_out_is_literal(self):
        # Method-reference: [-2,-2,+1,+1], mean -.5 percentage points.
        events = pd.DataFrame({'event': [4, 2, 3, 1], 'start_ms': [40, 20, 30, 10],
                               'method_loss': [11., 8., 11., 8.], 'reference_loss': [10.] * 4})
        result = statistics.paired_effect(events, draws=100, seed=42, blocks=(1, 3))
        self.assertEqual(result['event_order'], [1, 2, 3, 4])
        self.assertEqual(result['event_differences'], [-2., -2., 1., 1.])
        self.assertEqual(result['mean_difference_points'], -.5)
        self.assertEqual(result['improved_events'], 2)
        self.assertEqual(result['leave_one_out_min'], -1.)
        self.assertEqual(result['leave_one_out_max'], 0.)
        self.assertAlmostEqual(result['top1_absolute_share'], 1 / 3)
        self.assertEqual(set(result['bootstrap']), {'1', '3'})
        constant = events.assign(method_loss=8.)
        for interval in statistics.paired_effect(constant, draws=20, blocks=(1, 3))['bootstrap'].values():
            self.assertEqual(interval['ci95'], [-2., -2.])

    def test_long_blocks_are_explicitly_unavailable_for_small_groups(self):
        events = pd.DataFrame({'event': [1, 2], 'method_loss': [1., 2.], 'reference_loss': [2., 3.]})
        result = statistics.paired_effect(events, draws=10)
        self.assertEqual(result['bootstrap']['3']['status'], 'insufficient_events')


class MatchedCorrelationTests(unittest.TestCase):
    def test_correlation_bootstrap_uses_paired_activity_indices(self):
        first = pd.DataFrame({10: [1., 2., 3., 4.], 20: [4., 3., 2., 1.]})
        # Adding a constant leaves every resampled rank correlation unchanged.
        pairs, contrasts = statistics.correlation_bootstrap({'first': first, 'second': first + 7.},
                                                           draws=20, blocks=(1, 3))
        self.assertEqual(set(pairs.block), {1, 3})
        self.assertEqual(set(contrasts.reference), {'first'})
        np.testing.assert_allclose(contrasts.low95, 0., atol=1e-12)
        np.testing.assert_allclose(contrasts.high95, 0., atol=1e-12)

    def test_joint_missingness_is_frozen_across_all_targets(self):
        first = pd.DataFrame({10: [1., 2., 3., 4.], 20: [4., 3., 2., 1.]}, index=[1, 2, 3, 4])
        second = pd.DataFrame({10: [4., 2., 1., 3.], 20: [1., np.nan, 4., 2.]}, index=[4, 3, 2, 1])
        clock = {i: {'start_ms': 50 - 10 * i} for i in [1, 2, 3, 4]}
        panel, info = statistics.matched_panels({'first': first, 'second': second}, clock)
        self.assertEqual(info['event_ids'], [4, 2, 1])
        self.assertEqual(info['ranks'], [10, 20])
        self.assertEqual(info['excluded_events'], [3])
        self.assertEqual(list(panel['first'].index), [4, 2, 1])
        result = statistics.correlation_rows(panel)
        self.assertEqual(set(result.n), {3})
        self.assertAlmostEqual(result[result.panel == 'first'].iloc[0].spearman, -1.)

    def test_new_output_preserves_existing_artifacts_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'source.csv'
            source.write_text('hello\n')
            output = root / 'new'
            statistics.write_bundle(output, {'values.csv': pd.DataFrame({'x': [1]})},
                                    {'status': 'checked'}, sources=[source])
            self.assertTrue((output / 'manifest.json').exists())
            with self.assertRaises(FileExistsError):
                statistics.write_bundle(output, {}, {}, sources=[source])


if __name__ == '__main__':
    unittest.main()
