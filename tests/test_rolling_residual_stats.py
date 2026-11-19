"""Causal/statistical behavior checks, with no dataset or outcome files."""
import importlib
import json
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

try:
    stats = importlib.import_module('analysis.rolling_residual_stats')
except ModuleNotFoundError:
    stats = None


CUTOFF = 1706504400000
QUERY = CUTOFF + 10000


def row(event, error=.2, *, rank=1000, phase=0, progress=.1,
        event_type='a', band_group='x', suffix=''):
    start = CUTOFF + event * 100
    return {'case_id': f'{event}-{rank}-{phase}{suffix}', 'event': event,
            'rank': rank, 'phase': phase, 'progress': progress,
            'start_ms': start, 'end_ms': start + 50,
            'log_error': error, 'ewma_bias': 0.,
            'event_type': event_type, 'band_group': band_group}


def history(*rows):
    return pd.DataFrame(list(rows), columns=list(row(1)))


def target(**kwargs):
    result = row(99, **kwargs)
    result.pop('log_error')
    result['start_ms'] = QUERY
    result['end_ms'] = QUERY + 100
    return pd.DataFrame([result])


class ResidualStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(stats, 'Pure statistical correction module is missing')

    def predict(self, name, past, query=None):
        return stats.predict_bias(name, past, target() if query is None else query,
                                  query_start_ms=QUERY)

    def test_empty_history_means_zero_for_every_method(self):
        for name in stats.CONFIGS:
            with self.subTest(name=name):
                bias, audit = self.predict(name, history())
                np.testing.assert_array_equal(bias, [0.])
                json.dumps(audit, allow_nan=False)

    def test_reference_matches_real_online_bias_across_groups_and_event_means(self):
        from analysis.rolling_benchmark import OnlineBias
        past = history(row(2, -.2), row(1, .1), row(1, .3, suffix='b'),
                       row(1, -.1, rank=2000), row(3, .4, phase=1, progress=.5))
        online = OnlineBias()
        for event, part in past.sort_values(['end_ms', 'event']).groupby('event', sort=False):
            observations = part[['case_id', 'event', 'rank', 'phase']].copy()
            observations['predicted'] = np.exp(part.log_error.to_numpy())
            online.observe(observations, dict.fromkeys(part.case_id, 1.),
                           event=event, end_ms=int(part.end_ms.iloc[0]))
        query = pd.concat([target(), target(rank=2000),
                           target(phase=1, progress=.5), target(phase=2, progress=.95)],
                          ignore_index=True)
        actual, _ = self.predict('ewma_reference', past, query)
        expected = [online.bias(1000, 0), online.bias(2000, 0),
                    online.bias(1000, 1), online.bias(1000, 2)]
        np.testing.assert_allclose(actual, expected, atol=1e-16, rtol=0)
        self.assertAlmostEqual(actual[0], .013333333333333334)

    def test_fast_ewma_can_reverse_sign_after_new_negative_error(self):
        bias, _ = self.predict('ewma_fast', history(row(1, .2), row(2, -.2)))
        self.assertAlmostEqual(bias[0], -.006666666666666667)

    def test_window_uses_five_events_and_five_event_shrinkage(self):
        past = history(row(1, -9.), *[row(i, .1) for i in range(2, 7)])
        bias, _ = self.predict('window5', past)
        self.assertAlmostEqual(bias[0], .03333333333333333)

    def test_kalman_level_has_zero_prior_and_specified_observation_uncertainty(self):
        bias, audit = self.predict('kalman_level', history(row(1, .2)))
        self.assertAlmostEqual(bias[0], .18198198198198198)
        json.dumps(audit, allow_nan=False)

    def test_kalman_trend_projects_level_and_trend_one_event_ahead(self):
        bias, _ = self.predict('kalman_trend', history(row(1, .2)))
        self.assertAlmostEqual(bias[0], .2)
        negative, _ = self.predict('kalman_trend', history(row(1, -.2)))
        self.assertAlmostEqual(negative[0], -.2)

    def test_progress_kernel_shares_rank_across_phase_and_decays_by_event(self):
        past = history(row(1, .1), row(2, -.1, phase=1, progress=.5))
        query = target(phase=2, progress=.95)
        bias, _ = self.predict('kernel_progress', past, query)
        # One row per event: normalized event kernels equal its error; only .7 age remains.
        self.assertAlmostEqual(bias[0], -.0029411764705882353)
        other_rank, _ = self.predict('kernel_progress', past, target(rank=2000))
        self.assertEqual(other_rank[0], 0.)

    def test_kernel_progress_weights_prefixes_inside_each_event(self):
        past = history(row(1, .2), row(1, -.2, phase=1, progress=.4, suffix='b'))
        bias, _ = self.predict('kernel_progress', past)
        # Distance .3 at bandwidth .15 gives exp(-2), then one-event shrinkage 1/11.
        self.assertAlmostEqual(bias[0], .013847166471631905)

    def test_partial_pool_adds_small_matching_context_deviations(self):
        past = history(row(1, .2), row(2, -.2, event_type='b', band_group='y'))
        bias, _ = self.predict('context_partial_pool', past)
        self.assertAlmostEqual(bias[0], .01718560606060606)
        unseen, _ = self.predict('context_partial_pool', past,
                                 target(event_type='unseen', band_group='unseen'))
        self.assertAlmostEqual(unseen[0], .013333333333333334)

    def test_repeating_a_whole_event_prefix_panel_does_not_change_event_weight(self):
        past = history(row(1, .1), row(1, .3, suffix='b'), row(2, -.1))
        repeated = pd.concat([past, past[past.event == 1].assign(
            case_id=lambda frame: frame.case_id + '-replica')], ignore_index=True)
        for name in stats.CONFIGS:
            with self.subTest(name=name):
                first, _ = self.predict(name, past)
                second, _ = self.predict(name, repeated)
                np.testing.assert_allclose(first, second, rtol=0, atol=1e-16)

    def test_reordering_frames_preserves_prediction_order_and_does_not_mutate(self):
        past = history(row(2, -.2), row(1, .2))
        query = pd.concat([target(rank=2000), target()], ignore_index=True)
        past_copy, query_copy = past.copy(deep=True), query.copy(deep=True)
        for name in stats.CONFIGS:
            first, _ = self.predict(name, past, query)
            second, _ = self.predict(name, past.iloc[::-1], query.iloc[::-1])
            np.testing.assert_allclose(first, second[::-1], rtol=0, atol=1e-16)
        assert_frame_equal(past, past_copy)
        assert_frame_equal(query, query_copy)

    def test_all_methods_clip_total_bias_and_keep_correction_sign(self):
        for name in stats.CONFIGS:
            for sign in (-1, 1):
                with self.subTest(name=name, sign=sign):
                    bias, _ = self.predict(name, history(row(1, 100. * sign)))
                    self.assertEqual(bias[0], .3 * sign)
                    # Positive log(predicted/actual) must lower the base prediction.
                    self.assertEqual(np.sign(100. * np.exp(-bias[0]) - 100.), -sign)

    def test_rejects_equal_end_future_and_early_history_in_every_method(self):
        for name in stats.CONFIGS:
            for column, value in [('end_ms', QUERY), ('end_ms', QUERY + 1),
                                  ('start_ms', CUTOFF - 1)]:
                with self.subTest(name=name, column=column, value=value):
                    past = history(row(1))
                    past.loc[0, column] = value
                    with self.assertRaises(ValueError):
                        self.predict(name, past)

    def test_rejects_current_event_duplicate_case_and_inconsistent_event_clock(self):
        variants = [history(row(99)), history(row(1), row(1)),
                    history(row(1), row(1, suffix='b'))]
        variants[-1].loc[1, 'end_ms'] += 1
        for past in variants:
            with self.assertRaises(ValueError):
                self.predict('ewma_reference', past)

    def test_rejects_invalid_errors_progress_phase_and_target_answers(self):
        for column, value in [('log_error', np.inf), ('log_error', np.nan),
                              ('progress', -.01), ('phase', 2), ('rank', 1000.5)]:
            past = history(row(1))
            past[column] = [value]
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.predict('ewma_reference', past)
        for column in ('actual', 'target', 'log_error'):
            query = target().assign(**{column: 1.})
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.predict('ewma_reference', history(), query)
        with self.assertRaises(ValueError):
            self.predict('ewma_reference', history(), target().assign(start_ms=QUERY + 1))

    def test_unknown_method_fails_closed(self):
        with self.assertRaises(ValueError):
            self.predict('not_registered', history())


if __name__ == '__main__':
    unittest.main()
