"""Regression checks for leakage, time boundaries, and empirical error ranges."""
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from analysis.backtest import Curve
from analysis.data_policy import POLICY_ID

if importlib.util.find_spec('forecasting'):
    import forecasting
else:
    forecasting = None

HOUR = 3600000
START = 1706504400000


def curve(event, start, rank=2000, duration=200, points=None):
    hours = np.arange(duration + 1)
    return Curve(event, rank, 'live_try', start, start + duration * HOUR,
                 start + hours * HOUR, hours.astype(float) * 100 if points is None else points)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(forecasting, 'forecasting module is not implemented')
        self.history = [curve(226 + i, START + (i + 1) * 224 * HOUR) for i in range(30)]
        self.target = curve(328, START + 32 * 224 * HOUR)
        self.cutoff = self.target.start + 100 * HOUR

    def test_history_uses_dates_and_excludes_both_target_tiers(self):
        # Numeric IDs cannot replace availability timestamps or event grouping.
        early_high_id = curve(999, START + HOUR)
        late_low_id = curve(1, self.target.start + HOUR)
        same_event = curve(328, START + HOUR, rank=1000)
        chosen = forecasting.select_history(self.target, [early_high_id, late_low_id, same_event])
        self.assertEqual([c.event for c in chosen], [999])

    def test_suffix_scores_cannot_change_a_forecast(self):
        # A future-value feature or target-tier leakage would break this equality.
        altered = self.target.scores.copy()
        altered[self.target.times > self.cutoff] *= 1000
        first = forecasting.predict_cutoff(self.target, self.history, self.cutoff)
        second = forecasting.predict_cutoff(replace(self.target, scores=altered), self.history, self.cutoff)
        self.assertEqual(first['predicted_final_ep'], second['predicted_final_ep'])
        self.assertEqual(first['analog_final_ep'], second['analog_final_ep'])
        self.assertGreaterEqual(first['predicted_final_ep'], first['current_ep'])
        self.assertEqual(first['training_events'], 30)

    def test_stale_observations_are_rejected(self):
        truncated = replace(self.target, times=self.target.times[:101], scores=self.target.scores[:101])
        with self.assertRaisesRegex(ValueError, '60 分钟'):
            forecasting.predict_cutoff(truncated, self.history, self.cutoff + 2 * HOUR)

    def test_before_start_and_after_end_are_rejected(self):
        for cutoff in [self.target.start - HOUR, self.target.end]:
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                forecasting.predict_cutoff(self.target, self.history, cutoff)

    def test_insufficient_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '20'):
            forecasting.predict_cutoff(self.target, self.history[:2], self.cutoff)

    def test_missing_target_rank_templates_fail_cleanly(self):
        different_rank = [replace(c, rank=1000) for c in self.history]
        with self.assertRaisesRegex(ValueError, '同排名|参考'):
            forecasting.predict_cutoff(self.target, different_rank, self.cutoff)

    def test_templates_without_early_observations_fail_cleanly(self):
        delayed = [replace(c, times=c.times[80:], scores=c.scores[80:]) for c in self.history]
        with self.assertRaisesRegex(ValueError, '同排名|参考'):
            forecasting.predict_cutoff(self.target, delayed, self.target.start + 30 * HOUR)

    def calibration(self, count=20):
        frame = pd.DataFrame([{'event': 226 + i, 'rank': 2000, 'stage': '50%',
                              'method': 'xgb_growth', 'predicted': 100.,
                              'actual': 100. * np.exp(i / 100)} for i in range(count)])
        frame.attrs['data_policy_id'] = POLICY_ID
        return frame

    def test_calibration_uses_finite_sample_log_error_quantile(self):
        # 20 distinct errors 0..0.19: ceil(21 * .8)=17th, so q=.16.
        interval = forecasting.calibrate_range(100., 80., 2000, '50%', self.calibration(), self.history)
        self.assertEqual(interval['calibration_events'], 20)
        self.assertAlmostEqual(interval['lower_ep'], 100. * np.exp(-.16))
        self.assertAlmostEqual(interval['upper_ep'], 100. * np.exp(.16))
        self.assertEqual(interval['coverage_target'], .8)

    def test_future_and_target_calibration_rows_do_not_inflate_range(self):
        rows = self.calibration()
        extra = rows.iloc[[0]].copy()
        extra['event'] = self.target.event
        extra['actual'] = 1e10
        actual = forecasting.calibrate_range(100., 80., 2000, '50%', pd.concat([rows, extra]), self.history)
        expected = forecasting.calibrate_range(100., 80., 2000, '50%', rows, self.history)
        self.assertEqual(actual, expected)

    def test_repeated_rows_do_not_count_as_independent_events(self):
        rows = pd.concat([self.calibration(5)] * 5)
        self.assertIsNone(forecasting.calibrate_range(100., 80., 2000, '50%', rows, self.history))

    def test_range_respects_the_current_score(self):
        interval = forecasting.calibrate_range(100., 99., 2000, '50%', self.calibration(), self.history)
        self.assertEqual(interval['lower_ep'], 99.)

    def test_early_prediction_has_no_calibrated_range(self):
        result = forecasting.predict_cutoff(self.target, self.history, self.target.start + 30 * HOUR,
                                            calibration=self.calibration())
        self.assertIsNone(result['historical_error_range'])
        self.assertTrue(any('25%' in note for note in result['notes']))

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / 'missing.sqlite3'
            with self.assertRaises(ValueError):
                forecasting.load_local(missing, 328, 2000)
            self.assertFalse(missing.exists())


if __name__ == '__main__':
    unittest.main()
