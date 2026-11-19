import importlib.util
import unittest

import numpy as np
import pandas as pd


class NextReportTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.v2_next_report'))
        from analysis import v2_next_report
        return v2_next_report

    def test_bootstrap_identical_predictions_are_zero(self):
        m = self.module()
        result = m.paired_intervals(np.zeros(9), draws=100)
        self.assertEqual(result['mean_difference'], 0.)
        for interval in result['intervals'].values():
            self.assertEqual(interval, [0., 0.])

    def test_block_intervals_are_deterministic_and_capture_dependence(self):
        m = self.module()
        delta = np.repeat([-5., 5.], 20)
        result = m.paired_intervals(delta, draws=5000)
        self.assertEqual(result, m.paired_intervals(delta, draws=5000))
        widths = {k: b-a for k, (a, b) in result['intervals'].items()}
        self.assertGreater(widths['block_5'], widths['iid'])
        self.assertEqual(result['events'], 40)

    def test_errors_group_by_frozen_event_and_sort_actual_start(self):
        m = self.module()
        cases = pd.DataFrame({'case_id': ['a', 'b', 'c'], 'event': [99, 99, 1],
                              'start_ms': [0, 0, 100]})
        predictions = pd.DataFrame({'case_id': ['c', 'a', 'b'], 'predicted': [80., 110., 130.]})
        result = m.ordered_errors(cases, {'a': 100., 'b': 100., 'c': 100.}, predictions)
        self.assertEqual(list(result.index), [99, 1])
        np.testing.assert_allclose(result, [20., 20.])
        with self.assertRaises(ValueError):
            m.ordered_errors(cases, {'a': 100., 'b': 100., 'c': 100.}, predictions.iloc[:2])


if __name__ == '__main__':
    unittest.main()
