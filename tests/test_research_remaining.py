"""Contracts for remaining-growth development models, without frozen answers."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analysis import research_remaining as remaining


def fixture_arrays():
    current = np.array([100., 200., 150., 300., 80., 160., 120., 240.])
    progress = np.array([.25, .5, .5, .75, .25, .5, .5, .75])
    single = pd.DataFrame({
        'value': [1., 3., np.nan, 2., 5., 7., 4., 6.],
        'event_type': ['normal', 'normal', 'challenge', 'challenge'] * 2,
        'progress': progress,
    })
    cases = [{'case_id': f'train:{i//2}:1000:{i}', 'event': i//2,
              'rank': 1000 if i % 2 == 0 else 2000} for i in range(8)]
    return {'single': single, 'paired': single.assign(peer_value=np.arange(8.)),
            'current': current, 'progress': progress,
            'rank': np.arange(8) % 2, 'cases': cases, 'audit': pd.DataFrame(cases)}


class RemainingTargetTests(unittest.TestCase):
    def test_remaining_rate_decode_shrinks_to_current_at_event_end(self):
        # A constant relative remaining rate of two adds 6x, 2x, .02x
        # the observed score at the three hand-computed progress values.
        result = remaining.decode_target('remaining_log_rate', np.log([2., 2., 2., 2.]),
                                         np.array([100.] * 4),
                                         np.array([.25, .5, 100 / 101, 1.]))
        np.testing.assert_allclose(result, [700., 300., 102., 100.])

    def test_direct_remaining_ratio_cannot_predict_below_current(self):
        result = remaining.decode_target('final_ratio', np.array([.1, 1., 2.5]),
                                         np.array([100., 100., 100.]), np.array([.2, .5, .9]))
        np.testing.assert_allclose(result, [100., 100., 250.])

    def test_remaining_rate_round_trip_and_finite_extremes(self):
        arrays = fixture_arrays()
        actual = np.array([700., 600., 450., 400., 560., 480., 360., 320.])
        target = remaining.encode_target('remaining_log_rate', actual,
                                         arrays['current'], arrays['progress'])
        np.testing.assert_allclose(remaining.decode_target('remaining_log_rate', target,
                                                          arrays['current'], arrays['progress']), actual)
        result = remaining.decode_target('remaining_log_rate', np.array([1.e9, -1.e9]),
                                         np.array([100., 100.]), np.array([1 - 1.e-12, 1.]))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 100.).all())

    def test_training_preprocessing_is_fixed_and_bundle_replays_in_case_order(self):
        arrays = fixture_arrays()
        labels = {case['case_id']: current * 2 for case, current in zip(arrays['cases'], arrays['current'])}
        config = remaining.recipes()[1] | {'iterations': 8, 'depth': 2}
        bundle = remaining.fit_recipe(config, arrays, labels)
        # A median from validation or prediction rows would change this 4.
        numeric = bundle['preprocessor'].named_transformers_['num']
        self.assertEqual(numeric.statistics_[0], 4.)
        valid = {**arrays, 'paired': arrays['paired'].iloc[::-1].reset_index(drop=True).copy(),
                 'current': arrays['current'][::-1], 'progress': arrays['progress'][::-1],
                 'rank': arrays['rank'][::-1], 'cases': list(reversed(arrays['cases']))}
        valid['paired']['event_type'] = 'unseen'
        prediction = remaining.predict_recipe(bundle, valid)
        self.assertEqual(len(prediction), len(valid['cases']))
        self.assertTrue((prediction >= valid['current']).all())
        self.assertEqual(numeric.statistics_[0], 4.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(bundle, path)
            np.testing.assert_array_equal(remaining.predict_recipe(joblib.load(path), valid), prediction)
        # Restoring the known category isolates row-order preservation.
        valid['paired']['event_type'] = arrays['paired'].event_type.iloc[::-1].to_numpy()
        np.testing.assert_array_equal(remaining.predict_recipe(bundle, valid),
                                      remaining.predict_recipe(bundle, arrays)[::-1])

    def test_training_rejects_missing_labels_and_records_equal_event_mass(self):
        arrays = fixture_arrays()
        # Uneven case counts for event 0 ensure row weights are not uniform.
        arrays['cases'][2]['event'] = 0
        labels = {case['case_id']: value * 2 for case, value in zip(arrays['cases'], arrays['current'])}
        config = remaining.recipes()[0] | {'iterations': 5, 'depth': 2}
        with self.assertRaises(ValueError):
            remaining.fit_recipe(config, arrays, {k: v for k, v in list(labels.items())[1:]})
        bundle = remaining.fit_recipe(config, arrays, labels)
        table = pd.DataFrame({'event': [c['event'] for c in arrays['cases']],
                              'weight': bundle['training_weights']})
        np.testing.assert_allclose(table.groupby('event').weight.sum(), [2., 2., 2., 2.])


if __name__ == '__main__':
    unittest.main()
