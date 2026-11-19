"""Temporal shift research keeps the blind-prefix and equal-event contracts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from analysis import research_shift as shift


def fixture(events=12):
    """Actual dates deliberately disagree with numeric event order."""
    rows, cases, audit, current, progress, rank, labels = [], [], [], [], [], [], {}
    start0 = 1_600_000_000_000
    for k in range(events):
        event = 900-k
        start = start0+k*240*3_600_000
        for r in (1000, 2000):
            for p in (.12, .55, .97):
                score = 1000*(1+p)
                now = start+p*192*3_600_000
                case = f'train:{event}:{r}:{p}'
                rows.append({'event_type': 'normal' if k%2 else 'challenge',
                    'log_current': np.log1p(score), 'log_rank': np.log(r),
                    'elapsed_h': p*192, 'duration_h': 192., 'remaining_h': (1-p)*192,
                    'progress': p, 'prefix_00': None, 'prefix_16': .4, 'prefix_32': 1.,
                    'rate_3h_relative': 1+.2*(r == 1000)})
                cases.append({'case_id': case, 'event': event, 'rank': r})
                audit.append({'case_id': case, 'event': event, 'rank': r,
                    'cutoff': now+60000, 'observed_at': now})
                current.append(score); progress.append(p); rank.append(int(r == 2000))
                labels[case] = score/p*np.exp((.18 if r == 1000 else -.12)*(1-p))
    frame = pd.DataFrame(rows)
    return {'single': frame, 'paired': frame.assign(peer_available=0),
        'cases': cases, 'audit': pd.DataFrame(audit), 'current': np.array(current),
        'progress': np.array(progress), 'rank': np.array(rank)}, labels


class ShiftResearchTests(unittest.TestCase):
    def test_rank_validation_catches_inconsistent_blind_case(self):
        arrays, labels = fixture(4)
        arrays['rank'][0] = 1
        config = shift.recipes()[0] | {'iterations': 5}
        with self.assertRaisesRegex(ValueError, '[Rr]ank'):
            shift.fit_recipe(config, arrays, labels)

    def test_chronological_folds_exclude_unfinished_activity(self):
        arrays, _ = fixture(12)
        # First event runs into the evaluation era and must never be in a fit.
        arrays['single'].loc[:5, 'duration_h'] = 3000.
        folds = shift.chronological_folds(arrays, folds=3, warm_fraction=.4)
        self.assertEqual(len(folds), 3)
        for training, heldout, audit in folds:
            self.assertTrue(set(training).isdisjoint(heldout))
            self.assertLess(audit['train_latest_end_ms'], audit['validation_earliest_start_ms'])
            self.assertNotIn(900, audit['train_events'])
            self.assertTrue(set(audit['train_events']).isdisjoint(audit['validation_events']))

    def test_all_cases_preserved_with_equal_event_weight_and_persistence(self):
        arrays, labels = fixture(9)
        # Unequal prefix counts and a missing rank must not change event totals.
        indices = np.array([i for i, c in enumerate(arrays['cases'])
                            if not (c['event'] == 899 and c['rank'] == 2000)
                            and not (c['event'] == 897 and i%3 == 1)])
        arrays = shift.subset_arrays(arrays, indices)
        config = next(c for c in shift.recipes() if c['name'] == 'shift_calendar_rank')
        bundle = shift.fit_recipe(config | {'iterations': 20}, arrays, labels)
        weights = pd.DataFrame(bundle['training_weights'])
        totals = weights.groupby('event').weight.sum().to_numpy()
        np.testing.assert_allclose(totals, np.repeat(totals[0], len(totals)), atol=1e-12)
        pred = shift.predict_recipe(bundle, arrays)
        self.assertEqual(pred.shape, (len(indices),))
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((pred >= arrays['current']).all())
        with tempfile.TemporaryDirectory() as work:
            path = Path(work)/'bundle.joblib'
            joblib.dump(bundle, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(shift.predict_recipe(restored, arrays), pred)
        reverse = np.arange(len(indices))[::-1]
        np.testing.assert_allclose(shift.predict_recipe(bundle, shift.subset_arrays(arrays, reverse)), pred[::-1])

    def test_oof_calibration_uses_prior_events_and_preserves_rank_cells(self):
        arrays, labels = fixture(14)
        config = next(c for c in shift.recipes() if c['name'] == 'shift_rank_calibrated')
        bundle = shift.fit_recipe(config | {'iterations': 12, 'oof_folds': 2}, arrays, labels)
        self.assertEqual(len(bundle['oof_audit']), 2)
        for audit in bundle['oof_audit']:
            self.assertLess(audit['train_latest_end_ms'], audit['validation_earliest_start_ms'])
            self.assertTrue(set(audit['train_events']).isdisjoint(audit['validation_events']))
        self.assertEqual(set(bundle['calibration']['ranks']), {0, 1})
        pred = shift.predict_recipe(bundle, arrays)
        before = deepcopy(bundle['calibration'])
        # Unseen future cases supply only prefixes; prediction cannot learn from them.
        future = deepcopy(arrays)
        future['audit']['observed_at'] += 8*365*86_400_000
        future['audit']['cutoff'] += 8*365*86_400_000
        self.assertEqual(shift.predict_recipe(bundle, future).shape, pred.shape)
        self.assertEqual(bundle['calibration'], before)

    def test_mape_calibration_factor_resists_large_overprediction(self):
        # Event-weighted MAPE chooses .5: losses are .5 versus 1 for factors .5/1.
        factor = shift.mape_factor(np.array([100., 100.]), np.array([100., 200.]), np.ones(2))
        self.assertEqual(factor, .5)


if __name__ == '__main__':
    unittest.main()
