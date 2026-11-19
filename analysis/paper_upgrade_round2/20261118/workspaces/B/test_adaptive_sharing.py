import copy
import importlib
import math
import unittest

import pandas as pd

try:
    model = importlib.import_module('adaptive_sharing')
except ModuleNotFoundError:
    model = None

START = 1706504400000


def panel(event, start, errors=None, ranks=(1000, 2000), progress=.5):
    rows = []
    for rank in ranks:
        row = dict(case_id=f'{event}:{rank}', event=event, rank=rank,
                   phase=1, progress=progress, start_ms=start,
                   end_ms=start+10, cutoff=start+5, observed_at=start+5,
                   current=50., base_prediction=100., split='train')
        if errors is not None:
            row['log_error'] = errors[rank]
            row.update(issued_ind=0., issued_pair=0., issued_adjacent=0.,
                       issued_samepanel=0.)
        rows.append(row)
    return pd.DataFrame(rows)


class PureSharingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(model, 'pure adaptive_sharing module is missing')

    def test_quadratic_oracle_ridge_shrinks_before_clipping(self):
        # C/A=2; k=n=5 => regularized coefficient exactly 1, not .5.
        result = model.risk_lambda([(1., 2.)]*5, k=5., decay=1.)
        self.assertEqual(result['lambda'], 1.)
        self.assertEqual(result['support'], 5)
        self.assertEqual(result['n_eff'], 5.)

    def test_small_support_and_identical_experts_fall_back(self):
        for samples in [[(1., 1.)]*4, [(0., 0.)]*10]:
            self.assertEqual(model.risk_lambda(samples, k=4., decay=.85)['lambda'], 0.)

    def test_risk_uses_events_not_prefix_count(self):
        history = panel(1, START, {1000:.2, 2000:0.})
        history.loc[history['rank']==1000, 'issued_pair'] = .1
        extra = history[history['rank']==1000].copy()
        extra['case_id'] = 'extra'
        duplicated = pd.concat([history, extra], ignore_index=True)
        a = model.risk_observations(history, rank=1000, phase=1, pool='pair')
        b = model.risk_observations(duplicated, rank=1000, phase=1, pool='pair')
        self.assertEqual(a, b)
        self.assertAlmostEqual(a[0][0], .01)
        self.assertAlmostEqual(a[0][1], .02)

    def test_experts_pool_rank_equally_and_preserve_filter(self):
        history = panel(1, START, {1000:.2, 2000:0.})
        current = panel(2, START+20)
        result = model.predict(history, current,
                               {'kind':'fixed','lambda':.5,'pool':'pair'})
        self.assertAlmostEqual(result.iloc[0].expert_ind, .2/11)
        self.assertAlmostEqual(result.iloc[0].expert_shared, .1/11)
        self.assertAlmostEqual(result.iloc[0].bias, .15/11)
        self.assertAlmostEqual(result.iloc[0].prediction, 100*math.exp(-.15/11))
        self.assertAlmostEqual(result.iloc[1].bias, .05/11)

    def test_equal_end_boundary_is_rejected(self):
        history = panel(1, START, {1000:.2, 2000:0.})
        with self.assertRaisesRegex(ValueError, 'completed'):
            model.predict(history, panel(2, START+10),
                          {'kind':'fixed','lambda':0.,'pool':'pair'})

    def test_current_outcomes_unknown_fields_and_old_period_rejected(self):
        history = panel(1, START, {1000:.2, 2000:0.})
        for field in ['actual', 'log_error', 'expert__independent', 'mystery']:
            current = panel(2, START+20)
            current[field] = 1.
            with self.subTest(field=field), self.assertRaises(ValueError):
                model.predict(history, current, {'kind':'fixed','lambda':0.,'pool':'pair'})
        history['start_ms'] = START-1
        with self.assertRaises(ValueError):
            model.predict(history, panel(2, START+20), {'kind':'fixed','lambda':0.,'pool':'pair'})

    def test_future_data_not_needed_and_inputs_unmodified(self):
        history = panel(1, START, {1000:.2, 2000:0.})
        original = history.copy(deep=True)
        current = panel(2, START+20)
        one = model.predict(history, current, {'kind':'fixed','lambda':.5,'pool':'pair'})
        # Changing an unpassed future event cannot affect the current call.
        future = panel(3, START+40, {1000:99.,2000:-99.})
        future.loc[:,'log_error'] *= -1000
        two = model.predict(history, current, {'kind':'fixed','lambda':.5,'pool':'pair'})
        pd.testing.assert_frame_equal(one, two)
        pd.testing.assert_frame_equal(history, original)

    def test_only_one_rank_and_singleton_remain_independent(self):
        frames = [panel(j+1, START+j*20, {1000:.2,2000:0.}) for j in range(6)]
        history = pd.concat(frames, ignore_index=True)
        history['issued_ind'] = 0.
        history['issued_pair'] = .1
        current = panel(10, START+200)
        result = model.predict(history, current,
                               {'kind':'risk','k':8.,'decay':.85,'pool':'pair','apply_ranks':[1000]})
        self.assertGreater(result.iloc[0]['lambda'], 0.)
        self.assertEqual(result.iloc[1]['lambda'], 0.)
        single = model.predict(history, panel(10, START+200, ranks=(30000,)),
                               {'kind':'risk','k':8.,'decay':.85,'pool':'adjacent'})
        self.assertEqual(single.iloc[0]['lambda'], 0.)

    def test_constant_and_differential_drift_gate_closes(self):
        rows = []
        for j in range(10):
            rows.append(panel(j+1, START+j*20, {1000:.2,2000:.2}))
        gate = model.gate_state(pd.concat(rows, ignore_index=True), phase=1)
        self.assertFalse(gate['gate_open'])
        self.assertEqual(gate['rho'], 0.)


if __name__ == '__main__':
    unittest.main()
