import importlib.util
import unittest

import numpy as np


class InnovationCommonTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.innovation_common'))
        from analysis import innovation_common
        return innovation_common

    def test_chronological_folds_exclude_unfinished_and_preserve_events(self):
        m = self.module()
        rows = [{'event': i, 'start_ms': 100*i, 'end_ms': 100*i+90} for i in range(10)]
        rows[2]['end_ms'] = 650
        folds = m.chronological_folds(rows)
        self.assertEqual([e for f in folds for e in f['heldout_events']], list(range(5, 10)))
        self.assertNotIn(2, folds[0]['training_events'])
        self.assertTrue(all(max(rows[e]['end_ms'] for e in f['training_events']) <
                            min(rows[e]['start_ms'] for e in f['heldout_events']) for f in folds))

    def test_metric_is_activity_equal_and_prediction_shape_checked(self):
        m = self.module()
        a = {'cases': [{'case_id': str(i), 'event': e} for i,e in enumerate([1,1,1,2])]}
        labels = {str(i): 100. for i in range(4)}
        self.assertAlmostEqual(m.event_mape(a, labels, np.array([100,100,100,200.])), 50.)
        with self.assertRaises(ValueError):
            m.event_mape(a, labels, np.array([100.]))

    def test_subset_replaces_history_and_preserves_case_order(self):
        m = self.module()
        import pandas as pd
        a = {'cases':[{'case_id':'a'},{'case_id':'b'}], 'samples':['a','b'],
             'base':pd.DataFrame({'x':[10,20]}), 'current':np.array([10,20]),
             'band_shares':[{'1':1.},{'2':1.}], 'history':['not_allowed']}
        b = m.subset_arrays(a, [1,0], history=['allowed'])
        self.assertEqual([x['case_id'] for x in b['cases']], ['b','a'])
        self.assertEqual(b['history'], ['allowed'])
        self.assertEqual(b['base'].x.tolist(), [20,10])

    def test_context_is_separate_from_own_curve_ablation(self):
        m=self.module()
        from analysis.backtest import Curve, snapshot
        from types import SimpleNamespace
        hour=3600000
        c=Curve(1,1000,'story',0,10*hour,np.arange(4)*hour,np.array([1.,10.,20.,30.]))
        s=snapshot(c,3*hour);s['case_id']='train:1:1000:3h'
        a=m.prepare([s],SimpleNamespace(parties={1:{'bands':{2:1.}}}))
        self.assertNotIn('event_type',a['base'].columns)
        self.assertEqual(a['event_type'].tolist(),['story'])
        self.assertEqual(a['band_shares'],[{'2':1.}])


if __name__ == '__main__':
    unittest.main()
