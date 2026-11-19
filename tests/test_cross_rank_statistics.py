import unittest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from analysis.cross_rank_statistics import correlation, residualize, resample_indices, common_panel, legal_lag


class CrossRankStatisticsTests(unittest.TestCase):
    def test_rank_ties_and_constant_column(self):
        x = np.array([[1,4,2],[1,3,2],[3,2,2],[4,1,2]], float)
        c = correlation(x)
        self.assertAlmostEqual(c[0,1], spearmanr(x[:,0],x[:,1]).statistic)
        self.assertTrue(np.isnan(c[2]).all())

    def test_control_removes_shared_nuisance(self):
        nuisance = np.tile([-3.,-1.,1.,3.],4)
        a = np.tile([-1.,1.,1.,-1.],4)
        b = np.repeat([-1.,1.,1.,-1.],4)
        y = np.column_stack([100*nuisance+a,100*nuisance+b])
        self.assertGreater(correlation(y,rank=False)[0,1],.99)
        r = residualize(y,np.column_stack([np.ones(16),nuisance]))
        self.assertAlmostEqual(correlation(r,rank=False)[0,1],0.,places=10)

    def test_block_indices_keep_consecutive_activity_rows(self):
        idx = resample_indices(17,20,5,42)
        self.assertEqual(idx.shape,(20,17))
        for start in [0,5,10]:
            np.testing.assert_array_equal(np.diff(idx[:,start:start+5],axis=1)%17,np.ones((20,4)))

    def test_unsupported_rank_stays_missing_common_counts_explicit(self):
        frame=pd.DataFrame([[1,2,np.nan],[2,np.nan,np.nan],[3,4,np.nan]],index=[9,2,7],columns=[10,20,1500])
        values,events,ranks=common_panel(frame,[10,20,1500])
        self.assertEqual(events,[9,7]);self.assertEqual(ranks,[10,20]);self.assertEqual(values.shape,(2,2))
        self.assertTrue(frame[1500].isna().all())

    def test_lag_uses_actual_chronology_and_strict_completed_clock(self):
        matrix=pd.DataFrame([[1,2],[3,4],[5,6]],index=[90,4,8],columns=[10,20])
        clock={90:{'start_ms':10,'end_ms':20},4:{'start_ms':20,'end_ms':25},8:{'start_ms':30,'end_ms':35}}
        left,right,pairs=legal_lag(matrix,clock)
        self.assertEqual(pairs,[(4,8)])
        np.testing.assert_array_equal(left,[[3,4]]);np.testing.assert_array_equal(right,[[5,6]])


if __name__=='__main__':unittest.main()
