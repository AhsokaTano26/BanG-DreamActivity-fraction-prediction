"""Independent hand-calculated fixtures; changing weighting, alpha or units breaks them."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
import numpy as np
import pandas as pd

PATH = Path(__file__).with_name('statistics_audit.py')
spec = importlib.util.spec_from_file_location('e_statistics', PATH)
if PATH.exists():
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
else:
    audit = None

class StatisticsFixtures(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit, 'statistics implementation missing')

    def test_unique_case_then_event_and_rank_equal_are_distinct(self):
        df = pd.DataFrame(dict(case_id=['a','b','c','d'], event=[1,1,1,2],
             rank=[1000,1000,2000,1000], predicted=[100,100,130,120], actual=[100]*4))
        metrics, events = audit.loss_metrics(df)
        self.assertAlmostEqual(metrics['event_macro_mape_pct'], 15.)
        self.assertAlmostEqual(metrics['event_rank_macro_mape_pct'], 17.5)
        self.assertAlmostEqual(metrics['case_mape_pct'], 12.5)
        with self.assertRaises(ValueError): audit.loss_metrics(pd.concat([df,df.iloc[:1]]))

    def test_winkler_penalty_is_positive_both_sides_and_alpha_specific(self):
        score = audit.interval_score(np.array([10,10,10]),np.array([20,20,20]),np.array([15,5,25]),.2)
        np.testing.assert_allclose(score, [10,60,60])
        self.assertEqual(float(audit.interval_score(10,20,5,.1)),110)

    def test_full_group_joint_differs_from_point_and_whole_activity(self):
        df = pd.DataFrame(dict(case_id=list('abcd'),event=[1,1,1,2],rank=[1000,1000,2000,1000],
           phase=[0]*4,actual=[10]*4,lower=[9,11,9,9],upper=[11,12,11,11],own_support=[0]*4,
           support=[10]*4,source=['fallback']*4,available=[True]*4))
        metrics, groups = audit.interval_metrics(df, draws=40)
        self.assertAlmostEqual(metrics['case_coverage'],.75)
        self.assertAlmostEqual(metrics['event_equal_point_coverage'],5/6)
        self.assertAlmostEqual(metrics['event_equal_group_joint_coverage'],.75)
        self.assertAlmostEqual(metrics['whole_event_joint_coverage'],.5)
        self.assertEqual(metrics['n_groups'],3)
        self.assertEqual(metrics['own_support_zero_cases'],4)
        self.assertAlmostEqual(metrics['event_mean_p90_relative_width_pct'],20)
        self.assertAlmostEqual(metrics['event_mean_p90_relative_width_pct_low95'],20)

    def test_varying_support_keeps_full_joint_group_mixed(self):
        df=pd.DataFrame(dict(event=[1,1],rank=[1000]*2,phase=[0]*2,covered=[True,False],
                             own_support_bin=['0','1–9']))
        out=audit.full_joint_groups(df, 'own_support_bin')
        self.assertEqual(out.iloc[0]['condition'],'mixed')
        self.assertFalse(out.iloc[0]['covered'])
        self.assertEqual(out.iloc[0]['cases'],2)

    def test_unbounded_intervals_remain_and_have_infinite_score(self):
        df=pd.DataFrame(dict(case_id=['a'],event=[1],rank=[1000],phase=[0],actual=[10],
            lower=[0],upper=[np.inf],own_support=[0],support=[0],source=['unavailable'],available=[False]))
        metrics,_=audit.interval_metrics(df, draws=40)
        self.assertEqual(metrics['n_cases'],1)
        self.assertEqual(metrics['finite_interval_rate'],0)
        self.assertEqual(metrics['case_coverage'],1)
        self.assertTrue(np.isinf(metrics['winkler_score']))
        self.assertTrue(np.isinf(metrics['p90_width']))
        self.assertTrue(np.isinf(metrics['event_mean_p90_relative_width_pct']))

    def test_bootstrap_pairs_and_block_preserve_constant_differences(self):
        for block in [1,3,5,7]:
            out=audit.mean_bootstrap(np.ones(8)*2,draws=40,block=block)
            self.assertEqual(out['low95'],2)
            self.assertEqual(out['high95'],2)
        self.assertEqual(audit.mean_bootstrap(np.ones(2),draws=40,block=3)['status'],'insufficient_events')

    def test_power_decreases_n_with_larger_effect_and_increases_under_dependence(self):
        n1=audit.power_n(1.,.5,.8,rho=0)
        n2=audit.power_n(1.,1.,.8,rho=0)
        n3=audit.power_n(1.,.5,.8,rho=.6)
        self.assertGreater(n1,n2)
        self.assertGreater(n3,n1)
        self.assertGreaterEqual(audit.paired_t_power(n1,1.,.5),.8)
        self.assertLess(audit.paired_t_power(n1-1,1.,.5),.8)

    def test_relative_csv_path_has_same_hash_provenance_as_absolute_path(self):
        with tempfile.NamedTemporaryFile(dir=PATH.parent,suffix='.csv',mode='w',delete=False) as handle:
            handle.write('value\n7\n');name=Path(handle.name)
        self.addCleanup(name.unlink)
        relative=name.relative_to(audit.ROOT)
        frame=audit.read_csv(relative)
        self.assertEqual(frame.iloc[0]['value'],7)
        self.assertIn(str(relative),audit.INPUTS)

    def test_passed_manifest_rejects_modified_scoring_bundle(self):
        with tempfile.TemporaryDirectory(dir=PATH.parent) as dirname:
            root=Path(dirname);table=root/'scores.csv';table.write_text('mape\n1\n')
            manifest=root/'output_seal.json'
            manifest.write_text(json.dumps({'scores.csv':audit.sha(table)}))
            self.assertEqual(audit.passed_bundle(manifest),root)
            table.write_text('mape\n9\n')
            with self.assertRaises(ValueError):audit.passed_bundle(manifest)

    def test_hac_ordered_ar1_path_with_fixed_innovations(self):
        # x_t=.5*x_(t-1)+epsilon_t; fixed innovations [0,1,1.5,2].
        # Demeaned [-1.5,-.5,.5,1.5], gamma0=1.25,gamma1=.3125,gamma2=-.375.
        result=audit.hac_interval([0,1,2,3])
        self.assertEqual(result['bandwidth'],2)
        self.assertAlmostEqual(result['long_run_variance'],17/12)
        self.assertAlmostEqual(result['se_mean'],np.sqrt(17/48))
        self.assertEqual(result['mean'],1.5)
        self.assertGreater(result['low95'],0)

    def test_hac_constant_path_has_no_significance_decision(self):
        result=audit.hac_interval([2]*8)
        self.assertEqual(result['status'],'degenerate_variance_no_decision')
        self.assertIsNone(result['low95'])
        self.assertIsNone(result['high95'])
        self.assertIsNone(result['p_value'])

    def test_noncentral_t_extreme_tail_stays_finite_and_effect_monotone(self):
        # Exercises a Boost lower-CDF NaN at df19,nc10.6997; same exact distribution.
        self.assertGreater(audit.paired_t_power(20,.10449193168306388,.25),.999999)
        small=audit.power_n(.10449193168306388,.5,.9)
        large=audit.power_n(.10449193168306388,1.,.9)
        self.assertLessEqual(large,small)

if __name__=='__main__': unittest.main()
