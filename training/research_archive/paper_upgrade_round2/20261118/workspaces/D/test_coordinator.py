import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
if (HERE/'run_mechanism.py').exists():
    spec=importlib.util.spec_from_file_location('D_coordinator',HERE/'run_mechanism.py')
    C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
else:C=None


class CoordinatorFixtures(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(C,'Trusted coordinator implementation is missing')

    def test_real_order_uses_time_and_blind_current_detaches_outcomes(self):
        base=1706504400000
        queries=[{'event':312,'start_ms':base+200,'end_ms':base+250,'split':'train'},
                 {'event':293,'start_ms':base+300,'end_ms':base+350,'split':'train'}]
        f=pd.DataFrame([{'case_id':'a','event':293,'rank':1000,'phase':1,'progress':.25,'start_ms':base+300,'end_ms':base+350,'current':1.,'base_prediction':2.,'actual':2.,'log_error':0.,'split':'train'},
                        {'case_id':'b','event':312,'rank':2000,'phase':2,'progress':.9,'start_ms':base+200,'end_ms':base+250,'current':1.,'base_prediction':2.,'actual':2.,'log_error':0.,'split':'train'}])
        got=C.prepare_frame(f,queries)
        self.assertEqual(got['ordered_events'],[312,293])
        blind=C.blind_frame(f)
        self.assertNotIn('actual',blind);self.assertNotIn('log_error',blind)
        blind.loc[0,'base_prediction']=3.
        self.assertEqual(f.loc[0,'base_prediction'],2.)

    def test_wrong_stage_or_nontrain_input_is_rejected(self):
        base=1706504400000
        q=[{'event':1,'start_ms':base,'end_ms':base+10,'split':'train'}]
        f=pd.DataFrame([{'case_id':'a','event':1,'rank':1000,'phase':0,'progress':.25,'start_ms':base,'end_ms':base+10,'current':1.,'base_prediction':2.,'actual':2.,'log_error':0.,'split':'train'}])
        with self.assertRaisesRegex(ValueError,'phase'):C.prepare_frame(f,q)
        f['phase']=1;f['split']='test'
        with self.assertRaisesRegex(ValueError,'train'):C.prepare_frame(f,q)

    def test_pair_panel_keeps_time_gaps_and_complete_pair_counts(self):
        means=pd.DataFrame({'event':[312,312,293,294,294],'rank':[1000,2000,1000,1000,2000],
                            'phase':[1]*5,'z':[.1,.2,.4,.3,.5]})
        y,order,events=C.pair_panel(means,[312,293,294],1)
        self.assertEqual(events,[312,294]);np.testing.assert_array_equal(order,[0,2])
        np.testing.assert_allclose(y,[[.1,.2],[.3,.5]])

    def test_moment_diagnostics_exclude_nonadjacent_difference(self):
        y=np.array([[0.,0.],[1.,2.],[10.,20.],[11.,22.]])
        got=C.paired_statistics(y,np.array([0,1,3,4]))
        self.assertEqual(got['complete_pairs'],4)
        self.assertEqual(got['adjacent_differences'],2)
        self.assertEqual(got['difference_lag_pairs'],0)
        self.assertIsNone(got['Q_raw_00'])

    def test_bootstrap_linear_levels_never_create_delta_spikes_at_block_seams(self):
        y=np.c_[np.arange(10.),2*np.arange(10.)]
        got=C.paired_bootstrap(y,np.arange(10),np.random.default_rng(12))
        self.assertAlmostEqual(got['Gamma_delta0_00_block1_high'],0.)
        self.assertAlmostEqual(got['Gamma_delta0_11_block3_high'],0.)
        self.assertAlmostEqual(got['Q_raw_00_block5_low'],0.)

    def test_resampling_attached_contributions_keeps_original_gap_missing(self):
        y=np.array([[0.,0.],[1.,2.],[10.,20.],[11.,22.]])
        attached=C.pair_contributions(y,np.array([0,1,3,4]))
        got=C.contribution_statistics(attached,np.array([0,2,2,3]))
        self.assertEqual(got['adjacent_differences'],1)
        self.assertEqual(got['difference_lag_pairs'],0)
        self.assertIsNone(got['Q_raw_00'])


if __name__=='__main__':unittest.main(verbosity=2)
