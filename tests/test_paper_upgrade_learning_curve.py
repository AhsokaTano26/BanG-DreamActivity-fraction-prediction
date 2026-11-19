"""Training size and decoder contracts, with no model fitting."""
import importlib
import importlib.util
import os
import subprocess
import sys
import unittest
import numpy as np
import pandas as pd

MODULE='analysis.paper_upgrade_learning_curve'
MINIMUM=1706504400000

class LearningCurve(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE),'separate true fitting learning curve must exist')
        self.m=importlib.import_module(MODULE)

    def test_nested_training_subsets_use_actual_dates_not_event_ids(self):
        cases=pd.DataFrame({'event':[100,1,50],'start_ms':[MINIMUM,MINIMUM+300,MINIMUM+600],
                            'end_ms':[MINIMUM+100,MINIMUM+400,MINIMUM+700]})
        got=self.m.training_subsets(cases,MINIMUM+1000,sizes=[1,2,3])
        self.assertEqual(got,{1:[100],2:[100,1],3:[100,1,50]})

    def test_future_or_predense_training_events_rejected(self):
        cases=pd.DataFrame({'event':[1],'start_ms':[MINIMUM],'end_ms':[MINIMUM+1000]})
        with self.assertRaises(ValueError):self.m.training_subsets(cases,MINIMUM+1000,sizes=[1])
        cases.loc[0,'start_ms']=MINIMUM-1
        with self.assertRaises(ValueError):self.m.training_subsets(cases,MINIMUM+2000,sizes=[1])

    def test_growth_decoder_and_label_target_are_inverse_before_bounds(self):
        current=np.array([100.,200.]);progress=np.array([.5,.25]);actual=np.array([300.,800.])
        target=self.m.target_values(current,progress,actual)
        np.testing.assert_allclose(self.m.decode(target,current,progress),actual)
        np.testing.assert_allclose(self.m.decode(np.array([-100,100]),current,progress),[100.,200*np.exp(4)])

    def test_event_equal_weights_do_not_weight_dense_prefixes_as_events(self):
        weights=self.m.event_weights(np.array([1,1,2]))
        self.assertAlmostEqual(weights[:2].sum(),weights[2])

    def test_independent_network_does_not_transfer_other_rank_labels(self):
        # PyTorch integration remains isolated in a subprocess. Changing only
        # rank2000 supervision must leave rank1000's parameters/predictions fixed.
        result=subprocess.run([sys.executable,'-c',"""
import numpy as np
from analysis.paper_upgrade_learning_curve import fit_independent_networks
x=np.array([[1.,0.],[2.,0.],[1.,1.],[2.,1.]])
y=np.array([.1,.2,.3,.4]);events=np.array([1,2,1,2]);rank=np.array([0,0,1,1])
first,_=fit_independent_networks(x,y,events,rank,x,rank,seed=42)
y[2:]+=2.
second,_=fit_independent_networks(x,y,events,rank,x,rank,seed=42)
assert np.array_equal(first[:2],second[:2]),'other-rank labels changed independent predictions'
assert np.max(np.abs(first[2:]-second[2:]))>.01,'own rank did not learn changed labels'
"""],capture_output=True,text=True,env=dict(os.environ))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()
