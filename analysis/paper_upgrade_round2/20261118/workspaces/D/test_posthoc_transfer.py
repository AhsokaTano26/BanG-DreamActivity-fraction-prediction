import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
if (HERE/'posthoc_transfer.py').exists():
    spec=importlib.util.spec_from_file_location('D_transfer',HERE/'posthoc_transfer.py')
    T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
else:T=None


class TransferFixtures(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(T,'Posthoc transfer implementation missing')

    def frames(self):
        past=pd.DataFrame({'event':[1,1],'rank':[1000,2000],'phase':[1,1],
                           'z':[1.,3.],'start_ms':[100,100],'end_ms':[200,200],'ordinal':[0,0]})
        blind=pd.DataFrame({'case_id':['a','b'],'event':[2,2],'rank':[1000,2000],
                            'phase':[1,1],'start_ms':[300,300],'end_ms':[400,400],
                            'current':[1.,1.],'base_prediction':[2.,2.]})
        return past,blind

    def test_fixed_covariance_predicts_without_current_outcome(self):
        past,blind=self.frames()
        got=T.issue_fixed(past,blind,np.zeros((2,2)),np.eye(2),np.zeros(2))
        np.testing.assert_allclose(got.bias,[1/11,3/11])
        np.testing.assert_allclose(got.prediction,2*np.exp(-np.array([1/11,3/11])))
        self.assertTrue(got.history_complete_pairs.eq(1).all())

    def test_current_label_and_equal_end_clock_are_rejected(self):
        past,blind=self.frames();blind['actual']=2.
        with self.assertRaisesRegex(ValueError,'blind'):T.issue_fixed(past,blind,np.eye(2),np.eye(2),np.zeros(2))
        blind=blind.drop(columns='actual');past['end_ms']=300
        with self.assertRaisesRegex(ValueError,'strict'):T.issue_fixed(past,blind,np.eye(2),np.eye(2),np.zeros(2))


if __name__=='__main__':unittest.main(verbosity=2)
