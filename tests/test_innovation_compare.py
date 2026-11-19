import importlib.util
import unittest
import numpy as np
import pandas as pd


class InnovationCompareTests(unittest.TestCase):
    def test_bootstrap_groups_use_frozen_cases_not_prediction_event(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.innovation_compare'))
        from analysis.innovation_compare import compare_frames
        cases=pd.DataFrame({'case_id':['a','b','c'],'event':[1,1,2],'rank':[1000]*3})
        labels={'a':100.,'b':100.,'c':100.}
        new=pd.DataFrame({'case_id':['c','b','a'],'predicted':[100.,110.,110.],'event':[99]*3})
        old=pd.DataFrame({'case_id':['a','b','c'],'predicted':[100.,100.,120.],'event':[99]*3})
        result=compare_frames(cases,labels,new,old)
        self.assertEqual(result['events'],2)
        self.assertAlmostEqual(result['delta_mape_points'],-5.)
        self.assertEqual(result['improved_events'],1)
        with self.assertRaises(ValueError):compare_frames(cases,labels,new.iloc[:2],old)


if __name__=='__main__':unittest.main()
