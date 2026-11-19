import importlib.util,unittest
import pandas as pd

class MultilineEvaluationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.multiline_evaluate'),'independent multi-line scoring missing')
        return __import__('analysis.multiline_evaluate',fromlist=['panel_metrics'])

    def fixture(self):
        cases=pd.DataFrame({'case_id':['a','b','c','d'],'event':[901,901,2,2],'rank':[1000,1000,2000,1500],'observation_age_minutes':[0.,0.,0.,0.]})
        labels={c:100. for c in cases.case_id}
        predictions=pd.DataFrame({'case_id':cases.case_id,'predicted':[120.,120.,100.,110.],'event':[999]*4,'rank':[10]*4})
        return cases,labels,predictions

    def test_panels_use_frozen_activity_membership_and_keep_cold_rank(self):
        m=self.module();c,l,p=self.fixture()
        self.assertAlmostEqual(m.panel_metrics(c,l,p)['event_macro_mape_pct'],12.5)
        self.assertAlmostEqual(m.panel_metrics(c,l,p,ranks=[1000,2000])['event_macro_mape_pct'],10.)
        cold=m.panel_metrics(c,l,p,ranks=[1500]);self.assertAlmostEqual(cold['event_macro_mape_pct'],10.)
        self.assertEqual(cold['events'],1);self.assertEqual(cold['cases'],1)
        empty=m.panel_metrics(c,l,p,ranks=[5000]);self.assertEqual(empty['cases'],0)

    def test_extra_missing_and_duplicate_cases_rejected_even_for_subset_panel(self):
        m=self.module();c,l,p=self.fixture()
        for bad in [p.iloc[:-1],pd.concat([p,p.iloc[[0]]]),pd.concat([p,p.iloc[[0]].assign(case_id='extra')])]:
            with self.assertRaises(ValueError):m.panel_metrics(c,l,bad,ranks=[1500])
