"""Real file-boundary tests for the prospective shadow registry; all clocks are fixtures."""
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd


class ProspectiveTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.paper_upgrade_prospective'),
                             'Prospective seal/release implementation is required')
        self.m=importlib.import_module('analysis.paper_upgrade_prospective')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'shadow'
        self.t=1800000000000
        self.clock=patch.object(self.m,'now_ms',return_value=self.t)
        self.clock.start();self.addCleanup(self.clock.stop)
        self.m.register(self.root,{'reference':'a'*64,'paired':'b'*64},primary='reference')
        self.event={'event':900,'start_ms':self.t+1000,'end_ms':self.t+100000}

    def forecasts(self):
        return pd.DataFrame([{'case_id':'future:900:1000:one','event':900,'rank':1000,
            'start_ms':self.event['start_ms'],'end_ms':self.event['end_ms'],
            'cutoff':self.t+10000,'observed_at':self.t+9000,'current':100.,
            'method':m,'predicted':v} for m,v in [('reference',120.),('paired',110.)]])

    def issue(self):
        with patch.object(self.m,'now_ms',return_value=self.t+20000):
            return self.m.issue(self.root,self.event,self.forecasts())

    def test_empty_registry_waits_without_invented_scores(self):
        r=self.m.score(self.root)
        self.assertEqual(r['status'],'waiting_for_eligible_forecasts')
        self.assertNotIn('event_macro_mape_pct',r)

    def test_pre_registration_event_rejected(self):
        e=self.event|{'start_ms':self.t}
        with patch.object(self.m,'now_ms',return_value=self.t+20000),self.assertRaises(ValueError):
            self.m.issue(self.root,e,self.forecasts())

    def test_current_or_future_observation_rejected(self):
        f=self.forecasts();f['observed_at']=self.t+30000
        with patch.object(self.m,'now_ms',return_value=self.t+20000),self.assertRaises(ValueError):
            self.m.issue(self.root,self.event,f)

    def test_outcomes_in_forecast_rejected(self):
        f=self.forecasts();f['actual']=110.
        with patch.object(self.m,'now_ms',return_value=self.t+20000),self.assertRaises(ValueError):
            self.m.issue(self.root,self.event,f)

    def test_unknown_columns_are_not_a_hidden_answer_channel(self):
        f=self.forecasts();f['unrecognized_answer']=110.
        with patch.object(self.m,'now_ms',return_value=self.t+20000),self.assertRaises(ValueError):
            self.m.issue(self.root,self.event,f)

    def test_feedback_enters_future_history_only_after_actual_release(self):
        self.issue()
        with patch.object(self.m,'now_ms',return_value=self.t+110000):
            self.m.release(self.root,900,{1000:110.},captured_ms=self.t+100001,source='fixture')
        with patch.object(self.m,'now_ms',return_value=self.t+130000):
            self.assertTrue(self.m.released_history(self.root,query_start_ms=self.t+105000).empty)
            h=self.m.released_history(self.root,query_start_ms=self.t+120000)
            self.assertEqual(len(h),2)
            self.assertTrue((h.released_ms<self.t+120000).all())

    def test_method_case_support_must_match(self):
        f=self.forecasts();f.loc[1,'case_id']='different'
        with patch.object(self.m,'now_ms',return_value=self.t+20000),self.assertRaises(ValueError):
            self.m.issue(self.root,self.event,f)

    def test_same_case_cannot_be_issued_twice(self):
        self.issue()
        with patch.object(self.m,'now_ms',return_value=self.t+21000),self.assertRaises(ValueError):
            self.m.issue(self.root,self.event,self.forecasts())

    def test_relabeling_same_physical_cutoff_cannot_change_case_weight(self):
        self.issue()
        f=self.forecasts();f['case_id']='new-name-for-identical-observation'
        with patch.object(self.m,'now_ms',return_value=self.t+21000),self.assertRaises(ValueError):
            self.m.issue(self.root,self.event,f)

    def test_same_physical_cutoff_with_two_ids_in_one_batch_rejected(self):
        f=self.forecasts();other=f.copy();other['case_id']='alias'
        with patch.object(self.m,'now_ms',return_value=self.t+20000),self.assertRaises(ValueError):
            self.m.issue(self.root,self.event,pd.concat([f,other]))

    def test_integral_float_rank_survives_csv_and_outcome_roundtrip(self):
        f=self.forecasts();f['rank']=f['rank'].astype(float)
        with patch.object(self.m,'now_ms',return_value=self.t+20000):
            self.m.issue(self.root,self.event,f)
        with patch.object(self.m,'now_ms',return_value=self.t+110000):
            self.m.release(self.root,900,{1000:110.},captured_ms=self.t+100001,source='fixture')
            self.assertEqual(self.m.score(self.root)['cases'],1)

    def test_changed_local_implementation_rejected_after_registration(self):
        original=self.m.digest
        def altered(path):
            return 'f'*64 if Path(path)==Path(self.m.__file__) else original(path)
        with patch.object(self.m,'digest',side_effect=altered),self.assertRaises(ValueError):
            self.m.score(self.root)

    def test_frozen_grid_rejects_off_grid_forecast(self):
        root=Path(self.tmp.name)/'planned'
        self.m.register(root,{'reference':'a'*64,'paired':'b'*64},primary='reference',
                        plan={'stages':[{'axis':'progress','value':.5}],'ranks':[1000],
                              'target_events':20,'maximum_issue_lag_ms':900000})
        with patch.object(self.m,'now_ms',return_value=self.t+20000),self.assertRaises(ValueError):
            self.m.issue(root,self.event,self.forecasts())

    def test_registration_archives_code_and_frozen_stopping_rule(self):
        root=Path(self.tmp.name)/'planned'
        self.m.register(root,{'reference':'a'*64},primary='reference',
                        plan={'stages':[{'axis':'progress','value':.5}],'ranks':[1000],
                              'target_events':20,'maximum_issue_lag_ms':900000})
        self.assertEqual(self.m.config(root)['plan']['target_events'],20)
        self.assertEqual((root/'registry/interface_source.py').read_bytes(),Path(self.m.__file__).read_bytes())

    def planned_registry(self,name):
        root=Path(self.tmp.name)/name
        self.m.register(root,{'reference':'a'*64,'paired':'b'*64},primary='reference',
                        plan={'stages':[{'axis':'progress','value':.1}],'ranks':[1000],
                              'target_events':20,'maximum_issue_lag_ms':900000,'minimum_prefix_points':4})
        f=self.forecasts();f['cutoff']=self.event['start_ms']+round(.1*(self.event['end_ms']-self.event['start_ms']))
        return root,f

    def test_prefix_support_is_finite_and_integral_under_real_grid(self):
        for i,value in enumerate([float('nan'),float('inf'),4.5,-1]):
            with self.subTest(value=value):
                root,f=self.planned_registry('invalid'+str(i));f['prefix_points']=value
                with patch.object(self.m,'now_ms',return_value=self.t+20000),self.assertRaises(ValueError):
                    self.m.issue(root,self.event,f)

    def test_valid_frozen_grid_and_integer_support_are_accepted(self):
        root,f=self.planned_registry('valid');f['prefix_points']=4
        with patch.object(self.m,'now_ms',return_value=self.t+20000):
            path=self.m.issue(root,self.event,f)
        self.assertTrue((path/'seal.json').exists())

    def test_end_boundary_is_strict_and_predictions_remain_blind(self):
        self.issue()
        with patch.object(self.m,'now_ms',return_value=self.event['end_ms']),self.assertRaises(ValueError):
            self.m.release(self.root,900,{1000:110.},captured_ms=self.event['end_ms'],source='fixture')
        self.assertEqual(self.m.score(self.root)['status'],'waiting_for_verified_outcomes')

    def test_seal_then_release_and_independent_equal_activity_score(self):
        self.issue()
        with patch.object(self.m,'now_ms',return_value=self.t+110000):
            self.m.release(self.root,900,{1000:110.},captured_ms=self.t+100001,source='fixture')
            r=self.m.score(self.root)
        self.assertEqual(r['events'],1);self.assertEqual(r['cases'],1)
        self.assertAlmostEqual(r['methods']['reference']['event_macro_mape_pct'],1000/110)
        self.assertEqual(r['methods']['paired']['event_macro_mape_pct'],0.)

    def test_tampered_forecast_is_not_scored(self):
        batch=self.issue()
        f=batch/'forecasts.csv';f.write_text(f.read_text().replace('120.0','999.0'))
        with self.assertRaises(ValueError):
            self.m.score(self.root)


if __name__=='__main__':
    unittest.main()
