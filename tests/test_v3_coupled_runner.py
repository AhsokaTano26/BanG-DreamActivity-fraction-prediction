import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from test_rolling_v3 import fixture
from analysis import rolling_dataset,rolling_benchmark,v3_next_runner

class CoupledRunnerTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.v3_coupled_runner'),'new guarded runner is missing')
        return __import__('analysis.v3_coupled_runner',fromlist=['run_experiment'])

    def prepare(self,root):
        parent,curves=fixture(root)
        rolling_dataset.build_rolling(parent,root/'data',curves,source_identity={'fixture':True})
        data=rolling_dataset.RollingDataset(root/'data')
        rolling_benchmark.run_experiment(data,root/'base',iterations=2,allow_fixture=True)
        v3_next_runner.run_experiment(data,root/'cache',base_run=root/'base',warmup_events=2,iterations=2,allow_fixture=True)
        return data

    def test_causal_sealing_controls_choice_and_resume_without_answers(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.prepare(root)
            with patch.object(type(data),'labels',side_effect=AssertionError('Runner must not open answer files')):
                result=m.run_experiment(data,root/'run',source_run=root/'cache',allow_fixture=True)
            cfg=json.loads((root/'run/run_manifest.json').read_text())
            self.assertEqual(len(cfg['methods']),17)
            for event in [800,3,700,4,600,5]:
                a=json.loads((root/f'run/events/{event}/audit.json').read_text())
                self.assertFalse(a['current_outcome_released'])
                self.assertTrue(all(r['end_ms']<a['query_start_ms'] and r['event']!=event for r in a['released_forecasts']))
                if event in [600,5]:self.assertIsNotNone(a['selection_sha256'])
            new=pd.read_csv(root/'run/test_predictions.csv');old=pd.read_csv(root/'cache/test_predictions.csv')
            for ours,theirs in [('uncorrected','uncorrected'),('ewma_reference','ewma_reference'),('kernel_reference','mape_kernel'),('coupled_gap','coupled_gap')]:
                a=new.query('method==@ours').set_index('case_id').predicted.sort_index()
                b=old.query('method==@theirs').set_index('case_id').predicted.sort_index()
                np.testing.assert_allclose(a,b,rtol=1e-12)
            with patch.object(type(data),'reveal_event',side_effect=AssertionError('Completed resume cannot read feedback')):
                self.assertEqual(m.run_experiment(data,root/'run',source_run=root/'cache',allow_fixture=True,resume=True),result)

    def test_current_feedback_poison_does_not_affect_own_forecast_and_cache_guards(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.prepare(root)
            m.run_experiment(data,root/'clean',source_run=root/'cache',allow_fixture=True)
            reveal=data.reveal_event
            def poisoned(event,*,before_ms):
                curves=reveal(event,before_ms=before_ms)
                if event==800:
                    for c in curves:c.scores[-1]*=1.5
                return curves
            with patch.object(data,'reveal_event',side_effect=poisoned):
                m.run_experiment(data,root/'poison',source_run=root/'cache',allow_fixture=True)
            self.assertEqual((root/'clean/events/800/predictions.csv').read_bytes(),(root/'poison/events/800/predictions.csv').read_bytes())
            row=data.event_index[700];cfg=json.loads((root/'cache/run_manifest.json').read_text())
            seal=json.loads((root/'cache/events/700/event_manifest.json').read_text())
            audit=json.loads((root/'cache/events/700/audit.json').read_text())
            peer=pd.read_csv(root/'cache/events/700/peer_audit.csv')
            m.check_cache_event(data,row,cfg,seal,audit,peer)
            import copy
            bad=copy.deepcopy(audit);bad['released_history'][0]['sha256']='bad'
            with self.assertRaises(ValueError):m.check_cache_event(data,row,cfg,seal,bad,peer)
            bad=peer.copy();bad.loc[bad.peer_available==1,'peer_observed_at']=bad.loc[bad.peer_available==1,'observed_at']+1
            with self.assertRaises(ValueError):m.check_cache_event(data,row,cfg,seal,audit,bad)
