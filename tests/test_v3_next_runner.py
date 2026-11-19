import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
from test_rolling_v3 import fixture
from analysis import rolling_dataset,rolling_benchmark

class WarmResidualRunnerTests(unittest.TestCase):
    def prepare(self,root):
        parent,curves=fixture(root)
        rolling_dataset.build_rolling(parent,root/'data',curves,source_identity={'fixture':True})
        data=rolling_dataset.RollingDataset(root/'data')
        rolling_benchmark.run_experiment(data,root/'base',iterations=2,allow_fixture=True)
        return data

    def test_warm_forecasts_sealed_before_release_and_cold_baseline_exact(self):
        m=importlib.import_module('analysis.v3_next_runner')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.prepare(root)
            with patch.object(type(data),'labels',side_effect=AssertionError('Coordinator cannot open answer files')):
                result=m.run_experiment(data,root/'new',base_run=root/'base',allow_fixture=True,warmup_events=2,iterations=2)
            config=json.loads((root/'new/run_manifest.json').read_text())
            self.assertEqual(config['warm_forecast_events'],[800,3])
            for event in [800,3,700,4,600,5]:
                a=json.loads((root/f'new/events/{event}/audit.json').read_text())
                self.assertTrue(all(r['end_ms']<a['query_start_ms'] and r['event']!=event for r in a['released_forecasts']))
            a=json.loads((root/'new/events/700/audit.json').read_text())
            self.assertEqual(len(a['released_forecasts']),2)
            expected=pd.read_csv(root/'base/test_predictions.csv').query("method=='rolling_context_ewma'").set_index('case_id').predicted.sort_index()
            actual=pd.read_csv(root/'new/test_predictions.csv').query("method=='ewma_reference'").set_index('case_id').predicted.sort_index()
            np.testing.assert_allclose(actual,expected,rtol=1e-12)
            with patch.object(type(data),'reveal_event',side_effect=AssertionError('Resume must not read outcomes')):
                self.assertEqual(m.run_experiment(data,root/'new',base_run=root/'base',allow_fixture=True,warmup_events=2,iterations=2,resume=True),result)

    def test_current_warm_final_never_changes_its_own_forecast(self):
        m=importlib.import_module('analysis.v3_next_runner')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.prepare(root)
            m.run_experiment(data,root/'clean',base_run=root/'base',allow_fixture=True,warmup_events=2,iterations=2)
            original=data.reveal_event
            def poison(event,*,before_ms):
                curves=original(event,before_ms=before_ms)
                if event==800:
                    for c in curves:c.scores[-1]*=2
                return curves
            with patch.object(data,'reveal_event',side_effect=poison):
                m.run_experiment(data,root/'poison',base_run=root/'base',allow_fixture=True,warmup_events=2,iterations=2)
            self.assertEqual((root/'clean/events/800/predictions.csv').read_bytes(),(root/'poison/events/800/predictions.csv').read_bytes())

    def test_cached_history_hash_and_self_outcome_are_verified(self):
        m=importlib.import_module('analysis.v3_next_runner')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.prepare(root);row=data.event_index[700]
            seal=rolling_benchmark.verify_seal(root/'base/events/700/event_manifest.json')
            audit=json.loads((root/'base/events/700/audit.json').read_text())
            m.check_cached_event(data,row,seal,audit,seal['run_id'])
            import copy
            poison=copy.deepcopy(audit);poison['released_history'][0]['sha256']='bad'
            with self.assertRaisesRegex(ValueError,'hash'):m.check_cached_event(data,row,seal,poison,seal['run_id'])
            poison=copy.deepcopy(audit);poison['current_outcome_released']=True
            with self.assertRaisesRegex(ValueError,'identity/clock'):m.check_cached_event(data,row,seal,poison,seal['run_id'])
