import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from test_rolling_v3 import fixture
from analysis import rolling_dataset, rolling_benchmark

class ResidualCoordinatorTests(unittest.TestCase):
    def make(self, root):
        parent,curves=fixture(root)
        rolling_dataset.build_rolling(parent,root/'data',curves,source_identity={'fixture':True})
        data=rolling_dataset.RollingDataset(root/'data')
        rolling_benchmark.run_experiment(data,root/'base',iterations=2,allow_fixture=True)
        return data

    def test_seed_ensemble_averages_bounded_scores(self):
        m=importlib.import_module('analysis.rolling_residual_benchmark')
        result=m.final_predictions(np.array([100.]),np.array([200.]),{'residual_mlp_seed42':np.array([-.3]),'residual_mlp_seed123':np.array([0.]),'residual_mlp_seed2026':np.array([.3])})
        expected=np.mean([200*np.exp(.3),200,200*np.exp(-.3)])
        self.assertAlmostEqual(result['residual_mlp'][0],expected)
        self.assertNotAlmostEqual(result['residual_mlp'][0],200.)

    def test_prefix_cache_must_cover_current_cases_exactly(self):
        m=importlib.import_module('analysis.rolling_residual_benchmark')
        with self.assertRaisesRegex(ValueError,'case'):
            m.aligned_base(pd.DataFrame({'case_id':['a','a'],'predicted':[1.,2.]}),['a'])

    def test_event_seals_causal_feedback_baseline_resume_and_independent_score(self):
        m=importlib.import_module('analysis.rolling_residual_benchmark')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.make(root)
            with patch.object(type(data),'labels',side_effect=AssertionError('Coordinator cannot open answer files')):
                selected=m.run_experiment(data,root/'new',base_run=root/'base',allow_fixture=True,methods=['uncorrected','ewma_reference','ewma_fast'])
            expected=pd.read_csv(root/'base/test_predictions.csv')
            expected=expected[expected.method=='rolling_context_ewma'].set_index('case_id').predicted
            actual=pd.read_csv(root/'new/test_predictions.csv')
            actual=actual[actual.method=='ewma_reference'].set_index('case_id').predicted
            np.testing.assert_allclose(actual.sort_index(),expected.sort_index(),rtol=1e-12)
            for event in [700,4,600,5]:
                import json
                audit=json.loads((root/f'new/events/{event}/audit.json').read_text())
                self.assertTrue(all(r['end_ms']<audit['query_start_ms'] for r in audit['released_forecasts']))
                self.assertNotIn(event,[r['event'] for r in audit['released_forecasts']])
            with patch.object(type(data),'reveal_event',side_effect=AssertionError('Resume cannot read outcomes')):
                self.assertEqual(m.run_experiment(data,root/'new',base_run=root/'base',allow_fixture=True,methods=['uncorrected','ewma_reference','ewma_fast'],resume=True),selected)
            evaluator=importlib.import_module('analysis.rolling_residual_evaluate')
            result=evaluator.evaluate(root/'new')
            self.assertEqual(result['splits']['test']['ewma_reference']['overall']['cases'],len(actual))
            with patch.object(type(data),'labels',side_effect=AssertionError('Cached scorer cannot open outcomes')):
                self.assertEqual(evaluator.evaluate(root/'new'),result)

    def test_current_future_poisoning_cannot_change_unreleased_forecasts(self):
        m=importlib.import_module('analysis.rolling_residual_benchmark')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.make(root)
            methods=['uncorrected','ewma_reference','ewma_fast']
            m.run_experiment(data,root/'clean',base_run=root/'base',allow_fixture=True,methods=methods)
            original=data.reveal_event
            def poison(event,*,before_ms):
                values=original(event,before_ms=before_ms)
                if event==600:
                    for c in values:c.scores[-1]*=100
                return values
            with patch.object(data,'reveal_event',side_effect=poison):
                # Once poisoned event 600 really completes, its feedback is
                # allowed to change event 5. The fixed-cache replay guard then
                # correctly detects the changed reference, preserving its run.
                with self.assertRaisesRegex(ValueError,'EWMA replay'):
                    m.run_experiment(data,root/'poison',base_run=root/'base',allow_fixture=True,methods=methods)
            for event in [700,4,600]:
                self.assertEqual((root/f'clean/events/{event}/predictions.csv').read_bytes(),(root/f'poison/events/{event}/predictions.csv').read_bytes())
