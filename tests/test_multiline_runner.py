import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from test_rolling_v3 import fixture
from test_multiline_data_base import expanded_parent
from analysis import rolling_dataset,rolling_benchmark,v3_next_runner,multiline_dataset,multiline_base

class MultilineRunnerTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.multiline_runner'),'multi-line coordinator missing')
        return __import__('analysis.multiline_runner',fromlist=['run'])

    def prepare(self,root):
        parent,curves=fixture(root)
        rolling_dataset.build_rolling(parent,root/'pair',curves,source_identity={'fixture':True});pair=rolling_dataset.RollingDataset(root/'pair')
        rolling_benchmark.run_experiment(pair,root/'oldbase',iterations=2,allow_fixture=True)
        v3_next_runner.run_experiment(pair,root/'context',base_run=root/'oldbase',warmup_events=2,iterations=2,allow_fixture=True)
        parent,curves=expanded_parent(parent,curves,root/'parent19')
        multiline_dataset.build(parent,root/'data',curves,source_identity={'fixture':True});data=multiline_dataset.MultilineDataset(root/'data')
        multiline_base.run(data,root/'raw',context_run=root/'context',pair_dataset=pair,iterations=2,allow_fixture=True)
        return data

    def test_sealed_feedback_exact_pair_control_complete_case_panel_and_resume(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.prepare(root)
            with patch.object(type(data),'labels',side_effect=AssertionError('Coordinator cannot open answers')):
                result=m.run(data,root/'run',base_run=root/'raw',allow_fixture=True)
            cfg=json.loads((root/'run/run_manifest.json').read_text());self.assertEqual(len(cfg['methods']),15)
            for split in ['validation','test']:
                pred=pd.read_csv(root/f'run/{split}_predictions.csv');ids=set(data.case_index(split).case_id)
                for _,p in pred.groupby('method'):self.assertEqual(set(p.case_id),ids)
                old=pd.read_csv(root/f'context/{split}_predictions.csv').query("method=='coupled_gap'")
                current=pred.query("method=='legacy_pair' and rank in [1000,2000]").set_index('case_id').loc[old.case_id]
                np.testing.assert_allclose(current.predicted,old.predicted,rtol=1e-12)
            for event in [800,3,700,4,600,5]:
                a=json.loads((root/f'run/events/{event}/audit.json').read_text())
                self.assertFalse(a['current_outcome_released']);self.assertTrue(all(r['end_ms']<a['query_start_ms'] for r in a['released_forecasts']))
                if event in [600,5]:self.assertIsNotNone(a['selection_sha256'])
            with patch.object(type(data),'reveal_event',side_effect=AssertionError('Completed resume cannot decode outcomes')):
                self.assertEqual(m.run(data,root/'run',base_run=root/'raw',allow_fixture=True,resume=True),result)
            reveal=data.reveal_event
            def poison(event,*,before_ms):
                curves=reveal(event,before_ms=before_ms)
                if event==800:
                    for c in curves:c.scores[-1]*=2
                return curves
            with patch.object(data,'reveal_event',side_effect=poison):m.run(data,root/'poison',base_run=root/'raw',allow_fixture=True)
            self.assertEqual((root/'run/events/800/predictions.csv').read_bytes(),(root/'poison/events/800/predictions.csv').read_bytes())
