import importlib.util
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd


class InnovationRunnerTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.innovation_runner'))
        from analysis import innovation_runner
        return innovation_runner

    def test_selection_excludes_seed_winners_and_combines_final_scores(self):
        m = self.module()
        arrays = {'cases':[{'case_id':'a','event':1},{'case_id':'b','event':2}],
                  'current':np.array([10.,10.])}
        p = {'r':np.array([90.,110.]),'h':np.array([110.,90.]),
             'g':np.array([120.,120.]),'g__seed_42':np.array([100.,100.])}
        specs = {'r':{'family':'retrieval'},'h':{'family':'hierarchy'},'g':{'family':'growth'}}
        result = m.select_combinations(arrays, {'a':100.,'b':100.}, p, specs, control='r')
        self.assertNotIn('g__seed_42', result['candidates'])
        self.assertAlmostEqual(result['validation_scores']['family_mean'], 20/3)
        self.assertEqual(result['combination_specs']['family_mean']['members'], ['r','h','g'])

    def test_formal_run_rejects_wrong_dataset_before_creating_output(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        m=self.module()
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/'invalid'
            data=SimpleNamespace(manifest={'dataset_id':'wrong'})
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                m.run_experiment(data,output,protocol='v1',families=['retrieval'])
            self.assertFalse(output.exists())

    def test_run_is_blind_and_matches_temporal_scorer(self):
        m = self.module()
        import test_temporal_training as fixture
        from analysis.temporal_rank_subset import build_subset
        from analysis.temporal_dataset import TemporalDataset
        from analysis.temporal_evaluate import evaluate, verify_predictions
        helper=fixture.TemporalTrainingTests();self.addCleanup(helper.doCleanups)
        root,parent=helper.fixture();build_subset(parent,root/'subset')
        data=TemporalDataset(root/'subset');run=root/'innovation'
        old_labels,old_inputs=TemporalDataset.labels,TemporalDataset.inputs
        def labels(instance,split,**kw):
            if split=='test':raise AssertionError('No test labels in training')
            self.assertTrue((run/'run_manifest.json').exists())
            return old_labels(instance,split,**kw)
        def inputs(instance,split):
            if split=='test':self.assertTrue((run/'selection.json').exists())
            return old_inputs(instance,split)
        with patch.object(TemporalDataset,'labels',labels),patch.object(TemporalDataset,'inputs',inputs),\
             patch('sqlite3.connect',side_effect=AssertionError('No SQLite')):
            selected=m.run_experiment(data,run,protocol='v2',families=['retrieval'],
                limit_per_family=1,iterations_override=3,run_cv=False,allow_fixture=True)
        seal=verify_predictions(run)
        self.assertEqual(seal['reference_method'],selected['selected_recipe'])
        scored=evaluate(run)
        for name,score in scored['splits']['validation'].items():
            if name in selected['validation_scores']:
                self.assertAlmostEqual(score['overall']['event_macro_mape_pct'],selected['validation_scores'][name])
        w=pd.read_csv(run/'training_weights.csv')
        np.testing.assert_allclose(w.groupby('event').weight.sum(),w.groupby('event').weight.sum().iloc[0])
        with patch.object(TemporalDataset,'labels',side_effect=AssertionError('Cached scorer parsed labels')):
            self.assertEqual(evaluate(run),scored)
        with self.assertRaises(FileExistsError):m.run_experiment(data,run,protocol='v2')


if __name__=='__main__':unittest.main()
