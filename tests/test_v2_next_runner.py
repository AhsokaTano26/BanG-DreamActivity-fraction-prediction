import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class NextRunnerTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.v2_next_runner'))
        from analysis import v2_next_runner
        return v2_next_runner

    def test_selection_keeps_old_model_and_excludes_individual_seeds(self):
        m = self.module()
        a = {'cases': [{'case_id': 'a', 'event': 1}, {'case_id': 'b', 'event': 2}]}
        values = {'old_selected': np.array([100., 100.]), 'tree': np.array([120., 120.]),
                  'net': np.array([110., 110.]), 'net__seed_42': np.array([100., 100.])}
        recipes = {'tree': {'family': 'adaptation'}, 'net': {'family': 'growth'}}
        s = m.select_candidates(a, {'a': 100., 'b': 100.}, values, recipes)
        self.assertEqual(s['selected_recipe'], 'old_selected')
        self.assertNotIn('net__seed_42', s['candidates'])
        self.assertEqual(s['combination_specs']['family_mean']['members'], ['tree', 'net'])
        self.assertAlmostEqual(s['validation_scores']['blend_new_25'], 2.5)

    def test_alignment_rejects_duplicates_and_missing_cases(self):
        m = self.module()
        a = {'cases': [{'case_id': 'a'}, {'case_id': 'b'}], 'current': np.array([10., 10.])}
        frame = pd.DataFrame({'case_id': ['b', 'a'], 'predicted': [30., 20.]})
        np.testing.assert_array_equal(m.align_prediction(a, frame), [20., 30.])
        for bad in [frame.iloc[:1], pd.concat([frame, frame.iloc[:1]])]:
            with self.assertRaises(ValueError):
                m.align_prediction(a, bad)

    def test_wrong_fingerprint_rejected_before_creating_output(self):
        from tempfile import TemporaryDirectory
        m = self.module()
        with TemporaryDirectory() as root:
            run = Path(root)/'run'
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                m.run_experiment(SimpleNamespace(manifest={'dataset_id': 'wrong'}), run)
            self.assertFalse(run.exists())

    def test_development_blind_seal_resume_and_independent_evaluation(self):
        m = self.module()
        import test_temporal_training as fixture
        from analysis.holdout_dataset import sha256, write_json
        from analysis.temporal_rank_subset import build_subset
        from analysis.temporal_dataset import TemporalDataset
        from analysis.temporal_evaluate import evaluate, verify_predictions
        helper = fixture.TemporalTrainingTests(); self.addCleanup(helper.doCleanups)
        root, parent = helper.fixture(); build_subset(parent, root/'subset')
        data = TemporalDataset(root/'subset'); run = root/'next'; reference = root/'old'
        reference.mkdir()
        for split in ['train', 'validation', 'test']:
            rows = [{'case_id': s['case_id'], 'method': 'reference',
                     'predicted': s['score']/s['progress']} for s in data.inputs(split)]
            pd.DataFrame(rows).to_csv(reference/f'{split}_predictions.csv', index=False)
        write_json(reference/'blind_manifest.json', {'dataset_id': data.manifest['dataset_id'],
            'reference_method': 'reference', 'artifacts': {
                p.name: sha256(p) for p in reference.glob('*.csv')}})
        old_labels, old_inputs = TemporalDataset.labels, TemporalDataset.inputs
        def labels(instance, split, **kw):
            self.assertNotEqual(split, 'test')
            self.assertTrue((run/'run_manifest.json').exists())
            return old_labels(instance, split, **kw)
        def inputs(instance, split):
            if split == 'test':
                self.assertTrue((run/'development_manifest.json').exists())
            return old_inputs(instance, split)
        kw = {'reference': reference, 'families': ['retrieval'], 'allow_fixture': True,
              'run_cv': False, 'recipe_limit': 1, 'iterations_override': 3}
        with patch.dict(m.FAMILIES, {'retrieval': 'analysis.innovation_retrieval'}, clear=True), \
             patch.object(TemporalDataset, 'labels', labels), \
             patch.object(TemporalDataset, 'inputs', inputs), \
             patch('sqlite3.connect', side_effect=AssertionError('No SQLite')):
            s = m.run_experiment(data, run, phase='develop', **kw)
            self.assertFalse((run/'test_predictions.csv').exists())
            with patch.object(TemporalDataset, 'labels', side_effect=AssertionError('resume read labels')):
                self.assertEqual(m.run_experiment(data, run, phase='develop', resume=True, **kw), s)
                m.run_experiment(data, run, phase='blind', resume=True, **kw)
                self.assertEqual(m.run_experiment(data, run, phase='blind', resume=True, **kw), s)
            with self.assertRaises(FileExistsError):
                m.run_experiment(data, run, phase='develop', **kw)
        self.assertEqual(verify_predictions(run)['reference_method'], s['selected_recipe'])
        result = evaluate(run)
        for name, value in s['validation_scores'].items():
            self.assertAlmostEqual(result['splits']['validation'][name]['overall']['event_macro_mape_pct'], value)
        weights = pd.read_csv(run/'training_weights.csv').groupby('event').weight.sum()
        np.testing.assert_allclose(weights, weights.iloc[0])
        with patch.object(TemporalDataset, 'labels', side_effect=AssertionError('cached scorer read labels')):
            self.assertEqual(evaluate(run), result)
        from analysis.v2_next_report import evaluate_comparison, report
        comparison = evaluate_comparison(run)
        report_path = report(run)
        self.assertIn('测试120个冻结案例', report_path.read_text())
        with patch.object(TemporalDataset, 'labels', side_effect=AssertionError('cached comparison read labels')):
            self.assertEqual(evaluate_comparison(run), comparison)
            self.assertEqual(report(run), report_path)
        alternate = root/'audit'
        self.assertEqual(evaluate_comparison(run, alternate/'comparison')['pairs'], comparison['pairs'])
        self.assertEqual(report(run, alternate/'report', comparison=alternate/'comparison'), alternate/'report/README.md')
        self.assertTrue(report_path.exists())
        prediction = run/'validation_predictions.csv'
        prediction.write_text(prediction.read_text()+'\n')
        with patch.dict(m.FAMILIES, {'retrieval': 'analysis.innovation_retrieval'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'changed'):
                m.run_experiment(data, run, phase='blind', resume=True, **kw)


if __name__ == '__main__':
    unittest.main()
