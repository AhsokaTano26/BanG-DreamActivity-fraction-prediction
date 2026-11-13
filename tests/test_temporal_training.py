"""Guard the two-rank temporal experiment's cohort and blind training boundary."""
import gzip
import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import test_temporal_protocol as fixtures
from analysis.temporal_dataset import TemporalDataset, build_temporal


class TemporalTrainingTests(unittest.TestCase):
    def fixture(self):
        helper = fixtures.TemporalProtocolTests()
        self.addCleanup(helper.doCleanups)
        root, base, curves = helper.fixture()
        build_temporal(root / 'temporal', base, curves, source_identity={'fixture': True})
        return root, TemporalDataset(root / 'temporal')

    def subset_module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.temporal_rank_subset'),
                             'A sealed rank subset builder is required')
        from analysis import temporal_rank_subset
        return temporal_rank_subset

    def test_subset_preserves_every_selected_prefix_label_and_origin(self):
        module = self.subset_module()
        root, parent = self.fixture()
        module.build_subset(parent, root / 'subset', ranks=(1000, 2000))
        data = TemporalDataset(root / 'subset')
        self.assertEqual(data.manifest['protocol']['ranks'], [1000, 2000])
        self.assertEqual(data.manifest['parent_dataset_id'], parent.manifest['dataset_id'])
        for split in ('train', 'validation', 'test'):
            self.assertEqual(data.manifest['splits'][split]['event_ids'],
                             parent.manifest['splits'][split]['event_ids'])
            expected = parent.case_index(split).query('rank in [1000, 2000]').reset_index(drop=True)
            pd.testing.assert_frame_equal(data.case_index(split), expected)
            members = parent.memberships(split)
            pd.testing.assert_frame_equal(data.memberships(split),
                members[members.case_id.isin(expected.case_id)].reset_index(drop=True))
            labels = parent.labels(split, allow_test=True)
            self.assertEqual(data.labels(split, allow_test=True), {c: labels[c] for c in expected.case_id})
            with gzip.open(parent.root / f'{split}_inputs.jsonl.gz', 'rt') as f:
                source = [r for line in f if (r := json.loads(line))['rank'] in (1000, 2000)]
            with gzip.open(data.root / f'{split}_inputs.jsonl.gz', 'rt') as f:
                self.assertEqual([json.loads(line) for line in f], source)
        self.assertEqual({c.rank for c in data.train_curves()}, {1000, 2000})
        coverage = pd.read_csv(data.root / 'coverage.csv')
        missing = coverage.query("split == 'test' and stage_id == 'elapsed_1h'")
        self.assertEqual(missing.available_cases.tolist(), [0, 0])
        self.assertEqual(missing.missing_prefixes.tolist(), [3, 3])
        with self.assertRaises(ValueError):
            data.labels('test')
        with self.assertRaises(FileExistsError):
            module.build_subset(parent, data.root)
        f = data.root / 'test_cases.csv'
        f.write_text(f.read_text() + '\n')
        with self.assertRaises(ValueError):
            TemporalDataset(data.root)

    def test_subset_rejects_unknown_and_duplicate_ranks_without_output(self):
        module = self.subset_module()
        root, parent = self.fixture()
        for ranks in [(1000, 999999), (1000, 1000), ()]:
            with self.assertRaises(ValueError):
                module.build_subset(parent, root / 'invalid', ranks=ranks)
            self.assertFalse((root / 'invalid').exists())

    def test_training_is_blind_sealed_replayable_and_validation_selected(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.temporal_train'),
                             'Temporal full-training entry point is required')
        from analysis.temporal_train import run_training
        from analysis.temporal_evaluate import evaluate, verify_predictions
        from analysis.multirank_model import event_weights, prepare, model_prediction
        import joblib
        root, parent = self.fixture()
        self.subset_module().build_subset(parent, root / 'subset')
        data = TemporalDataset(root / 'subset')
        original_labels = TemporalDataset.labels

        def guarded_labels(instance, split, **kwargs):
            if split == 'test':
                raise AssertionError('Training must not read test answers')
            return original_labels(instance, split, **kwargs)

        with patch.object(TemporalDataset, 'labels', guarded_labels), \
             patch('sqlite3.connect', side_effect=AssertionError('Model cannot read SQLite')):
            selection = run_training(data, root / 'run', iterations=5)
        self.assertEqual(selection['selected_recipe'], min(selection['candidates'],
            key=lambda m: selection['validation'][m]['overall']['event_macro_mape_pct']))
        self.assertEqual(set(selection['candidates']),
            {'hold_current', 'linear', 'catboost_rank', 'catboost_context'})
        seal = verify_predictions(root / 'run')
        self.assertEqual(seal['reference_method'], selection['selected_recipe'])
        metadata = joblib.load(root / 'run/models/metadata.joblib')
        arrays = prepare(data.inputs('test'), metadata)
        predictions = pd.read_csv(root / 'run/test_predictions.csv')
        for method in ('catboost_rank', 'catboost_context'):
            bundle = joblib.load(root / f'run/models/{method}.joblib')
            actual = model_prediction(bundle, arrays)
            saved = predictions[predictions.method == method].set_index('case_id')
            np.testing.assert_allclose(actual, saved.loc[[c['case_id'] for c in arrays['cases']], 'predicted'])
        weights = event_weights([1, 1, 2])
        self.assertAlmostEqual(weights[:2].sum(), weights[2])
        scored = evaluate(root / 'run')
        self.assertEqual(scored['splits']['test']['catboost_rank']['overall']['events'], 3)
        with patch.object(TemporalDataset, 'labels', side_effect=AssertionError('Cached scorer must not reread labels')):
            self.assertEqual(evaluate(root / 'run'), scored)
        with self.assertRaises(FileExistsError):
            run_training(data, root / 'run', iterations=5)
        model = root / 'run/models/catboost_rank.joblib'
        model.write_bytes(model.read_bytes() + b'x')
        with self.assertRaises(ValueError):
            verify_predictions(root / 'run')

    def test_frozen_comparison_is_same_parent_and_same_cases_without_answer_reads(self):
        from analysis.temporal_train import reference_predictions, run_training, FROZEN_METHODS
        from analysis.temporal_benchmark import run_frozen
        from analysis.multirank_dataset import MultirankDataset
        from analysis.multirank_model import run_experiment
        root, parent = self.fixture()
        self.subset_module().build_subset(parent, root / 'subset')
        data = TemporalDataset(root / 'subset')
        run_experiment(MultirankDataset(root / 'ranks'), root / 'old', iterations=5)
        run_frozen(parent, root / 'old', root / 'frozen')
        with patch.object(TemporalDataset, 'labels', side_effect=AssertionError('No reference answer reads')):
            reference = reference_predictions(data, root / 'frozen')
        for split, frame in reference.items():
            original = pd.read_csv(root / f'frozen/{split}_predictions.csv')
            expected = original[original.case_id.isin(data.case_index(split).case_id)
                                & original.method.isin(FROZEN_METHODS)].reset_index(drop=True)
            pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected)
        # Unrelated data identity must fail before creating a new training run.
        data.manifest['parent_dataset_id'] = 'wrong-parent'
        with self.assertRaisesRegex(ValueError, 'parent dataset'):
            run_training(data, root / 'invalid_run', iterations=5, reference_run=root / 'frozen')
        self.assertFalse((root / 'invalid_run').exists())
        data = TemporalDataset(root / 'subset')
        selection = run_training(data, root / 'with_reference', iterations=5, reference_run=root / 'frozen')
        self.assertTrue(set(FROZEN_METHODS).isdisjoint(selection['candidates']))
        saved = pd.read_csv(root / 'with_reference/test_predictions.csv')
        pd.testing.assert_frame_equal(saved[saved.method.isin(FROZEN_METHODS)].reset_index(drop=True),
                                      reference['test'].reset_index(drop=True))
        source = root / 'frozen/test_predictions.csv'
        source.write_text(source.read_text() + '\n')
        with self.assertRaises(ValueError):
            reference_predictions(data, root / 'frozen')


if __name__ == '__main__':
    unittest.main()
