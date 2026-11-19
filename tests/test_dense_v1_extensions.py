import importlib.util
import tempfile
import unittest
from pathlib import Path


class DenseV1ExtensionTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec('analysis.dense_v1_extensions')
        self.assertIsNotNone(spec, 'Dense V1 adapter must enforce the population and answer boundary')
        from analysis import dense_v1_extensions
        return dense_v1_extensions

    def test_model_dataset_never_releases_test_answers(self):
        m = self.module()
        class Answers:
            def labels(self, split, **kwargs):
                return {'answer': 123}
        guarded = m.BlindDataset(Answers())
        self.assertEqual(guarded.labels('train'), {'answer': 123})
        with self.assertRaisesRegex(PermissionError, 'test'):
            guarded.labels('test', allow_test=True)

    def test_time_folds_use_actual_dates_and_strictly_earlier_ends(self):
        m = self.module()
        from types import SimpleNamespace
        curves = [SimpleNamespace(event=e, start=100*i, end=100*i+90)
                  for i,e in enumerate([9, 2, 80, 7, 3, 60, 1, 30], 1)]
        folds = m.dense_folds(curves)
        self.assertEqual(folds[0]['training_events'], [9, 2, 80, 7])
        self.assertEqual(folds[0]['heldout_events'], [3])
        self.assertEqual(folds[-1]['heldout_events'], [30])
        for fold in folds:
            self.assertLess(fold['training_latest_end'], fold['heldout_earliest_start'])

    def test_reference_rejects_other_cohort_even_with_valid_seal(self):
        m = self.module()
        from analysis.holdout_dataset import write_json, sha256
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/'selection.json', {'selected_recipe': 'chosen'})
            for split in ['validation', 'test']:
                (root/f'{split}_predictions.csv').write_text('method,case_id,predicted\nchosen,x,1\n')
            write_json(root/'blind_test_manifest.json', {'dataset_id':'old', 'run_id':'old',
                       'artifacts':{p.name:sha256(p) for p in root.iterdir()}})
            with self.assertRaisesRegex(ValueError, 'dataset|cohort'):
                m.reference_identity(root, 'dense')

    def test_training_blends_replay_frozen_reference_without_refitting(self):
        spec = importlib.util.find_spec('analysis.dense_v1_training_replay')
        self.assertIsNotNone(spec, 'Replay must restore reference-dependent training diagnostics')
        from analysis.dense_v1_training_replay import training_combinations
        import numpy as np
        actual = training_combinations('relationship', {'new_top3':np.array([20.])},
                                       np.array([100.]))
        self.assertEqual(actual['blend_new_0.25'].tolist(), [80.])
        actual = training_combinations('structured', {'neural_top3':np.array([20.])},
                                       np.array([100.]))
        self.assertEqual(actual['blend_neural_0.75'].tolist(), [40.])
        actual = training_combinations('quality', {'time_gru_seed42':np.array([10.]),
                                       'time_gru_ensemble3':np.array([20.])},np.array([100.]))
        self.assertEqual(actual['blend25_time_gru_ensemble3'].tolist(), [80.])
        self.assertNotIn('blend25_time_gru_seed42', actual)

    def test_training_coverage_only_allows_declared_history_baselines(self):
        from analysis import dense_v1_training_replay as replay
        self.assertTrue(hasattr(replay, 'check_training_coverage'),
                        'Replay must distinguish unavailable causal baselines from missing fitted models')
        check = replay.check_training_coverage
        check({'model'}, ['model','template','shape_analog'], 'model', ['template','shape_analog'])
        with self.assertRaises(ValueError):
            check({'model'}, ['model','template'], 'model')
        with self.assertRaises(ValueError):
            check({'model'}, ['model','other_model'], 'model', ['other_model'])
        with self.assertRaises(ValueError):
            check({'model'}, ['model','template'], 'template', ['template'])

    def test_mps_restart_requires_explicit_available_mps(self):
        spec = importlib.util.find_spec('analysis.dense_v1_mps')
        self.assertIsNotNone(spec, 'MPS restart needs an explicit device guard')
        from analysis.dense_v1_mps import require_mps
        require_mps('mps', True)
        with self.assertRaises(ValueError):
            require_mps('cpu', True)
        with self.assertRaises(RuntimeError):
            require_mps('mps', False)


if __name__ == '__main__':
    unittest.main()
