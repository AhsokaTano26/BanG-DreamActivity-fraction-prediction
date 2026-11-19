"""Policy and provenance boundaries for the dense two-rank full retraining adapter."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from analysis import dense_v2 as dense
from analysis.data_policy import MIN_EVENT_START_MS, POLICY_ID


class DenseV2Tests(unittest.TestCase):
    def fixture(self, start=MIN_EVENT_START_MS):
        tmp = TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pd.DataFrame([{'event': 1, 'start_ms': start}]).to_csv(root/'partitions.csv', index=False)
        curve = SimpleNamespace(event=1, start=start)
        return SimpleNamespace(root=root,
            manifest={'dataset_id': dense.DATASET_ID,
                      'protocol': {'ranks': [1000, 2000], 'data_policy_id': POLICY_ID,
                                   'minimum_event_start_ms': MIN_EVENT_START_MS},
                      'splits': {'train': {'event_ids': [1]}}},
            train_curves=lambda: iter([curve]), inputs=lambda split: iter([{'curve': curve}]),
            labels=lambda split, **kwargs: {'train': 100.})

    def test_exact_dataset_policy_and_actual_start_required(self):
        data = self.fixture()
        dense.DenseTrainingData(data)
        data.manifest['dataset_id'] = 'legacy'
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            dense.DenseTrainingData(data)
        with self.assertRaisesRegex(ValueError, 'Pre-dense'):
            dense.DenseTrainingData(self.fixture(MIN_EVENT_START_MS-1))
        data = self.fixture(); data.manifest['protocol']['data_policy_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'policy'):
            dense.DenseTrainingData(data)

    def test_model_facade_never_releases_test_answers(self):
        proxy = dense.DenseTrainingData(self.fixture())
        self.assertEqual(proxy.labels('train'), {'train': 100.})
        with self.assertRaises(PermissionError):
            proxy.labels('test', allow_test=True)
        self.assertEqual(len(list(proxy.inputs('test'))), 1)

    def test_history_cohort_is_checked(self):
        data = self.fixture()
        data.train_curves = lambda: iter([SimpleNamespace(event=2, start=MIN_EVENT_START_MS)])
        with self.assertRaisesRegex(ValueError, 'history'):
            dense.DenseTrainingData(data)

    def test_legacy_guards_restored_and_adapter_hashed(self):
        from analysis import innovation_runner, v2_next_runner
        original_ids = innovation_runner.DATASET_IDS.copy()
        original_next = v2_next_runner.DATASET_ID
        for stage in ['innovation', 'next']:
            with dense.adapted_runner(stage) as runner:
                if stage == 'innovation':
                    self.assertEqual(runner.DATASET_IDS['v2'], dense.DATASET_ID)
                else:
                    self.assertEqual(runner.DATASET_ID, dense.DATASET_ID)
                cfg = {}; runner.identity(cfg)
                self.assertEqual(cfg['execution_adapter']['data_policy_id'], POLICY_ID)
                self.assertEqual(cfg['execution_adapter']['cpu_threads'], 2)
                self.assertIn('training/analysis/dense_v2.py', runner.local_sources([]))
        self.assertEqual(innovation_runner.DATASET_IDS, original_ids)
        self.assertEqual(v2_next_runner.DATASET_ID, original_next)

    def test_stage_guard_and_cpu_budget(self):
        from analysis import temporal_train
        original = temporal_train.CatBoostRegressor
        with dense.adapted_runner('base') as runner:
            self.assertEqual(runner.CatBoostRegressor(thread_count=4).get_param('thread_count'), 2)
            self.assertIn('training/analysis/dense_v2.py', runner.SOURCES)
        self.assertIs(temporal_train.CatBoostRegressor, original)
        with self.assertRaisesRegex(ValueError, 'Only'):
            dense.run_stage('base', resume=True)

    def test_new_run_receives_guarded_data_and_same_cohort_reference(self):
        from analysis import temporal_train
        with patch.object(dense, 'TemporalDataset', return_value=self.fixture()), \
             patch.object(temporal_train, 'run_training', return_value={'selected_recipe': 'x'}) as run:
            self.assertEqual(dense.run_stage('base'), {'selected_recipe': 'x'})
            self.assertIsInstance(run.call_args.args[0], dense.DenseTrainingData)
            self.assertEqual(run.call_args.args[1], dense.RUNS['base'])


if __name__ == '__main__':
    unittest.main()
