"""Deployment boundaries; fixture archives below are not trained models."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest


class DeploymentTests(unittest.TestCase):
    def test_deployment_entry_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('production.deployment'))

    def test_model_release_entry_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('production.model_release'))

    def test_configuration_uses_portable_paths_and_explicit_training(self):
        from production.deployment import configure
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'state'
            config = configure(root)
            self.assertEqual(config['root'], str(root.resolve()))
            self.assertIsNone(config['selected_seed_run'])
            self.assertFalse(config['automatic_training'])
            self.assertEqual(config['forecast_method'], 'legacy_pair')
            self.assertEqual(config['ranks'][-1], 30000)
            self.assertEqual(configure(root), config)

    def test_current_activity_pin_survives_path_migration(self):
        from production.deployment import relocate_pins
        from production.state import Ledger
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = Ledger(root)
            event = {'event': 329, 'start_ms': 1800000000000, 'end_ms': 1800000100000}
            ledger.create_pin(event, '/old-computer/models/versions/abc', 'abc',
                              {'calibration_id': 'pinned'}, now_ms=1800000000001)
            relocate_pins(root)
            pin = ledger.pin(329)
            self.assertEqual(pin['model_id'], 'abc')
            self.assertEqual(pin['calibration'], {'calibration_id': 'pinned'})
            self.assertEqual(pin['model_path'], str(root.resolve() / 'models/versions/abc'))

    def test_archive_rejects_path_traversal_before_deserialization(self):
        from production.model_release import inspect_release
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'evil.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                member = tarfile.TarInfo('../escape')
                member.size = 1
                archive.addfile(member, io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError, 'path|路径'):
                inspect_release(path)
            self.assertFalse((Path(tmp).parent / 'escape').exists())

    def test_archive_rejects_symbolic_links(self):
        from production.model_release import inspect_release
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'link.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                member = tarfile.TarInfo('model/link')
                member.type = tarfile.SYMTYPE
                member.linkname = '/etc/passwd'
                archive.addfile(member)
            with self.assertRaisesRegex(ValueError, 'link|链接|regular'):
                inspect_release(path)

    def test_archive_rejects_duplicate_members(self):
        from production.model_release import inspect_release
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'duplicate.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                for _ in range(2):
                    member = tarfile.TarInfo('release.json')
                    member.size = 2
                    archive.addfile(member, io.BytesIO(b'{}'))
            with self.assertRaisesRegex(ValueError, 'Duplicate|重复'):
                inspect_release(path)

    def test_feedback_refuses_future_or_old_policy_rows(self):
        from production.model_release import validate_feedback
        row = {'event': 1, 'rank': 1000, 'phase': 1, 'start_ms': 1706504400000,
               'end_ms': 1706504500000, 'mean_log_error': .01, 'cases': 4,
               'actual_final': 1000, 'origin': 'live', 'source': None}
        self.assertEqual(len(validate_feedback([row], 1706504500001)), 1)
        with self.assertRaises(ValueError):
            validate_feedback([row], row['end_ms'])
        with self.assertRaises(ValueError):
            validate_feedback([row | {'start_ms': 1706504399999}], 1706504500001)
        with self.assertRaises(ValueError):
            validate_feedback([row | {'mean_log_error': float('nan')}], 1706504500001)

    def test_worker_is_web_deployment_cycle(self):
        from production.deployment import DeploymentWorker
        worker = DeploymentWorker({'root': '/tmp/test-state'}, '/tmp/test-state/config.json', offline=True)
        command = worker.cycle_command()
        self.assertIn('entrypoint.py', command[2])
        self.assertIn('cycle', command)
        self.assertEqual(command[-1], '--offline')

    def test_training_requires_all_previous_history_pairs(self):
        from production.deployment import history_covers_model
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'curve_support.json').write_text(json.dumps([
                {'event': 1, 'rank': 1000}, {'event': 1, 'rank': 2000}]))
            from types import SimpleNamespace
            self.assertFalse(history_covers_model(path, [SimpleNamespace(event=1, rank=1000)]))
            self.assertTrue(history_covers_model(path, [SimpleNamespace(event=1, rank=r) for r in (1000, 2000)]))

    def test_inference_only_cycle_never_calls_training(self):
        from production import selected_runtime as runtime
        from production.state import Ledger
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Ledger(root).fact('bootstrap_complete', True)
            fake_store = SimpleNamespace(metadata_directory=root / 'metadata',
                                         events=lambda now: [], load_curves=lambda *args, **kwargs: [])
            manifest = {'model_id': 'released', 'history_max_end_ms': 1706504400001}
            config = {'root': str(root), 'ranks': [1000], 'forecast_method': 'legacy_pair',
                      'automatic_training': False, 'selected_seed_run': None}
            with patch.object(runtime, 'Store', return_value=fake_store), \
                 patch.object(runtime, '_previous', return_value=(root / 'weights', manifest)), \
                 patch.object(runtime, 'train_pack', side_effect=AssertionError('must not train')) as train:
                result = runtime.run_cycle(config, now_ms=1706504500000, sync=False)
            train.assert_not_called()
            self.assertFalse(result['automatic_model_updates'])
            self.assertEqual(result['model_id'], 'released')

    def test_incomplete_history_retains_released_weights_without_training(self):
        from production import selected_runtime as runtime
        from production.state import Ledger
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Ledger(root).fact('bootstrap_complete', True)
            weights = root / 'weights'
            weights.mkdir()
            (weights / 'curve_support.json').write_text('[{"event":1,"rank":1000}]')
            store = SimpleNamespace(metadata_directory=root / 'metadata',
                                    events=lambda now: [], load_curves=lambda *args, **kwargs: [])
            manifest = {'model_id': 'released', 'history_max_end_ms': 1706504400001}
            config = {'root': str(root), 'ranks': [1000], 'forecast_method': 'legacy_pair',
                      'automatic_training': True, 'require_history_coverage': True, 'selected_seed_run': None}
            with patch.object(runtime, 'Store', return_value=store), \
                 patch.object(runtime, '_previous', return_value=(weights, manifest)), \
                 patch.object(runtime, 'train_pack', side_effect=AssertionError('must not train')) as train:
                result = runtime.run_cycle(config, now_ms=1706504500000, sync=False)
            train.assert_not_called()
            self.assertEqual(result['model_id'], 'released')
            self.assertTrue(any('Incomplete training history' in item['error'] for item in result['errors']))


if __name__ == '__main__':
    unittest.main()
