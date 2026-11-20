"""Resource reuse must preserve predictions, time guards and corruption checks."""
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import joblib

from test_production_model import model_fixture, live_sample as future_sample, H


class InferenceReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from production import model
        cls.model = model
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.curves, cls.metadata = model_fixture(cls.root)
        cls.before = max(c.end for c in cls.curves) + H
        cls.pack = model.train_pack(cls.curves, cls.metadata, cls.root / 'versions',
                                   before_ms=cls.before, ranks=[1000, 2000, 5000], iterations=3)

    def test_cached_predictions_match_and_load_each_artifact_once(self):
        sample = future_sample(self.curves, self.metadata, 1000)
        expected = self.model.predict_pack(self.pack, [sample], metadata_directory=self.metadata)
        cache = self.model.InferenceCache()
        with patch.object(joblib, 'load', wraps=joblib.load) as loads, \
             patch.object(self.model, '_verify_pack', wraps=self.model._verify_pack) as checks:
            for _ in range(2):
                with self.model.inference_session(cache):
                    self.model.verify_pack(self.pack)
                    actual = self.model.predict_pack(self.pack, [sample], metadata_directory=self.metadata)
                    self.assertEqual(actual, expected)
            names = [Path(call.args[0]).name for call in loads.call_args_list]
        self.assertCountEqual(names, ['metadata.joblib', 'context.joblib', 'shared.joblib'])
        self.assertEqual(checks.call_count, 1)

    def test_changed_compatibility_registry_forces_full_verification(self):
        registry = self.root / 'registry.json'
        registry.write_text('{}')
        cache = self.model.InferenceCache()
        with patch.object(self.model, 'REGISTRY', registry), self.model.inference_session(cache), \
             patch.object(self.model, '_verify_pack', wraps=self.model._verify_pack) as checks:
            self.model.verify_pack(self.pack)
            self.model.verify_pack(self.pack)
            registry.write_text('{"changed":true}')
            self.model.verify_pack(self.pack)
        self.assertEqual(checks.call_count, 2)

    def test_evicted_model_is_reverified_when_selected_again(self):
        import shutil
        other = self.root / 'second-model'
        shutil.copytree(self.pack, other)
        cache = self.model.InferenceCache()
        with self.model.inference_session(cache), \
             patch.object(self.model, '_verify_pack', wraps=self.model._verify_pack) as checks:
            self.model.verify_pack(self.pack)
            self.model.verify_pack(other)
            self.model.verify_pack(self.pack)
        self.assertEqual(checks.call_count, 3)

    def test_warm_cache_rejects_artifact_change_and_extra_file(self):
        import shutil
        path = self.root / 'tampered'
        shutil.copytree(self.pack, path)
        cache = self.model.InferenceCache()
        with self.model.inference_session(cache):
            self.model.verify_pack(path)
            weights = path / 'context.joblib'
            original = weights.read_bytes()
            weights.write_bytes(original + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'artifact'):
                self.model.verify_pack(path)
            weights.write_bytes(original)
            self.model.verify_pack(path)
            (path / 'unexpected').write_text('extra')
            with self.assertRaisesRegex(ValueError, 'artifact'):
                self.model.verify_pack(path)

    def test_warm_cache_refreshes_query_metadata_and_keeps_future_guard(self):
        sample = future_sample(self.curves, self.metadata, 1000)
        cache = self.model.InferenceCache()
        with self.model.inference_session(cache):
            self.model.predict_pack(self.pack, [sample], metadata_directory=self.metadata)
            path = self.metadata / 'events.json'
            events = json.loads(path.read_text())
            events.pop(str(sample['curve'].event))
            path.write_text(json.dumps(events))
            try:
                with self.assertRaisesRegex(ValueError, 'missing metadata'):
                    self.model.predict_pack(self.pack, [sample], metadata_directory=self.metadata)
            finally:
                events[str(sample['curve'].event)] = {'characters': [{'characterId': 7}]}
                path.write_text(json.dumps(events))
            with self.assertRaises(ValueError):
                self.model.predict_pack(self.pack, [dict(sample, cutoff=sample['curve'].start - 1)],
                                        metadata_directory=self.metadata)


class WorkerReuseTests(unittest.TestCase):
    def test_web_parent_does_not_import_numerical_libraries(self):
        code = ("import production.app,sys,json; "
                "print(json.dumps([n for n in ('numpy','pandas','catboost','xgboost','torch') if n in sys.modules]))")
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])

    def test_successive_updates_reuse_child_and_restart_after_exit(self):
        from production.web import CycleWorker
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipts = root / 'calls.jsonl'
            code = ("import sys,json,os\n"
                    "for line in sys.stdin:\n"
                    f" with open({str(receipts)!r},'a') as f: f.write(json.dumps(os.getpid())+'\\n')\n"
                    " print(json.dumps({'status':'ok'}),flush=True)\n")
            worker = CycleWorker({'root': str(root)}, root / 'config.json', interval=.05)
            with patch.object(worker, 'cycle_command', return_value=[sys.executable, '-u', '-c', code]):
                worker.start()
                try:
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        calls = receipts.read_text().splitlines() if receipts.exists() else []
                        if len(calls) >= 2:
                            break
                        time.sleep(.01)
                    self.assertGreaterEqual(len(calls), 2)
                    self.assertEqual(len(set(calls)), 1)
                    os.kill(int(calls[-1]), 15)
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        calls = receipts.read_text().splitlines()
                        if len(set(calls)) == 2:
                            break
                        time.sleep(.01)
                    self.assertEqual(len(set(calls)), 2)
                finally:
                    worker.stop()
                self.assertFalse(worker.thread.is_alive())
                self.assertEqual(worker.snapshot()['state'], 'stopped')


class RuntimeReadTests(unittest.TestCase):
    def test_single_version_does_not_fall_back_to_an_old_directory(self):
        from production.runtime import _previous
        from analysis.data_policy import MIN_EVENT_START_MS as S
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/'models/versions/old'
            old.mkdir(parents=True)
            (root/'models/current.json').write_text(json.dumps({'path':str(root/'missing-new')}))
            def verify(path):
                if Path(path)==old:
                    return dict(production_ready=True, history_max_end_ms=S+1)
                raise ValueError('Missing installed model')
            with patch('production.runtime.verify_pack', side_effect=verify):
                with self.assertRaises(ValueError):
                    _previous(root,before_ms=S+H,current_only=True)

    def test_replacing_installed_model_switches_existing_activity_without_reading_old_weights(self):
        from production import selected_runtime as runtime
        from production.state import Ledger
        from test_production_runtime import FakeStore
        from analysis.data_policy import MIN_EVENT_START_MS as S
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = Ledger(root)
            ledger.fact('bootstrap_complete', True)
            event = dict(event=9, start_ms=S+3*H, end_ms=S+4*H)
            ledger.create_pin(event, root/'deleted-old', 'old', {'calibration_id':'existing'}, now_ms=S+3*H)
            config = dict(root=str(root), ranks=[1000], forecast_method='legacy_pair',
                          automatic_training=False, selected_seed_run=None)
            manifest = dict(model_id='new', history_max_end_ms=S+H)
            FakeStore.index=[event]; FakeStore.curves=[]
            with patch.object(runtime, 'Store', FakeStore), \
                 patch.object(runtime, '_previous', return_value=(root/'current', manifest)), \
                 patch.object(runtime, 'verify_pack', side_effect=AssertionError('must not open deleted old model')):
                result = runtime.run_cycle(config, now_ms=S+3*H+H//2, sync=False)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(ledger.pin(9)['model_id'], 'new')
            self.assertEqual(ledger.pin(9)['calibration'], {'calibration_id':'existing'})
            self.assertEqual(result['active_event_models']['9']['model_id'], 'new')
            with ledger.connect() as con:
                self.assertEqual(con.execute('SELECT COUNT(*) FROM pin_history').fetchone()[0], 1)

    def test_web_cycle_queries_only_issued_feedback_and_active_curves(self):
        from production import selected_runtime as runtime
        from production.data import Store
        from production.state import Ledger
        from analysis.data_policy import MIN_EVENT_START_MS as S
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root)
            Ledger(root).fact('bootstrap_complete', True)
            config = dict(root=str(root), ranks=[1000], forecast_method='legacy_pair',
                          automatic_training=False, selected_seed_run=None)
            with store._connect() as con:
                for event, start in [(900, S), (9, S + 3 * H)]:
                    con.execute('INSERT INTO events VALUES(?,?,?,?,?)', (event, start, start + H, 'normal', str(event)))
                    # No issued forecast needs the old curve. It must not be decoded.
                    if event == 900:
                        con.execute('INSERT INTO curves(event,rank,points_json,updated_ms) VALUES(?,?,?,?)',
                                    (event, 1000, 'not json', S))
            manifest = dict(model_id='released', history_max_end_ms=S + H)
            with patch.object(runtime, '_previous', return_value=(root / 'weights', manifest)), \
                 patch.object(runtime, 'verify_pack', return_value=manifest), \
                 patch.object(runtime, 'train_pack', side_effect=AssertionError('must not fit')):
                result = runtime.run_cycle(config, now_ms=S + 3 * H + H // 2, sync=False)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['active_events'], [9])
            self.assertFalse(any(row['available'] for row in result['predictions']))
