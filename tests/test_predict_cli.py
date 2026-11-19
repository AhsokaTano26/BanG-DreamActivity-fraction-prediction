"""Exercise real CLI outputs; only the external HTTP transport is simulated."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx
import analysis.backtest as backtest

if importlib.util.find_spec('predict'):
    import predict
else:
    predict = None

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'


class PredictCommandTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(predict, 'prediction command is not implemented')

    def test_naive_dates_use_china_time(self):
        self.assertEqual(predict.parse_as_of('2026-01-01T08:00:00'),
                         predict.parse_as_of('2026-01-01T00:00:00Z'))

    def test_invalid_as_of_exits_without_a_traceback(self):
        command = subprocess.run([str(PYTHON), str(ROOT / 'training/legacy/predict.py'), '--event', '328',
                                  '--as-of', 'not-a-date'], capture_output=True, text=True)
        self.assertNotEqual(command.returncode, 0)
        self.assertIn('--as-of', command.stderr)
        self.assertNotIn('Traceback', command.stderr)

    def test_offline_json_replay_is_readonly_and_works_from_another_folder(self):
        database = ROOT / 'data/db.sqlite3'
        before = hashlib.sha256(database.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            command = subprocess.run([str(PYTHON), str(ROOT / 'training/legacy/predict.py'), '--event', '327',
                                      '--rank', '2000', '--offline', '--as-of', '2026-11-02T16:59:59+08:00',
                                      '--json'], cwd=folder, capture_output=True, text=True)
        self.assertEqual(command.returncode, 0, command.stderr)
        payload = json.loads(command.stdout)
        self.assertEqual(payload['event_id'], 327)
        self.assertEqual(payload['rank'], 2000)
        self.assertGreaterEqual(payload['predicted_final_ep'], payload['current_ep'])
        self.assertLessEqual(payload['observed_at'], payload['as_of'])
        self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())

    def test_empty_http_payload_is_rejected(self):
        def response(request):
            if request.url.path.endswith('/328.json'):
                return httpx.Response(200, json={'eventType': 'live_try',
                    'startAt': [None, None, None, '1000', None],
                    'endAt': [None, None, None, '2000', None]})
            return httpx.Response(200, json={'result': True, 'cutoffs': []})
        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            with self.assertRaisesRegex(ValueError, '观测'):
                predict.fetch_current(328, 2000, client=client)

    def test_http_result_false_is_not_used_as_a_forecast(self):
        def response(request):
            if request.url.path.endswith('/328.json'):
                return httpx.Response(200, json={'eventType': 'live_try',
                    'startAt': [None, None, None, '1000', None],
                    'endAt': [None, None, None, '2000', None]})
            return httpx.Response(200, json={'result': False})
        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            with self.assertRaises(ValueError):
                predict.fetch_current(328, 2000, client=client)

    def test_api_points_are_sorted_and_identical_duplicates_are_collapsed(self):
        def response(request):
            if request.url.path.endswith('/328.json'):
                return httpx.Response(200, json={'eventType': 'live_try',
                    'startAt': [None, None, None, '1000', None],
                    'endAt': [None, None, None, '10000', None]})
            return httpx.Response(200, json={'result': True,
                'cutoffs': [{'time': 3000, 'ep': 2}, {'time': 2000, 'ep': 1}, {'time': 3000, 'ep': 2}]})
        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            current = predict.fetch_current(328, 2000, client=client)
        self.assertEqual(current.times.tolist(), [2000, 3000])
        self.assertEqual(current.scores.tolist(), [1., 2.])

    def test_calibration_database_mismatch_suppresses_range(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'backtest.json').write_text(json.dumps({'database_sha256': 'wrong'}))
            (path / 'predictions.csv').write_text('event,rank,stage,method,predicted,actual\n')
            calibration, note = predict.load_calibration(ROOT / 'data/db.sqlite3', directory=path)
        self.assertIsNone(calibration)
        self.assertIn('回测', note)

    def test_unavailable_calibration_does_not_prevent_point_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            calibration, note = predict.load_calibration(ROOT / 'data/db.sqlite3', directory=Path(folder))
        self.assertIsNone(calibration)
        self.assertTrue(note)

    def test_calibration_model_mismatch_suppresses_range(self):
        database = ROOT / 'data/db.sqlite3'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'backtest.json').write_text(json.dumps({
                'database_sha256': hashlib.sha256(database.read_bytes()).hexdigest(),
                'model_signature': 'old-features-or-parameters'}))
            (path / 'predictions.csv').write_text('event,rank,stage,method,predicted,actual\n')
            calibration, note = predict.load_calibration(database, directory=path)
        self.assertIsNone(calibration)
        self.assertIn('模型', note)

    def test_changing_categorical_features_invalidates_calibration(self):
        database = ROOT / 'data/db.sqlite3'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'backtest.json').write_text(json.dumps({
                'database_sha256': hashlib.sha256(database.read_bytes()).hexdigest(),
                'model_signature': backtest.model_signature()}))
            (path / 'predictions.csv').write_text('event,rank,stage,method,predicted,actual\n')
            with patch.object(backtest, 'CAT', ['event_type']):
                calibration, note = predict.load_calibration(database, directory=path)
        self.assertIsNone(calibration)
        self.assertIn('模型', note)


if __name__ == '__main__':
    unittest.main()
