"""Production must exclude manual-report history by actual start, including artifacts."""
from dataclasses import replace
from contextlib import closing
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.backtest import Curve


START = 1706504400000
HOUR = 3600000


def curve(event, start, rank=2000):
    times = start + np.arange(25, dtype=np.int64) * HOUR
    return Curve(event, rank, 'normal', start, int(times[-1]), times,
                 np.arange(25, dtype=float) * 100 + 1)


class DataPolicyTests(unittest.TestCase):
    def test_history_cutoff_is_inclusive_and_never_uses_event_id(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.data_policy'),
                             'shared start-time data policy is missing')
        from analysis.data_policy import eligible_history
        old = curve(999, START - 1)
        boundary = curve(2, START)
        recent = curve(1, START + HOUR)
        values = [recent, old, boundary, replace(boundary, rank=1000)]
        selected = eligible_history(iter(values))
        self.assertEqual([(c.event, c.rank) for c in selected],
                         [(1, 2000), (2, 2000), (2, 1000)])
        self.assertIs(selected[0], recent)
        self.assertEqual(len(values), 4)

    def test_production_history_rejects_old_and_unfinished_events(self):
        from forecasting import select_history
        target = curve(2, START + 100 * HOUR)
        old = curve(999, START - 1)
        boundary = curve(1, START)
        unfinished = curve(3, target.start - HOUR)
        same_event = curve(2, START, 1000)
        selected = select_history(target, [old, boundary, unfinished, same_event])
        self.assertEqual([c.event for c in selected], [1])

    def test_production_loader_uses_dates_instead_of_event_number(self):
        import forecasting
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / 'history.sqlite3'
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute('CREATE TABLE event (EventID INTEGER, EventType TEXT, '
                    'StartAt INTEGER, EndAt INTEGER, Rank INTEGER, Country TEXT, PointRank TEXT)')
                for item in [curve(999, START - 1), curve(1, START), curve(2, START + HOUR)]:
                    points = [{'time': int(t), 'ep': int(s)} for t, s in zip(item.times, item.scores)]
                    connection.execute('INSERT INTO event VALUES (?,?,?,?,?,?,?)',
                        (item.event, item.kind, item.start, item.end, item.rank, '中国大陆', str(points)))
            curves, _ = forecasting.load_curves(database, now_ms=START + 100 * HOUR)
        self.assertEqual([c.event for c in curves], [1, 2])

    def test_old_calibration_events_cannot_supply_the_minimum_event_count(self):
        from forecasting import calibrate_range
        from analysis.data_policy import POLICY_ID
        old = [curve(i, START - (i + 1) * HOUR) for i in range(20)]
        rows = pd.DataFrame([{'event': c.event, 'rank': 2000, 'stage': '50%',
            'method': 'xgb_growth', 'predicted': 100., 'actual': 120.} for c in old])
        rows.attrs['data_policy_id'] = POLICY_ID
        self.assertIsNone(calibrate_range(100., 80., 2000, '50%', rows, old))

    def test_direct_calibration_cannot_bypass_policy_provenance(self):
        from forecasting import calibrate_range
        history = [curve(i, START + i * 48 * HOUR) for i in range(20)]
        rows = pd.DataFrame([{'event': c.event, 'rank': 2000, 'stage': '50%',
            'method': 'xgb_growth', 'predicted': 100., 'actual': 120.} for c in history])
        self.assertIsNone(calibrate_range(100., 80., 2000, '50%', rows, history))

    def test_policy_attested_calibration_can_be_loaded_and_used(self):
        import predict
        from forecasting import calibrate_range
        from analysis.backtest import model_signature
        from analysis.data_policy import POLICY_ID
        history = [curve(i, START + i * 48 * HOUR) for i in range(20)]
        rows = pd.DataFrame([{'event': c.event, 'rank': 2000, 'stage': '50%',
            'method': 'xgb_growth', 'predicted': 100., 'actual': 120.} for c in history])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); database = root / 'db'; database.write_bytes(b'new fixture')
            (root / 'backtest.json').write_text(json.dumps({
                'database_sha256': hashlib.sha256(database.read_bytes()).hexdigest(),
                'model_signature': model_signature(), 'data_policy_id': POLICY_ID}))
            rows.to_csv(root / 'predictions.csv', index=False)
            loaded, note = predict.load_calibration(database, directory=root)
        self.assertEqual(loaded.attrs.get('data_policy_id'), POLICY_ID)
        interval = calibrate_range(100., 80., 2000, '50%', loaded, history)
        self.assertEqual(interval['calibration_events'], 20)
        self.assertAlmostEqual(interval['upper_ep'], 120.)
        self.assertIsNone(note)

    def test_matching_legacy_calibration_is_rejected_without_policy_provenance(self):
        import predict
        from analysis.backtest import model_signature
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / 'db'; database.write_bytes(b'calibration fixture')
            (root / 'backtest.json').write_text(json.dumps({
                'database_sha256': hashlib.sha256(database.read_bytes()).hexdigest(),
                'model_signature': model_signature()}))
            (root / 'predictions.csv').write_text('event,rank,stage,method,predicted,actual\n')
            calibration, note = predict.load_calibration(database, directory=root)
        self.assertIsNone(calibration)
        self.assertTrue(note)

    def test_multirank_cli_rejects_pre_cutoff_training_artifacts(self):
        from analysis.holdout_dataset import curve_record, sha256, write_json
        from analysis.multirank_dataset import identity
        from predict_multirank import forecast_rows
        for old_in_partitions in (True, False):
            with self.subTest(old_in_partitions=old_in_partitions), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); data = root / 'data'; data.mkdir()
                old = curve(999, START - 1)
                start = old.start if old_in_partitions else START
                pd.DataFrame([{'split': 'train', 'event': 999, 'rank': 2000,
                    'start_ms': start, 'end_ms': old.end}]).to_csv(data / 'partitions.csv', index=False)
                with gzip.open(data / 'train_curves.jsonl.gz', 'wt') as stream:
                    stream.write(json.dumps(curve_record(old)) + '\n')
                manifest = {'artifacts': {p.name: sha256(p) for p in data.iterdir()}}
                manifest['dataset_id'] = identity(manifest)
                write_json(data / 'manifest.json', manifest)
                run = root / 'run'; (run / 'models').mkdir(parents=True)
                (run / 'models/rank_coverage.csv').write_text('rank,train_events,validation_events,test_events\n')
                config = {'dataset_directory': str(data), 'dataset_id': manifest['dataset_id'],
                          'run_id': 'fixture'}
                write_json(run / 'run_manifest.json', config)
                write_json(run / 'blind_test_manifest.json', config | {
                    'artifacts': {'run_manifest.json': sha256(run / 'run_manifest.json')}})
                with self.assertRaisesRegex(ValueError, '2024-01-29|policy|population'):
                    forecast_rows(run, [], START + 100 * HOUR)

    def test_multirank_seal_cannot_repoint_a_model_to_a_different_population(self):
        from analysis.holdout_dataset import curve_record, sha256, write_json
        from analysis.multirank_dataset import identity
        from predict_multirank import forecast_rows
        for field in ('dataset_directory', 'dataset_id', 'run_id', 'unprotected_config'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); data = root / 'dense'; data.mkdir()
                current = curve(1, START)
                pd.DataFrame([{'split': 'train', 'event': 1, 'rank': 2000,
                    'start_ms': current.start, 'end_ms': current.end}]).to_csv(data / 'partitions.csv', index=False)
                with gzip.open(data / 'train_curves.jsonl.gz', 'wt') as stream:
                    stream.write(json.dumps(curve_record(current)) + '\n')
                manifest = {'artifacts': {p.name: sha256(p) for p in data.iterdir()}}
                manifest['dataset_id'] = identity(manifest)
                write_json(data / 'manifest.json', manifest)
                run = root / 'run'; (run / 'models').mkdir(parents=True)
                (run / 'models/rank_coverage.csv').write_text('rank,train_events,validation_events,test_events\n')
                seal = {'dataset_directory': str(data), 'dataset_id': manifest['dataset_id'],
                        'run_id': 'dense-run'}
                config = dict(seal)
                if field != 'unprotected_config':
                    config[field] = str(root / 'legacy') if field == 'dataset_directory' else 'legacy-value'
                write_json(run / 'run_manifest.json', config)
                seal['artifacts'] = {} if field == 'unprotected_config' else {
                    'run_manifest.json': sha256(run / 'run_manifest.json')}
                write_json(run / 'blind_test_manifest.json', seal)
                with self.assertRaisesRegex(ValueError, 'provenance|population|manifest'):
                    forecast_rows(run, [], START + 100 * HOUR)

    def test_context_feature_history_cannot_bypass_filtered_training_pool(self):
        # Torch stays in a subprocess. The real feature builder runs; only the
        # expensive regressor is replaced so the causal feature arrays are visible.
        root = Path(__file__).resolve().parents[1]
        program = '''
import numpy as np
from unittest.mock import patch
import predict_context as p
from tests.test_data_policy import curve, START, HOUR
history = [curve(226+i, START+i*48*HOUR) for i in range(20)]
target = curve(246, START+30*48*HOUR)
peer = curve(246, target.start, 1000)
old = curve(225, START-48*HOUR)
old.scores *= 1000000
arrays = []
builders = []
class ObservedBuilder(p.ContextBuilder):
    def __init__(self, curves):
        super().__init__(curves)
        builders.append(self)
def inspect_fit(kind, values, train, test):
    arrays.append(values)
    return np.zeros(len(test)), {}
with patch.object(p, 'fit_context', side_effect=inspect_fit), patch.object(p, 'ContextBuilder', ObservedBuilder):
    p.forecast(target, history, target.start+12*HOUR, peer)
    p.forecast(target, [old]+history, target.start+12*HOUR, peer)
np.testing.assert_array_equal(arrays[0]['train_tab'], arrays[1]['train_tab'])
np.testing.assert_array_equal(arrays[0]['test_tab'], arrays[1]['test_tab'])
for builder in builders:
    assert set(builder.metadata.events) == {str(c.event) for c in history+[target]}
    assert builder.metadata.character_ids == sorted({
        c for event in {item.event for item in history}
        for c in builder.metadata.parties[event]['characters']})
'''
        result = subprocess.run([str(root / '.venv/bin/python'), '-c', program],
                                cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_target_from_manual_reporting_period_is_not_a_new_forecast(self):
        from forecasting import predict_cutoff
        old = curve(999, START - HOUR)
        with self.assertRaisesRegex(ValueError, '2024-01-29'):
            predict_cutoff(old, [], old.start + 12 * HOUR)

    def test_context_rejects_an_old_peer_even_with_recent_observations(self):
        root = Path(__file__).resolve().parents[1]
        program = '''
from dataclasses import replace
import predict_context as p
from tests.test_data_policy import curve, START, HOUR
target = curve(246, START+30*48*HOUR)
peer = replace(target, rank=1000, start=START-1)
try:
    p.forecast(target, [], target.start+12*HOUR, peer)
except ValueError as exc:
    assert '2024-01-29' in str(exc), str(exc)
else:
    raise AssertionError('Old peer entered the production model boundary')
'''
        result = subprocess.run([str(root / '.venv/bin/python'), '-c', program],
                                cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
