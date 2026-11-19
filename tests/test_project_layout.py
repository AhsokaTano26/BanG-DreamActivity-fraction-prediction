"""Executable boundaries for uv-managed web inference and explicit training."""
from contextlib import closing, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ProjectLayoutTests(unittest.TestCase):
    def test_benchmark_forwards_dataset_and_training_controls_before_fitting(self):
        from training.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                main(['benchmark', '--protocol', 'v3', '--run', str(Path(tmp) / 'run'),
                      '--device', 'cpu', '--max-epochs', '1', '--dataset', str(Path(tmp) / 'absent')])

    def test_raw_training_history_clips_postevent_tail_without_rewriting_database(self):
        import hashlib
        import sqlite3
        from analysis.data_policy import MIN_EVENT_START_MS
        from training.cli import load_histories
        start = MIN_EVENT_START_MS
        end = start + 48 * 3600000
        points = [{'time': start + index * 3600000, 'ep': index + 1} for index in range(49)]
        points.append({'time': end + 3600000, 'ep': 9999})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'raw.sqlite3'
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('CREATE TABLE event (EventID INTEGER, EventType TEXT, '
                           'StartAt INTEGER, EndAt INTEGER, Rank INTEGER, Country TEXT, PointRank TEXT)')
                db.execute('INSERT INTO event VALUES (?,?,?,?,?,?,?)',
                           (2, 'normal', start, end, 1000, '中国大陆', str(points)))
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            curves = load_histories(path, end + 2 * 3600000, [1000])
            self.assertEqual(curves[0].event, 2)
            self.assertEqual(int(curves[0].times[-1]), end)
            self.assertEqual(curves[0].final, 49)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_training_source_directory_cannot_be_used_as_runtime_storage(self):
        from production.runtime import validate_root
        from training.paths import ROOT
        with self.assertRaises(ValueError):
            validate_root(ROOT / 'training')

    def test_default_web_entry_disables_fitting_even_for_existing_training_config(self):
        from production.app import main
        from production.deployment import configure
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'state'
            configure(root, auto_train=True)
            with patch('production.app.serve', return_value=0) as web:
                result = main(['--state', str(root), '--offline'])
            self.assertEqual(result, 0)
            self.assertFalse(web.call_args.args[0]['automatic_training'])
            self.assertTrue(web.call_args.kwargs['offline'])

    def test_web_entry_does_not_allow_an_automatic_training_flag(self):
        from production.app import main
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['--auto-train'])
        self.assertEqual(error.exception.code, 2)

    def test_registered_legacy_source_profile_requires_both_complete_hash_sets(self):
        from production.source_compat import sources_compatible
        old = {'training/analysis/example.py': 'a' * 64}
        current = {'training/analysis/example.py': 'b' * 64}
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / 'profiles.json'
            registry.write_text(json.dumps({'schema': 1, 'profiles': [
                {'original_sources': old, 'relocated_sources': current}]}))
            self.assertTrue(sources_compatible(old, current, registry=registry))
            self.assertFalse(sources_compatible(old, current | {'extra.py': 'c' * 64}, registry=registry))
            self.assertFalse(sources_compatible(old | {'extra.py': 'd' * 64}, current, registry=registry))
            self.assertFalse(sources_compatible(old, {'training/analysis/example.py': 'e' * 64}, registry=registry))

    def test_unregistered_source_fingerprint_is_rejected(self):
        from production.source_compat import sources_compatible
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / 'profiles.json'
            registry.write_text(json.dumps({'schema': 1, 'profiles': []}))
            self.assertFalse(sources_compatible({'old.py': 'a' * 64}, {'new.py': 'b' * 64}, registry=registry))

    def test_training_entry_fits_an_immutable_fixture_without_promoting_it(self):
        from training.cli import main
        from test_production_model import model_fixture
        from production.model import verify_pack
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curves, metadata = model_fixture(root, ranks=(1000, 2000))
            database = root / 'history.sqlite3'
            with closing(sqlite3.connect(database)) as db, db:
                db.execute('CREATE TABLE event (EventID INTEGER, EventType TEXT, '
                           'StartAt INTEGER, EndAt INTEGER, Rank INTEGER, Country TEXT, PointRank TEXT)')
                for curve in curves:
                    points = [{'time': int(t), 'ep': int(round(s))} for t, s in zip(curve.times, curve.scores)]
                    db.execute('INSERT INTO event VALUES (?,?,?,?,?,?,?)',
                        (curve.event, curve.kind, curve.start, curve.end, curve.rank, '中国大陆', str(points)))
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(['fit', '--database', str(database), '--metadata', str(metadata),
                    '--before', str(max(c.end for c in curves) + 3600000),
                    '--output', str(root / 'versions'), '--rank', '1000', '--rank', '2000',
                    '--fixture-iterations', '2'])
            self.assertEqual(result, 0)
            report = json.loads(output.getvalue())
            manifest = verify_pack(report['model_path'])
            self.assertFalse(manifest['production_ready'])
            self.assertEqual(manifest['training_events'], 3)
            self.assertFalse((root / 'current.json').exists())


if __name__ == '__main__':
    unittest.main()
