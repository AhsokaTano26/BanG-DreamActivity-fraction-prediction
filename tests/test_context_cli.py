"""Check the usable metadata-aware predictor without importing torch here."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ContextCommandTests(unittest.TestCase):
    def test_metadata_aware_replay_outputs_a_readonly_point_forecast(self):
        command=ROOT/'training/legacy/predict_context.py'
        self.assertTrue(command.exists(),'metadata-aware prediction entry is missing')
        database=ROOT/'data/db.sqlite3';before=hashlib.sha256(database.read_bytes()).hexdigest()
        result=subprocess.run([str(ROOT/'.venv/bin/python'),str(command),'--event','327','--rank','2000',
            '--as-of','2026-11-02T16:59:59+08:00','--offline','--json'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        value=json.loads(result.stdout)
        self.assertEqual(value['model'],'catboost_context')
        self.assertTrue(value['event_bands'])
        self.assertGreaterEqual(value['predicted_final_ep'],value['current_ep'])
        self.assertIsNone(value['historical_error_range'])
        self.assertEqual(before,hashlib.sha256(database.read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
