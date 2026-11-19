import os
from pathlib import Path
import subprocess
import sys
import unittest

class DenseCoreReplayTests(unittest.TestCase):
    def test_cold_start_templates_are_unavailable_without_dropping_rows(self):
        code='''
from types import SimpleNamespace
import numpy as np
from analysis.dense_core_replay import training_baselines
sample={'case_id':'cold','curve':SimpleNamespace(event=226,start=100,rank=1000),'score':100.,'progress':.5,'features':{'rate_24h_relative':1.,'elapsed_h':10.,'remaining_h':10.}}
values, unavailable=training_baselines([sample],[])
assert set(values)=={'linear','recent_24h'}
assert values['linear'].tolist()==[200.]
assert values['recent_24h'].tolist()==[200.]
assert unavailable==['cold']
'''
        env=os.environ.copy();env['DYLD_LIBRARY_PATH']='/opt/homebrew/opt/libomp/lib'
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
