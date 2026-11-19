"""Launch the uv-managed framework with bounded numerical threads and safe imports."""
import os
from pathlib import Path
import runpy
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import xgboost  # third-party library, resolved with Python's -P safe path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == 'model':
        sys.argv = ['model_release', *args[1:]]
        runpy.run_module('production.model_release', run_name='__main__')
    elif args and args[0] == 'test':
        sys.argv = ['unittest', 'discover', '-s', '/app/tests', '-p', 'test_deployment*.py', '-v']
        runpy.run_module('unittest', run_name='__main__')
    else:
        sys.argv = ['deployment', *(args or ['serve'])]
        runpy.run_module('production.deployment', run_name='__main__')
