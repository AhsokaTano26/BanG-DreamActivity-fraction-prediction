"""uv run app.py: serve forecasts from existing model weights."""
import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

if __name__ == '__main__':
    from production.app import main
    raise SystemExit(main())
