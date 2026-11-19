"""uv run training/train.py: explicit training, evaluation and research commands."""
import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

if __name__ == '__main__':
    from training.cli import main
    raise SystemExit(main())
