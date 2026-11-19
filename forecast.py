"""Manual entry: forecast.py --rank 1000, or forecast.py serve."""
import os

# Set limits before importing numerical libraries. Run politely on a laptop.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
if __name__ == '__main__':
    try:
        os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    except (AttributeError, OSError):
        pass
    from production.interactive import main
    raise SystemExit(main())
