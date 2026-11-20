"""Serial, long-lived inference child; one request and one response per cycle."""
import argparse
from contextlib import redirect_stdout
import json
import sys


def run_requests(config_path, *, offline=False, deployment=False):
    from production.model import InferenceCache, inference_session
    from production.runtime import read_config
    if deployment:
        from production.deployment import run_cycle
    else:
        from production.selected_runtime import run_cycle
    cache = InferenceCache()
    for command in sys.stdin:
        try:
            if command.strip() != 'cycle':
                raise ValueError('Unknown worker request')
            # Numerical/model diagnostics must not enter the response stream.
            with redirect_stdout(sys.stderr), inference_session(cache):
                config = read_config(config_path)
                result = run_cycle(config | {'automatic_training': False}, sync=not offline)
            response = {'status': result['status']}
        except Exception as exc:
            response = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
            print(response['error'], file=sys.stderr, flush=True)
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    return run_requests(args.config, offline=args.offline)


if __name__ == '__main__':
    raise SystemExit(main())
