"""Web-only deployment of the fixed recipe; image and model releases are independent."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time

# Apply before importing numerical libraries, including when invoked via Docker.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

from download_ranks import RANKS
from production.runtime import cycle_lock, read_config, validate_root
from production.state import Ledger, atomic_json
from production.web import CycleWorker, serve


def configure(root, *, bootstrap=None, auto_train=None, interval=None):
    root = validate_root(root)
    path = root / 'config.json'
    old = json.loads(path.read_text()) if path.is_file() else {}
    interval = old.get('interval_seconds', 900) if interval is None else int(interval)
    if interval < 60:
        raise ValueError('Update interval must be at least 60 seconds')
    bootstrap = Path(bootstrap or '/bootstrap').resolve()
    config = old | {'schema': 1, 'root': str(root), 'ranks': list(RANKS),
                    'interval_seconds': interval, 'forecast_method': 'legacy_pair',
                    'bootstrap_database': str(bootstrap / 'db.sqlite3'),
                    'bootstrap_metadata': str(bootstrap / 'metadata'),
                    'seed_run': None, 'selected_seed_run': None,
                    'feedback_grace_ms': 600000,
                    'automatic_training': old.get('automatic_training', False) if auto_train is None else bool(auto_train),
                    'require_history_coverage': True}
    # Selection provenance lives in released manifests/feedback; never read a developer path.
    config.pop('selection_manifest', None)
    config.pop('selection_manifest_sha256', None)
    if config != old:
        atomic_json(path, config)
    return config


def relocate_pins(root):
    """Resolve stored identities under this installation, preserving every event snapshot."""
    root = validate_root(root)
    ledger = Ledger(root)
    with ledger.connect() as con:
        for row in con.execute('SELECT event,model_id FROM pins').fetchall():
            model_id = row['model_id']
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', model_id):
                raise ValueError('Invalid pinned model identifier')
            con.execute('UPDATE pins SET model_path=? WHERE event=?',
                        (str(root / 'models/versions' / model_id), row['event']))
    for filename in ('current.json', 'manual_override.json'):
        path = root / 'models' / filename
        if path.is_file():
            pointer = json.loads(path.read_text())
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', pointer['model_id']):
                raise ValueError('Invalid model pointer')
            atomic_json(path, pointer | {'path': str(root / 'models/versions' / pointer['model_id'])})


def history_covers_model(model_path, histories):
    expected = {(row['event'], row['rank'])
                for row in json.loads((Path(model_path) / 'curve_support.json').read_text())}
    return expected <= {(curve.event, curve.rank) for curve in histories}


class DeploymentWorker(CycleWorker):
    def cycle_command(self):
        entry = Path(__file__).resolve().parents[1] / 'deploy/entrypoint.py'
        return [sys.executable, '-P', str(entry), 'cycle',
                '--config', str(self.config_path)] + (['--offline'] if self.offline else [])


def run_cycle(config, *, sync=True):
    """Cold start may use released weights without a full local training database."""
    from production.data import Store
    from production.selected_runtime import run_cycle as selected_cycle
    root = validate_root(config['root'])
    with cycle_lock(root) as acquired:
        if not acquired:
            return {'status': 'busy', 'predictions': [], 'as_of_ms': int(time.time() * 1000)}
        ledger = Ledger(root)
        if not ledger.fact('bootstrap_complete') and not Path(config['bootstrap_database']).is_file():
            store = Store(root)
            pointer = root / 'models/current.json'
            model_metadata = None
            if pointer.is_file():
                identity = json.loads(pointer.read_text())['model_id']
                if not re.fullmatch(r'[0-9a-f]{64}', identity):
                    raise ValueError('Invalid bootstrap model identity')
                model_metadata = root / 'models/versions' / identity / 'metadata'
            for name in ('events', 'characters', 'bands', 'cards'):
                target = store.metadata_directory / (name + '.json')
                if not target.exists():
                    source = None if model_metadata is None else model_metadata / target.name
                    if source is not None and source.is_file():
                        shutil.copyfile(source, target)
                    else:
                        atomic_json(target, {})
            ledger.fact('bootstrap_complete', True)
            ledger.fact('deployment_bootstrap', {'origin': 'online_cache',
                'training_history_complete': False, 'note': 'Automatic training requires complete prior model support.'})
    return selected_cycle(config, sync=sync)


def export_state(source, output):
    """SQLite backups and immutable weights; the original installation is untouched."""
    source, output = validate_root(source), Path(output).expanduser().resolve()
    if output.exists() or source == output or source in output.parents:
        raise FileExistsError('Export to a new directory outside the source runtime')
    with cycle_lock(source) as acquired:
        if not acquired:
            raise RuntimeError('Cycle is busy; retry state export later')
        stage = output.with_name('.' + output.name + '.exporting')
        stage.mkdir(parents=True, exist_ok=False)
        try:
            state = stage / 'state'
            state.mkdir()
            for name in ('data.sqlite3', 'state.sqlite3'):
                database = source / name
                if not database.is_file():
                    raise ValueError('Missing production database: ' + name)
                with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as original:
                    with closing(sqlite3.connect(state / name)) as destination:
                        original.backup(destination)
            shutil.copytree(source / 'metadata', state / 'metadata')
            shutil.copytree(source / 'models', stage / 'models',
                            ignore=shutil.ignore_patterns('.training-*', '.install-*'))
            # Model paths are resolved at startup; don't change calibration or issued forecasts.
            atomic_json(stage / 'migration.json', {'schema': 1, 'origin': str(source),
                'created_ms': int(time.time() * 1000), 'container_root': '/state',
                'model_storage': '/state/models', 'contains_private_credentials': False})
            stage.rename(output)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return {'status': 'exported', 'directory': str(output),
            'state': str(output / 'state'), 'models': str(output / 'models')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    web = commands.add_parser('serve')
    web.add_argument('--root', type=Path, default=Path(os.environ.get('STATE_DIR', '/state')))
    web.add_argument('--host', default=os.environ.get('HOST', '0.0.0.0'))
    web.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8765)))
    web.add_argument('--interval', type=int, default=None)
    web.add_argument('--bootstrap', type=Path, default=Path(os.environ.get('BOOTSTRAP_DIR', '/bootstrap')))
    web.add_argument('--auto-train', action=argparse.BooleanOptionalAction, default=None)
    web.add_argument('--offline', action='store_true')
    cycle = commands.add_parser('cycle')
    cycle.add_argument('--config', type=Path, required=True)
    cycle.add_argument('--offline', action='store_true')
    export = commands.add_parser('export-state')
    export.add_argument('--root', type=Path, default=Path('runtime/live'))
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'serve':
            if not 1 <= args.port <= 65535:
                raise ValueError('Invalid port')
            auto_train = args.auto_train
            if auto_train is None and 'AUTO_TRAIN' in os.environ:
                value = os.environ['AUTO_TRAIN'].lower()
                if value not in ('true', 'false', '1', '0'):
                    raise ValueError('AUTO_TRAIN must be true or false')
                auto_train = value in ('true', '1')
            interval = args.interval
            if interval is None and 'UPDATE_INTERVAL' in os.environ:
                interval = int(os.environ['UPDATE_INTERVAL'])
            with cycle_lock(args.root) as acquired:
                if not acquired:
                    raise RuntimeError('Cycle is busy; retry startup later')
                config = configure(args.root, bootstrap=args.bootstrap, auto_train=auto_train, interval=interval)
                relocate_pins(args.root)
            return serve(config, args.root / 'config.json', host=args.host, port=args.port,
                         offline=args.offline, worker_factory=DeploymentWorker)
        if args.command == 'cycle':
            result = run_cycle(read_config(args.config), sync=not args.offline)
        else:
            result = export_state(args.root, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
