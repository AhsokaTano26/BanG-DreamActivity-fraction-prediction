"""Versioned weight releases for the fixed production recipe, separate from images."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time

from analysis.data_policy import MIN_EVENT_START_MS, POLICY_ID
from analysis.holdout_dataset import sha256
from production.model import _identity, verify_pack
from production.runtime import _activate, cycle_lock, validate_root
from production.selected_model import METHOD, RECIPE_ID
from production.state import Ledger, atomic_json

MAX_BYTES = 1024 ** 3
MAX_FILES = 10000
FEEDBACK_FIELDS = ('event', 'rank', 'phase', 'start_ms', 'end_ms', 'mean_log_error',
                   'cases', 'actual_final', 'origin', 'source')


def validate_feedback(rows, as_of_ms):
    result, seen = [], set()
    for supplied in rows:
        row = {key: supplied.get(key) for key in FEEDBACK_FIELDS}
        for key in ('event', 'rank', 'phase', 'start_ms', 'end_ms', 'cases'):
            value = row[key]
            try:
                valid = not isinstance(value, bool) and int(value) == value
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError('Invalid feedback integer')
            row[key] = int(value)
        for key in ('mean_log_error', 'actual_final'):
            row[key] = float(row[key])
        key = (row['event'], row['rank'], row['phase'])
        if (key in seen or row['event'] <= 0 or row['rank'] <= 0 or row['phase'] not in (0, 1, 2)
                or not MIN_EVENT_START_MS <= row['start_ms'] < row['end_ms'] < as_of_ms
                or row['cases'] < 1 or not math.isfinite(row['mean_log_error'])
                or not math.isfinite(row['actual_final']) or row['actual_final'] <= 0
                or row['origin'] not in ('live', 'retrospective_v3_seed')
                or not (row['source'] is None or isinstance(row['source'], str))):
            raise ValueError('Invalid, duplicate or noncausal feedback row')
        seen.add(key)
        result.append(row)
    return sorted(result, key=lambda row: (row['end_ms'], row['event'], row['rank'], row['phase']))


def build_release(model_path, output, *, feedback_rows=(), created_ms=None):
    output = Path(output).resolve()
    if output.exists() or Path(str(output) + '.sha256').exists():
        raise FileExistsError('Release assets cannot be overwritten')
    manifest = verify_pack(model_path)
    if not manifest.get('production_ready') or manifest['config'].get('fixture'):
        raise ValueError('Only production-trained models may be released')
    now = int(time.time() * 1000) if created_ms is None else int(created_ms)
    feedback = validate_feedback(feedback_rows, now)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.model-release-', dir=output.parent) as tmp:
        stage = Path(tmp)
        shutil.copytree(model_path, stage / 'model')
        verify_pack(stage / 'model')
        atomic_json(stage / 'feedback.json', feedback)
        release = {'schema_version': 1, 'kind': 'fixed_recipe_weights',
                   'forecast_method': METHOD, 'recipe_id': RECIPE_ID, 'policy_id': POLICY_ID,
                   'model_id': manifest['model_id'], 'created_ms': now,
                   'history_max_end_ms': manifest['history_max_end_ms'],
                   'training_events': manifest['training_events'], 'ranks': manifest['ranks'],
                   'software': manifest['software'], 'feedback_groups': len(feedback),
                   'artifacts': {str(p.relative_to(stage)): sha256(p)
                                 for p in sorted(stage.rglob('*')) if p.is_file()}}
        release['release_id'] = _identity(release)
        atomic_json(stage / 'release.json', release)
        archive_path = stage / 'archive.tar.gz'
        with archive_path.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for path in sorted(stage.rglob('*')):
                    if not path.is_file() or path == archive_path:
                        continue
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(stage)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open('rb') as source:
                        archive.addfile(info, source)
        # Exclusive output creation avoids a concurrent publisher replacing a release.
        with archive_path.open('rb') as source, output.open('xb') as destination:
            shutil.copyfileobj(source, destination)
    checksum = sha256(output)
    Path(str(output) + '.sha256').write_text(checksum + '  ' + output.name + '\n')
    return release | {'archive_sha256': checksum, 'archive': str(output)}


@contextmanager
def extracted_release(archive_path):
    """Reject links, duplicate names and escaping paths before writing any member."""
    with tempfile.TemporaryDirectory(prefix='bangdream-release-') as tmp:
        stage = Path(tmp)
        with tarfile.open(archive_path, 'r:gz') as archive:
            members = archive.getmembers()
            names, total = set(), 0
            if len(members) > MAX_FILES:
                raise ValueError('Model archive contains too many files')
            for member in members:
                path = PurePosixPath(member.name)
                if (path.is_absolute() or '..' in path.parts or '\\' in member.name
                        or not path.parts or str(path) != member.name):
                    raise ValueError('Unsafe archive path')
                if member.name in names:
                    raise ValueError('Duplicate archive member')
                names.add(member.name)
                if not member.isfile():
                    raise ValueError('Only regular files allowed; links are forbidden')
                total += member.size
                if member.size < 0 or total > MAX_BYTES:
                    raise ValueError('Model archive exceeds size limit')
            for member in members:
                target = stage / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as destination:
                    shutil.copyfileobj(source, destination)
        try:
            release = json.loads((stage / 'release.json').read_text())
            if (release.get('schema_version') != 1 or release.get('kind') != 'fixed_recipe_weights'
                    or release.get('forecast_method') != METHOD or release.get('recipe_id') != RECIPE_ID
                    or release.get('policy_id') != POLICY_ID
                    or _identity({k: v for k, v in release.items() if k != 'release_id'}) != release['release_id']):
                raise ValueError('Wrong release recipe, policy or fingerprint')
            actual = {str(p.relative_to(stage)) for p in stage.rglob('*')
                      if p.is_file() and p != stage / 'release.json'}
            if actual != set(release['artifacts']):
                raise ValueError('Release artifact set differs')
            for name, digest in release['artifacts'].items():
                if sha256(stage / name) != digest:
                    raise ValueError('Release artifact changed: ' + name)
            manifest = verify_pack(stage / 'model')
            if (not manifest.get('production_ready') or manifest['config'].get('fixture')
                    or manifest['model_id'] != release['model_id']
                    or manifest['history_max_end_ms'] != release['history_max_end_ms']
                    or manifest['training_events'] != release['training_events']
                    or manifest['software'] != release['software'] or manifest['ranks'] != release['ranks']):
                raise ValueError('Model and release identities differ')
            feedback = validate_feedback(json.loads((stage / 'feedback.json').read_text()), release['created_ms'])
            if len(feedback) != release['feedback_groups']:
                raise ValueError('Feedback group count differs')
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            raise ValueError('Invalid model release') from exc
        yield stage, release, feedback


def inspect_release(archive_path):
    with extracted_release(archive_path) as (_, release, _):
        return release


def install_release(archive_path, root):
    root = validate_root(root)
    # All verification completes before changing the active pointer or ledger.
    with extracted_release(archive_path) as (stage, release, feedback):
        with cycle_lock(root) as acquired:
            if not acquired:
                raise RuntimeError('Prediction/training cycle is busy; retry installation later')
            versions = root / 'models/versions'
            versions.mkdir(parents=True, exist_ok=True)
            model_id = release['model_id']
            if not isinstance(model_id, str) or len(model_id) != 64 or any(c not in '0123456789abcdef' for c in model_id):
                raise ValueError('Invalid model identifier')
            target = versions / model_id
            if target.exists():
                verify_pack(target)
            else:
                temporary = Path(tempfile.mkdtemp(prefix='.install-', dir=versions))
                try:
                    shutil.copytree(stage / 'model', temporary / 'model')
                    verify_pack(temporary / 'model')
                    (temporary / 'model').rename(target)
                finally:
                    shutil.rmtree(temporary)
            # Same fixed base recipe: imported issued errors keep their origin.
            # Existing local feedback always takes precedence, including late corrections.
            ledger = Ledger(root)
            with ledger.connect() as con:
                for row in feedback:
                    con.execute('INSERT OR IGNORE INTO feedback VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                                tuple(row.get(k) for k in FEEDBACK_FIELDS) + (release['created_ms'],))
            pointer = _activate(root, target, verify_pack(target))
            # Explicit installation supersedes an earlier manual rollback choice.
            # The next inference cycle adopts this installed version. The ledger
            # archives a superseded binding and preserves already issued results.
            (root / 'models/manual_override.json').unlink(missing_ok=True)
            atomic_json(root / 'models/releases' / (release['release_id'] + '.json'), release)
            return {'status': 'installed', 'release_id': release['release_id'], **pointer}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--root', type=Path, default=Path('runtime/live'))
    build.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('archive', type=Path)
    install = commands.add_parser('install')
    install.add_argument('archive', type=Path)
    install.add_argument('--root', type=Path, default=Path('/state'))
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            root = validate_root(args.root)
            with cycle_lock(root) as acquired:
                if not acquired:
                    raise RuntimeError('Cycle is busy; retry after it finishes')
                pointer = json.loads((root / 'models/current.json').read_text())
                now = int(time.time() * 1000)
                result = build_release(pointer['path'], args.output,
                                       feedback_rows=Ledger(root).feedback_rows(before_ms=now), created_ms=now)
        elif args.command == 'verify':
            result = inspect_release(args.archive)
        else:
            result = install_release(args.archive, args.root)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
