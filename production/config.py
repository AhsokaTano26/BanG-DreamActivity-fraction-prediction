"""Runtime paths and locking without importing the numerical/model stack."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from download_ranks import RANKS

ROOT = Path(__file__).resolve().parents[1]


def validate_root(root):
    path = Path(root).expanduser().resolve()
    if path == ROOT or path in ROOT.parents or any(
            path == ROOT / d or ROOT / d in path.parents
            for d in ('data', 'analysis', 'training', 'production', 'tests', 'docs')):
        raise ValueError('Runtime storage cannot replace source or frozen research directories')
    return path


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    validate_root(cfg['root'])
    if not cfg['ranks'] or len(set(cfg['ranks'])) != len(cfg['ranks']) or not set(cfg['ranks']) <= set(RANKS):
        raise ValueError('Invalid rank configuration')
    if cfg['interval_seconds'] < 60:
        raise ValueError('Invalid schedule interval')
    return cfg


@contextmanager
def cycle_lock(root):
    root = validate_root(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'cycle.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid()))
            stream.flush()
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
