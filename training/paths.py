"""Repository locations and import-source resolution after the training split."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module_source(name):
    relative = Path(*name.split('.'))
    if name == 'analysis' or name.startswith('analysis.'):
        bases = [ROOT / 'training' / relative]
    else:
        bases = [ROOT / relative, ROOT / 'training/legacy' / relative]
    for base in bases:
        for candidate in [base.with_suffix('.py'), base / '__init__.py']:
            # Match Python module names exactly even on case-insensitive macOS volumes.
            if candidate.is_file() and candidate.name in {path.name for path in candidate.parent.iterdir()}:
                return candidate.resolve()
    return None
