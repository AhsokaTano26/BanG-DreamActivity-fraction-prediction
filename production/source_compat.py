"""Explicit, complete source profiles for reviewed layout-only model compatibility."""
import json
from pathlib import Path

REGISTRY = Path(__file__).with_name('layout-compatibility.json')


def sources_compatible(recorded, current, *, registry=REGISTRY):
    if recorded == current:
        return True
    try:
        content = json.loads(Path(registry).read_text())
        return content['schema'] == 1 and any(
            profile['original_sources'] == recorded and profile['relocated_sources'] == current
            for profile in content['profiles'])
    except (OSError, ValueError, KeyError, TypeError):
        return False
