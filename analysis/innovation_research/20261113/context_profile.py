"""Training-only metadata sparsity diagnostic; no model fitting or answer access."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
from analysis.event_features import EventFeatures
from analysis.holdout_dataset import HoldoutDataset, sha256
from analysis.temporal_dataset import TemporalDataset


def main(output):
    if output.exists():
        raise FileExistsError(output)
    original = HoldoutDataset()
    data = TemporalDataset(ROOT / 'data/temporal_holdout_v2_t1000_t2000')
    metadata = EventFeatures.load(data.metadata_directory)
    rows = {}
    curves = list(data.train_curves())
    for curve in curves:
        group = metadata.band_only(curve)
        rows[curve.event] = {
            'event': curve.event, 'event_type': curve.kind,
            'band': group['band_group'],
            'duration_days': curve.duration / 86400000,
            'year': pd.Timestamp(curve.start, unit='ms', tz='Asia/Shanghai').year,
        }
    frame = pd.DataFrame(rows.values()).sort_values('event')
    cells = frame.groupby(['event_type', 'band']).size()
    active = set(data.case_index('train').event)
    payload = {
        'scope': 'Training metadata diagnosis only; not a model comparison or causal estimate',
        'v1_dataset_id': original.manifest['dataset_id'],
        'v2_dataset_id': data.manifest['dataset_id'],
        'training_events_with_curves': len(frame), 'training_curves': len(curves),
        'training_events_with_prefixes': len(active),
        'events_without_grid_prefixes': sorted(set(rows) - active),
        'type_counts': frame.event_type.value_counts().to_dict(),
        'band_counts': frame.band.value_counts().to_dict(),
        'type_band_cells': len(cells),
        'cells_with_at_most_5_events': int((cells <= 5).sum()),
        'type_band_counts': {'|'.join(k): int(v) for k, v in cells.items()},
        'year_counts': frame.year.value_counts().sort_index().to_dict(),
        'duration_days_range': [float(frame.duration_days.min()), float(frame.duration_days.max())],
        'sources': {str(path.relative_to(ROOT)): sha256(path) for path in [
            Path(__file__).resolve(), ROOT / 'analysis/event_features.py',
            ROOT / 'analysis/temporal_dataset.py', ROOT / 'analysis/multirank_dataset.py',
            ROOT / 'analysis/holdout_dataset.py']},
        'limitations': [
            'Frozen test cohort was previously exposed in exploratory experiments.',
            'Metadata is retrospective, without point-in-time archives.',
            'Type/band frequencies do not identify causal effects on scores.',
        ],
    }
    output.mkdir(parents=True)
    frame.to_csv(output / 'training_event_context.csv', index=False)
    (output / 'profile.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: payload[k] for k in [
        'training_events_with_curves', 'training_events_with_prefixes',
        'type_band_cells', 'cells_with_at_most_5_events']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'context')
    main(parser.parse_args().output)
