"""Describe sampling density from verified TRAIN curves only; never reads test labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.holdout_dataset import HoldoutDataset, sha256

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
if (OUT / 'summary.json').exists():
    raise FileExistsError('Preserve this diagnostic; use a new output directory.')
data = HoldoutDataset(ROOT / 'data/holdout_v1')
assert data.manifest['dataset_id'] == 'd0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb'
rows = []
for curve in data.train_curves():
    times = np.unique(curve.times[(curve.times >= curve.start) & (curve.times <= curve.end)])
    assert len(times) > 1
    rows.append({
        'event': curve.event, 'rank': curve.rank, 'start_ms': curve.start,
        'start_cn': pd.Timestamp(curve.start, unit='ms', tz='UTC').tz_convert('Asia/Shanghai').isoformat(),
        'first_observation_cn': pd.Timestamp(times[0], unit='ms', tz='UTC').tz_convert('Asia/Shanghai').isoformat(),
        'observations_within_activity': len(times),
        'median_gap_minutes': float(np.median(np.diff(times)) / 60000),
        'observations_per_day': len(times) / ((curve.end - curve.start) / 86400000),
    })
frame = pd.DataFrame(rows).sort_values(['start_ms', 'rank'])
assert len(frame) == 330 and frame.event.nunique() == 198
boundary = int(frame.loc[frame.event == 226, 'start_ms'].iloc[0])
periods = {}
for name, part in [('before_activity_226', frame[frame.start_ms < boundary]),
                   ('activity_226_onward_in_training', frame[frame.start_ms >= boundary])]:
    periods[name] = {
        'events': int(part.event.nunique()), 'curves': len(part),
        **{column: float(part[column].median()) for column in
           ['observations_within_activity', 'median_gap_minutes', 'observations_per_day']},
    }
assert periods['activity_226_onward_in_training']['events'] == 12
assert np.allclose(frame[frame.start_ms >= boundary].median_gap_minutes, 30, atol=.02)
frame.to_csv(OUT / 'train_curve_density.csv', index=False)
frame[frame.start_ms >= pd.Timestamp('2023-12-01', tz='Asia/Shanghai').value // 1000000].to_csv(
    OUT / 'boundary_details.csv', index=False)
result = {
    'dataset_id': data.manifest['dataset_id'],
    'source': 'Frozen complete TRAIN curves only; no validation/test answers or database snapshots read.',
    'source_sha256': sha256(data.root / 'train_curves.jsonl'),
    'definitions': 'Unique timestamps within start<=time<=end; gap median per curve then median over curves. '
                   'Descriptive boundary inspection, not a formal change-point test or causal analysis.',
    'boundary_activity': 226, 'boundary_activity_start_cn': '2024-01-29T13:00:00+08:00',
    'first_saved_observation_in_boundary_activity_cn': frame.loc[frame.event == 226, 'first_observation_cn'].iloc[0],
    'periods': periods,
    'limits': 'Only eligible frozen training curves, ending May2024. Does not establish exact collector rollout time '
              'or why sampling changed. Earlier sparse observations may miss nonlinear growth. '
              'Existing test exposure and retrospective metadata limitations remain unchanged.',
}
(OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
