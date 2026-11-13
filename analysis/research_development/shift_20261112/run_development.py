"""One fixed train/validation pass; does not load test inputs or answers."""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from analysis.holdout_dataset import sha256
from analysis.research_shift import fit_recipe, predict_recipe
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_evaluate import score_temporal
from analysis.temporal_paired_features import prepare_paired

run = Path('analysis/research_development/shift_20261112')
manifest = json.loads((run/'predeclared_manifest.json').read_text())
for name, digest in manifest['source_hashes'].items():
    if sha256(Path(name)) != digest:
        raise ValueError('Source changed after predeclaration: '+name)
data = TemporalDataset(manifest['dataset_directory'])
if data.manifest['dataset_id'] != manifest['dataset_id']:
    raise ValueError('Dataset fingerprint differs')
arrays = {split: prepare_paired(data.inputs(split)) for split in ('train', 'validation')}
labels = {split: data.labels(split) for split in arrays}
indexes = {split: data.case_index(split) for split in arrays}
members = {split: data.memberships(split) for split in arrays}
for split, values in arrays.items():
    joblib.dump(values, run/f'{split}_arrays.joblib')
    values['audit'].to_csv(run/f'{split}_prefix_audit.csv', index=False)
all_metrics, summary = {}, []
for config in manifest['recipes']:
    started = time.monotonic()
    name = config['name']
    print(json.dumps({'starting': name}), flush=True)
    bundle = fit_recipe(config, arrays['train'], labels['train'])
    joblib.dump(bundle, run/'models'/f'{name}.joblib')
    pd.DataFrame(bundle['training_weights']).to_csv(run/f'{name}_training_weights.csv', index=False)
    (run/f'{name}_oof_audit.json').write_text(json.dumps(bundle['oof_audit'], indent=2)+'\n')
    score = {}
    for split in arrays:
        prediction = predict_recipe(bundle, arrays[split])
        frame = pd.DataFrame({'case_id': [c['case_id'] for c in arrays[split]['cases']],
            'method': name, 'predicted': prediction})
        frame.to_csv(run/f'{name}_{split}_predictions.csv', index=False)
        score[split] = score_temporal(indexes[split], members[split], labels[split], frame)
        summary.append({'method': name, 'split': split, **score[split]['overall']})
    score['fit_elapsed_s'] = time.monotonic()-started
    all_metrics[name] = score
    (run/f'{name}_metrics.json').write_text(json.dumps(score, indent=2)+'\n')
    print(json.dumps({'method': name, 'elapsed_s': score['fit_elapsed_s'],
        'train': score['train']['overall'], 'validation': score['validation']['overall']}), flush=True)
(run/'all_metrics.json').write_text(json.dumps(all_metrics, indent=2)+'\n')
pd.DataFrame(summary).to_csv(run/'summary.csv', index=False)
selected = min(all_metrics, key=lambda n: all_metrics[n]['validation']['overall']['event_macro_mape_pct'])
(run/'development_selection.json').write_text(json.dumps({'selected_recipe': selected,
    'criterion': 'validation event-macro MAPE', 'all_recipes_reported': True,
    'test_labels_read': False, 'validation_used_for_fitting': False}, indent=2)+'\n')
artifacts = {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*')) if p.is_file()}
(run/'development_manifest.json').write_text(json.dumps({'dataset_id': manifest['dataset_id'],
    'predeclared_manifest_sha256': sha256(run/'predeclared_manifest.json'),
    'test_labels_read': False, 'artifacts': artifacts}, indent=2)+'\n')
print(json.dumps({'selected_recipe': selected}), flush=True)
