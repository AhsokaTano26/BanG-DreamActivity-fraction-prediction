"""Replay sealed validation-selected models on this archived live prefix only."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import json
from datetime import datetime, timezone
import copy
import joblib
import numpy as np
import pandas as pd
from analysis import structured_neural_benchmark as nn
from analysis import holdout_benchmark as base
from analysis import relationship_benchmark as relation
from analysis.holdout_dataset import HoldoutDataset, sha256, write_json
from analysis.event_features import EventFeatures
from analysis.holdout_features import HoldoutFeatureBuilder
from analysis.relationship_features import relationship_frame, select_features
from analysis.backtest import snapshot, TZ
from forecasting import curve_from_points
import torch

HERE = Path(__file__).resolve().parent
RUNS = {key: ROOT/'analysis/holdout_results'/key for key in ['v1', 'relationships_v3', 'structured_neural_v1']}
CONFIG = {key: json.loads((path/'run_manifest.json').read_text()) for key, path in RUNS.items()}
SELECTION = {key: json.loads((path/'selection.json').read_text()) for key, path in RUNS.items()}


def verify(data):
    audit = {}
    for key, path in RUNS.items():
        config = CONFIG[key]
        assert config['dataset_id'] == data.manifest['dataset_id']
        relation.verify_software(config['software'])
        # The old scorer was revised after v1; it is not used in live inference.
        skipped = []
        for source, digest in config['source_hashes'].items():
            if source == 'analysis/holdout_evaluate.py':
                if sha256(ROOT/source) != digest: skipped.append(source)
                continue
            assert sha256(ROOT/source) == digest, ('Changed inference source', source)
        seal = json.loads((path/'blind_test_manifest.json').read_text())
        for name, digest in seal['artifacts'].items():
            assert sha256(path/name) == digest, ('Changed sealed artifact', key, name)
        audit[key] = {'run_id': config['run_id'], 'manifest_sha256': sha256(path/'blind_test_manifest.json'),
                      'selection': SELECTION[key]['selected_recipe'], 'unused_changed_scorer': skipped}
    nn.verify_config(CONFIG['structured_neural_v1'], data)
    return audit


def predict(samples, metadata, data):
    device = CONFIG['structured_neural_v1']['device']
    builder = HoldoutFeatureBuilder(data.train_curves(), metadata)
    sequence = np.stack([base.prefix_sequence(s) for s in samples])
    offset = -np.log(np.array([s['progress'] for s in samples]))
    tabs = {mode: joblib.load(RUNS['v1']/f'models/preprocessor_{mode}.joblib').transform(builder.frame(samples, mode)).astype(np.float32)
            for mode in ['baseline', 'peer']}
    old = {}
    for name in SELECTION['v1']['top3_members']:
        if name.endswith('_ensemble3'):
            family = name.removesuffix('_ensemble3'); mode, _ = base.NEURAL_RECIPES[family]
            old[name] = np.mean([base.final_predictions(samples, base.neural_predict(
                RUNS['v1']/f'models/{family}_seed{seed}.pt', tabs[mode], sequence, device, 256))
                for seed in CONFIG['v1']['seeds']], axis=0)
        else:
            mode, _ = base.TREE_RECIPES[name]
            old[name] = base.final_predictions(samples, base.tree_predict(joblib.load(RUNS['v1']/f'models/{name}.joblib'), tabs[mode], sequence, offset))
    assert SELECTION['v1']['selected_recipe'] == 'validation_top3_ensemble'
    old_selected = np.mean(list(old.values()), axis=0)
    frame = relationship_frame(samples, metadata)
    tree = {name: relation.predict_bundle(joblib.load(RUNS['relationships_v3']/f'models/{name}.joblib'), frame, samples)
            for name in SELECTION['relationships_v3']['top3_members']}
    weight = float(SELECTION['relationships_v3']['selected_recipe'].removeprefix('blend_new_'))
    reference = weight*np.mean(list(tree.values()), axis=0)+(1-weight)*old_selected
    config = CONFIG['structured_neural_v1']; selected = SELECTION['structured_neural_v1']
    compact = select_features(frame, 'invariant')
    inputs = nn.tensor_arrays(nn.arrays(samples, compact, joblib.load(RUNS['structured_neural_v1']/'preprocessor.joblib')), device)
    neural = {}
    for ensemble in selected['top3_members']:
        name = ensemble.removesuffix('_ensemble3')
        neural[ensemble] = np.mean([relation.decode(samples, nn.infer(nn.load_model(
            RUNS['structured_neural_v1']/f'models/{name}_{seed}.pt', config), inputs), 'remaining')
            for seed in config['seeds']], axis=0)
    neural_mean = np.mean(list(neural.values()), axis=0)
    weight = float(selected['selected_recipe'].removeprefix('blend_neural_'))
    final = weight*neural_mean+(1-weight)*reference
    return {'v1': old_selected, 'relationships_v3': reference, 'structured_neural_v1': final,
            'neural_top3': neural_mean, **neural, **tree}


def main():
    if (HERE/'prediction.json').exists(): raise FileExistsError('Keep the archived forecast unchanged')
    torch.set_num_threads(4); data = HoldoutDataset(); audit = verify(data)
    # No label access is required for replay or live inference.
    data.labels = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('No labels in live inference'))
    metadata = EventFeatures.load(data.metadata_directory)
    validation = data.inputs('validation'); replay = predict(validation, metadata, data); checks = {}
    for key, path in RUNS.items():
        expected = pd.read_csv(path/'validation_predictions.csv')
        expected = expected[expected.method == SELECTION[key]['selected_recipe']].set_index('case_id')
        expected = expected.loc[[s['case_id'] for s in validation], 'predicted'].to_numpy()
        assert np.allclose(expected, replay[key], rtol=3e-6, atol=.1), ('Replay mismatch', key)
        checks[key] = {'cases': len(validation), 'max_relative_difference': float(np.max(np.abs(replay[key]/expected-1)))}
    fetch = json.loads((HERE/'fetch_manifest.json').read_text()); as_of = int(datetime.fromisoformat(fetch['requested_at']).timestamp()*1000)
    events = json.loads((HERE/'events.json').read_text())
    eligible = []
    for event, info in events.items():
        try: start, end = int(info['startAt'][3]), int(info['endAt'][3])
        except (KeyError, IndexError, TypeError, ValueError): continue
        if start <= as_of: eligible.append((start, int(event), end))
    _, event, end = max(eligible)
    assert event == 328 and as_of < end, 'The archived latest event must be ongoing'
    info = json.loads((HERE/f'event_{event}.json').read_text())
    start = int(info['startAt'][3]); end = int(info['endAt'][3])
    # Preserve the frozen historical metadata and training vocabulary; update the current event only.
    current_metadata = copy.deepcopy(metadata.events); current_metadata[str(event)] = info
    live_metadata = EventFeatures(current_metadata, metadata.characters, metadata.bands, metadata.cards)
    assert live_metadata.band_ids == metadata.band_ids and live_metadata.character_ids == metadata.character_ids
    curves = {}
    for rank in [1000, 2000]:
        payload = json.loads((HERE/f'rank_{rank}.json').read_text()); assert payload['result'] is True
        curve = curve_from_points(event, rank, info['eventType'], start, end, payload['cutoffs'])
        assert not (curve.times > as_of).any(), 'Live data contain future observations'
        curves[rank] = curve
    samples = []
    for rank in [1000, 2000]:
        sample = snapshot(curves[rank], as_of)
        if sample is None: raise ValueError('Prefix is stale or too short')
        sample.update(case_id=f'live_{event}_{rank}_{as_of}', stage='live', peer_curve=curves[3000-rank])
        samples.append(sample)
    values = predict(samples, live_metadata, data)
    rows = []
    for i, sample in enumerate(samples):
        predicted = float(values['structured_neural_v1'][i])
        assert np.isfinite(predicted) and predicted >= sample['score']
        rows.append({'rank': sample['curve'].rank, 'observed_at': datetime.fromtimestamp(sample['now']/1000, TZ).isoformat(),
                     'current_ep': int(sample['score']), 'progress': float(sample['progress']),
                     'hours_to_end_from_observation': (end-sample['now'])/3600000,
                     'predicted_final_ep': int(round(predicted)),
                     'predicted_remaining_ep': int(round(predicted-sample['score'])),
                     'components_ep': {name: float(v[i]) for name, v in values.items()}})
    result = {'event_id': event, 'event_name': info['eventName'][3], 'event_type': info['eventType'],
              'bands': list(live_metadata.describe(event)['band_names'].values()),
              'as_of': datetime.fromtimestamp(as_of/1000, TZ).isoformat(),
              'event_start': datetime.fromtimestamp(start/1000, TZ).isoformat(),
              'event_end': datetime.fromtimestamp(end/1000, TZ).isoformat(),
              'model': SELECTION['structured_neural_v1']['selected_recipe'], 'model_retrained': False,
              'predictions': rows, 'validation_replay': checks, 'models_verified': audit,
              'dataset_id': data.manifest['dataset_id'], 'sources': fetch['sources'],
              'notes': ['This is a forward prediction, not a fixed-test accuracy estimate.',
                        'No calibrated prediction interval is available for this mixture.',
                        'Model evaluation reused previously explored test events and retrospective metadata.']}
    write_json(HERE/'prediction.json', result)
    write_json(HERE/'prediction_manifest.json', {'artifacts': {p.name: sha256(p) for p in HERE.iterdir() if p.is_file()},
               'source_hashes': {n: sha256(ROOT/n) for n in ['predict.py', 'forecasting.py']},
               'source_database_sha256': sha256(ROOT/'data/db.sqlite3')})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
