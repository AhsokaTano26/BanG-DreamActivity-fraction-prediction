"""Preregistered paired neural study: develop, seal blind predictions, score separately."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import numpy as np
import pandas as pd

# Load the existing numeric/tree libraries before torch on macOS.
from analysis.relationship_benchmark import (
    prep_for, weights, target_values, decode, rows as prediction_rows, verify_software,
)
from analysis.holdout_dataset import ROOT, HoldoutDataset, sha256, write_json
from analysis.event_features import EventFeatures
from analysis.relationship_features import relationship_frame, select_features
from analysis.paired_prefix import paired_sequence, thin_sample
import torch
from torch.nn import functional as F
from analysis.structured_neural_models import StructuredForecaster

SEEDS = [42, 123, 2026]
RECIPES = {name: {'architecture': name, 'loss': 'log_rate', 'thin': False} for name in
           ['shared_mlp', 'rank_mlp', 'dual_gru', 'dual_tcn', 'multiscale_tcn', 'cross_patch', 'gated_tcn']}
RECIPES.update(dual_tcn_metric={'architecture': 'dual_tcn', 'loss': 'relative_final', 'thin': False},
               dual_tcn_thin={'architecture': 'dual_tcn', 'loss': 'log_rate', 'thin': True})
SOURCE_NAMES = ['training/analysis/structured_neural_benchmark.py', 'training/analysis/structured_neural_models.py',
                'training/analysis/paired_prefix.py', 'training/analysis/relationship_features.py',
                'training/analysis/relationship_benchmark.py', 'training/analysis/holdout_dataset.py',
                'training/analysis/backtest.py', 'training/analysis/event_features.py', 'training/analysis/holdout_evaluate.py']


def reference_identity(root):
    seal = json.loads((root/'blind_test_manifest.json').read_text())
    names = ['selection.json', 'validation_predictions.csv', 'test_predictions.csv']
    for name in names:
        if sha256(root/name) != seal['artifacts'][name]:
            raise ValueError('Reference seal mismatch: '+name)
    return {'directory': str(root.resolve()), 'dataset_id': seal['dataset_id'], 'run_id': seal['run_id'],
            'manifest_sha256': sha256(root/'blind_test_manifest.json'),
            'hashes': {name: sha256(root/name) for name in names},
            'method': json.loads((root/'selection.json').read_text())['selected_recipe']}


def reference_prediction(identity, split, samples):
    path = Path(identity['directory'])/f'{split}_predictions.csv'
    if sha256(path) != identity['hashes'][path.name]: raise ValueError('Reference predictions changed')
    frame = pd.read_csv(path)
    frame = frame[frame.method == identity['method']].set_index('case_id')
    ids = [s['case_id'] for s in samples]
    if not frame.index.is_unique or set(frame.index) != set(ids): raise ValueError('Reference cases differ')
    return frame.loc[ids, 'predicted'].to_numpy()


def config_for(args, data):
    config = {'dataset_id': data.manifest['dataset_id'], 'dataset_directory': str(data.root.resolve()),
              'recipes': RECIPES, 'seeds': SEEDS, 'device': args.device, 'epochs': args.epochs,
              'batch_size': args.batch_size, 'width': 48, 'dropout': .2,
              'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .005, 'gradient_clip': 1},
              'early_stop': {'minimum_epochs': 30, 'patience': 20, 'metric': 'validation event-macro MAPE'},
              'loss': 'Huber(log-rate,beta=.2); metric variant .8*sqrt(relative_final_error^2+1e-4)+.2*Huber',
              'thinning': '50% training rows per epoch use max(4,n//4) evenly spaced observations, keep first/last',
              'weights': 'equal total weight per event across both ranks; global weights unchanged in batches',
              'target': 'remaining log-rate clipped [-6,4]; decode current*(1+exp(log-rate)*(1-p)/p)',
              'input': '39 compact invariant features, train-only median/standardization/one-hot; full and last48h paired grids 64x8',
              'selection': 'validation only: each seed mean, top3 neural means, 25/50/75% top3 vs reference, reference; no single seed selection',
              'reference': reference_identity(args.reference),
              'source_hashes': {n: sha256(ROOT/n) for n in SOURCE_NAMES},
              'software': {n: importlib.metadata.version(n) for n in
                           ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'lightgbm', 'joblib', 'torch']},
              'python': platform.python_version(), 'previous_test_exposure': True,
              'metadata_limitation': 'retrospective metadata without historical snapshots'}
    if config['reference']['dataset_id'] != config['dataset_id']: raise ValueError('Reference dataset differs')
    config['run_id'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config


def verify_config(config, data):
    if config['dataset_id'] != data.manifest['dataset_id']: raise ValueError('Dataset changed')
    if config['python'] != platform.python_version(): raise ValueError('Python version changed')
    verify_software(config['software'])
    for name, digest in config['source_hashes'].items():
        if sha256(ROOT/name) != digest: raise ValueError('Source changed: '+name)
    if reference_identity(Path(config['reference']['directory'])) != config['reference']:
        raise ValueError('Reference identity changed')


def verify_artifacts(run, seal):
    for name, digest in seal['artifacts'].items():
        if sha256(run/name) != digest: raise ValueError('Artifact changed: '+name)


def arrays(samples, frame, prep):
    return {'pair': np.stack([paired_sequence(s) for s in samples]),
            'recent': np.stack([paired_sequence(s, window_hours=48) for s in samples]),
            'tab': np.asarray(prep.transform(frame), dtype=np.float32),
            'rank': np.array([0 if s['curve'].rank == 1000 else 1 for s in samples], dtype=np.int64)}


def tensor_arrays(values, device):
    return {key: torch.as_tensor(value, device=device) for key, value in values.items()}


def forward(model, values):
    return model(values['pair'], values['recent'], values['tab'], values['rank'])


def infer(model, values, batch_size=256):
    model.eval()
    with torch.no_grad():
        output = [forward(model, {k: v[i:i+batch_size] for k, v in values.items()}).cpu().numpy()
                  for i in range(0, len(values['tab']), batch_size)]
    result = np.concatenate(output).astype(float)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite network prediction')
    return result


def event_mape(samples, labels, prediction):
    actual = np.array([labels[s['case_id']] for s in samples])
    _, index, counts = np.unique([s['curve'].event for s in samples], return_inverse=True, return_counts=True)
    return float(np.mean(np.bincount(index, weights=np.abs(prediction/actual-1)*100)/counts))


def train_one(name, recipe, seed, config, train, val, thin, samples, validation, labels, val_labels, run):
    started = time.perf_counter(); torch.manual_seed(seed)
    device = config['device']; width = config['width']; features = train['tab'].shape[1]
    model = StructuredForecaster(recipe['architecture'], features, width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.005)
    y = torch.as_tensor(target_values(samples, labels, 'remaining'), dtype=torch.float32, device=device)
    w = torch.as_tensor(weights(samples), dtype=torch.float32, device=device)
    remaining = torch.as_tensor([(1-s['progress'])/s['progress'] for s in samples], dtype=torch.float32, device=device)
    actual_ratio = torch.as_tensor([labels[s['case_id']]/s['score'] for s in samples], dtype=torch.float32, device=device)
    rng = torch.Generator().manual_seed(seed); best = float('inf'); best_epoch = 0; best_state = None; trace = []
    for epoch in range(1, config['epochs']+1):
        model.train(); order = torch.randperm(len(samples), generator=rng); total_loss = 0.
        for begin in range(0, len(samples), config['batch_size']):
            indices = order[begin:begin+config['batch_size']].to(device)
            batch = {key: value[indices] for key, value in train.items()}
            if recipe['thin']:
                augment = (torch.rand(len(indices), generator=rng) < .5).to(device)
                batch = {key: torch.where(augment.reshape((-1,)+(1,)*(value.ndim-1)), thin[key][indices], value)
                         for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True); output = forward(model, batch)
            loss = F.smooth_l1_loss(output, y[indices], beta=.2, reduction='none')
            if recipe['loss'] == 'relative_final':
                predicted_ratio = 1+remaining[indices]*output.clamp(-6, 4).exp()
                relative = predicted_ratio/actual_ratio[indices]-1
                loss = .8*torch.sqrt(relative.square()+1e-4)+.2*loss
            loss = (loss*w[indices]).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            total_loss += loss.item()*len(indices)
        prediction = decode(validation, infer(model, val), 'remaining')
        score = event_mape(validation, val_labels, prediction)
        trace.append({'epoch': epoch, 'loss': total_loss/len(samples), 'validation_mape': score})
        if score < best:
            best = score; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({'method': name, 'seed': seed, 'epoch': epoch, 'best_epoch': best_epoch, 'validation_mape': best}), flush=True)
        if epoch >= 30 and epoch-best_epoch >= 20: break
    path = run/'models'/f'{name}_{seed}.pt'
    torch.save({'state': best_state, 'architecture': recipe['architecture'], 'features': features, 'width': width,
                'run_id': config['run_id'], 'seed': seed}, path)
    model.load_state_dict(best_state)
    original = decode(validation, infer(model, val), 'remaining')
    reloaded = load_model(path, config)
    prediction = decode(validation, infer(reloaded, val), 'remaining')
    if not np.allclose(original, prediction, rtol=1e-6, atol=1e-4): raise ValueError('Checkpoint replay changed predictions')
    pred_path = run/'models'/f'{name}_{seed}_validation.npy'; np.save(pred_path, prediction)
    record = {'method': name, 'seed': seed, 'run_id': config['run_id'], 'best_epoch': best_epoch,
              'epochs_run': epoch, 'validation_mape': event_mape(validation, val_labels, prediction),
              'seconds': time.perf_counter()-started, 'trace': trace,
              'artifacts': {str(p.relative_to(run)): sha256(p) for p in [path, pred_path]}}
    write_json(run/'models'/f'{name}_{seed}.json', record)
    print('FIT '+json.dumps({k: v for k, v in record.items() if k not in ['trace', 'artifacts']}), flush=True)
    return prediction, record


def load_model(path, config):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['run_id'] != config['run_id']: raise ValueError('Checkpoint run identity differs')
    model = StructuredForecaster(checkpoint['architecture'], checkpoint['features'], checkpoint['width'])
    model.load_state_dict(checkpoint['state']); return model.to(config['device']).eval()


def ensembles(values, config, top3=None):
    names = []
    for name in config['recipes']:
        label = f'{name}_ensemble{len(config["seeds"])}'; names.append(label)
        values[label] = np.mean([values[f'{name}_seed{s}'] for s in config['seeds']], axis=0)
    if top3 is not None:
        values['neural_top3'] = np.mean([values[n] for n in top3], axis=0)
        for weight in [.25, .5, .75]:
            values[f'blend_neural_{weight}'] = weight*values['neural_top3']+(1-weight)*values['reference']
    return names


def develop(args, data):
    run = args.run; config = config_for(args, data); torch.set_num_threads(4)
    if run.exists():
        if not args.resume: raise FileExistsError(run)
        if json.loads((run/'run_manifest.json').read_text()) != config: raise ValueError('Resume configuration changed')
        verify_config(config, data)
        if (run/'development_seal.json').exists():
            verify_artifacts(run, json.loads((run/'development_seal.json').read_text()))
            print('Development already sealed; returning saved selection.'); return
    else:
        run.mkdir(parents=True); (run/'models').mkdir(); (run/'sources').mkdir()
        write_json(run/'run_manifest.json', config)
        for name in SOURCE_NAMES: shutil.copyfile(ROOT/name, run/'sources'/Path(name).name)
    samples = data.inputs('train'); validation = data.inputs('validation')
    labels = data.labels('train'); val_labels = data.labels('validation')
    meta = EventFeatures.load(data.metadata_directory)
    frame = select_features(relationship_frame(samples, meta), 'invariant')
    val_frame = select_features(relationship_frame(validation, meta), 'invariant')
    prep_path = run/'preprocessor.joblib'; receipt_path = run/'preprocessor.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text()); verify_artifacts(run, receipt)
        if receipt['run_id'] != config['run_id']: raise ValueError('Preprocessor run differs')
        prep = joblib.load(prep_path)
    else:
        prep = prep_for(frame); prep.fit(frame); joblib.dump(prep, prep_path)
        write_json(receipt_path, {'run_id': config['run_id'], 'columns': list(frame),
                                 'artifacts': {'preprocessor.joblib': sha256(prep_path)}})
    train = tensor_arrays(arrays(samples, frame, prep), config['device'])
    val = tensor_arrays(arrays(validation, val_frame, prep), config['device'])
    thin_samples = [thin_sample(s) for s in samples]
    thin_frame = select_features(relationship_frame(thin_samples, meta), 'invariant')
    thin = tensor_arrays(arrays(thin_samples, thin_frame, prep), config['device'])
    values = {'reference': reference_prediction(config['reference'], 'validation', validation)}; records = []
    for name, recipe in config['recipes'].items():
        for seed in config['seeds']:
            record_path = run/'models'/f'{name}_{seed}.json'
            if record_path.exists():
                record = json.loads(record_path.read_text()); verify_artifacts(run, record)
                if record['run_id'] != config['run_id']: raise ValueError('Cached fit identity differs')
                pred = np.load(run/'models'/f'{name}_{seed}_validation.npy')
                if len(pred) != len(validation) or not np.isfinite(pred).all(): raise ValueError('Invalid cached predictions')
            else:
                pred, record = train_one(name, recipe, seed, config, train, val, thin, samples, validation, labels, val_labels, run)
            values[f'{name}_seed{seed}'] = pred; records.append(record)
    neural_names = ensembles(values, config)
    top3 = sorted(neural_names, key=lambda n: (event_mape(validation, val_labels, values[n]), n))[:3]
    ensembles(values, config, top3)
    scores = {name: event_mape(validation, val_labels, pred) for name, pred in values.items()}
    candidates = neural_names+['reference', 'neural_top3']+[f'blend_neural_{w}' for w in [.25, .5, .75]]
    selected = min(candidates, key=lambda n: (scores[n], n))
    selection = {'dataset_id': config['dataset_id'], 'run_id': config['run_id'], 'selected_recipe': selected,
                 'top3_members': top3, 'validation_scores': scores, 'selection_candidates': candidates,
                 'reported_methods': sorted(values), 'test_labels_used_for_selection': False}
    write_json(run/'selection.json', selection); write_json(run/'training_records.json', records)
    pd.DataFrame([row for name, pred in values.items() for row in prediction_rows(validation, name, pred)]).to_csv(run/'validation_predictions.csv', index=False)
    write_json(run/'development_seal.json', {'artifacts': {str(p.relative_to(run)): sha256(p) for p in run.rglob('*') if p.is_file()}})
    print('SELECTION '+json.dumps(selection), flush=True)


def predict(args, data):
    run = args.run
    if (run/'blind_test_manifest.json').exists() or (run/'test_predictions.csv').exists():
        raise FileExistsError('Blind predictions already exist; cannot overwrite')
    config = json.loads((run/'run_manifest.json').read_text()); verify_config(config, data)
    verify_artifacts(run, json.loads((run/'development_seal.json').read_text()))
    torch.set_num_threads(4)
    selection = json.loads((run/'selection.json').read_text()); samples = data.inputs('test')
    frame = select_features(relationship_frame(samples, EventFeatures.load(data.metadata_directory)), 'invariant')
    inputs = tensor_arrays(arrays(samples, frame, joblib.load(run/'preprocessor.joblib')), config['device'])
    values = {'reference': reference_prediction(config['reference'], 'test', samples)}
    for name in config['recipes']:
        for seed in config['seeds']:
            model = load_model(run/'models'/f'{name}_{seed}.pt', config)
            values[f'{name}_seed{seed}'] = decode(samples, infer(model, inputs), 'remaining')
    ensembles(values, config, selection['top3_members'])
    if set(values) != set(selection['reported_methods']): raise ValueError('Prediction method list changed')
    pd.DataFrame([row for name, pred in values.items() for row in prediction_rows(samples, name, pred)]).to_csv(run/'test_predictions.csv', index=False)
    write_json(run/'blind_test_manifest.json', {'run_id': config['run_id'], 'dataset_id': config['dataset_id'],
               'dataset_directory': str(data.root.resolve()), 'test_cases': len(samples),
               'artifacts': {str(p.relative_to(run)): sha256(p) for p in run.rglob('*') if p.is_file()}})
    print('Blind predictions sealed; no test labels read.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['develop', 'predict']); parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/holdout_v1')
    parser.add_argument('--reference', type=Path, default=ROOT/'analysis/holdout_results/relationships_v3')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=100); parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1: parser.error('epochs and batch size must be positive')
    data = HoldoutDataset(args.dataset)
    (develop if args.phase == 'develop' else predict)(args, data)


if __name__ == '__main__': main()
