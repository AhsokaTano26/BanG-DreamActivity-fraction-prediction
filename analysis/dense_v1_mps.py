"""Explicit MPS operational restart of the fixed dense V1 structured menu.

The incomplete CPU attempt is retained and excluded from model selection.  Every
recipe is rerun with the original three seeds; no CPU/MPS result comparison is
used to select a device or model.  The original dense adapter remains immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from analysis.current_benchmark import load_dataset
from analysis.data_policy import MIN_EVENT_START_MS, POLICY_ID
from analysis.dense_v1_extensions import (BlindDataset, DATASET_ID, configure_runtime,
    reference_identity, save_predictions, read_values)
from analysis.holdout_dataset import ROOT, sha256, write_json


def require_mps(device, available):
    if device != 'mps':
        raise ValueError('This operational restart requires explicit MPS execution')
    if not available:
        raise RuntimeError('MPS is unavailable; silent CPU fallback is forbidden')


def run_mps(run, reference, interrupted_run):
    from analysis import structured_neural_benchmark as b
    from analysis.event_features import EventFeatures
    from analysis.innovation_common import local_sources
    from analysis.relationship_benchmark import score, decode
    require_mps('mps',b.torch.backends.mps.is_available())
    run, reference, interrupted_run = map(Path,[run,reference,interrupted_run])
    if run.exists():
        raise FileExistsError(run)
    if (interrupted_run/'blind_test_manifest.json').exists():
        raise ValueError('Operational restart cannot replace a sealed CPU experiment')
    data = BlindDataset(load_dataset('v1'))
    if data.manifest['dataset_id'] != DATASET_ID:
        raise ValueError('Dense V1 dataset fingerprint differs')
    reference_identity(reference,DATASET_ID)
    configure_runtime()
    b.SOURCE_NAMES = [*b.SOURCE_NAMES,'analysis/dense_v1_extensions.py','analysis/dense_v1_mps.py']
    child = run/'development'
    args = SimpleNamespace(run=child,reference=reference,device='mps',epochs=100,batch_size=256,resume=False)
    sources = local_sources([Path(__file__)])
    config = b.config_for(args,data)
    config.update(family='structured',cpu_threads=2,minimum_event_start_ms=MIN_EVENT_START_MS,
                  data_policy_id=POLICY_ID,operational_restart_of=str(interrupted_run.resolve()),
                  restart_reason='CPU convolution runtime; MPS became free; no device comparison or test selection',
                  source_hashes={name:sha256(ROOT/name) for name in sources})
    config.pop('run_id')
    config['run_id'] = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    run.mkdir(parents=True); write_json(run/'run_manifest.json',config)
    for name in sources:
        target = run/'sources'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    started = time.perf_counter()
    b.develop(args,data)
    train = data.inputs('train'); valid = data.inputs('validation')
    metadata = EventFeatures.load(data.metadata_directory)
    frame = b.select_features(b.relationship_frame(train,metadata),'invariant')
    prep = joblib.load(child/'preprocessor.joblib')
    child_config = json.loads((child/'run_manifest.json').read_text())
    require_mps(child_config['device'],b.torch.backends.mps.is_available())
    arrays = b.tensor_arrays(b.arrays(train,frame,prep),child_config['device'])
    if any(value.device.type != 'mps' for value in arrays.values()):
        raise ValueError('Training replay tensors must share the MPS model device')
    values = {}
    for name in child_config['recipes']:
        for seed in child_config['seeds']:
            model = b.load_model(child/f'models/{name}_{seed}.pt',child_config)
            values[f'{name}_seed{seed}'] = decode(train,b.infer(model,arrays),'remaining')
    b.ensembles(values,child_config)
    selection = json.loads((child/'selection.json').read_text())
    values['neural_top3'] = np.mean([values[name] for name in selection['top3_members']],axis=0)
    save_predictions(child/'train_predictions.csv',train,values)
    b.predict(args,data)
    test = data.inputs('test')
    val_values = read_values(child/'validation_predictions.csv',valid)
    test_values = read_values(child/'test_predictions.csv',test)
    if set(val_values) != set(test_values):
        raise ValueError('Blind prediction methods differ from validation')
    selection.update(run_id=config['run_id'],dataset_id=DATASET_ID)
    write_json(run/'selection.json',selection)
    for split in ['train','validation','test']:
        shutil.copyfile(child/f'{split}_predictions.csv',run/f'{split}_predictions.csv')
    labels = data.labels('train'); val_labels = data.labels('validation')
    pd.DataFrame([{'method':name,'train_fit_mape_pct':score(train,labels,pred),
                   'validation_mape_pct':score(valid,val_labels,val_values[name])}
                  for name,pred in values.items()]).to_csv(run/'train_validation_summary.csv',index=False)
    write_json(run/'runtime.json',{'seconds':time.perf_counter()-started,'device':'mps',
               'training_cases':len(train),'validation_cases':len(valid),'test_cases':len(test),
               'reported_methods':len(test_values),'train_fit_methods':len(values)})
    for name,digest in config['source_hashes'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Source changed during retraining: '+name)
    write_json(run/'blind_test_manifest.json',{'run_id':config['run_id'],'dataset_id':DATASET_ID,
               'dataset_directory':str(data.root.resolve()),'test_cases':len(test),
               'artifacts':{str(p.relative_to(run)):sha256(p) for p in sorted(run.rglob('*')) if p.is_file()}})
    print('DENSE_MPS_BLIND_SEALED '+selection['selected_recipe'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--interrupted-run',type=Path,required=True)
    args = parser.parse_args()
    run_mps(args.run,args.reference,args.interrupted_run)
