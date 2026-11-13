"""Replay sealed V1 models on Temporal V2; no fitting or label reads."""
from __future__ import annotations
import argparse,importlib.metadata,json,platform,shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import SPLITS,sha256,write_json
from analysis.multirank_dataset import identity
from analysis.multirank_model import verify_run,prepare,model_prediction
from analysis.temporal_dataset import TemporalDataset,DEFAULT

METHODS=['hold_current','linear','frozen_catboost_rank','frozen_catboost_context']
SOURCES=['analysis/temporal_benchmark.py','analysis/temporal_dataset.py','analysis/multirank_model.py',
    'analysis/multirank_dataset.py','analysis/event_features.py','analysis/backtest.py','analysis/holdout_dataset.py']


def run_frozen(data,reference_run,run):
    reference_run=Path(reference_run);run=Path(run)
    if run.exists():raise FileExistsError(run)
    seal=verify_run(reference_run)
    if data.manifest['base_dataset_id']!=seal['dataset_id']:raise ValueError('Model and temporal cohort have different source datasets')
    original=json.loads((reference_run/'run_manifest.json').read_text())
    for name,digest in original['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Frozen inference source changed: '+name)
    software={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','catboost','joblib']}
    if software!=original['software'] or platform.python_version()!=original['python']:
        raise ValueError('Frozen-model inference requires the recorded dependency versions')
    selected=json.loads((reference_run/'selection.json').read_text())['selected_recipe']
    reference_method=selected if selected=='linear' else 'frozen_'+selected
    config={'dataset_directory':str(data.root.resolve()),'dataset_id':data.manifest['dataset_id'],
        'reference_run':str(reference_run.resolve()),'reference_blind_manifest_sha256':sha256(reference_run/'blind_test_manifest.json'),
        'reference_method':reference_method,'methods':METHODS,'policy':'frozen models; no refit, no selection, no label reads',
        'python':platform.python_version(),'software':software,'source_hashes':{n:sha256(ROOT/n) for n in SOURCES}}
    config['run_id']=identity(config);run.mkdir(parents=True);(run/'sources').mkdir()
    shutil.copytree(reference_run/'models',run/'reference_models')
    shutil.copyfile(reference_run/'run_manifest.json',run/'reference_run_manifest.json')
    write_json(run/'run_manifest.json',config)
    for name in SOURCES:shutil.copyfile(ROOT/name,run/'sources'/Path(name).name)
    metadata=joblib.load(run/'reference_models/metadata.joblib')
    models={name:joblib.load(run/'reference_models'/f'{name}.joblib') for name in ['catboost_rank','catboost_context']}
    for split in SPLITS:
        arrays=prepare(data.inputs(split),metadata)
        frozen=data.case_index(split)
        if set(frozen.case_id)!={c['case_id'] for c in arrays['cases']}:raise ValueError('Prefix/case index mismatch')
        predictions={'hold_current':arrays['current'],'linear':arrays['current']/arrays['progress']}
        predictions.update({'frozen_'+name:model_prediction(bundle,arrays) for name,bundle in models.items()})
        rows=[]
        for method,values in predictions.items():
            if len(values)!=len(arrays['cases']) or not np.isfinite(values).all() or (values<=0).any():raise ValueError('Invalid prediction vector')
            rows.extend({'case_id':c['case_id'],'method':method,'predicted':float(p)} for c,p in zip(arrays['cases'],values))
        pd.DataFrame(rows).to_csv(run/f'{split}_predictions.csv',index=False)
        print(json.dumps({'split':split,'unique_cases':len(frozen),'methods':METHODS}),flush=True)
    result={k:config[k] for k in ['run_id','dataset_id','dataset_directory','methods','reference_method']}
    result['artifacts']={str(p.relative_to(run)):sha256(p) for p in sorted(run.rglob('*')) if p.is_file()}
    write_json(run/'blind_manifest.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DEFAULT)
    p.add_argument('--reference-run',type=Path,default=ROOT/'analysis/multirank_results/v2')
    p.add_argument('--run',required=True,type=Path);a=p.parse_args();run_frozen(TemporalDataset(a.dataset),a.reference_run,a.run)
