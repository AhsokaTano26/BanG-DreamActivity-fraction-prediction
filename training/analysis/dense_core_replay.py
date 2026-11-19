"""Replay dense V1 fitted models for training diagnostics; never requests test labels."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from analysis.current_benchmark import load_dataset
from analysis.holdout_dataset import ROOT, sha256, write_json
from analysis.holdout_features import HoldoutFeatureBuilder
from analysis.event_features import EventFeatures
from analysis.holdout_benchmark import (BASELINES, TREE_RECIPES, NEURAL_RECIPES, SEEDS,
    baseline_predictions, prefix_sequence, tree_predict, neural_predict,
    final_predictions, prediction_records)
from analysis.holdout_evaluate import case_metadata, score_predictions


def training_baselines(samples, history):
    values={name:[] for name in BASELINES}; unavailable=[]
    for sample in samples:
        try:
            prediction=baseline_predictions([sample],history)
        except AssertionError as exc:
            if str(exc)!='No historical templates available':raise
            unavailable.append(sample['case_id'])
            score,p=sample['score'],sample['progress'];linear=score/p
            rate=sample['features']['rate_24h_relative']
            recent=score+max(0.,rate)*score/sample['features']['elapsed_h']*sample['features']['remaining_h'] if np.isfinite(rate) else linear
            prediction={'linear':[max(score,linear)],'recent_24h':[max(score,recent)]}
        for name,pred in prediction.items():values[name].append(float(pred[0]))
    # A historical method with any undefined cold-start row has no full-panel
    # training score. Never drop the row, use future history or invent a fallback.
    return {name:np.asarray(v) for name,v in values.items() if len(v)==len(samples)},unavailable


def replay_training(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    seal=json.loads((run/'blind_test_manifest.json').read_text())
    config=json.loads((run/'run_manifest.json').read_text())
    for name,digest in seal['artifacts'].items():
        if sha256(run/name)!=digest:
            raise ValueError('Sealed artifact changed: '+name)
    for name,digest in config['source_hashes'].items():
        if sha256(ROOT/name)!=digest:
            raise ValueError('Replay source changed: '+name)
    for name,version in config['software'].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('Replay dependency changed: '+name)
    data=load_dataset('v1',seal['dataset_directory'])
    if data.manifest['dataset_id'] != seal['dataset_id']:
        raise ValueError('Replay dataset changed')
    selection=json.loads((run/'selection.json').read_text())
    metadata=EventFeatures.load(data.metadata_directory)
    history=data.train_curves();builder=HoldoutFeatureBuilder(history,metadata)
    torch.set_num_threads(2)
    scores={};frames={};replay_errors={};unavailable=[]
    for split in ['train','validation']:
        samples=data.inputs(split)
        seq=np.stack([prefix_sequence(s) for s in samples])
        base=-np.log(np.array([s['progress'] for s in samples]))
        modes={mode for mode,_ in TREE_RECIPES.values()}|{mode for mode,_ in NEURAL_RECIPES.values()}
        tab={mode:joblib.load(run/f'models/preprocessor_{mode}.joblib').transform(builder.frame(samples,mode)).astype(np.float32) for mode in modes}
        if split=='train':values,unavailable=training_baselines(samples,history)
        else:values=baseline_predictions(samples,history)
        for name,(mode,backend) in TREE_RECIPES.items():
            values[name]=final_predictions(samples,tree_predict(joblib.load(run/f'models/{name}.joblib'),tab[mode],seq,base))
        for name,(mode,architecture) in NEURAL_RECIPES.items():
            members=[]
            for seed in SEEDS:
                key=f'{name}_seed{seed}'
                values[key]=final_predictions(samples,neural_predict(run/f'models/{key}.pt',tab[mode],seq,config['device'],config['batch_size']))
                members.append(values[key])
            values[name+'_ensemble3']=np.mean(members,axis=0)
        values['validation_top3_ensemble']=np.mean([values[n] for n in selection['top3_members']],axis=0)
        frame=pd.DataFrame([row for name,pred in values.items() for row in prediction_records(samples,name,pred)])
        missing=set(selection['reported_methods'])-set(values)
        allowed={'template','same_type_template','shape_analog'} if split=='train' and unavailable else set()
        if missing!=allowed or set(values)-set(selection['reported_methods']):
            raise ValueError('Replay model coverage differs')
        if split=='validation':
            expected=pd.read_csv(run/'validation_predictions.csv').set_index(['case_id','method']).predicted.sort_index()
            actual=frame.set_index(['case_id','method']).predicted.sort_index()
            if not expected.index.equals(actual.index):
                raise ValueError('Replay cases differ')
            relative=np.abs(actual.to_numpy()/expected.to_numpy()-1)
            replay_errors[split]=float(relative.max())
            if not np.allclose(actual,expected,rtol=1e-5,atol=1e-3):
                raise ValueError('Saved-model replay differs')
        labels=data.labels(split)
        cases=case_metadata(samples,metadata)
        scores[split]={name:score_predictions(cases,labels,group.to_dict('records')) for name,group in frame.groupby('method')}
        frames[split]=frame
    output.mkdir(parents=True)
    for split,frame in frames.items():
        frame.to_csv(output/f'{split}_predictions.csv',index=False)
    write_json(output/'metrics.json',{'splits':scores,'caveat':'Training scores are resubstitution fit, not out-of-sample accuracy. Three pure historical template baselines have no complete training-panel score because initial activities lack past templates; no rows are deleted or filled.'})
    write_json(output/'replay.json',{'run_id':config['run_id'],'dataset_id':seal['dataset_id'],'test_labels_read':False,'max_relative_replay_error':replay_errors,'methods':len(values),'blind_seal_sha256':sha256(run/'blind_test_manifest.json'),
        'unavailable_training_methods':['template','same_type_template','shape_analog'] if unavailable else [],
        'cold_start_case_ids':unavailable,'cold_start_reason':'No historical templates available; full training metric undefined, no case filtering or fallback.'})
    (output/'source.py').write_text(Path(__file__).read_text())
    write_json(output/'manifest.json',{'artifacts':{str(p.relative_to(output)):sha256(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'methods':len(values),'replay_errors':replay_errors,'output':str(output)}),flush=True)
    return scores

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();replay_training(a.run,a.output)
