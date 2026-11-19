"""Independent, sealed final comparison to historically selected references."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import HoldoutDataset, sha256, write_json
from analysis.temporal_dataset import TemporalDataset
from analysis.holdout_evaluate import case_metadata, score_predictions
from analysis.temporal_evaluate import aligned_frame, score_temporal
from analysis.event_features import EventFeatures
from analysis.multirank_dataset import identity
from analysis.innovation_common import local_sources


def verify_run(run, seal_name):
    run=Path(run);seal=json.loads((run/seal_name).read_text())
    for name,digest in seal['artifacts'].items():
        if sha256(run/name)!=digest:raise ValueError('Changed sealed artifact: '+str(run/name))
    return seal


def compare_frames(cases, labels, prediction, reference):
    errors=[]
    for p in [prediction,reference]:
        frame=aligned_frame(cases,labels,p)
        frame['ape']=abs(frame.predicted/frame.actual-1)*100
        errors.append(frame.groupby('event').ape.mean().sort_index())
    delta=(errors[0]-errors[1]).to_numpy();n=len(delta)
    rng=np.random.default_rng(42)
    boot=delta[rng.integers(0,n,(5000,n))].mean(axis=1)
    return {'events':n,'delta_mape_points':float(delta.mean()),
        'paired_activity_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
        'improved_events':int((delta<0).sum()),'worse_events':int((delta>0).sum()),
        'unchanged_events':int((delta==0).sum())}


def evaluate(run, output=None):
    run=Path(run);output=Path(output) if output else run/'comparison'
    cfg=json.loads((run/'run_manifest.json').read_text());protocol=cfg['protocol']
    seal_name='blind_test_manifest.json' if protocol=='v1' else 'blind_manifest.json'
    seal=verify_run(run,seal_name)
    data=HoldoutDataset(cfg['dataset_directory']) if protocol=='v1' else TemporalDataset(cfg['dataset_directory'])
    if data.manifest['dataset_id']!=cfg['dataset_id']:raise ValueError('Dataset identity differs')
    if protocol=='v1':
        refs={'old_catboost_peer':(ROOT/'analysis/holdout_results/v1','catboost_peer'),
              'old_selected':(ROOT/'analysis/holdout_results/structured_neural_v1',None)}
    else:
        parent=ROOT/'analysis/temporal_results/v2_multiagent_20261112'
        refs={n:(parent,m) for n,m in [('old_single','reference_single'),('old_paired','reference_paired'),('old_selected',None)]}
    ref_manifest={}
    for name,(path,method) in refs.items():
        old=verify_run(path,seal_name)
        if old['dataset_id']!=cfg['dataset_id']:raise ValueError('Reference dataset differs')
        if method is None:method=json.loads((path/'selection.json').read_text())['selected_recipe']
        ref_manifest[name]={'path':str(path),'method':method,'seal_sha256':sha256(path/seal_name)}
    sources=local_sources([Path(__file__)])
    ident={'dataset_id':cfg['dataset_id'],'run_seal_sha256':sha256(run/seal_name),
        'references':ref_manifest,'source_hashes':{n:sha256(ROOT/n) for n in sources},
        'python':platform.python_version(),'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas']},
        'bootstrap_seed':42,'bootstrap_draws':5000}
    evaluation_id=identity(ident);receipt=output/'evaluation_manifest.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['evaluation_id']!=evaluation_id:raise FileExistsError('Changed scorer; use a new --output directory')
        for name,digest in saved['artifacts'].items():
            if sha256(output/name)!=digest:raise ValueError('Changed comparison artifact')
        return json.loads((output/'comparison.json').read_text())
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    # Answers are parsed only after all seals, identities and cache checks.
    labels=data.labels('test',allow_test=True)
    cases=(pd.DataFrame(case_metadata(data.inputs('test'),EventFeatures.load(data.metadata_directory)))
           if protocol=='v1' else data.case_index('test'))
    selection=json.loads((run/'selection.json').read_text());selected=selection['selected_recipe']
    frame=pd.read_csv(run/'test_predictions.csv')
    chosen=frame[frame.method==selected]
    predictions={n:frame[frame.method==n] for n in frame.method.unique()}
    for name,r in ref_manifest.items():
        old=pd.read_csv(Path(r['path'])/'test_predictions.csv')
        predictions[name]=old[old.method==r['method']]
    scores={}
    for name,prediction in predictions.items():
        scores[name]=(score_predictions(cases.to_dict('records'),labels,prediction.to_dict('records'))
            if protocol=='v1' else score_temporal(cases,data.memberships('test'),labels,prediction))
    control=cfg['control']
    comparisons={name:compare_frames(cases,labels,chosen,predictions[name]) for name in [*ref_manifest,control]}
    result={'evaluation_id':evaluation_id,'identity':ident,'selected':selected,'scores':scores,
        'comparisons':comparisons,'limitations':['Previously exposed test cohort; not a pristine blind test.',
        'Retrospective metadata without point-in-time archives.',
        'Unadjusted paired activity intervals are descriptive; no post-test reselection.']}
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'comparison.json',result)
    pd.DataFrame([{'method':m,**s['overall']} for m,s in scores.items()]).to_csv(output/'summary.csv',index=False)
    for name in sources:
        dest=output/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write_json(receipt,{'evaluation_id':evaluation_id,'artifacts':{
        str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'protocol':protocol,'selected':selected,'comparisons':comparisons}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path);a=p.parse_args();evaluate(a.run,a.output)
