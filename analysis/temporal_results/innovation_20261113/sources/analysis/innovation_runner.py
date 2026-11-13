"""Frozen V1/V2 innovation recipes, chronological training diagnostics, blind test."""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from analysis.backtest import ROOT
from analysis.event_features import EventFeatures
from analysis.holdout_dataset import HoldoutDataset, SPLITS, sha256, write_json
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import TemporalDataset
from analysis.innovation_common import (prepare, subset_arrays, chronological_folds,
    event_mape, event_weights, prediction_rows, local_sources)

FAMILIES = {name: 'analysis.innovation_'+name for name in ['retrieval', 'hierarchy', 'growth']}
COMBINATIONS = ['family_mean', 'blend_control_50']
DATASET_IDS = {'v1':'d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb',
               'v2':'1f76f39af8151bba185496411bbcee064b16a4c5aca5c515c1d957a5056fe854'}


def combine(spec, predictions):
    values = np.stack([predictions[n] for n in spec['members']])
    return values.mean(axis=0)


def select_combinations(arrays, labels, predictions, recipes, *, control):
    scores = {n: event_mape(arrays, labels, predictions[n]) for n in recipes}
    family_best = {f: min((n for n,s in recipes.items() if s['family']==f),key=scores.get)
                   for f in dict.fromkeys(s['family'] for s in recipes.values())}
    best = min(scores,key=scores.get)
    specs = {'family_mean': {'kind':'mean','members':list(family_best.values())},
             'blend_control_50': {'kind':'mean','members':[best,control]}}
    values = dict(predictions)
    values.update({n:combine(s,predictions) for n,s in specs.items()})
    candidates = [n for n in ['hold_current','linear'] if n in predictions]+list(recipes)+COMBINATIONS
    validation = {n:event_mape(arrays,labels,values[n]) for n in candidates}
    return {'selected_recipe':min(validation,key=validation.get),'candidates':candidates,
            'validation_scores':validation,'family_best':family_best,'combination_specs':specs,
            'reported_methods':list(values),'test_labels_used_for_selection':False,
            'refit_after_selection':False,'criterion':'validation event-macro MAPE; individual NN seeds excluded'}


def predict_all(module, bundle, arrays, name):
    values = {name:np.asarray(module.predict_recipe(bundle,arrays))}
    if hasattr(module,'predict_members'):
        values.update({name+'__'+k:np.asarray(v) for k,v in module.predict_members(bundle,arrays).items()})
    for method,prediction in values.items():
        prediction_rows(arrays,method,prediction)
    return values


def diagnostic_rows(arrays, labels, predictions, **extra):
    rows=[]
    groups=[('overall','all',np.arange(len(arrays['cases'])))]
    for r in [0,1]:groups.append(('rank',str(1000+1000*r),np.flatnonzero(arrays['rank']==r)))
    phase=np.searchsorted([.25,.9],arrays['progress'],side='right')
    for phase_id in range(3):groups.append(('phase',['early','middle','late'][phase_id],np.flatnonzero(phase==phase_id)))
    for axis,group,indices in groups:
        if not len(indices):continue
        part={'cases':[arrays['cases'][i] for i in indices]}
        for method,prediction in predictions.items():
            rows.append({**extra,'axis':axis,'group':group,'method':method,
                'event_macro_mape_pct':event_mape(part,labels,np.asarray(prediction)[indices]),
                'events':len({c['event'] for c in part['cases']}),'cases':len(indices)})
    return rows


def run_experiment(data, run, *, protocol, families=None, limit_per_family=None,
                   iterations_override=None, epochs_override=None, run_cv=True, allow_fixture=False):
    run=Path(run)
    if run.exists():raise FileExistsError(run)
    if protocol not in ('v1','v2'):raise ValueError('Unknown protocol')
    if not allow_fixture and data.manifest['dataset_id']!=DATASET_IDS[protocol]:
        raise ValueError('Formal dataset fingerprint differs from frozen protocol')
    modules={f:importlib.import_module(FAMILIES[f]) for f in (families or FAMILIES)}
    recipes={}
    for family,module in modules.items():
        configs=module.recipes()
        if limit_per_family is not None:configs=configs[:limit_per_family]
        for c in configs:
            c=dict(c)
            if iterations_override is not None:c['iterations']=iterations_override
            if epochs_override is not None:c['epochs']=epochs_override
            recipes[family+'__'+c['name']]={'family':family,'config':c}
    control=next((n for n in recipes if n.endswith('__retrieval_control')),next(iter(recipes)))
    sources=local_sources([Path(__file__),*(Path(m.__file__) for m in modules.values()),
                          ROOT/'analysis/holdout_evaluate.py',ROOT/'analysis/temporal_evaluate.py'])
    cfg={'dataset_id':data.manifest['dataset_id'],'dataset_directory':str(data.root.resolve()),
         'protocol':protocol,'synthetic_fixture':bool(allow_fixture),
         'recipes':recipes,'combination_menu':COMBINATIONS,'control':control,
         'internal_cv':bool(run_cv),'fold_protocol':'first half warmup, three later contiguous activity blocks; fixed earlier-completed training history per block',
         'selection':'validation only; all predeclared candidates; no seed selection; no refit',
         'inputs':'own-line blind prefix; context/retrieval additions explicitly ablated',
         'weights':'equal total per training activity across both ranks and all prefixes',
         'python':platform.python_version(),
         'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','catboost','torch','joblib']},
         'source_hashes':{n:sha256(ROOT/n) for n in sources}}
    cfg['run_id']=identity(cfg)
    run.mkdir(parents=True)
    for name in ['sources','models','features','cv']:(run/name).mkdir()
    write_json(run/'run_manifest.json',cfg)
    for name in sources:
        dest=run/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    metadata=EventFeatures.load(data.metadata_directory)
    joblib.dump(metadata,run/'models/metadata.joblib')
    history=list(data.train_curves())
    train=prepare(data.inputs('train'),metadata);train['history']=history
    labels=data.labels('train')
    expected=set(data.manifest['splits']['train']['event_ids'])
    if {c.event for c in history}!=expected:raise ValueError('Training history event cohort differs from frozen manifest')
    weights=event_weights(train['cases'])
    pd.DataFrame(train['cases']).assign(weight=weights).to_csv(run/'training_weights.csv',index=False)
    train['base'].to_csv(run/'features/train_base.csv.gz',index=False)
    event_rows=[{'event':c.event,'start_ms':c.start,'end_ms':c.end} for c in history]
    folds=chronological_folds(event_rows) if run_cv else []
    write_json(run/'cv/folds.json',folds)
    cv_rows=[];cv_predictions=[]
    for fold in folds:
        directory=run/f"cv/fold_{fold['fold']}";directory.mkdir()
        train_ids=set(fold['training_events']);heldout=set(fold['heldout_events'])
        part_history=[c for c in history if c.event in train_ids]
        fitting=subset_arrays(train,[i for i,c in enumerate(train['cases']) if c['event'] in train_ids],history=part_history)
        scoring=subset_arrays(train,[i for i,c in enumerate(train['cases']) if c['event'] in heldout],history=[])
        fold_labels={c['case_id']:labels[c['case_id']] for c in fitting['cases']}
        values={'hold_current':scoring['current'],'linear':scoring['current']/scoring['progress']}
        for name,spec in recipes.items():
            started=time.perf_counter();module=modules[spec['family']]
            bundle=module.fit_recipe(spec['config'],fitting,fold_labels)
            path=directory/(name+'.joblib');joblib.dump(bundle,path)
            values.update(predict_all(module,bundle,scoring,name))
            np.testing.assert_allclose(module.predict_recipe(joblib.load(path),scoring),values[name],rtol=1e-9)
            print(json.dumps({'phase':'cv','fold':fold['fold'],'method':name,
                'event_mape':event_mape(scoring,labels,values[name]),'seconds':round(time.perf_counter()-started,2)}),flush=True)
        part=pd.concat([prediction_rows(scoring,n,p) for n,p in values.items()],ignore_index=True)
        part['fold']=fold['fold'];part.to_csv(directory/'predictions.csv',index=False)
        cv_predictions.append(part)
        cv_rows.extend(diagnostic_rows(scoring,labels,values,fold=fold['fold']))
        pd.DataFrame(cv_rows).to_csv(run/'cv/metrics.csv',index=False)
    if cv_predictions:pd.concat(cv_predictions,ignore_index=True).to_csv(run/'cv/predictions.csv',index=False)
    # Full fixed training fit. Validation never participates in fit_recipe.
    valid=prepare(data.inputs('validation'),metadata)
    valid_labels=data.labels('validation')
    arrays={'train':train,'validation':valid}
    predictions={split:{'hold_current':a['current'],'linear':a['current']/a['progress']} for split,a in arrays.items()}
    bundles={};ledger=[]
    for name,spec in recipes.items():
        module=modules[spec['family']];started=time.perf_counter()
        bundle=module.fit_recipe(spec['config'],train,labels)
        path=run/f'models/{name}.joblib';joblib.dump(bundle,path);bundles[name]=bundle
        for split,a in arrays.items():predictions[split].update(predict_all(module,bundle,a,name))
        np.testing.assert_allclose(module.predict_recipe(joblib.load(path),valid),predictions['validation'][name],rtol=1e-9)
        ledger.append({'method':name,'train_fit_mape':event_mape(train,labels,predictions['train'][name]),
            'validation_mape':event_mape(valid,valid_labels,predictions['validation'][name]),
            'seconds':round(time.perf_counter()-started,2)})
        write_json(run/'development_ledger.json',ledger);print(json.dumps(ledger[-1]),flush=True)
    selection=select_combinations(valid,valid_labels,predictions['validation'],recipes,control=control)
    selection.update({'run_id':cfg['run_id'],'dataset_id':cfg['dataset_id']})
    for split in arrays:
        predictions[split].update({n:combine(s,predictions[split]) for n,s in selection['combination_specs'].items()})
    write_json(run/'selection.json',selection)
    print('SELECTED '+selection['selected_recipe'],flush=True)
    # No fit, parameter choice, or answer reads beyond this boundary.
    arrays['test']=prepare(data.inputs('test'),metadata)
    predictions['test']={'hold_current':arrays['test']['current'],'linear':arrays['test']['current']/arrays['test']['progress']}
    for name,spec in recipes.items():predictions['test'].update(predict_all(modules[spec['family']],bundles[name],arrays['test'],name))
    predictions['test'].update({n:combine(s,predictions['test']) for n,s in selection['combination_specs'].items()})
    for split,a in arrays.items():
        pd.concat([prediction_rows(a,n,p) for n,p in predictions[split].items()],ignore_index=True).to_csv(run/f'{split}_predictions.csv',index=False)
        a['base'].to_csv(run/f'features/{split}_base.csv.gz',index=False)
    metrics=diagnostic_rows(train,labels,predictions['train'],split='train')
    metrics+=diagnostic_rows(valid,valid_labels,predictions['validation'],split='validation')
    pd.DataFrame(metrics).to_csv(run/'development_metrics.csv',index=False)
    for name,digest in cfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Source changed during fit: '+name)
    seal={k:cfg[k] for k in ['run_id','dataset_id','dataset_directory']}
    seal.update({'methods':selection['reported_methods'],'reference_method':selection['selected_recipe'],
        'test_cases':len(arrays['test']['cases']),
        'artifacts':{str(p.relative_to(run)):sha256(p) for p in sorted(run.rglob('*')) if p.is_file()}})
    write_json(run/('blind_test_manifest.json' if protocol=='v1' else 'blind_manifest.json'),seal)
    return selection


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',choices=['v1','v2'],required=True)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    data=HoldoutDataset() if args.protocol=='v1' else TemporalDataset(ROOT/'data/temporal_holdout_v2_t1000_t2000')
    run_experiment(data,args.run,protocol=args.protocol)
