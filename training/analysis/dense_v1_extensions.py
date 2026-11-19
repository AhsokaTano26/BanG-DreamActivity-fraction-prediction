"""Retrain the registered V1 extension menus on the frozen dense-era cohort.

Old fit implementations are reused without editing their sources.  Each new run
archives this adapter, its runtime overrides, models, selection, and predictions.
The independent holdout evaluator remains the only reader of test answers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from analysis.current_benchmark import load_dataset
from analysis.holdout_dataset import ROOT, sha256, write_json
from analysis.data_policy import MIN_EVENT_START_MS, POLICY_ID

DATASET_ID = '80e354c3a5340df6724a40ef1df490f92c08007311d55849342de509eaf5ee50'
FAMILIES = ['relationship', 'structured', 'deadline', 'quality', 'quality_extended', 'innovation']


class BlindDataset:
    """Enforce the answer boundary even if an adapted trainer requests opt-in."""
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        return getattr(self._data, name)

    def labels(self, split, **kwargs):
        if split == 'test':
            raise PermissionError('Model code cannot read test answers')
        return self._data.labels(split, **kwargs)


def dense_folds(curves):
    """Four chronological OOF blocks after a roughly half-cohort warmup."""
    events = {}
    for curve in curves:
        dates = (int(curve.start), int(curve.end))
        if curve.event in events and events[curve.event] != dates:
            raise ValueError('Inconsistent activity dates')
        events[curve.event] = dates
    ordered = sorted(events, key=lambda event: (events[event][0], event))
    size = len(ordered)//8
    if size < 1:
        raise ValueError('At least eight activities required')
    warmup = len(ordered)-4*size
    folds = []
    for number in range(4):
        held = ordered[warmup+number*size:warmup+(number+1)*size]
        start = min(events[event][0] for event in held)
        train = [event for event in ordered if events[event][1] < start]
        if len(train) != warmup+number*size:
            raise ValueError('Activity overlap prevents legacy contiguous OOF adapter')
        folds.append({'fold':number, 'training_events':train, 'heldout_events':held,
                      'training_latest_end':max(events[event][1] for event in train),
                      'heldout_earliest_start':start})
    return folds


def reference_identity(root, dataset_id):
    root = Path(root)
    seal = json.loads((root/'blind_test_manifest.json').read_text())
    if seal['dataset_id'] != dataset_id:
        raise ValueError('Reference dataset/cohort differs')
    from analysis.structured_neural_benchmark import reference_identity as inspect
    return inspect(root)


def configure_runtime():
    """Historical recipes differ only in execution resource limits."""
    from analysis import relationship_benchmark as relationship
    from analysis import deadline_benchmark as deadline
    import torch
    original_cat = relationship.CatBoostRegressor
    original_lgb = relationship.LGBMRegressor
    relationship.CatBoostRegressor = lambda **kw: original_cat(**(kw | {'thread_count':2}))
    relationship.LGBMRegressor = lambda **kw: original_lgb(**(kw | {'n_jobs':2}))
    deadline.TREE_PARAMETERS['thread_count'] = 2
    original_threads = torch.set_num_threads
    torch.set_num_threads = lambda count: original_threads(min(int(count), 2))
    torch.set_num_threads(2)


def save_predictions(path, samples, values):
    from analysis.relationship_benchmark import rows
    pd.DataFrame([row for name,pred in values.items() for row in rows(samples,name,pred)]).to_csv(path,index=False)


def read_values(path, samples):
    frame = pd.read_csv(path)
    ids = [sample['case_id'] for sample in samples]
    values = {}
    for method, group in frame.groupby('method', sort=False):
        group = group.set_index('case_id')
        if not group.index.is_unique or set(group.index) != set(ids):
            raise ValueError('Prediction coverage differs: '+method)
        values[method] = group.loc[ids, 'predicted'].to_numpy()
    return values


def relationship_run(run, data, reference, folds):
    from analysis import relationship_benchmark as b
    from analysis.event_features import EventFeatures
    from analysis.structured_neural_benchmark import reference_prediction
    run.mkdir(); (run/'models').mkdir()
    train = data.inputs('train'); valid = data.inputs('validation')
    labels = data.labels('train'); val_labels = data.labels('validation')
    meta = EventFeatures.load(data.metadata_directory)
    frame = b.relationship_frame(train,meta); vf = b.relationship_frame(valid,meta)
    rid = reference_identity(reference,data.manifest['dataset_id'])
    config = {'dataset_id':data.manifest['dataset_id'],'recipes':b.RECIPES,'reference':rid,
              'folds':folds,'tree_threads':2,'fold_variants':['full','no_band','no_peer','no_type','no_quality','invariant']}
    write_json(run/'run_manifest.json',config)
    cv = []
    for fold in folds:
        ti = [i for i,s in enumerate(train) if s['curve'].event in fold['training_events']]
        vi = [i for i,s in enumerate(train) if s['curve'].event in fold['heldout_events']]
        ts = [train[i] for i in ti]; vs = [train[i] for i in vi]
        for variant in config['fold_variants']:
            name = 'catboost_'+variant
            bundle = b.fit_recipe(b.RECIPES[name],frame.iloc[ti],ts,labels)
            joblib.dump(bundle,run/f'models/cv_{fold["fold"]}_{name}.joblib')
            pred = b.predict_bundle(bundle,frame.iloc[vi],vs)
            cv.extend(row | {'fold':fold['fold']} for row in b.rows(vs,name,pred))
        print('RELATIONSHIP_CV '+str(fold['fold']),flush=True)
    pd.DataFrame(cv).to_csv(run/'training_oof_predictions.csv',index=False)
    train_values = {}; values = {'reference':reference_prediction(rid,'validation',valid)}; records = []
    for name,recipe in b.RECIPES.items():
        started = time.perf_counter(); bundle = b.fit_recipe(recipe,frame,train,labels)
        path = run/f'models/{name}.joblib'; joblib.dump(bundle,path); bundle = joblib.load(path)
        train_values[name] = b.predict_bundle(bundle,frame,train)
        values[name] = b.predict_bundle(bundle,vf,valid)
        record = {'method':name,'validation_mape':b.score(valid,val_labels,values[name]),
                  'seconds':time.perf_counter()-started,'model_sha256':sha256(path)}
        records.append(record); print(json.dumps(record),flush=True)
    top = sorted(b.RECIPES,key=lambda n:(b.score(valid,val_labels,values[n]),n))[:3]
    for collection in [train_values,values]:
        collection['new_top3'] = np.mean([collection[n] for n in top],axis=0)
    for weight in [.25,.5,.75]:
        values[f'blend_new_{weight}'] = weight*values['new_top3']+(1-weight)*values['reference']
    scores = {name:b.score(valid,val_labels,pred) for name,pred in values.items()}
    selection = {'selected_recipe':min(scores,key=lambda n:(scores[n],n)), 'top3_members':top,
                 'validation_scores':scores,'reported_methods':list(values),'test_labels_used_for_selection':False}
    write_json(run/'selection.json',selection); write_json(run/'training_records.json',records)
    save_predictions(run/'train_predictions.csv',train,train_values)
    save_predictions(run/'validation_predictions.csv',valid,values)
    # All choices above are finalized before the blind test is requested.
    test = data.inputs('test'); tf = b.relationship_frame(test,meta)
    test_values = {'reference':reference_prediction(rid,'test',test)}
    for name in b.RECIPES:
        test_values[name] = b.predict_bundle(joblib.load(run/f'models/{name}.joblib'),tf,test)
    test_values['new_top3'] = np.mean([test_values[n] for n in top],axis=0)
    for weight in [.25,.5,.75]:
        test_values[f'blend_new_{weight}'] = weight*test_values['new_top3']+(1-weight)*test_values['reference']
    save_predictions(run/'test_predictions.csv',test,test_values)


def adapted_run(family, run, data, reference, folds):
    """Execute existing fixed menus and add complete blind inference to dev-only ones."""
    from analysis.event_features import EventFeatures
    from analysis.structured_neural_benchmark import reference_prediction
    from analysis.relationship_benchmark import decode
    if family == 'innovation':
        from analysis import innovation_runner as b
        b.DATASET_IDS = b.DATASET_IDS | {'v1':DATASET_ID}
        old_sources = b.local_sources
        b.local_sources = lambda paths: old_sources([*paths,Path(__file__)])
        b.run_experiment(data,run,protocol='v1')
        return
    if family == 'relationship':
        return relationship_run(run,data,reference,folds)
    rid = reference_identity(reference,DATASET_ID)
    if family == 'structured':
        from analysis import structured_neural_benchmark as b
        b.SOURCE_NAMES = [*b.SOURCE_NAMES,'training/analysis/dense_v1_extensions.py']
        args = SimpleNamespace(run=run,reference=reference,device='cpu',epochs=100,batch_size=256,resume=False)
        b.develop(args,data)
        train = data.inputs('train'); metadata = EventFeatures.load(data.metadata_directory)
        frame = b.select_features(b.relationship_frame(train,metadata),'invariant')
        prep = joblib.load(run/'preprocessor.joblib')
        arrays = b.tensor_arrays(b.arrays(train,frame,prep),'cpu')
        config = json.loads((run/'run_manifest.json').read_text()); values = {}
        for name in config['recipes']:
            for seed in config['seeds']:
                model = b.load_model(run/f'models/{name}_{seed}.pt',config)
                values[f'{name}_seed{seed}'] = decode(train,b.infer(model,arrays),'remaining')
        b.ensembles(values,config)
        selection = json.loads((run/'selection.json').read_text())
        values['neural_top3'] = np.mean([values[n] for n in selection['top3_members']],axis=0)
        save_predictions(run/'train_predictions.csv',train,values)
        b.predict(args,data)
        return
    if family == 'deadline':
        from analysis import deadline_benchmark as b
    else:
        from analysis import quality_irregular_benchmark as b
        if family == 'quality_extended':
            b.NEURAL_SETTINGS = b.NEURAL_SETTINGS | {'epochs':240}
    b.FOLD_LEFT = [len(fold['training_events']) for fold in folds]
    b.FOLD_SIZE = len(folds[0]['heldout_events'])
    b.SOURCES = [*b.SOURCES,'training/analysis/dense_v1_extensions.py']
    b.develop(run,data,reference)
    metadata = EventFeatures.load(data.metadata_directory); history = data.train_curves()
    selection = json.loads((run/'selection.json').read_text())
    config = json.loads((run/'run_manifest.json').read_text())
    for split in ['train','test']:
        samples = data.inputs(split)
        values = {} if split == 'train' else {'current_reference':reference_prediction(rid,split,samples)}
        if family == 'deadline':
            base,extra = b.frame_for(samples,history,metadata)
            for name in b.RECIPES:
                bundle = joblib.load(run/f'models/final_{name}.joblib')
                values[name] = b.predict(bundle,samples,base if name=='remaining_rank' else extra)
            calibration = joblib.load(run/'models/final_calibration.joblib')
            for shrink in b.SHRINKAGES:
                values[f'oof_correction_{shrink}'] = b.calibrated(calibration,samples,values['remaining_rank'],shrink)
        else:
            base,quality = b.frames(samples,history,metadata)
            for name in b.hierarchy.RECIPES:
                bundle = joblib.load(run/f'models/final_{name}.joblib')
                values[name] = b.hierarchy.predict(bundle,samples,base,quality)
            prep = joblib.load(run/'models/neural_preprocessor.joblib')
            arrays = b.arrays(samples,base,prep,b.NEURAL_SETTINGS['nodes'])
            for name in b.NEURAL:
                for seed in b.SEEDS:
                    model = b.load_model(run/f'models/{name}_{seed}.pt',config['run_id'])
                    values[f'{name}_seed{seed}'] = decode(samples,b.infer(model,arrays),'remaining')
                values[f'{name}_ensemble3'] = np.mean([values[f'{name}_seed{s}'] for s in b.SEEDS],axis=0)
            if split == 'test':
                for name in [*b.hierarchy.RECIPES,*[f'{n}_ensemble3' for n in b.NEURAL]]:
                    values[f'blend25_{name}'] = .25*values[name]+.75*values['current_reference']
        save_predictions(run/f'{split}_predictions.csv',samples,values)


def run_family(family, run, reference=None, dataset=None):
    if family not in FAMILIES:
        raise ValueError('Unknown extension family')
    data = BlindDataset(load_dataset('v1',dataset))
    if data.manifest['dataset_id'] != DATASET_ID:
        raise ValueError('Dense V1 fingerprint differs')
    if family != 'innovation':
        if reference is None:
            raise ValueError('A sealed reference from the same dense cohort is required')
        reference = Path(reference)
        reference_identity(reference,DATASET_ID)
    run = Path(run)
    if run.exists():
        raise FileExistsError('New run directory required; preserved existing run: '+str(run))
    folds = dense_folds(data.train_curves())
    configure_runtime()
    from analysis.innovation_common import local_sources
    from analysis.relationship_benchmark import score
    sources = local_sources([Path(__file__)])
    config = {'family':family,'dataset_id':DATASET_ID,'dataset_directory':str(data.root.resolve()),
              'minimum_event_start_ms':MIN_EVENT_START_MS,'data_policy_id':POLICY_ID,
              'cpu_threads':2,'folds':folds,'seeds':[42,123,2026],
              'reference':reference_identity(reference,DATASET_ID) if reference else None,
              'selection':'original registered menu and validation criterion; no test-label access',
              'previous_test_exposure':True,'validation_repeatedly_used':True,
              'metadata_limitation':'retrospective snapshot without historical-time archives',
              'source_hashes':{name:sha256(ROOT/name) for name in sources},
              'software':{name:importlib.metadata.version(name) for name in
                          ['numpy','pandas','scipy','scikit-learn','catboost','lightgbm','joblib','torch']}}
    config['run_id'] = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    run.mkdir(parents=True); write_json(run/'run_manifest.json',config)
    for name in sources:
        target = run/'sources'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    started = time.perf_counter()
    child = run/'development'
    adapted_run(family,child,data,reference,folds)
    selected = json.loads((child/'selection.json').read_text())
    valid = data.inputs('validation'); train = data.inputs('train'); test = data.inputs('test')
    validation_values = read_values(child/'validation_predictions.csv',valid)
    test_values = read_values(child/'test_predictions.csv',test)
    if set(validation_values) != set(test_values):
        raise ValueError('Blind test method coverage differs from validation')
    selection = selected | {'run_id':config['run_id'],'dataset_id':DATASET_ID,
                           'selected_recipe':selected.get('selected_recipe',selected.get('selected_on_validation')),
                           'reported_methods':list(validation_values),'test_labels_used_for_selection':False}
    write_json(run/'selection.json',selection)
    for split in ['train','validation','test']:
        shutil.copyfile(child/f'{split}_predictions.csv',run/f'{split}_predictions.csv')
    fit_values = read_values(run/'train_predictions.csv',train)
    fit_labels = data.labels('train'); val_labels = data.labels('validation')
    summary = [{'method':name,'train_fit_mape_pct':score(train,fit_labels,pred),
                'validation_mape_pct':score(valid,val_labels,validation_values[name])}
               for name,pred in fit_values.items()]
    pd.DataFrame(summary).to_csv(run/'train_validation_summary.csv',index=False)
    write_json(run/'runtime.json',{'seconds':time.perf_counter()-started,'training_cases':len(train),
                                  'validation_cases':len(valid),'test_cases':len(test),
                                  'reported_methods':len(test_values),'train_fit_methods':len(fit_values)})
    for name,digest in config['source_hashes'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Source changed during retraining: '+name)
    write_json(run/'blind_test_manifest.json',{'run_id':config['run_id'],'dataset_id':DATASET_ID,
               'dataset_directory':str(data.root.resolve()),'test_cases':len(test),
               'artifacts':{str(p.relative_to(run)):sha256(p) for p in sorted(run.rglob('*')) if p.is_file()}})
    print('DENSE_BLIND_SEALED '+json.dumps({'family':family,'run':str(run),'selected':selection['selected_recipe']}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=FAMILIES,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--dataset',type=Path)
    args = parser.parse_args()
    run_family(args.family,args.run,args.reference,args.dataset)


if __name__ == '__main__':
    main()
