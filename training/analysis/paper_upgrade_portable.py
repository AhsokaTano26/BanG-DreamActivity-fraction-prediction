"""Trusted export and portable replay of the already sealed fixed experiment.

The capsule contains archived blind targets and a deduplicated historical
feedback vault. Only this coordinator opens that vault and releases rows with
end strictly before the query start. Frozen predictors receive detached frames,
never the global feedback table, file paths, raw labels or a dataset object.

Runtime dependencies: Python standard library, NumPy, pandas, and the two exact
frozen pure source modules shipped inside the capsule. No tree/neural weights,
SQLite, original workspace, internet access or raw answer files are required.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import types
import numpy as np
import pandas as pd

MIN_START=1706504400000
TARGET_COLUMNS=['case_id','event','rank','phase','progress','start_ms','end_ms',
                'cutoff','observed_at','current','event_type','band_group','split',
                'duration_h','remaining_h','rate_6h_relative','rate_24h_relative','base_prediction']
HISTORY_COLUMNS=TARGET_COLUMNS+['log_error','expert__independent']
FORMAL_RUN='dcc493f82e83933d6963e2d39a02e6c624057ad6c0932ad24801efbd252b1f09'
DEFAULT_REPLAY=Path(__file__).resolve().parents[2]/'analysis/paper_upgrade_records/20261118/runs/fixed_replay_v2'
DEFAULT_CAPSULE=Path(__file__).resolve().parents[2]/'analysis/paper_upgrade_records/20261118/portable_v1/capsule'


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def load_json(path):return json.loads(Path(path).read_text())


def read_table(path):
    return pd.read_csv(path,dtype={'case_id':'str','event_type':'str','band_group':'str'},low_memory=False)


def verify_seal(path):
    """Compatible with existing seals; reject path traversal and changed bytes."""
    path=Path(path);body=load_json(path)
    if identity({k:v for k,v in body.items() if k!='seal_id'})!=body.get('seal_id'):raise ValueError('Sealed manifest changed: '+str(path))
    for name,digest in body['artifacts'].items():
        artifact=path.parent/name
        if not artifact.resolve().is_relative_to(path.parent.resolve()):raise ValueError('Artifact escaped its sealed directory')
        if sha256(artifact)!=digest:raise ValueError('Sealed artifact changed: '+str(artifact))
    return body


def seal_directory(directory,name,fields):
    directory=Path(directory)
    body=dict(fields)|{'artifacts':{str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}}
    body['seal_id']=identity(body);write_json(directory/name,body);return body


def case_ids_hash(case_ids):
    ids=list(map(str,case_ids))
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate physical case IDs')
    return identity(sorted(ids))


def _unique(frame):
    if 'case_id' not in frame or frame.case_id.isna().any() or frame.case_id.duplicated().any():raise ValueError('Unique nonmissing physical case IDs required')


def _compare_frames(actual,expected,*,numeric_tolerance=False):
    """Compare aligned case-indexed records, retaining real category conflicts."""
    if not actual.index.equals(expected.index) or list(actual.columns)!=list(expected.columns):raise ValueError('Case/schema alignment differs')
    for column in actual:
        a=actual[column];b=expected[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av=a.to_numpy(float);bv=b.to_numpy(float)
            same=np.allclose(av,bv,rtol=1e-12,atol=1e-6,equal_nan=True) if numeric_tolerance else np.array_equal(av,bv,equal_nan=True)
        else:same=a.fillna('__missing__').astype(str).equals(b.fillna('__missing__').astype(str))
        if not same:raise ValueError('Conflicting value for case column: '+column)


def merge_history(existing,incoming):
    """Repeated snapshots may repeat cases, but never change their stored values."""
    _unique(existing);_unique(incoming)
    if list(existing.columns)!=list(incoming.columns):raise ValueError('Historical snapshot schema differs')
    if existing.empty:return incoming.copy(deep=True)
    old=existing.set_index('case_id',drop=False);new=incoming.set_index('case_id',drop=False)
    common=new.index.intersection(old.index)
    _compare_frames(new.loc[common],old.loc[common])
    return pd.concat([existing,incoming[~incoming.case_id.isin(existing.case_id)]],ignore_index=True)


def history_for_query(global_history,query):
    """Guard before numerical errors/center values are decoded by a predictor."""
    _unique(global_history);start=int(query['start_ms']);event=int(query['event'])
    if start<MIN_START:raise ValueError('Query violates dense-period policy')
    if len(global_history) and ((global_history.start_ms<MIN_START).any() or (global_history.end_ms<=global_history.start_ms).any()):raise ValueError('Invalid historical event clock')
    permitted=global_history[(global_history.end_ms<start)&(global_history.event!=event)].copy(deep=True)
    if len(permitted)!=query['history_cases'] or case_ids_hash(permitted.case_id)!=query['history_case_ids_sha256']:raise ValueError('Missing or additional historical cases at query start')
    if (permitted.end_ms>=start).any() or (permitted.event==event).any():raise ValueError('Current/future feedback was released')
    detached=permitted[TARGET_COLUMNS+['log_error']].copy(deep=True)
    if not np.isfinite(detached.log_error.to_numpy(float)).all():raise ValueError('Invalid released forecast errors')
    expert=permitted.expert__independent.to_numpy(float)
    if not np.isfinite(expert).all() or (expert<=0).any():raise ValueError('Invalid issued interval center')
    detached['center_log_error']=np.log(expert/detached.base_prediction.to_numpy(float))+detached.log_error.to_numpy(float)
    return detached


def _load_frozen_models(capsule):
    """Load exact capsule bytes in a temporary package, without repo imports."""
    directory=Path(capsule)/'analysis';names=['analysis','analysis.rolling_residual_stats','analysis.paper_upgrade_replay_models']
    saved={name:sys.modules.get(name) for name in names}
    old_bytecode_policy=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    package=types.ModuleType('analysis');package.__path__=[str(directory)];sys.modules['analysis']=package
    try:
        modules={}
        for short in ['rolling_residual_stats','paper_upgrade_replay_models']:
            name='analysis.'+short;spec=importlib.util.spec_from_file_location(name,directory/(short+'.py'))
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;setattr(package,short,module);spec.loader.exec_module(module);modules[short]=module
        result=modules['paper_upgrade_replay_models']
        if result.TARGET_COLUMNS!=TARGET_COLUMNS:raise ValueError('Frozen target schema differs')
        return result
    finally:
        sys.dont_write_bytecode=old_bytecode_policy
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value


def _write_compact(frame,path):
    frame.to_csv(path,index=False,compression={'method':'gzip','compresslevel':9,'mtime':0})


def _check_case_blind_values(history,target):
    _unique(history);_unique(target)
    if not set(history.case_id)<=set(target.case_id):raise ValueError('History case lacks a sealed blind target')
    actual=history.set_index('case_id')[TARGET_COLUMNS[1:]]
    expected=target.set_index('case_id').loc[actual.index,TARGET_COLUMNS[1:]]
    _compare_frames(actual,expected)


def export_capsule(replay_run,capsule,*,allow_fixture=False):
    """Trusted one-time export; all source hashes and time memberships are checked."""
    replay_run,capsule=map(Path,[replay_run,capsule])
    if capsule.exists():raise FileExistsError('Preserve existing/failed capsule; choose a new destination')
    blind=verify_seal(replay_run/'blind_manifest.json');cfg=load_json(replay_run/'run_manifest.json')
    if blind['run_id']!=cfg['run_id'] or (not allow_fixture and cfg['run_id']!=FORMAL_RUN):raise ValueError('Wrong fixed sealed replay')
    source=Path(cfg['source_run']);source_blind=verify_seal(source/'blind_manifest.json')
    if (source_blind['run_id']!=cfg['source_run_id'] or sha256(source/'blind_manifest.json')!=cfg['source_blind_sha256']):raise ValueError('Wrong original released forecast source')
    capsule.mkdir(parents=True);(capsule/'analysis').mkdir();(capsule/'data').mkdir()
    (capsule/'analysis/__init__.py').write_text('"""Frozen portable replay modules."""\n')
    frozen={}
    for name in ['paper_upgrade_replay_models.py','rolling_residual_stats.py']:
        origin=replay_run/'sources/analysis'/name
        if sha256(origin)!=cfg['source_hashes']['analysis/'+name]:raise ValueError('Frozen pure implementation differs')
        shutil.copyfile(origin,capsule/'analysis'/name);frozen['analysis/'+name]=sha256(origin)
    shutil.copyfile(__file__,capsule/'analysis/paper_upgrade_portable.py')
    models=_load_frozen_models(capsule)
    if set(cfg['methods'])!=set(models.menu()):raise ValueError('Frozen method menu differs')
    event_seals=[]
    for path in (replay_run/'events').glob('*/event_manifest.json'):
        sealed=verify_seal(path)
        if sealed['run_id']!=cfg['run_id']:raise ValueError('Query run binding differs')
        event_seals.append((sealed,path))
    event_seals.sort(key=lambda item:(item[0]['query_start_ms'],item[0]['event']))
    if not allow_fixture and len(event_seals)!=65:raise ValueError('Expected all65 original query activities')
    targets=[];references=[];intervals=[];queries=[];provenance=[]
    unique=pd.DataFrame(columns=HISTORY_COLUMNS);repeated_history_rows=0
    query_events={int(s['event']):s for s,_ in event_seals}
    for sealed,path in event_seals:
        event=int(sealed['event']);start=int(sealed['query_start_ms']);directory=path.parent;original=source/'events'/str(event)
        source_event=verify_seal(original/'event_manifest.json');audit=load_json(directory/'audit.json')
        if (source_event['run_id']!=cfg['source_run_id'] or source_event['query_start_ms']!=start or source_event['event']!=event
                or source_event['end_ms']!=sealed['end_ms'] or source_event['split']!=sealed['split']
                or audit.get('current_outcome_released') is not False):raise ValueError('Invalid query/source clock')
        if audit['source_event_manifest_sha256']!=sha256(original/'event_manifest.json'):raise ValueError('Source event binding differs')
        available=[r for r in audit['released_forecasts']]
        expected={e for e,r in query_events.items() if r['end_ms']<start}
        if len(available)!=len(expected) or {int(r['event']) for r in available}!=expected:raise ValueError('Incorrect forecast release cohort')
        # Validate metadata and all prior forecast seals before opening errors.
        for release in available:
            r=query_events[int(release['event'])]
            if (release['start_ms']<MIN_START or release['end_ms']>=start or release['start_ms']!=r['query_start_ms'] or release['end_ms']!=r['end_ms']):raise ValueError('Illegal forecast feedback clock')
            if release['prediction_seal_sha256']!=sha256(source/'events'/str(release['event'])/'event_manifest.json'):raise ValueError('Prior forecast seal binding differs')
        history_path=original/'released_residuals.csv'
        if sha256(history_path)!=audit['history_artifact_sha256']:raise ValueError('Released snapshot binding differs')
        target=read_table(directory/'features.csv')[TARGET_COLUMNS]
        if set(target.event)!={event} or not (target.start_ms==start).all() or len(target)!=sealed['cases']:raise ValueError('Blind target query differs')
        _unique(target);targets.append(target)
        history=read_table(history_path)
        # The original first query has zero feedback rows and no expert column
        # yet. Add only an empty typed column, never impute a nonempty history.
        if history.empty and 'expert__independent' not in history:
            history['expert__independent']=pd.Series(dtype=float)
        if set(HISTORY_COLUMNS)-set(history):raise ValueError('Missing compact feedback fields')
        history=history[HISTORY_COLUMNS].copy();_unique(history)
        if set(history.event)!=expected or len(history)!=audit['history_cases']:raise ValueError('Source snapshot cases/events differ')
        query={'event':event,'split':sealed['split'],'start_ms':start,'end_ms':int(sealed['end_ms']),
               'target_cases':len(target),'target_case_ids_sha256':case_ids_hash(target.case_id),
               'history_cases':len(history),'history_case_ids_sha256':case_ids_hash(history.case_id),
               'history_events':len(expected),'history_event_ids_sha256':identity(sorted(expected))}
        history_for_query(history,query)
        # Merge and verify now; retaining all65 repeated snapshots would defeat
        # the exporter's bounded-memory purpose. Query membership is hashed.
        unique=merge_history(unique,history);repeated_history_rows+=len(history);queries.append(query)
        prediction=read_table(directory/'predictions.csv')
        if prediction.duplicated(['case_id','method']).any() or set(prediction.method)!=set(cfg['methods']):raise ValueError('Incomplete original prediction grid')
        wide=prediction.pivot(index='case_id',columns='method',values='predicted')
        if set(wide.index)!=set(target.case_id) or wide.isna().any().any():raise ValueError('Missing original method/case')
        wide=wide.reindex(index=target.case_id,columns=cfg['methods']).reset_index();references.append(wide)
        interval=read_table(directory/'intervals.csv')
        if interval.duplicated(['case_id','level','method']).any() or set(interval.case_id)!=set(target.case_id) or len(interval)!=2*len(target):raise ValueError('Incomplete original interval panel')
        intervals.append(interval)
        provenance.append({'event':event,'query_start_ms':start,'fixed_event_manifest_sha256':sha256(path),
                           'source_event_manifest_sha256':sha256(original/'event_manifest.json'),'released_snapshot_sha256':sha256(history_path),
                           'fixed_target_sha256':sha256(directory/'features.csv'),'fixed_predictions_sha256':sha256(directory/'predictions.csv'),
                           'fixed_intervals_sha256':sha256(directory/'intervals.csv')})
    target_table=pd.concat(targets,ignore_index=True);_unique(target_table)
    _check_case_blind_values(unique,target_table)
    for query in queries:history_for_query(unique,query)
    reference_table=pd.concat(references,ignore_index=True);_unique(reference_table)
    issued=reference_table.set_index('case_id').loc[unique.case_id,'ewma_reference'].to_numpy(float)
    if not np.allclose(issued,unique.expert__independent.to_numpy(float),rtol=1e-12,atol=1e-6):raise ValueError('Historical issued interval center differs')
    _write_compact(target_table,capsule/'data/targets.csv.gz');_write_compact(unique,capsule/'data/released_history.csv.gz')
    _write_compact(reference_table,capsule/'data/reference_predictions_wide.csv.gz')
    _write_compact(pd.concat(intervals,ignore_index=True),capsule/'data/reference_intervals.csv.gz')
    write_json(capsule/'queries.json',queries);write_json(capsule/'source_provenance.json',provenance)
    shutil.copyfile(replay_run/'frozen_dlm_parameters.json',capsule/'frozen_dlm_parameters.json')
    parameters=load_json(capsule/'frozen_dlm_parameters.json')
    first_validation=next(q for q in queries if q['split']=='validation')
    train_history=history_for_query(unique,first_validation)
    if set(train_history['split'])!={'train'}:raise ValueError('DLM fit boundary is not training only')
    starts=tuple(q['start_ms'] for q in queries);reestimated={'complete':models.dlm_parameters(train_history)}
    for name,policy in models.POLICIES.items():reestimated[name]=models.dlm_parameters(models.feedback_history(train_history,query_start_ms=first_validation['start_ms'],policy=policy,query_starts=starts))
    if parameters!=reestimated:raise ValueError('Frozen DLM training parameters are not reproduced exactly')
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas']}
    (capsule/'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name,version in versions.items()))
    (capsule/'README.md').write_text('''# Portable fixed replay capsule

This reproduces archived54-method forecasts on65 query events; it does not fit
the original base trees or calculate new accuracy scores. Source answer files,
history-vault curves, old model weights, SQLite and the original repo are not
needed. The archived history includes outcomes encoded as previously released
forecast errors. Only the trusted coordinator reads it, applies strict
end<query-start release, and passes detached permitted rows to frozen predictors.

Use Python with the exact NumPy/pandas versions in requirements.txt. From the
extracted capsule directory (keep output outside sealed inputs):

```text
python -m analysis.paper_upgrade_portable verify --capsule .
python -m analysis.paper_upgrade_portable replay --capsule . --output ../portable_replay_output
```

No internet/network operation is performed. A fresh output directory is required;
failed outputs are preserved. The receipt compares every54-method point forecast
and every80/90% interval with the original sealed reference, including infinities,
availability and support. This is forecast parity, not accuracy model selection.
Independent scoring requires the separately packaged scorer and frozen answers;
the new two-rank learning-curve data/protocol are a separate reproducibility task.
''')
    summary={'schema_version':1,'protocol':'portable_fixed_replay_capsule_v1','source_run_id':cfg['run_id'],
             'dataset_id':cfg['dataset_id'],'source_blind_manifest_sha256':sha256(replay_run/'blind_manifest.json'),
             'released_source_blind_manifest_sha256':sha256(source/'blind_manifest.json'),'events':len(queries),'methods':cfg['methods'],
             'target_cases':len(target_table),'unique_history_cases':len(unique),'repeated_history_rows':repeated_history_rows,
             'query_membership_sha256':identity(queries),'frozen_model_sources':frozen,'software':versions,'python':platform.python_version(),
             'created_utc':datetime.now(timezone.utc).isoformat(),'model_access':'detached legal frames only; coordinator owns archived historical errors',
             'not_included':['base-model training reproducibility','accuracy labels/scorer','new learning-curve fits'],
             'dependency_scope':'standard library + NumPy + pandas + bundled exact models/stats/portable modules; no original repo required for replay'}
    seal_directory(capsule,'capsule_manifest.json',summary)
    return summary


def _interval_parity(actual,reference):
    keys=['case_id','level','method']
    if actual.duplicated(keys).any() or reference.duplicated(keys).any():raise ValueError('Duplicate interval key')
    a=actual.set_index(keys).sort_index();b=reference.set_index(keys).sort_index()
    if set(a.columns)!=set(b.columns):raise ValueError('Interval schema differs')
    _compare_frames(a,b[a.columns],numeric_tolerance=True)
    return float(max([np.max(np.abs(a[c].to_numpy(float)[np.isfinite(a[c].to_numpy(float))]-b[c].to_numpy(float)[np.isfinite(a[c].to_numpy(float))]),initial=0.) for c in ['lower','upper','q_log']]))


def replay_capsule(capsule,output):
    capsule,output=map(Path,[capsule,output]);manifest=verify_seal(capsule/'capsule_manifest.json')
    if output.exists():raise FileExistsError('Preserve old/failed replay output; choose new directory')
    if any(importlib.metadata.version(k)!=v for k,v in manifest['software'].items()):raise ValueError('Runtime NumPy/pandas differs from frozen capsule requirements')
    queries=load_json(capsule/'queries.json')
    if identity(queries)!=manifest['query_membership_sha256']:raise ValueError('Query schedule changed')
    models=_load_frozen_models(capsule);targets=read_table(capsule/'data/targets.csv.gz');global_history=read_table(capsule/'data/released_history.csv.gz')
    references=read_table(capsule/'data/reference_predictions_wide.csv.gz');interval_references=read_table(capsule/'data/reference_intervals.csv.gz')
    _unique(targets);_unique(global_history);_unique(references);_check_case_blind_values(global_history,targets)
    if len(targets)!=manifest['target_cases'] or len(global_history)!=manifest['unique_history_cases'] or set(references.case_id)!=set(targets.case_id):raise ValueError('Capsule global case coverage differs')
    if set(references)-{'case_id'}!=set(manifest['methods']):raise ValueError('Reference methods differ')
    parameters=load_json(capsule/'frozen_dlm_parameters.json');starts=tuple(q['start_ms'] for q in queries)
    output.mkdir(parents=True);max_relative=0.;max_interval=0.;rows=[];interval_rows=[];sealed_events=set();began=time.monotonic()
    for query in queries:
        event=query['event'];target=targets[targets.event==event][TARGET_COLUMNS].copy(deep=True)
        if len(target)!=query['target_cases'] or case_ids_hash(target.case_id)!=query['target_case_ids_sha256'] or not (target.start_ms==query['start_ms']).all():raise ValueError('Target query membership differs')
        history=history_for_query(global_history,query)
        if not set(history.event)<=sealed_events:raise ValueError('Feedback forecast has not been sealed in this replay')
        supplied=None if query['split']=='train' else parameters
        scores,_=models.predict_all(history,target,query_start_ms=query['start_ms'],parameters=supplied,query_starts=starts)
        if set(scores)!=set(manifest['methods']):raise ValueError('Predictor menu differs')
        reference=references.set_index('case_id').loc[target.case_id,manifest['methods']]
        # Some frozen outputs are Series with source integer indices; preserve
        # row order explicitly instead of allowing case-ID index alignment.
        predicted=pd.DataFrame({k:np.asarray(v,float) for k,v in scores.items()},index=target.case_id).reindex(columns=manifest['methods'])
        if not np.allclose(predicted.to_numpy(),reference.to_numpy(float),rtol=1e-12,atol=1e-6):raise ValueError('Portable point prediction parity failed at event '+str(event))
        relative=float(np.max(np.abs(predicted.to_numpy()/reference.to_numpy(float)-1)));max_relative=max(max_relative,relative)
        interval=models.empirical_intervals(history,target,scores['ewma_reference'])
        original=interval_references[interval_references.case_id.isin(target.case_id)]
        interval_delta=_interval_parity(interval,original);max_interval=max(max_interval,interval_delta)
        # Seal the complete event's issued outputs before any later feedback.
        event_dir=output/'events'/str(event);event_dir.mkdir(parents=True)
        _write_compact(predicted.reset_index(),event_dir/'predictions_wide.csv.gz');_write_compact(interval,event_dir/'intervals.csv.gz')
        write_json(event_dir/'audit.json',{'event':event,'query_start_ms':query['start_ms'],'history_cases':len(history),
                   'history_case_ids_sha256':case_ids_hash(history.case_id),'history_event_ids_sha256':identity(sorted(map(int,history.event.unique()))),
                   'current_outcome_released':False,'prediction_max_relative_difference':relative,'interval_max_absolute_difference':interval_delta})
        seal_directory(event_dir,'event_manifest.json',{'event':event,'query_start_ms':query['start_ms'],'end_ms':query['end_ms'],'cases':len(target),'methods':manifest['methods']})
        sealed_events.add(event);rows.append(predicted.reset_index());interval_rows.append(interval)
        print(json.dumps({'portable_sealed_event':event,'split':query['split'],'cases':len(target),'history_cases':len(history),'max_relative':relative}),flush=True)
    _write_compact(pd.concat(rows,ignore_index=True),output/'predictions_wide.csv.gz');_write_compact(pd.concat(interval_rows,ignore_index=True),output/'intervals.csv.gz')
    receipt={'status':'pass','capsule_manifest_sha256':sha256(capsule/'capsule_manifest.json'),'events':len(queries),'methods':len(manifest['methods']),
             'cases':len(targets),'point_forecast_max_relative_difference':max_relative,'interval_max_absolute_difference':max_interval,
             'rtol':1e-12,'atol':1e-6,'seconds':time.monotonic()-began,'original_repo_or_weights_required':False,
             'accuracy_scoring_performed':False,'global_history_exposed_to_predictor':False,'python':platform.python_version(),'software':manifest['software']}
    write_json(output/'replay_receipt.json',receipt);seal_directory(output,'replay_manifest.json',{'protocol':'portable_forecast_parity_v1','status':'pass','capsule_manifest_sha256':sha256(capsule/'capsule_manifest.json')})
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['export','replay','verify']);parser.add_argument('--replay-run',type=Path,default=DEFAULT_REPLAY);parser.add_argument('--capsule',type=Path,default=DEFAULT_CAPSULE);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='export':result=export_capsule(args.replay_run,args.capsule)
    elif args.action=='verify':m=verify_seal(args.capsule/'capsule_manifest.json');result={'status':'pass','events':m['events'],'methods':len(m['methods'])}
    else:
        if args.output is None:parser.error('replay requires a fresh --output directory')
        result=replay_capsule(args.capsule,args.output)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
