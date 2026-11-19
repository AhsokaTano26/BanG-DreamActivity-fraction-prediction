"""Warm prequential residual learning on unchanged dense V3 Context forecasts."""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.current_benchmark import load_dataset
from analysis.event_features import EventFeatures
from analysis.holdout_dataset import sha256
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import (OnlineBias,verify_seal,seal_directory,write_new_json,
    fit_context,_training,_save_model,_weight_rows)
from analysis.rolling_residual_benchmark import aligned_base,_frame,FRAME_COLUMNS
from analysis.temporal_paired_features import prepare_paired
from analysis import v2_next_adaptation as adaptation
from analysis import v3_next_calibration as calibration

BASE=ROOT/'analysis/rolling_results/v3_dense_full_retrain_20261115'
RUN=ROOT/'analysis/rolling_results/v3_next_warm_20261116'
DATASET_ID='85f5924e0b90deb0e1a30f3aa27d2237640cca4d968d96612dda8fee1d9c72ec'
CANDIDATES=['uncorrected','ewma_reference',*calibration.CONFIGS]
METHODS=CANDIDATES+[f'factorized_mlp_seed{s}' for s in [42,123,2026]]
EXPERT_COLUMNS=['expert__'+name for name in calibration.EXPERT_NAMES]
FEATURE_COLUMNS=FRAME_COLUMNS+['cutoff','observed_at','peer_log_ratio']


def _configuration(data,base,*,warmup_events,iterations,allow_fixture):
    seal=verify_seal(base/'blind_manifest.json')
    base_cfg=json.loads((base/'run_manifest.json').read_text())
    if seal['dataset_id']!=data.manifest['dataset_id']:raise ValueError('Base cache uses different cases')
    if (base_cfg['run_id']!=seal['run_id'] or identity({k:v for k,v in base_cfg.items() if k!='run_id'})!=seal['run_id']
            or seal.get('protocol')!='rolling_holdout_v3'):raise ValueError('Base cache configuration identity differs')
    if not allow_fixture and (base_cfg.get('allow_fixture') or base_cfg['base_recipe']!=(adaptation.recipes()[0]|{'iterations':400})):
        raise ValueError('Formal base cache must use the exact unchanged400-treeContext recipe')
    train=sorted([r for r in data.event_index.values() if r['split']=='train'],key=lambda r:(r['start_ms'],r['event']))
    if not 1<=warmup_events<len(train):raise ValueError('Warmup must leave at least one forecast activity')
    if not allow_fixture and (data.manifest['dataset_id']!=DATASET_ID or warmup_events!=32 or iterations!=400):
        raise ValueError('Formal run requires frozen dense V3,32 initial events and400 iterations')
    if any(r['end_ms']>=train[warmup_events]['start_ms'] for r in train[:warmup_events]):raise ValueError('Initial histories overlap warm forecast start')
    sources=local_sources([Path(__file__)])
    cfg={'protocol':'rolling_residual_v3','experiment':'warm_prequential_residual_optimization',
         'dataset_id':data.manifest['dataset_id'],'dataset_directory':str(data.root.resolve()),
         'base_run':str(base.resolve()),'base_run_id':seal['run_id'],'base_blind_sha256':sha256(base/'blind_manifest.json'),
         'warmup_events':[r['event'] for r in train[:warmup_events]],'warm_forecast_events':[r['event'] for r in train[warmup_events:]],
         'iterations':iterations,'warm_base_recipe':adaptation.recipes()[0]|{'iterations':iterations},
         'methods':METHODS,'candidates':CANDIDATES,'calibration_recipes':calibration.CONFIGS,'experts':list(calibration.EXPERT_NAMES),
         'feedback':'strictly completed already sealed forecasts; no in-sample residuals',
         'cold_control':'original Context EWMA updated from validation/test only, exact frozen replay',
         'warm_control':'same Context plus35earlier train prequential forecast errors',
         'selection':'minimum validation event-macro MAPE; fixed candidate order ties; never individual NN seeds',
         'scored_splits':['validation','test'],'training_scores':'Warm forecast errors are out-of-sample chronological diagnostics, not training resubstitution',
         'exposure':'Exploratory follow-up on previously exposed test/reused validation',
         'metadata_caveat':data.manifest['protocol']['metadata_caveat'],'allow_fixture':bool(allow_fixture),
         'python':platform.python_version(),'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','catboost','torch','joblib','scipy']},
         'source_hashes':{n:sha256(ROOT/n) for n in sources},
         'specification_sha256':sha256(ROOT/'docs/superpowers/specs/2026-11-16-v3-next-design.md')}
    cfg['run_id']=identity(cfg)
    return cfg


def check_cached_event(data,row,seal,audit,base_run_id):
    if (seal['event']!=row['event'] or seal['query_start_ms']!=row['start_ms'] or seal['end_ms']!=row['end_ms']
            or seal['split']!=row['split'] or seal['run_id']!=base_run_id
            or audit['query_start_ms']!=row['start_ms'] or audit.get('current_outcome_released') is not False):
        raise ValueError('Unsafe cached event identity/clock')
    expected={e:r for e,r in data.event_index.items() if r['end_ms']<row['start_ms']}
    history=audit['released_history']
    if len(history)!=len(expected) or {r['event'] for r in history}!=set(expected):raise ValueError('Cached history cohort differs')
    if set(audit['rolling_training_events'])!=set(expected):raise ValueError('Cached fitted events differ')
    for r in history:
        if any(r[k]!=expected[r['event']][k] for k in ['start_ms','end_ms','sha256']):raise ValueError('Cached history date/hash differs')


def _features(samples,metadata,base,state):
    frame=_frame(samples,metadata,base,state)
    paired=prepare_paired(samples)
    frame['cutoff']=[s['cutoff'] for s in samples]
    frame['observed_at']=[s['now'] for s in samples]
    ratio=pd.to_numeric(paired['paired'].pair_log_score_ratio,errors='coerce').to_numpy(float)
    frame['peer_log_ratio']=ratio*np.where(frame['rank'].to_numpy()==1000,1.,-1.)
    return frame,paired['audit']


def run_experiment(data,run,*,base_run=BASE,warmup_events=32,iterations=400,allow_fixture=False,resume=False):
    run,base_run=Path(run),Path(base_run)
    cfg=_configuration(data,base_run,warmup_events=warmup_events,iterations=iterations,allow_fixture=allow_fixture)
    if run.exists():
        if not resume:raise FileExistsError('Preserve existing run; use a new directory')
        if json.loads((run/'run_manifest.json').read_text())!=cfg:raise ValueError('Source/configuration changed')
        if not (run/'blind_manifest.json').exists():raise FileExistsError('Incomplete causal run cannot resume; preserve and use a new run')
        verify_seal(run/'blind_manifest.json');return json.loads((run/'selection.json').read_text())
    run.mkdir(parents=True);(run/'events').mkdir()
    write_new_json(run/'run_manifest.json',cfg)
    for name in cfg['source_hashes']:
        path=run/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    shutil.copyfile(ROOT/'docs/superpowers/specs/2026-11-16-v3-next-design.md',run/'specification.md')
    metadata=EventFeatures.load(data.metadata_directory)
    events=sorted(data.event_index.values(),key=lambda r:(r['start_ms'],r['event']))
    warm_ids=set(cfg['warm_forecast_events']);initial_ids=set(cfg['warmup_events'])
    queries=[r for r in events if r['event'] in warm_ids or r['split']!='train']
    samples_by_event=defaultdict(list)
    for split in ['train','validation','test']:
        for sample in data.inputs(split):samples_by_event[sample['curve'].event].append(sample)
    first_start=queries[0]['start_ms']
    initial={r['event']:data.reveal_event(r['event'],before_ms=first_start) for r in events if r['event'] in initial_ids}
    released_curves=dict(initial);released_forecasts={}
    history=pd.DataFrame(columns=FEATURE_COLUMNS+EXPERT_COLUMNS+['log_error'])
    warm_state=OnlineBias();cold_state=OnlineBias();predictions={}

    def release_before(start):
        nonlocal history
        for row in sorted(queries,key=lambda r:(r['end_ms'],r['event'])):
            event=row['event']
            if row['end_ms']>=start or event in released_forecasts:continue
            directory=run/'events'/str(event);seal=verify_seal(directory/'event_manifest.json')
            if seal['event']!=event or seal['run_id']!=cfg['run_id']:raise ValueError('Feedback seal identity differs')
            curves=data.reveal_event(event,before_ms=start);released_curves[event]=curves
            finals={c.rank:c.final for c in curves}
            raw=pd.read_csv(directory/'predictions.csv').query("method == 'uncorrected'")
            labels={r.case_id:finals[int(r.rank)] for r in raw.itertuples()}
            warm_state.observe(raw,labels,event=event,end_ms=row['end_ms'])
            if row['split']!='train':cold_state.observe(raw,labels,event=event,end_ms=row['end_ms'])
            frame=pd.read_csv(directory/'features.csv')
            frame['log_error']=[float(np.log(r.base_prediction)-np.log(finals[int(r.rank)])) for r in frame.itertuples()]
            history=pd.concat([history,frame],ignore_index=True) if len(history) else frame
            released_forecasts[event]={'event':event,'split':row['split'],'start_ms':row['start_ms'],'end_ms':row['end_ms'],
                'prediction_seal_sha256':sha256(directory/'event_manifest.json'),'outcome_sha256':row['sha256']}
        return [c for row in events if row['event'] in released_curves for c in released_curves[row['event']]]

    def predict_event(row):
        event,start=row['event'],row['start_ms'];curves=release_before(start)
        directory=Path(tempfile.mkdtemp(prefix=f'.event-{event}-',dir=run/'events'))
        samples=samples_by_event[event];ids=[s['case_id'] for s in samples]
        cached=None
        if row['split']=='train':
            fitting,labels=_training(samples_by_event,curves,metadata)
            bundle=fit_context(fitting,labels,iterations=iterations)
            _save_model(bundle,directory/'base.joblib');_weight_rows(fitting).to_csv(directory/'base_training_weights.csv',index=False)
            write_new_json(directory/'base_diagnostics.json',bundle['diagnostics'])
            from analysis.innovation_common import prepare
            base=adaptation.predict_recipe(bundle,prepare(samples,metadata))
        else:
            path=base_run/'events'/str(event);cached=verify_seal(path/'event_manifest.json')
            audit=json.loads((path/'audit.json').read_text())
            check_cached_event(data,row,cached,audit,cfg['base_run_id'])
            saved=pd.read_csv(path/'predictions.csv');base=aligned_base(saved.query("method == 'rolling_context'"),ids)
            aligned=saved.query("method == 'rolling_context'").set_index('case_id').loc[ids]
            if (list(aligned['rank'])!=[s['curve'].rank for s in samples] or not np.allclose(aligned.current,[s['score'] for s in samples],rtol=1e-12,atol=1e-6)):
                raise ValueError('Cached rank/current differs from blind prefix')
        target,peer_audit=_features(samples,metadata,base,warm_state)
        values,diagnostics,experts=calibration.predict_all(history.copy(deep=True),target.copy(deep=True),query_start_ms=start,model_directory=directory/'calibration')
        values={'uncorrected':base,'ewma_reference':np.maximum(target.current,np.minimum(target.current*np.exp(4),base*np.exp(-np.array([cold_state.bias(r.rank,r.phase) for r in target.itertuples()])))),**values}
        if set(values)!=set(METHODS) or set(experts)!=set(calibration.EXPERT_NAMES):raise ValueError('Method/expert coverage differs')
        if cached:
            reference=aligned_base(saved.query("method == 'rolling_context_ewma'"),ids)
            if not np.allclose(values['ewma_reference'],reference,rtol=1e-12,atol=1e-6):raise ValueError('Cold baseline EWMA replay differs')
        output=[]
        for method,pred in values.items():
            pred=np.asarray(pred,float)
            if pred.shape!=base.shape or not np.isfinite(pred).all() or np.any(pred<target.current.to_numpy()-1e-6):raise ValueError('Invalid bounded forecast')
            for case,value,raw in zip(target.to_dict('records'),pred,base):
                output.append({k:case[k] for k in ['case_id','event','rank','phase','current']}|{'method':method,'predicted':float(value),'log_bias':float(np.log(raw/value))})
        features=target.copy()
        for name,value in experts.items():features['expert__'+name]=value
        features.to_csv(directory/'features.csv',index=False)
        peer_audit.to_csv(directory/'peer_audit.csv',index=False)
        pd.DataFrame(output).to_csv(directory/'predictions.csv',index=False)
        history.to_csv(directory/'released_residuals.csv',index=False)
        write_new_json(directory/'calibration_diagnostics.json',diagnostics)
        write_new_json(directory/'audit.json',{'event':event,'split':row['split'],'query_start_ms':start,
            'base_origin':'new prequential train fit' if row['split']=='train' else 'identical sealed Context cache',
            'base_event_seal_sha256':sha256(base_run/'events'/str(event)/'event_manifest.json') if cached else None,
            'base_history_events':[r['event'] for r in events if r['event'] in released_curves],
            'released_history':[{k:r[k] for k in ['event','start_ms','end_ms','sha256']} for r in events if r['event'] in released_curves],
            'released_forecasts':list(released_forecasts.values()),'current_outcome_released':False,
            'selection_sha256':sha256(run/'selection.json') if row['split']=='test' else None})
        seal_directory(directory,'event_manifest.json',{'run_id':cfg['run_id'],'event':event,'split':row['split'],'query_start_ms':start,'end_ms':row['end_ms'],'cases':len(samples),'methods':METHODS})
        directory.rename(run/'events'/str(event))
        print(json.dumps({'sealed_event':event,'split':row['split'],'forecast_history_events':len(released_forecasts),'base_history_events':len(released_curves),'methods':len(values)}),flush=True)
        return pd.DataFrame(output)

    for split in ['train','validation','test']:
        rows=[r for r in queries if r['split']==split]
        if split=='test':
            release_before(rows[0]['start_ms']);truth=history.set_index('case_id');scores={}
            for method in CANDIDATES:
                part=predictions['validation'].query('method == @method').copy()
                expected=truth.loc[part.case_id]
                actual=np.exp(np.log(expected.base_prediction.to_numpy(float))-expected.log_error.to_numpy(float))
                part['ape']=np.abs(part.predicted.to_numpy()/actual-1)*100
                scores[method]=float(part.groupby('event').ape.mean().mean())
            selected=min(CANDIDATES,key=lambda name:(scores[name],CANDIDATES.index(name)))
            write_new_json(run/'selection.json',{'run_id':cfg['run_id'],'selected_recipe':selected,'methods':METHODS,'candidates':CANDIDATES,'validation_scores':scores,
                'test_labels_used_for_selection':False,'criterion':cfg['selection'],'sealed_before_test_start_ms':rows[0]['start_ms'],
                'validation_event_seals':{str(r['event']):sha256(run/'events'/str(r['event'])/'event_manifest.json') for r in queries if r['split']=='validation'}})
        predictions[split]=pd.concat([predict_event(r) for r in rows],ignore_index=True)
        expected={s['case_id'] for r in rows for s in samples_by_event[r['event']]}
        for method,part in predictions[split].groupby('method'):
            if part.case_id.duplicated().any() or set(part.case_id)!=expected:raise ValueError('Frozen case coverage differs')
        name='warm_predictions.csv' if split=='train' else f'{split}_predictions.csv'
        predictions[split].to_csv(run/name,index=False)
    for name,digest in cfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Source changed during run')
    seal_directory(run,'blind_manifest.json',{'protocol':'rolling_residual_v3','experiment':cfg['experiment'],'run_id':cfg['run_id'],
        'dataset_id':cfg['dataset_id'],'dataset_directory':cfg['dataset_directory'],'methods':METHODS,'scored_splits':['validation','test'],
        'reference_method':'ewma_reference','selected_recipe':selected,'base_blind_sha256':cfg['base_blind_sha256']})
    return json.loads((run/'selection.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--base-run',type=Path,default=BASE);p.add_argument('--resume',action='store_true')
    a=p.parse_args();print(json.dumps(run_experiment(load_dataset('v3'),a.run,base_run=a.base_run,resume=a.resume),ensure_ascii=False))
