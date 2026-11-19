"""Trusted immutable replay of fixed paper rules; never opens answer files or vaults.

Prepare only seals configuration and source. Execute only after the resource
owner permits a real replay. Outcomes enter solely as already released and
sealed historical forecast errors in the source event's causal snapshot.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import verify_seal,seal_directory,write_new_json
from analysis import paper_upgrade_replay_models as models

SOURCE=ROOT/'analysis/rolling_results/v3_multiline_calibration_20261117'
BASE=ROOT/'analysis/rolling_results/v3_multiline_base_20261117'
DATA=ROOT/'data/rolling_holdout_v3_dense_19rank_20261117'
RUN=ROOT/'analysis/paper_upgrade_records/20261118/runs/fixed_replay_v1'
FORMAL_DATASET='db4e9b9c84bd64a6de5ee30a090cfac83efbe85c2f6dd899304516e7d031cdf1'
FORMAL_SOURCE='14888ceeff945c1fd387a0e03265ddde1283901a654a678b9b81ac4a71699f5d'
FORMAL_BASE='7ea967c5b7ba83f9389fbffe6ede0ac3bd1ec18412e694d674c54d05f8efdfbe'


def load(path):return json.loads(Path(path).read_text())


def read(path):
    # Category codes are textual IDs. Chunked inference can make the same
    # event's code integer in one block and text in another (e.g. 2 vs '2').
    # Explicit dtypes preserve the raw CSV spelling and real conflicts remain.
    return pd.read_csv(path,dtype={'event_type':'str','band_group':'str'},low_memory=False)


def release_guard(rows,query_start_ms,current_event):
    if int(query_start_ms)!=query_start_ms or query_start_ms<models.stats.MIN_START_MS:raise ValueError('Invalid query start')
    seen=set()
    for row in rows:
        if (row['event'] in seen or row['event']==current_event or row['start_ms']<models.stats.MIN_START_MS
                or row['end_ms']>=query_start_ms or row['end_ms']<=row['start_ms']):
            raise ValueError('History is not strictly completed before current event')
        seen.add(row['event'])


def case_parity(actual,expected):
    if actual.case_id.duplicated().any() or expected.case_id.duplicated().any() or set(actual.case_id)!=set(expected.case_id):
        raise ValueError('Frozen physical case panel differs')
    a=actual.set_index('case_id');b=expected.set_index('case_id').loc[a.index]
    required=['event','rank','start_ms','end_ms','cutoff','observed_at','current']
    if any(k not in a or k not in b for k in required):raise ValueError('Missing blind identity fields')
    for col in required+['phase','progress','base_prediction']:
        if col not in a or col not in b:continue
        av=a[col].to_numpy(float);bv=b[col].to_numpy(float)
        if col in ['current','base_prediction','progress']:
            good=np.allclose(av,bv,rtol=1e-12,atol=1e-6 if col!='progress' else 1e-12,equal_nan=True)
        else:good=np.array_equal(av,bv)
        if not good:raise ValueError('Frozen blind/base parity differs: '+col)
    for col in ['split','event_type','band_group']:
        if col in a and col in b and not a[col].fillna('__missing__').astype(str).equals(b[col].fillna('__missing__').astype(str)):
            raise ValueError('Frozen blind context differs: '+col)


class SourceGateway:
    """Only this coordinator reads source snapshots; predictors get detached frames."""
    def __init__(self,source=SOURCE,base=BASE,dataset=DATA,*,allow_fixture=False):
        self.source,self.base,self.dataset=map(Path,[source,base,dataset])
        self.source_seal=verify_seal(self.source/'blind_manifest.json');self.base_seal=verify_seal(self.base/'blind_manifest.json')
        self.cfg=load(self.source/'run_manifest.json');self.bcfg=load(self.base/'run_manifest.json');self.data=load(self.dataset/'manifest.json')
        if (self.cfg['run_id']!=self.source_seal['run_id'] or self.bcfg['run_id']!=self.base_seal['run_id']
                or self.cfg['base_run_id']!=self.bcfg['run_id']
                or len({self.cfg['dataset_id'],self.bcfg['dataset_id'],self.data['dataset_id']})!=1):raise ValueError('Source identities differ')
        if not allow_fixture and (self.data['dataset_id']!=FORMAL_DATASET or self.cfg['run_id']!=FORMAL_SOURCE or self.bcfg['run_id']!=FORMAL_BASE):raise ValueError('Wrong formal sealed prediction source')
        self.events={int(r['event']):r for r in self.data['history_index']}
        self.queries=sorted([r for r in self.events.values() if r['event'] in self.cfg['warm_forecast_events'] or r['split']!='train'],key=lambda r:(r['start_ms'],r['event']))
        self.cases={s:read(self.dataset/f'{s}_cases.csv') for s in ['train','validation','test']}
        self.targets={};self.source_records={}

    def blind_event(self,event):
        if event in self.targets:return self.targets[event].copy(deep=True)
        row=self.events[event];sp=self.source/'events'/str(event);bp=self.base/'events'/str(event)
        ss=verify_seal(sp/'event_manifest.json');bs=verify_seal(bp/'event_manifest.json');audit=load(sp/'audit.json')
        for seal,run_id in [(ss,self.cfg['run_id']),(bs,self.bcfg['run_id'])]:
            if any(seal.get(k)!=v for k,v in {'event':event,'run_id':run_id,'query_start_ms':row['start_ms'],'end_ms':row['end_ms'],'split':row['split']}.items()):raise ValueError('Wrong source event clock')
        if audit.get('current_outcome_released') is not False or audit['query_start_ms']!=row['start_ms']:raise ValueError('Current source outcome was released')
        if audit['base_event_manifest_sha256']!=sha256(bp/'event_manifest.json'):raise ValueError('Source base seal binding differs')
        expected_history={r['event']:r for r in self.events.values() if r['end_ms']<row['start_ms']}
        ba=load(bp/'audit.json')
        if ba.get('current_outcome_released') is not False:raise ValueError('Cached base outcome was released')
        release_guard(ba['released_history'],row['start_ms'],event)
        if {r['event'] for r in ba['released_history']}!=set(expected_history):raise ValueError('Cached base history cohort differs')
        raw=read(sp/'features.csv')
        if set(models.TARGET_COLUMNS)-set(raw):raise ValueError('Blind source schema differs')
        target=raw[models.TARGET_COLUMNS].copy()
        frozen=self.cases[row['split']].query('event==@event');case_parity(target,frozen)
        routed=read(bp/'predictions.csv');routed['base_prediction']=routed.predicted
        case_parity(target,routed)
        issued=read(sp/'predictions.csv');uncorrected=issued[issued.method=='uncorrected'].set_index('case_id')
        if set(uncorrected.index)!=set(target.case_id) or not np.allclose(uncorrected.loc[target.case_id].predicted,target.base_prediction,rtol=1e-12,atol=1e-6):raise ValueError('Source uncorrected parity differs')
        self.targets[event]=target;self.source_records[event]={'audit':audit,'issued':issued,'source_seal_sha256':sha256(sp/'event_manifest.json'),'base_seal_sha256':sha256(bp/'event_manifest.json')}
        return target.copy(deep=True)

    def event_record(self,event):
        row=self.events[event];target=self.blind_event(event);record=self.source_records[event];audit=record['audit']
        releases=audit['released_forecasts'];release_guard(releases,row['start_ms'],event)
        expected={r['event']:r for r in self.queries if r['end_ms']<row['start_ms']}
        if {r['event'] for r in releases}!=set(expected):raise ValueError('Released forecast cohort differs')
        for release in releases:
            original=expected[release['event']]
            if any(release[k]!=original[k] for k in ['start_ms','end_ms']):raise ValueError('Release metadata clock differs')
            if release.get('prediction_seal_sha256')!=sha256(self.source/'events'/str(release['event'])/'event_manifest.json'):raise ValueError('Released prediction seal differs')
        # All history end guards pass BEFORE decoding the snapshot's errors.
        historical=read(self.source/'events'/str(event)/'released_residuals.csv')
        if len(historical):
            if set(historical.event)!=set(expected):raise ValueError('Residual history event set differs')
            for old,part in historical.groupby('event',sort=False):case_parity(part,self.blind_event(int(old)))
        elif expected:raise ValueError('Missing released residual history')
        if 'log_error' not in historical or not np.isfinite(historical.log_error.to_numpy(float)).all():raise ValueError('Invalid historical forecast errors')
        keep=historical[models.TARGET_COLUMNS+['log_error']].copy()
        if len(keep):
            if 'expert__independent' not in historical:raise ValueError('Missing original issued interval center')
            expert=historical.expert__independent.to_numpy(float)
            if not np.isfinite(expert).all() or (expert<=0).any():raise ValueError('Invalid historical center scores')
            for old,part in historical.groupby('event',sort=False):
                original=self.source_records[int(old)]['issued']
                reference=original[original.method=='ewma_reference'].set_index('case_id')
                if not np.allclose(part.expert__independent,reference.loc[part.case_id].predicted,rtol=1e-12,atol=1e-6):
                    raise ValueError('Historical issued interval center parity differs')
            keep['center_log_error']=np.log(expert/keep.base_prediction.to_numpy())+keep.log_error.to_numpy()
        else:keep['center_log_error']=pd.Series(dtype=float)
        models.validate(keep,target,row['start_ms'])
        return {'target':target,'history':keep,'row':row,'audit':audit,'issued':record['issued'],
                'source_event_manifest_sha256':record['source_seal_sha256'],'base_event_manifest_sha256':record['base_seal_sha256'],
                'history_artifact_sha256':sha256(self.source/'events'/str(event)/'released_residuals.csv')}


def configuration(source=SOURCE,base=BASE,dataset=DATA):
    cfg=load(Path(source)/'run_manifest.json');data=load(Path(dataset)/'manifest.json')
    names=['training/analysis/paper_upgrade_replay.py','training/analysis/paper_upgrade_replay_models.py',
           'training/analysis/rolling_residual_stats.py','training/analysis/rolling_benchmark.py','training/analysis/holdout_dataset.py','training/analysis/multirank_dataset.py']
    result={'protocol':'rolling_multiline_calibration_v3','experiment':'paper_fixed_sealed_base_replay',
            'dataset_id':data['dataset_id'],'dataset_directory':str(Path(dataset).resolve()),
            'source_run':str(Path(source).resolve()),'source_run_id':cfg['run_id'],'source_blind_sha256':sha256(Path(source)/'blind_manifest.json'),
            'base_run':str(Path(base).resolve()),'base_blind_sha256':sha256(Path(base)/'blind_manifest.json'),
            'warmup_events':cfg['warmup_events'],'warm_forecast_events':cfg['warm_forecast_events'],'ranks':data['protocol']['ranks'],
            'methods':list(models.menu()),'recipes':models.menu(),'candidates':['ewma_reference'],'reference_method':'ewma_reference',
            'pair_recipe':models.PAIR,'sensitivity_sweeps':models.SWEEPS,
            'statistical_recipes':models.stats.CONFIGS,'phase_edges':[.25,.9],
            'simple_baselines':{'time_proportion':'bounded current/actual_progress',
                                'history_growth':'same-rank per-event normalized Gaussian-progress mean log_growth; bandwidth .15; equal event mean; no support -> time proportion'},
            'primary_fixed':'ewma_reference','test_selection':False,'secondary_selection':'none; all methods exploratory diagnostics',
            'dlm_recipe':{'name':'dlm_fullcov','Q':'.1*(.5*C+.5*diag(diag(C)))','R':'diag(.5*diag(sampleCov(diff raw residual levels)))','P0':'C+1e-6I','floor':1e-6,'minimum_pairs':8,'initial_mean':[0,0],
                          'fallback':{'Q':.0001,'R':.01,'P0':.01},'freeze':'estimate train-only available history for warm diagnostics; freeze separately per feedback policy at first validation query; never reestimate heldout'},
            'interval_recipe':{'center':'original independently corrected forecast; recomputed parity checked','levels':[.8,.9],'score':'max absolute log center error per event/rank/phase','quantile':'ceil((n+1)*level) order statistic','fallback':'all released event-phase maxima pool, then explicit unavailable/infinite','coverage_claim':'historical grouped empirical interval, no temporal conformal guarantee'},
            'feedback_policies':models.POLICIES,'source_hashes':{n:sha256(ROOT/n) for n in names},
            'resource_policy':{'processes':1,'threads':1,'nice':19,'fit_base':False},'exposure':cfg['exposure'],
            'metadata_caveat':cfg['metadata_caveat'],'python':platform.python_version(),
            'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy']}}
    result['run_id']=identity(result);return result


def prepare(run=RUN,*,source=SOURCE,base=BASE,dataset=DATA):
    run=Path(run)
    if run.exists():raise FileExistsError('Preserve existing run; choose a new directory')
    cfg=configuration(source,base,dataset);run.mkdir(parents=True)
    write_new_json(run/'run_manifest.json',cfg)
    write_new_json(run/'freeze_receipt.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'run_id':cfg['run_id'],'real_numeric_job_started':False})
    for name in cfg['source_hashes']:
        dest=run/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    seal_directory(run,'configuration_seal.json',{'run_id':cfg['run_id'],'stage':'prepared_only'})
    return cfg


def execute(run=RUN):
    run=Path(run);cfg=load(run/'run_manifest.json');verify_seal(run/'configuration_seal.json')
    if cfg!=configuration(cfg['source_run'],cfg['base_run'],cfg['dataset_directory']):raise ValueError('Frozen configuration/source changed')
    if (run/'events').exists() or (run/'blind_manifest.json').exists():raise FileExistsError('Preserve partial or completed replay; never overwrite')
    gateway=SourceGateway(cfg['source_run'],cfg['base_run'],cfg['dataset_directory']);(run/'events').mkdir()
    starts=tuple(r['start_ms'] for r in gateway.queries);parameters=None;counts={};seen=set()
    selection={'run_id':cfg['run_id'],'selected_recipe':'ewma_reference','methods':cfg['methods'],'candidates':['ewma_reference'],
               'validation_scores':{},'test_labels_used_for_selection':False,'criterion':'fixed primary independent warm; no new selection','sealed_before_test_replay':True}
    write_new_json(run/'selection.json',selection)
    for row in gateway.queries:
        began=time.monotonic();event=row['event'];record=gateway.event_record(event)
        target,history=record['target'],record['history']
        required=set(history.event)
        if not required<=seen:raise ValueError('New run has not sealed all histories before feedback')
        for prior in required:verify_seal(run/'events'/str(int(prior))/'event_manifest.json')
        if row['split']=='validation' and parameters is None:
            if set(history['split'])!={'train'}:raise ValueError('DLM parameter fit must be training only')
            parameters={'complete':models.dlm_parameters(history)}
            for name,policy in models.POLICIES.items():
                masked=models.feedback_history(history,query_start_ms=row['start_ms'],policy=policy,query_starts=starts)
                parameters[name]=models.dlm_parameters(masked)
            write_new_json(run/'frozen_dlm_parameters.json',parameters)
        scores,states=models.predict_all(history,target,query_start_ms=row['start_ms'],parameters=parameters,query_starts=starts)
        if set(scores)!=set(cfg['methods']):raise ValueError('Fixed method menu changed')
        parity={}
        for method in ['uncorrected','ewma_reference','legacy_pair']:
            old=record['issued'].query('method==@method').set_index('case_id')
            values=old.loc[target.case_id].predicted.to_numpy(float)
            parity[method]=float(np.max(np.abs(scores[method]/values-1)))
            if not np.allclose(scores[method],values,rtol=1e-12,atol=1e-6):raise ValueError('Original issued baseline parity failed: '+method)
        rows=[]
        for method,values in scores.items():
            for c,value in zip(target.to_dict('records'),values):
                rows.append({k:c[k] for k in ['case_id','event','rank','split','start_ms','end_ms','phase','progress','cutoff','observed_at','current']}|{'method':method,'predicted':float(value),'log_bias':float(np.log(c['base_prediction']/value))})
        forecasts=pd.DataFrame(rows);intervals=models.empirical_intervals(history,target,scores['ewma_reference'])
        path=Path(tempfile.mkdtemp(prefix=f'.event-{event}-',dir=run/'events'))
        target.to_csv(path/'features.csv',index=False);forecasts.to_csv(path/'predictions.csv',index=False);intervals.to_csv(path/'intervals.csv',index=False)
        write_new_json(path/'states.json',states)
        write_new_json(path/'audit.json',{'event':event,'split':row['split'],'query_start_ms':row['start_ms'],'current_outcome_released':False,
             'released_forecasts':record['audit']['released_forecasts'],'history_cases':len(history),'history_events':int(history.event.nunique()),
             'source_event_manifest_sha256':record['source_event_manifest_sha256'],'base_event_manifest_sha256':record['base_event_manifest_sha256'],
             'history_artifact_sha256':record['history_artifact_sha256'],'selection_sha256':sha256(run/'selection.json'),'baseline_parity_max_relative':parity})
        seal_directory(path,'event_manifest.json',{'run_id':cfg['run_id'],'event':event,'split':row['split'],'query_start_ms':row['start_ms'],'end_ms':row['end_ms'],'cases':len(target),'methods':cfg['methods']})
        path.rename(run/'events'/str(event));seen.add(event)
        split=row['split'];fname='warm_predictions.csv' if split=='train' else f'{split}_predictions.csv'
        forecasts.to_csv(run/fname,index=False,mode='a',header=counts.get(split,0)==0)
        intervals.to_csv(run/('warm_intervals.csv' if split=='train' else f'{split}_intervals.csv'),index=False,mode='a',header=counts.get(split,0)==0)
        counts[split]=counts.get(split,0)+len(target)
        print(json.dumps({'sealed_event':event,'split':split,'cases':len(target),'history_events':len(required),'methods':len(scores),'seconds':round(time.monotonic()-began,3)}),flush=True)
    for split in ['validation','test']:
        if counts[split]!=len(gateway.cases[split]):raise ValueError('Frozen case coverage differs')
    for name,digest in cfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Source changed during replay')
    return seal_directory(run,'blind_manifest.json',{'protocol':cfg['protocol'],'run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],
         'dataset_directory':cfg['dataset_directory'],'methods':cfg['methods'],'selected_recipe':'ewma_reference','reference_method':'ewma_reference','scored_splits':['validation','test']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','execute']);parser.add_argument('--run',type=Path,default=RUN)
    args=parser.parse_args();result=prepare(args.run) if args.action=='prepare' else execute(args.run)
    print(json.dumps({'run_id':result['run_id'],'action':args.action,'methods':len(result['methods']) if 'methods' in result else None},ensure_ascii=False))
