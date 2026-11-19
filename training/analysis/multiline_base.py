"""Fixed causal shared-rank base cache with original two-rank Context routing."""
from __future__ import annotations
import argparse,importlib.metadata,json,platform,shutil,tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import identity
from analysis.multirank_model import feature_row,make_preprocessor,event_weights
from analysis.rolling_benchmark import verify_seal,seal_directory,write_new_json
from analysis.rolling_dataset import RollingDataset
from analysis.multiline_dataset import MultilineDataset,DEFAULT
from analysis.v3_coupled_runner import check_cache_event
from download_ranks import RANKS

CONTEXT=ROOT/'analysis/rolling_results/v3_next_warm_20261116'
RUN=ROOT/'analysis/rolling_results/v3_multiline_base_20261117'
SPEC=ROOT/'docs/superpowers/specs/2026-11-17-multiline-base-design.md'
CONTEXT_ID='1a2aea3070785aaa377817f105239961b6b8582749ffe6ba3369fe6ffe81a9a7'

def feature_cache(data):
    frames=[];cases=[];schema=None
    for split in ['train','validation','test']:
        index=data.case_index(split).set_index('case_id')
        for s in data.inputs(split):
            row=feature_row(s)
            if schema is None:schema=set(row)
            if set(row)!=schema:raise ValueError('Blind feature schema drift')
            frames.append(row);c=s['curve'];rec=index.loc[s['case_id']].to_dict()
            if rec['event']!=c.event or rec['rank']!=c.rank or rec['cutoff']!=s['cutoff'] or not np.isclose(rec['current'],s['score']):raise ValueError('Blind input and case index disagree')
            cases.append(rec|{'case_id':s['case_id'],'split':split,'observed_at':s['now'],'progress':s['progress'],
                'phase':int(np.searchsorted([.25,.9],s['progress'],side='right')),
                'duration_h':c.duration/3600000,'remaining_h':(c.end-s['now'])/3600000,
                'rate_6h_relative':s['features'].get('rate_6h_relative',np.nan),
                'rate_24h_relative':s['features'].get('rate_24h_relative',np.nan)})
    return pd.DataFrame(frames),pd.DataFrame(cases)

def run(data,run,*,context_run=CONTEXT,pair_dataset=None,iterations=600,allow_fixture=False,resume=False):
    run,context=Path(run),Path(context_run);pair=pair_dataset or RollingDataset()
    context_seal=verify_seal(context/'blind_manifest.json');ccfg=json.loads((context/'run_manifest.json').read_text())
    if identity({k:v for k,v in ccfg.items() if k!='run_id'})!=ccfg['run_id'] or ccfg['run_id']!=context_seal['run_id'] or ccfg['dataset_id']!=pair.manifest['dataset_id']:raise ValueError('Wrong Context cache identity')
    if not allow_fixture and (ccfg['run_id']!=CONTEXT_ID or iterations!=600 or data.manifest['protocol']['ranks']!=RANKS or ccfg['allow_fixture']):raise ValueError('Wrong formal recipe/cohort')
    for split in ['train','validation','test']:
        a=data.case_index(split).query('rank in [1000,2000]');b=pair.case_index(split)
        if set(a.case_id)!=set(b.case_id):raise ValueError('Two-rank comparison panel changed')
        if data.manifest['splits'][split]['event_ids']!=pair.manifest['splits'][split]['event_ids']:raise ValueError('Activity split changed')
    sources=local_sources([Path(__file__)])
    cfg={'protocol':'rolling_multiline_base_v3','dataset_id':data.manifest['dataset_id'],'dataset_directory':str(data.root.resolve()),
        'context_run':str(context.resolve()),'context_run_id':ccfg['run_id'],'context_blind_sha256':sha256(context/'blind_manifest.json'),
        'warmup_events':ccfg['warmup_events'],'warm_forecast_events':ccfg['warm_forecast_events'],'ranks':data.manifest['protocol']['ranks'],
        'base_recipe':{'iterations':iterations,'depth':5,'learning_rate':.035,'l2_leaf_reg':12,'seed':42,'threads':1},
        'routing':'Context400 exact cached predictions at1000/2000; newly fit shared-rank600 otherwise',
        'weights':'equal total per legally completed activity; preprocessor history only','allow_fixture':allow_fixture,
        'source_hashes':{n:sha256(ROOT/n) for n in sources},'specification_sha256':sha256(SPEC),'python':platform.python_version(),
        'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','catboost','joblib','scikit-learn']}}
    cfg['run_id']=identity(cfg)
    if run.exists():
        if not resume:raise FileExistsError(run)
        if json.loads((run/'run_manifest.json').read_text())!=cfg or not (run/'blind_manifest.json').exists():raise ValueError('Only identical completed run can resume')
        return verify_seal(run/'blind_manifest.json')
    run.mkdir(parents=True);(run/'events').mkdir();write_new_json(run/'run_manifest.json',cfg)
    for n in sources:
        p=run/'sources'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    x,cases=feature_cache(data);joblib.dump(x,run/'blind_features.joblib',compress=3);cases.to_csv(run/'blind_cases.csv',index=False)
    events=sorted(data.event_index.values(),key=lambda r:(r['start_ms'],r['event']))
    queries=[r for r in events if r['event'] in cfg['warm_forecast_events'] or r['split']!='train']
    released={};finals={};query_ids={r['event'] for r in queries}
    for row in queries:
        start,event=row['start_ms'],row['event']
        for previous in sorted(events,key=lambda r:(r['end_ms'],r['event'])):
            e=previous['event']
            if previous['end_ms']>=start or e in released:continue
            if e in query_ids:verify_seal(run/'events'/str(e)/'event_manifest.json')
            for c in data.reveal_event(e,before_ms=start):finals[(c.event,c.rank)]=c.final
            released[e]=previous
        training=cases.event.isin(released);target=cases.event==event
        train=cases[training];current=cases[target];weights=event_weights(train.event.tolist())
        y=np.log(np.array([finals[(int(r.event),int(r.rank))] for r in train.itertuples()])/train.current.to_numpy())+np.log(train.progress.to_numpy())
        prep=make_preprocessor(x.loc[training]);matrix=prep.fit_transform(x.loc[training])
        model=CatBoostRegressor(iterations=iterations,depth=5,learning_rate=.035,l2_leaf_reg=12,random_seed=42,thread_count=1,verbose=False,allow_writing_files=False)
        model.fit(matrix,y,sample_weight=weights)
        shared=current.current.to_numpy()*np.exp(np.clip(model.predict(prep.transform(x.loc[target]))-np.log(current.progress.to_numpy()),0,4))
        directory=Path(tempfile.mkdtemp(prefix=f'.event-{event}-',dir=run/'events'))
        joblib.dump({'model':model,'preprocessor':prep,'features':'rank'},directory/'shared.joblib',compress=3)
        pd.DataFrame({'case_id':train.case_id,'event':train.event,'rank':train['rank'],'weight':weights}).to_csv(directory/'training_weights.csv',index=False)
        src=context/'events'/str(event);src_seal=verify_seal(src/'event_manifest.json');src_audit=json.loads((src/'audit.json').read_text())
        peer=pd.read_csv(src/'peer_audit.csv');check_cache_event(pair,pair.event_index[event],ccfg,src_seal,src_audit,peer)
        original=pd.read_csv(src/'predictions.csv').query("method=='uncorrected'").set_index('case_id')
        frame=current.copy();frame['shared_prediction']=shared;frame['predicted']=shared;frame['model_family']='shared_rank600'
        paired=frame['rank'].isin([1000,2000])
        for i,r in frame[paired].iterrows():
            source=original.loc[r.case_id]
            if source['rank']!=r['rank'] or not np.isclose(source.current,r.current,rtol=1e-12,atol=1e-6):raise ValueError('Cached Context prefix mismatch')
            frame.loc[i,'predicted']=source.predicted;frame.loc[i,'model_family']='context400'
        if not np.isfinite(frame.predicted).all() or (frame.predicted<frame.current).any():raise ValueError('Invalid base predictions')
        frame.to_csv(directory/'predictions.csv',index=False)
        write_new_json(directory/'audit.json',{'event':event,'query_start_ms':start,'current_outcome_released':False,
            'released_history':[{k:r[k] for k in ['event','start_ms','end_ms','sha256']} for r in released.values()],
            'context_event_manifest_sha256':sha256(src/'event_manifest.json'),'context_history':src_audit['released_history'],
            'training_cases':len(train),'target_cases':len(current),'target_rank_support':{str(r):int((train['rank']==r).sum()) for r in cfg['ranks']}})
        seal_directory(directory,'event_manifest.json',{'run_id':cfg['run_id'],'event':event,'split':row['split'],'query_start_ms':start,'end_ms':row['end_ms'],'cases':len(current)})
        directory.rename(run/'events'/str(event));print(json.dumps({'sealed_event':event,'split':row['split'],'train_activities':len(released),'train_cases':len(train),'cases':len(current)}),flush=True)
    for name,digest in cfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Source changed during raw cache run')
    return seal_directory(run,'blind_manifest.json',{'protocol':cfg['protocol'],'run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],
        'dataset_directory':cfg['dataset_directory'],'context_blind_sha256':cfg['context_blind_sha256']})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DEFAULT);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--resume',action='store_true');a=p.parse_args()
    run(MultilineDataset(a.dataset),a.run,resume=a.resume)
