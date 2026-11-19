"""Fetch a common-time live prefix and forecast all 19 CN ranks in one call."""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime,timezone
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import httpx
import pandas as pd
from analysis.backtest import ROOT,TZ,snapshot
from analysis.data_policy import MIN_EVENT_START_MS,MIN_EVENT_START_ISO,POLICY,POLICY_ID
from analysis.holdout_dataset import from_record,curve_record,prefix_curve,sha256,write_json
from analysis.multirank_model import predict_samples,verify_run
from analysis.multirank_dataset import MultirankDataset
from download_ranks import RANKS,Limiter,request,normalize_cutoffs,normalize_top,started_events

def runtime_identity():
    return {'python':platform.python_version(),'data_policy_id':POLICY_ID,'data_policy':POLICY,
        'software':{n:importlib.metadata.version(n) for n in
        ['numpy','pandas','scikit-learn','catboost','joblib','httpx']}}

def parse_replay_time(value):
    try:
        dt=datetime.fromisoformat(value.replace('Z','+00:00'))
        if dt.tzinfo is None:dt=dt.replace(tzinfo=TZ)
        return int(dt.timestamp()*1000)
    except (ValueError,OverflowError,OSError) as exc:raise argparse.ArgumentTypeError('Invalid ISO8601 replay time') from exc

def production_dataset(run,seal):
    """Old sealed models stay reproducible but cannot bypass today's population."""
    manifest_path=Path(run)/'run_manifest.json'
    manifest_hash=seal.get('artifacts',{}).get('run_manifest.json')
    if not manifest_hash or sha256(manifest_path)!=manifest_hash:
        raise ValueError('Model provenance requires a hash-protected run manifest')
    config=json.loads(manifest_path.read_text())
    for field in ('dataset_id','dataset_directory','run_id'):
        if not isinstance(config.get(field),str) or not isinstance(seal.get(field),str):
            raise ValueError('Model provenance is missing '+field)
        expected,actual=config[field],seal[field]
        if field=='dataset_directory':expected,actual=Path(expected).resolve(),Path(actual).resolve()
        if expected!=actual:raise ValueError('Model provenance differs from run manifest: '+field)
    data=MultirankDataset(config['dataset_directory'])
    if data.manifest['dataset_id']!=seal['dataset_id']:
        raise ValueError('Model population dataset identity differs from its seal')
    partitions=pd.read_csv(data.root/'partitions.csv')
    starts=pd.to_numeric(partitions['start_ms'],errors='coerce')
    if starts.isna().any() or (starts<MIN_EVENT_START_MS).any():
        raise ValueError(f'Model population includes activities before {MIN_EVENT_START_ISO}; retrain on dense data')
    if any(c.start<MIN_EVENT_START_MS for c in data.train_curves()):
        raise ValueError(f'Model training history violates population policy {MIN_EVENT_START_ISO}')
    return data,partitions

def forecast_rows(run,curves,cutoff):
    seal=verify_run(run)
    data,partitions=production_dataset(run,seal)
    training_end=int(partitions.loc[partitions['split']=='train','end_ms'].max())
    coverage=pd.read_csv(Path(run)/'models/rank_coverage.csv').set_index('rank')
    curves=list(curves)
    if any(c.start<=training_end for c in curves):
        raise ValueError('Replay activity must begin after the model training history ends')
    if len({c.event for c in curves})>1:raise ValueError('Live curves must describe one activity')
    mapping={c.rank:c for c in curves}
    if len(mapping)!=len(curves):raise ValueError('Duplicate current rank')
    samples=[];rows=[]
    for rank in RANKS:
        c=mapping.get(rank);s=None
        if c is not None and c.start<=cutoff<c.end:s=snapshot(c,cutoff)
        if s is None:
            rows.append({'rank':rank,'status':'unavailable','reason':'missing, stale, insufficient, or outside event time'});continue
        # No model receives a current activity's future suffix.
        s=snapshot(prefix_curve(c,cutoff),cutoff)
        s.update({'case_id':f'live:{c.event}:{rank}:{cutoff}','stage':'live'});samples.append(s)
        rows.append({'rank':rank,'status':'ok','event':c.event,'current_ep':int(s['score']),
            'progress':float(s['progress']),'observed_at':datetime.fromtimestamp(s['now']/1000,TZ).isoformat(),
            'as_of':datetime.fromtimestamp(cutoff/1000,TZ).isoformat(),'event_end':datetime.fromtimestamp(c.end/1000,TZ).isoformat()})
        counts={name:int(coverage.loc[rank,column]) for name,column in
            [('training_events','train_events'),('validation_events','validation_events'),('test_events','test_events')]}
        rows[-1].update(counts)
        missing=[name for name,value in counts.items() if value==0]
        rows[-1].update(provisional=bool(missing),provisional_reason='No support in '+', '.join(missing) if missing else '')
    if samples:
        forecasts=predict_samples(run,samples);by_rank={s['curve'].rank:float(p) for s,p in zip(samples,forecasts)}
        for row in rows:
            if row['status']=='ok':row['predicted_final_ep']=int(round(by_rank[row['rank']]))
    return rows

async def fetch_prefixes(output,*,event=None,as_of=None,transport=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    now=int(datetime.now(timezone.utc).timestamp()*1000)
    cutoff=now if as_of is None else as_of
    if cutoff>now:raise ValueError('Replay time cannot be in the future')
    output.mkdir(parents=True);(output/'raw').mkdir();(output/'prefixes').mkdir()
    limiter=Limiter(2.)
    async with httpx.AsyncClient(timeout=30,transport=transport,follow_redirects=True) as client:
        response,_=await request(client,'https://bestdori.com/api/events/all.5.json',{},limiter)
        (output/'raw/events.json').write_bytes(response.content);metadata=response.json()
        candidates=[e for e in started_events(metadata,cutoff) if int(metadata[str(e)]['endAt'][3])>cutoff]
        if event is None:
            if not candidates:raise ValueError('No ongoing CN activity')
            event=candidates[-1]
        info=metadata[str(event)];start,end=int(info['startAt'][3]),int(info['endAt'][3])
        if not start<=cutoff<end:raise ValueError('Requested event is not ongoing at the cutoff')
        curves=[];failures=[]
        for rank in RANKS:
            url='https://bestdori.com/api/eventtop/data' if rank==10 else 'https://bestdori.com/api/tracker/data'
            params={'server':3,'event':event}|({'mid':0,'interval':0} if rank==10 else {'tier':rank})
            try:
                response,_=await request(client,url,params,limiter)
                with gzip.open(output/f'raw/{rank}.json.gz','wb') as f:f.write(response.content)
                points,_=(normalize_top if rank==10 else normalize_cutoffs)(response.json())
                points=[p for p in points if p['time']<=cutoff]
                if not points:raise ValueError('No prefix observations')
                c=from_record({'event':event,'rank':rank,'kind':info['eventType'],'start':start,'end':end,
                    'times':[p['time'] for p in points],'scores':[p['ep'] for p in points]})
                curves.append(c);write_json(output/f'prefixes/{rank}.json',curve_record(c))
            except (httpx.HTTPError,ValueError,KeyError,TypeError) as exc:failures.append({'rank':rank,'error':str(exc)})
    write_json(output/'fetch.json',{'event':event,'cutoff':cutoff,'ranks':RANKS,'failures':failures})
    return curves,cutoff

async def run_live(model_run,output,*,event=None,as_of=None):
    seal=verify_run(model_run)
    production_dataset(model_run,seal)
    curves,cutoff=await fetch_prefixes(output,event=event,as_of=as_of)
    rows=forecast_rows(model_run,curves,cutoff)
    selected=json.loads((Path(model_run)/'selection.json').read_text())['selected_recipe']
    result={'model_run':str(Path(model_run).resolve()),'run_id':seal['run_id'],'method':selected,
        'runtime':runtime_identity(),
        'mode':'historical_replay' if as_of is not None else 'live',
        'as_of':datetime.fromtimestamp(cutoff/1000,TZ).isoformat(),'predictions':rows,
        'notes':['All ranks use the same requested cutoff; observations may differ in timestamp.',
            'Supplementary fixed-event model; no refit on validation/test.',
            'No calibrated intervals; sparse ranks require cautious interpretation.',
            'Test cohort previously explored; metadata is retrospective.']}
    write_json(output/'prediction.json',result);pd.DataFrame(rows).to_csv(output/'predictions.csv',index=False)
    shutil.copyfile(__file__,output/'predict_source.py');shutil.copyfile(ROOT/'training/legacy/download_ranks.py',output/'download_source.py')
    shutil.copyfile(ROOT/'training/analysis/data_policy.py',output/'data_policy_source.py')
    write_json(output/'prediction_manifest.json',{'model_blind_manifest_sha256':sha256(Path(model_run)/'blind_test_manifest.json'),
        'artifacts':{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(pd.DataFrame(rows).to_string(index=False));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--event',type=int)
    p.add_argument('--as-of',type=parse_replay_time,help='Explicit historical replay cutoff; never includes later scores')
    a=p.parse_args()
    try:asyncio.run(run_live(a.run,a.output,event=a.event,as_of=a.as_of))
    except (ValueError,OSError,KeyError,httpx.HTTPError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(1)
