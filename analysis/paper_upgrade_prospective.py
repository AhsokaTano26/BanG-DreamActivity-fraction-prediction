"""Append-only prospective shadow submission and delayed-outcome scoring.

This records supplied forecast outputs; it does not fetch data or train a model.
Only events that start after registration are eligible. CLI clocks use wall time.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd

LOWER=1706504400000
RANKS={10,20,30,40,50,100,200,300,400,500,1000,1500,2000,3000,4000,5000,10000,20000,30000}
ANSWERS={'actual','target','final','final_score','actual_score','actual_final','label','labels','log_error'}


def now_ms():
    return time.time_ns()//1000000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def seal(path):
    hashes={f.name:digest(f) for f in Path(path).iterdir() if f.is_file() and f.name!='seal.json'}
    write_new(Path(path)/'seal.json',hashes)


def verify(path):
    p=Path(path);hashes=json.loads((p/'seal.json').read_text())
    if set(hashes)!={f.name for f in p.iterdir() if f.is_file() and f.name!='seal.json'}:
        raise ValueError('Sealed file list changed')
    if any(digest(p/name)!=h for name,h in hashes.items()):
        raise ValueError('Sealed content changed')


def register(root,methods,*,primary,plan=None):
    root=Path(root)
    if primary not in methods or not methods or any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in methods.values()):
        raise ValueError('Methods require exact source/config SHA256 values')
    if plan is not None:
        if not plan.get('stages') or not set(plan.get('ranks',[])).issubset(RANKS) or int(plan.get('target_events',0))<1:
            raise ValueError('A research plan requires stages, ranks and an event stopping count')
        if any(s['axis'] not in ['elapsed','progress','remaining'] or not math.isfinite(float(s['value'])) or float(s['value'])<=0 for s in plan['stages']):
            raise ValueError('Invalid frozen time grid')
    root.mkdir(parents=True,exist_ok=False)
    (root/'registry').mkdir();(root/'events').mkdir()
    config={'registered_ms':now_ms(),'methods':methods,'primary':primary,
            'minimum_start_ms':LOWER,'eligibility':'actual start strictly after registered_ms',
            'feedback_eligibility':'end_ms and actual released_ms strictly before query start',
            'primary_metric':'event-macro MAPE percent; unique cases',
            'status':'prospective_shadow_submission_interface',
            'source_sha256':digest(__file__),
            'software':{'numpy':np.__version__,'pandas':pd.__version__},
            'plan':plan,
            'limits':['Method identifiers are recorded commitments; forecasting is an upstream step.',
                      'Verified outcome capture provenance is supplied by the trusted collector.',
                      'No new activity outcome or prospective accuracy exists at registration.']}
    write_new(root/'registry/config.json',config)
    (root/'registry/interface_source.py').write_bytes(Path(__file__).read_bytes())
    seal(root/'registry')
    return config


def config(root):
    verify(Path(root)/'registry')
    value=json.loads((Path(root)/'registry/config.json').read_text())
    if value['source_sha256']!=digest(__file__) or value['software']!={'numpy':np.__version__,'pandas':pd.__version__}:
        raise ValueError('Registered interface implementation or dependencies changed; create a new version')
    return value


def batches(root,event=None):
    parent=Path(root)/'events'
    folders=[parent/str(event)] if event is not None else sorted(parent.iterdir())
    out=[]
    for folder in folders:
        if (folder/'batches').exists():
            for p in sorted((folder/'batches').iterdir()):
                verify(p);out.append(p)
    return out


def issue(root,event,forecasts):
    root=Path(root);cfg=config(root);issued=now_ms()
    event={k:int(event[k]) for k in ['event','start_ms','end_ms']}
    if event['event']<=0 or event['start_ms']<LOWER or event['start_ms']<=cfg['registered_ms'] or not event['start_ms']<=issued<event['end_ms']:
        raise ValueError('Only a live event starting after registration is eligible')
    required={'case_id','event','rank','start_ms','end_ms','cutoff','observed_at','current','method','predicted'}
    optional={'lower80','upper80','lower90','upper90','base_prediction','prefix_points'}
    if not isinstance(forecasts,pd.DataFrame) or forecasts.empty or required-set(forecasts) or set(forecasts)-(required|optional) or ANSWERS&set(forecasts):
        raise ValueError('A nonempty blind forecast panel is required')
    f=forecasts.copy()
    if f[list(required)].isna().any().any() or f.duplicated(['method','case_id']).any() or set(f.method)!=set(cfg['methods']):
        raise ValueError('Missing values, duplicated cases or method menu mismatch')
    numbers=list(required-{'case_id','method'})
    if not np.isfinite(f[numbers].to_numpy(float)).all():
        raise ValueError('Forecast numeric fields must be finite')
    for key in ['event','rank','start_ms','end_ms','cutoff','observed_at']:
        values=pd.to_numeric(f[key]).to_numpy(float)
        if np.any(values!=np.floor(values)):raise ValueError('Identifiers and clocks must be integral')
        f[key]=values.astype(np.int64)
    if 'prefix_points' in f:
        support=pd.to_numeric(f.prefix_points,errors='raise').to_numpy(float)
        if not np.isfinite(support).all() or np.any(support!=np.floor(support)) or np.any(support<0):
            raise ValueError('Prefix support must be finite nonnegative integer counts')
        f['prefix_points']=support.astype(np.int64)
    physical=['event','rank','cutoff']
    if f.duplicated(['method']+physical).any():
        raise ValueError('A physical forecast cutoff must have one case ID')
    if any(not (f[k]==v).all() for k,v in event.items()) or not set(f['rank']).issubset(RANKS):
        raise ValueError('Event clocks or ranks disagree')
    if ((f.observed_at<f.start_ms)|(f.observed_at>f.cutoff)|(f.cutoff>issued)|(f.cutoff>=f.end_ms)|(f.current<=0)|(f.predicted<f.current)|(f.predicted>f.current*np.exp(4))).any():
        raise ValueError('Future observation or invalid forecast bounds')
    plan=cfg.get('plan')
    if plan is not None:
        allowed=planned_cutoffs(event,plan)
        if not set(f.cutoff).issubset(allowed) or not set(f['rank']).issubset(plan['ranks']):
            raise ValueError('Forecast is outside the frozen physical grid')
        if (issued-f.cutoff>int(plan['maximum_issue_lag_ms'])).any():
            raise ValueError('Forecast was issued too late for its nominal origin')
        if (f.cutoff-f.observed_at>int(plan.get('maximum_observation_age_ms',3600000))).any():
            raise ValueError('Prefix observation is too stale')
        minimum=int(plan.get('minimum_prefix_points',0))
        if minimum and ('prefix_points' not in f or (f.prefix_points<minimum).any()):
            raise ValueError('Prefix support is below the frozen eligibility rule')
    base=f[f.method==cfg['primary']].set_index('case_id').sort_index()
    identity=['event','rank','start_ms','end_ms','cutoff','observed_at','current']
    for method,part in f.groupby('method'):
        pp=part.set_index('case_id').sort_index()
        if not pp[identity].equals(base[identity]):
            raise ValueError('Methods must share identical blind cases')
    prior=batches(root,event['event']);used=set();observations=set()
    for p in prior:
        old=json.loads((p/'event.json').read_text())
        if {k:old[k] for k in event}!=event:raise ValueError('Event metadata changed')
        oldframe=pd.read_csv(p/'forecasts.csv')
        used.update(oldframe.case_id)
        observations.update(map(tuple,oldframe[physical].to_numpy(np.int64)))
    if used&set(f.case_id):raise ValueError('Previously issued cases cannot be replaced or counted again')
    if observations.intersection(map(tuple,f[physical].to_numpy(np.int64))):
        raise ValueError('A physical forecast cutoff has already been sealed')
    parent=root/'events'/str(event['event'])/'batches';parent.mkdir(parents=True,exist_ok=True)
    dest=parent/str(issued)
    if dest.exists():raise FileExistsError(dest)
    temp=Path(tempfile.mkdtemp(prefix='.pending-',dir=parent))
    try:
        f.sort_values(['method','case_id']).to_csv(temp/'forecasts.csv',index=False)
        write_new(temp/'event.json',event|{'issued_ms':issued,'registry_seal_sha256':digest(root/'registry/seal.json')})
        seal(temp);temp.rename(dest)
    except BaseException:
        # Preserve pending output for audit; it is never accepted as a sealed batch.
        raise
    return dest


def planned_cutoffs(event,plan):
    start,end=event['start_ms'],event['end_ms'];out=set()
    for s in plan['stages']:
        value=float(s['value'])
        t=start+round(value*(end-start)) if s['axis']=='progress' else (
            start+round(value*3600000) if s['axis']=='elapsed' else end-round(value*3600000))
        if start<t<end:out.add(t)
    return out


def release(root,event,outcomes,*,captured_ms,source):
    root=Path(root);config(root);known=batches(root,event)
    if not known:raise ValueError('A sealed prior forecast is required before feedback')
    info=json.loads((known[0]/'event.json').read_text());now=now_ms()
    if now<=info['end_ms'] or not info['end_ms']<=int(captured_ms)<=now or not source:
        raise ValueError('Feedback requires an ended event and actual capture provenance')
    ranks=set()
    for batch in known:
        if json.loads((batch/'event.json').read_text())['issued_ms']>=captured_ms:
            raise ValueError('Outcome capture must follow the sealed prediction')
        ranks.update(pd.read_csv(batch/'forecasts.csv')['rank'])
    actual={str(int(k)):float(v) for k,v in outcomes.items()}
    if set(map(int,actual))!=ranks or any(not math.isfinite(v) or v<=0 for v in actual.values()):
        raise ValueError('Outcomes must cover exactly the issued ranks and be positive')
    dest=root/'events'/str(event)/'outcomes';dest.mkdir(exist_ok=False)
    write_new(dest/'outcomes.json',{'actual':actual,'captured_ms':int(captured_ms),'released_ms':now,
              'source':str(source),'event':event,'end_ms':info['end_ms']})
    seal(dest)
    return dest


def released_history(root,*,query_start_ms):
    """Trusted feedback export, gated by actual recorded release rather than event end alone."""
    root=Path(root);config(root);query=int(query_start_ms)
    if query>now_ms() or query<LOWER:raise ValueError('History query clock is invalid')
    frames=[]
    for batch in batches(root):
        event=json.loads((batch/'event.json').read_text());outcome=batch.parent.parent/'outcomes'
        if not outcome.exists():continue
        verify(outcome);out=json.loads((outcome/'outcomes.json').read_text())
        if event['end_ms']>=query or out['released_ms']>=query:continue
        if out['captured_ms']<event['end_ms'] or out['captured_ms']<=event['issued_ms'] or out['captured_ms']>out['released_ms']:
            raise ValueError('Invalid actual feedback chronology')
        f=pd.read_csv(batch/'forecasts.csv')
        f['actual']=f['rank'].map(lambda r:out['actual'][str(int(r))])
        f['released_ms']=out['released_ms']
        f['issued_log_error']=np.log(f.predicted/f.actual)
        frames.append(f)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def score(root):
    root=Path(root);cfg=config(root);issued=batches(root)
    if not issued:return {'status':'waiting_for_eligible_forecasts','events':0}
    panels=[]
    for p in issued:
        event=json.loads((p/'event.json').read_text());outcome=p.parent.parent/'outcomes'
        if not outcome.exists():continue
        verify(outcome);out=json.loads((outcome/'outcomes.json').read_text())
        if out['released_ms']<=event['end_ms'] or out['captured_ms']<=event['issued_ms'] or out['captured_ms']<event['end_ms'] or out['released_ms']>now_ms():
            raise ValueError('Invalid release chronology')
        f=pd.read_csv(p/'forecasts.csv');f['actual']=f['rank'].map(lambda r:out['actual'][str(int(r))])
        panels.append(f)
    if not panels:return {'status':'waiting_for_verified_outcomes','events':0,'sealed_batches':len(issued)}
    f=pd.concat(panels,ignore_index=True)
    if f.duplicated(['method','case_id']).any():raise ValueError('Repeated scored cases')
    f['ape']=100*(f.predicted/f.actual-1).abs();f['ae']=(f.predicted-f.actual).abs();f['se']=(f.predicted-f.actual)**2
    metrics={}
    for method,x in f.groupby('method'):
        metrics[method]={'event_macro_mape_pct':float(x.groupby('event').ape.mean().mean()),'case_mape_pct':float(x.ape.mean()),
                         'mae_ep':float(x.ae.mean()),'rmse_ep':float(np.sqrt(x.se.mean())),'p90_ape_pct':float(x.ape.quantile(.9))}
    result={'status':'observed_prospective_results','events':int(f.event.nunique()),'cases':int(f.case_id.nunique()),
            'primary':cfg['primary'],'methods':metrics,'sealed_batches':len(issued)}
    if cfg.get('plan'):
        coverage=[]
        for event,x in f[f.method==cfg['primary']].groupby('event'):
            info={key:int(x.iloc[0][key]) for key in ['start_ms','end_ms']}
            expected=len(planned_cutoffs(info,cfg['plan']))*len(cfg['plan']['ranks'])
            coverage.append({'event':int(event),'planned_physical_cases':expected,'received_physical_cases':len(x),
                             'unreceived_cases':expected-len(x)})
        result['grid_coverage']=coverage
        result['target_events']=cfg['plan']['target_events']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    sub=p.add_subparsers(dest='action',required=True)
    r=sub.add_parser('register');r.add_argument('--methods-json',type=Path,required=True);r.add_argument('--primary',default='ewma_reference');r.add_argument('--plan-json',type=Path,required=True)
    s=sub.add_parser('seal');s.add_argument('--event-json',type=Path,required=True);s.add_argument('--forecasts-csv',type=Path,required=True)
    o=sub.add_parser('release');o.add_argument('--outcomes-json',type=Path,required=True)
    h=sub.add_parser('history');h.add_argument('--query-start-ms',type=int,required=True);h.add_argument('--output',type=Path,required=True)
    sub.add_parser('score');a=p.parse_args()
    if a.action=='register':result=register(a.root,json.loads(a.methods_json.read_text()),primary=a.primary,plan=json.loads(a.plan_json.read_text()))
    elif a.action=='seal':result={'sealed':str(issue(a.root,json.loads(a.event_json.read_text()),pd.read_csv(a.forecasts_csv)))}
    elif a.action=='release':
        x=json.loads(a.outcomes_json.read_text());result={'released':str(release(a.root,int(x['event']),x['actual'],captured_ms=x['captured_ms'],source=x['source']))}
    elif a.action=='history':
        f=released_history(a.root,query_start_ms=a.query_start_ms)
        with a.output.open('x') as stream:f.to_csv(stream,index=False)
        result={'released_rows':len(f),'output':str(a.output)}
    else:result=score(a.root)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
