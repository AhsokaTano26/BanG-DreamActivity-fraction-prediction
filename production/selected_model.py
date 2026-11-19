"""User-selected legacy_pair correction over unchanged, verified base packs."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.holdout_dataset import sha256
from analysis import v3_next_calibration as canonical
from analysis import rolling_residual_stats as stats
from production import model as base

METHOD='legacy_pair'
RECIPE={'pair':canonical.CONFIGS['coupled_gap'],'others':canonical.CONFIGS['ewma_warm']}
RECIPE_ID=base._identity(RECIPE)

def build_bias(feedback_rows,*,before_ms,exclude_event_id=None):
    checked=base.build_bias(feedback_rows,before_ms=before_ms,exclude_event_id=exclude_event_id)
    records=sorted(checked['records'],key=lambda r:(r['end_ms'],r['start_ms'],r['event'],r['rank'],r['phase']))
    dates={}
    for r in records:
        if dates.setdefault(r['event'],(r['start_ms'],r['end_ms']))!=(r['start_ms'],r['end_ms']):
            raise ValueError('Feedback ranks disagree on activity dates')
    past=pd.DataFrame([{**r,'log_error':r['mean_log_error'],'case_id':f"{r['event']}:{r['rank']}:{r['phase']}"} for r in records],
        columns=['event','rank','phase','start_ms','end_ms','log_error','case_id'])
    means=stats._event_means(past);groups={}
    for (rank,phase),part in means.groupby(['rank','phase']):
        mean,n=stats._ewma(part.error.to_numpy(float),.3)
        groups[f'{int(rank)}:{int(phase)}']={'mean':mean,'count':n,'bias':float(np.clip(mean*n/(n+10),-.3,.3))}
    target=pd.DataFrame([{'case_id':f'{r}:{p}','rank':r,'phase':p} for r in [1000,2000] for p in range(3)])
    warm=np.array([groups.get(f'{r.rank}:{r.phase}',{}).get('bias',0.) for r in target.itertuples()])
    values,details=canonical._coupled(past,target,warm,kalman=False)
    pair={r.case_id:{**d,'bias':float(v)} for r,d,v in zip(target.itertuples(),details['target_states'],values)}
    state={'schema_version':2,'forecast_method':METHOD,'recipe_id':RECIPE_ID,'recipe':RECIPE,
        'policy_id':checked['policy_id'],'before_ms':int(before_ms),'latest_end_ms':checked['latest_end_ms'],
        'records':records,'groups':groups,'coupled':pair,'source_sha256':sha256(Path(__file__)),
        'source':'verified issued raw forecast errors; one observation per event/rank/phase'}
    state['calibration_id']=base._identity(state);return state

def _validate(state):
    if (state.get('schema_version')!=2 or state.get('forecast_method')!=METHOD or state.get('recipe_id')!=RECIPE_ID
        or state.get('policy_id')!=base.POLICY_ID
        or base._identity({k:v for k,v in state.items() if k!='calibration_id'})!=state.get('calibration_id')):
        raise ValueError('Invalid selected calibration identity')

def bias_for(state,rank,phase):
    _validate(state);key=f'{int(rank)}:{int(phase)}'
    group=(state['coupled'] if int(rank) in [1000,2000] else state['groups']).get(key)
    return 0. if group is None else float(group['bias'])

def predict_pack(path,samples,*,metadata_directory=None,calibration=None):
    if not calibration:raise ValueError('Selected predictions require a pinned calibration state')
    if calibration.get('schema_version')==1:
        rows=base.predict_pack(path,samples,metadata_directory=metadata_directory,calibration=calibration)
        return [r|{'forecast_method':'pinned_independent_ewma','calibration_id':calibration.get('calibration_id')} for r in rows]
    _validate(calibration)
    rows=base.predict_pack(path,samples,metadata_directory=metadata_directory,calibration=None);out=[]
    for row in rows:
        if (calibration['before_ms']>row['start_ms'] or (calibration['latest_end_ms'] is not None
            and calibration['latest_end_ms']>=row['start_ms']) or any(r['event']==row['event'] for r in calibration['records'])):
            raise ValueError('Selected calibration contains own or future feedback')
        raw=row['raw_predicted_final_ep'];bias=bias_for(calibration,row['rank'],row['phase'])
        predicted=None if raw is None else max(row['current_ep'],math.exp(min(math.log(row['current_ep'])+4,math.log(raw)-bias,math.log(np.finfo(float).max))))
        out.append(row|{'predicted_final_ep':predicted,'log_bias':bias if raw is not None else 0.,
            'forecast_method':METHOD,'calibration_id':calibration['calibration_id']})
    return out
