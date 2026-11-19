"""Fixed paper diagnostics over detached, causally released forecast records.

No datasets, paths, raw labels, fitted trees or mutable persistent state enter
this module. Positive log_error denotes an overprediction of the routed base.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
import pandas as pd
from analysis import rolling_residual_stats as stats

TARGET_COLUMNS = ['case_id','event','rank','phase','progress','start_ms','end_ms',
                  'cutoff','observed_at','current','event_type','band_group','split',
                  'duration_h','remaining_h','rate_6h_relative','rate_24h_relative','base_prediction']
PAIR = {'anchor_alpha':.2,'difference_alpha':.6,'anchor_shrinkage':10.,
        'difference_shrinkage':5.,'minimum_pairs':1,'target_rank':1000,'anchor_rank':2000}
SWEEPS = {'anchor_alpha':[.1,.3], 'difference_alpha':[.3,.9],
          'anchor_shrinkage':[0.,5.,20.], 'difference_shrinkage':[0.,2.5,10.],
          'minimum_pairs':[3,5,10]}
POLICIES = {'delay24h':{'delay_ms':86400000}, 'delay7d':{'delay_ms':604800000},
            'delay_one_query':{'query_delay':1}, 'drop10pct':{'drop_fraction':.1},
            'drop30pct':{'drop_fraction':.3}, 'missing_anchor2000':{'drop_rank':2000},
            'missing_target1000':{'drop_rank':1000}}
CORE = ['uncorrected','ewma_reference','legacy_pair','time_proportion','history_growth',
        'residual_mean','window5','window10','ewma_fast','pair_anchor_only','pair_difference_only',
        'pair_swapped_anchor','same_filter_pair','same_panel_independent',
        'kalman_level','kalman_trend','dlm_fullcov']


def menu():
    recipes={name:{'kind':name} for name in CORE}
    for key,values in SWEEPS.items():
        for value in values:
            name=f'pair_{key}_{value:g}'.replace('.','p')
            recipes[name]={'kind':'pair_sensitivity',key:value}
    for target,anchor in [(400,500),(1000,3000),(2000,3000)]:
        recipes[f'pair_T{target}_T{anchor}']={'kind':'other_pair','target_rank':target,'anchor_rank':anchor}
    for policy in POLICIES:
        for method in ['ewma_reference','legacy_pair','dlm_fullcov']:
            recipes[f'{method}__{policy}']={'kind':'feedback','method':method,'policy':policy}
    return recipes


def validate(history,target,query_start_ms):
    past,current,clock=stats._frames(history,target,query_start_ms,context=False)
    if current.empty or (set(current)-set(TARGET_COLUMNS)):
        raise ValueError('Only allowlisted blind target fields are accepted')
    for f in [past,current]:
        for col in ['current','base_prediction']:
            if col not in f or not np.isfinite(f[col].to_numpy(float)).all():raise ValueError('Invalid base score')
        if ((f.current<=0).any() or (f.base_prediction<f.current-1e-6).any()
                or (f.base_prediction>f.current*np.exp(4)+1e-6).any()):raise ValueError('Invalid score bounds')
        if {'cutoff','observed_at'}<=set(f):
            if ((f.observed_at<f.start_ms).any() or (f.observed_at>f.cutoff).any()
                    or (f.cutoff>=f.end_ms).any()):raise ValueError('Invalid blind prefix clocks')
    return past,current,clock


def final(target,bias):
    return np.clip(target.base_prediction.to_numpy(float)*np.exp(-np.clip(bias,-.3,.3)),
                   target.current.to_numpy(float),target.current.to_numpy(float)*np.exp(4))


def shrink(value,n,k):
    return float(value)*n/(n+k) if n else 0.


def _warm(means,target,*,alpha=.3,window=None,mean=False):
    state={}
    for (rank,phase),part in means.groupby(['rank','phase'],sort=False):
        e=part.error.to_numpy(float)
        if window is not None:e=e[-window:]
        raw=float(e.mean()) if mean else stats._ewma(e,alpha)[0]
        state[(int(rank),int(phase))]=shrink(raw,len(e),10.)
    return np.array([state.get((int(r.rank),int(r.phase)),0.) for r in target.itertuples()])


def pair_bias(means,target,warm,*,recipe=None,mode='complete'):
    cfg=PAIR|dict(recipe or {});anchor_rank=cfg['anchor_rank'];target_rank=cfg['target_rank']
    output=np.asarray(warm,float).copy();details=[]
    for phase in target.phase.unique():
        part=means[means.phase==phase]
        panel=part.pivot(index='event',columns='rank',values='error').reindex(columns=[target_rank,anchor_rank])
        times=part.drop_duplicates('event').set_index('event')[['end_ms','start_ms']]
        panel=panel.join(times).sort_values(['end_ms','start_ms']);panel=panel.reset_index().sort_values(['end_ms','start_ms','event'])
        complete=panel.dropna(subset=[target_rank,anchor_rank]);n=len(complete)
        anchors=panel[anchor_rank].dropna().to_numpy(float);na=len(anchors)
        a=shrink(stats._ewma(anchors,cfg['anchor_alpha'])[0],na,cfg['anchor_shrinkage'])
        d=shrink(stats._ewma((complete[target_rank]-complete[anchor_rank]).to_numpy(float),cfg['difference_alpha'])[0],n,cfg['difference_shrinkage'])
        for i,row in enumerate(target.itertuples()):
            if row.phase!=phase or row.rank not in [target_rank,anchor_rank]:continue
            fallback=False
            if mode in ['same_filter','same_panel']:
                if n:
                    a0=shrink(stats._ewma(complete[anchor_rank].to_numpy(float),.3)[0],n,10.)
                    d0=shrink(stats._ewma((complete[target_rank]-complete[anchor_rank]).to_numpy(float),.3)[0],n,10.)
                    value=(a0+d0 if row.rank==target_rank else a0) if mode=='same_filter' else shrink(stats._ewma(complete[row.rank].to_numpy(float),.3)[0],n,10.)
                else:value=float(warm[i]);fallback=True
            elif row.rank==anchor_rank:
                if na:value=0. if mode=='difference_only' else a
                else:value=float(warm[i]);fallback=True
            elif n>=cfg['minimum_pairs'] and na:
                value=a if mode=='anchor_only' else d if mode=='difference_only' else a+d
            else:value=float(warm[i]);fallback=True
            output[i]=value
            # One state per phase/rank; not one repeated state per prefix.
            if not any(x['phase']==int(phase) and x['rank']==int(row.rank) for x in details):
                details.append({'rank':int(row.rank),'phase':int(phase),'anchor_events':na,
                                'paired_events':n,'anchor':a,'difference':d,'fallback':fallback,'bias':float(value)})
    return output,details


def feedback_history(history,*,query_start_ms,policy,query_starts=()):
    result=history.copy(deep=True)
    if result.empty:return result
    available=result.end_ms.to_numpy(np.int64)+int(policy.get('delay_ms',0))
    if 'feedback_available_at_ms' in result:
        available=np.maximum(available,result.feedback_available_at_ms.to_numpy(np.int64))
    if policy.get('query_delay'):
        starts=np.array(sorted(set(query_starts)),dtype=np.int64)
        if not len(starts):raise ValueError('Query-delay policy requires frozen query starts')
        for i,end in enumerate(result.end_ms):
            positions=np.flatnonzero(starts>int(end));j=policy['query_delay']-1
            available[i]=max(available[i],int(starts[positions[j]]) if len(positions)>j else np.iinfo(np.int64).max)
    keep=available<int(query_start_ms)
    if 'drop_rank' in policy:keep &= result['rank'].to_numpy()!=policy['drop_rank']
    if 'drop_fraction' in policy:
        keep &= np.array([int.from_bytes(hashlib.sha256(f'{r.event}:{r.rank}:{r.phase}:42'.encode()).digest()[:8],'big')/2**64>=policy['drop_fraction'] for r in result.itertuples()])
    return result.loc[keep].copy(deep=True)


def _psd(matrix,floor=1e-6):
    matrix=np.asarray(matrix,float);values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    return (vectors*np.maximum(values,floor))@vectors.T


def dlm_parameters(history):
    training=history[history['split']=='train'] if 'split' in history else history
    means=stats._event_means(training);parameters={}
    for phase in range(3):
        part=means[means.phase==phase]
        table=part.pivot(index='event',columns='rank',values='error').reindex(columns=[1000,2000])
        times=part.drop_duplicates('event').set_index('event')[['end_ms','start_ms']]
        table=table.join(times).reset_index().sort_values(['end_ms','start_ms','event']).dropna(subset=[1000,2000])
        levels=table[[1000,2000]].to_numpy(float);n=len(levels)
        if n<8:q=np.eye(2)*.0001;r=np.eye(2)*.01;p=np.eye(2)*.01
        else:
            c=_psd(np.cov(levels,rowvar=False,ddof=1))
            differences=np.cov(np.diff(levels,axis=0),rowvar=False,ddof=1)
            r=np.diag(np.maximum(.5*np.diag(differences),1e-6))
            q=_psd(.1*(.5*c+.5*np.diag(np.diag(c))))
            p=_psd(c+np.eye(2)*1e-6)
        parameters[str(phase)]={'Q':q.tolist(),'R':r.tolist(),'P0':p.tolist(),
                                'training_pairs':n,'fallback_constants':n<8}
    return parameters


def dlm_filter(errors,parameters,*,anchor_coordinates=False):
    """Random-walk filtering with correlated process noise and missing coordinates."""
    errors=np.asarray(errors,float).reshape(-1,2)
    transform=np.array([[0.,1.],[1.,-1.]]) if anchor_coordinates else np.eye(2)
    inverse=np.linalg.inv(transform);state=np.zeros(2)
    q0=np.asarray(parameters['Q'],float);r0=np.asarray(parameters['R'],float)
    q=transform@q0@transform.T;p=transform@np.asarray(parameters['P0'],float)@transform.T
    for observation in errors:
        p=p+q;valid=np.isfinite(observation)
        if not valid.any():continue
        if valid.all() and anchor_coordinates:
            h=np.eye(2);y=transform@observation;r=transform@r0@transform.T
        else:h=np.eye(2)[valid]@inverse;y=observation[valid];r=r0[np.ix_(valid,valid)]
        innovation=h@p@h.T+r;gain=np.linalg.solve(innovation,h@p).T
        state=state+gain@(y-h@state)
        residual=np.eye(2)-gain@h
        p=residual@p@residual.T+gain@r@gain.T;p=(p+p.T)/2
    raw=inverse@state;covariance=inverse@p@inverse.T
    return raw,{'mean':raw.tolist(),'covariance':covariance.tolist(),
                'coordinate_covariance':p.tolist(),'anchor_coordinates':bool(anchor_coordinates)}


def _dlm(means,target,warm,parameters):
    out=np.asarray(warm).copy();details={}
    for phase in target.phase.unique():
        part=means[(means.phase==phase)&means['rank'].isin([1000,2000])]
        table=part.pivot(index='event',columns='rank',values='error').reindex(columns=[1000,2000])
        times=part.drop_duplicates('event').set_index('event')[['end_ms','start_ms']]
        table=table.join(times).reset_index().sort_values(['end_ms','start_ms','event'])
        value,state=dlm_filter(table[[1000,2000]].to_numpy(float),parameters[str(int(phase))])
        details[str(int(phase))]=state|{'events':list(map(int,table.event)),'parameters':parameters[str(int(phase))]}
        for i,row in enumerate(target.itertuples()):
            if row.phase==phase and row.rank in [1000,2000]:out[i]=value[[1000,2000].index(row.rank)]
    return out,details


def empirical_intervals(history,target,center):
    """Grouped historical intervals; no exchangeability/coverage guarantee is claimed."""
    if len(center)!=len(target):raise ValueError('Wrong interval center panel')
    records=[]
    if 'center_log_error' in history:
        h=history.copy();h['absolute']=np.abs(h.center_log_error.to_numpy(float))
        if not np.isfinite(h.absolute).all():raise ValueError('Nonfinite released center error')
        scores=h.groupby(['event','rank','phase']).absolute.max().reset_index()
        pool=scores.groupby(['event','phase']).absolute.max().to_numpy(float)
    else:scores=pd.DataFrame(columns=['event','rank','phase','absolute']);pool=np.array([])
    for row,point in zip(target.itertuples(),center):
        own=scores[(scores['rank']==row.rank)&(scores.phase==row.phase)].absolute.to_numpy(float)
        for level in [.8,.9]:
            use=own;source='rank_phase';k=math.ceil((len(use)+1)*level)
            if k>len(use):use=pool;source='all_event_phase_pool';k=math.ceil((len(use)+1)*level)
            available=k<=len(use) and len(use)>0
            q=float(np.sort(use)[k-1]) if available else float('inf')
            lower=float(point*np.exp(-q));upper=float(point*np.exp(q))
            records.append({'case_id':row.case_id,'event':int(row.event),'rank':int(row.rank),'phase':int(row.phase),
                            'method':'ewma_reference','level':level,'nominal_coverage':level,
                            'center':float(point),'lower':lower,'upper':upper,'q_log':q,
                            'support':len(use),'own_support':len(own),'order_index':k,
                            'source':source if available else 'unavailable','available':bool(available)})
    return pd.DataFrame(records)


def predict_all(history,target,*,query_start_ms,parameters=None,query_starts=()):
    past,current,clock=validate(history,target,query_start_ms);means=stats._event_means(past)
    # Standalone synthetic calls can use the observed query starts; the trusted
    # replay coordinator always supplies the complete frozen issuance schedule.
    query_starts=query_starts or tuple(sorted(set(map(int,past.start_ms))|{clock}))
    warm=_warm(means,current);base=current.base_prediction.to_numpy(float)
    out={'uncorrected':base,'ewma_reference':final(current,warm)};details={}
    bias,details['legacy_pair']=pair_bias(means,current,warm);out['legacy_pair']=final(current,bias)
    out['time_proportion']=np.clip(current.current.to_numpy()/current.progress.to_numpy(),current.current,current.current*np.exp(4))
    growth=[]
    for row in current.itertuples():
        summaries=[]
        for _,part in past[past['rank']==row.rank].groupby('event',sort=False):
            logw=-.5*((part.progress.to_numpy()-row.progress)/.15)**2;w=np.exp(logw-logw.max())
            g=np.log(part.base_prediction.to_numpy()/part.current.to_numpy())-part.log_error.to_numpy()
            summaries.append(float(w@g/w.sum()))
        growth.append(float(np.clip(np.mean(summaries),0,4)) if summaries else float(np.clip(-np.log(row.progress),0,4)))
    out['history_growth']=current.current.to_numpy()*np.exp(growth)
    for name,kwargs in [('residual_mean',{'mean':True}),('window5',{'window':5,'mean':True}),('window10',{'window':10,'mean':True}),('ewma_fast',{'alpha':.6})]:
        out[name]=final(current,_warm(means,current,**kwargs))
    for name,mode,recipe in [('pair_anchor_only','anchor_only',{}),('pair_difference_only','difference_only',{}),
                              ('pair_swapped_anchor','complete',{'anchor_rank':1000,'target_rank':2000}),
                              ('same_filter_pair','same_filter',{}),('same_panel_independent','same_panel',{})]:
        bias,details[name]=pair_bias(means,current,warm,recipe=recipe,mode=mode);out[name]=final(current,bias)
    for name in ['kalman_level','kalman_trend']:
        bias,details[name]=stats.predict_bias(name,past,current,query_start_ms=clock);out[name]=final(current,bias)
    pars=parameters or {'complete':dlm_parameters(past)}
    bias,details['dlm_fullcov']=_dlm(means,current,warm,pars['complete']);out['dlm_fullcov']=final(current,bias)
    for name,recipe in menu().items():
        if recipe['kind'] in ['pair_sensitivity','other_pair']:
            bias,details[name]=pair_bias(means,current,warm,recipe={k:v for k,v in recipe.items() if k!='kind'});out[name]=final(current,bias)
        elif recipe['kind']=='feedback':
            policy_name=recipe['policy'];masked=feedback_history(past,query_start_ms=clock,policy=POLICIES[policy_name],query_starts=query_starts)
            mm=stats._event_means(masked);mw=_warm(mm,current);method=recipe['method']
            if method=='ewma_reference':bias=mw;state={'feedback_events':int(masked.event.nunique())}
            elif method=='legacy_pair':bias,state=pair_bias(mm,current,mw)
            else:bias,state=_dlm(mm,current,mw,pars.get(policy_name) or dlm_parameters(masked))
            out[name]=final(current,bias);details[name]={'policy':policy_name,'feedback_events':int(masked.event.nunique()),'state':state}
    for name,value in out.items():
        if len(value)!=len(current) or not np.isfinite(value).all():raise ValueError('Nonfinite or missing forecast: '+name)
    return out,details
