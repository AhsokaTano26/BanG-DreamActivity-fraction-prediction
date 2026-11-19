"""Pure past-only residual sharing; no files, model objects or hidden state.

Positive z=log(base/actual) is an overprediction. Risk records use expert
forecasts originally issued before z was released, never refitted experts.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

MIN_START_MS = 1706504400000
RANKS = (10,20,30,40,50,100,200,300,400,500,1000,1500,2000,3000,
         4000,5000,10000,20000,30000)
PAIR = (1000,2000)
ADJACENT = tuple(tuple(RANKS[i:i+2]) for i in range(0,len(RANKS),2))
BLIND_COLUMNS = ('case_id','event','rank','phase','progress','start_ms','end_ms',
                 'cutoff','observed_at','current','event_type','band_group','split',
                 'duration_h','remaining_h','rate_6h_relative','rate_24h_relative',
                 'base_prediction')
CORE = ('case_id','event','rank','phase','progress','start_ms','end_ms',
        'current','base_prediction')
ISSUED = ('issued_ind','issued_pair','issued_adjacent','issued_samepanel')


def _validated(frame, historical=False):
    if not isinstance(frame,pd.DataFrame):
        raise ValueError('Expected detached pandas frame')
    allowed = set(BLIND_COLUMNS) | (set(ISSUED)|{'log_error'} if historical else set())
    required = set(CORE) | ({'log_error'} if historical else set())
    if set(frame)-allowed or required-set(frame):
        raise ValueError('Only allowlisted blind fields and released historical errors are accepted')
    result=frame.copy(deep=True)
    if result.case_id.isna().any() or result.case_id.duplicated().any():
        raise ValueError('Case identifiers must be nonmissing and unique')
    ints=('event','rank','phase','start_ms','end_ms')
    for column in set(ints)|{'progress','current','base_prediction'}|(
            {'log_error'}|set(ISSUED).intersection(result) if historical else set()):
        try: values=pd.to_numeric(result[column],errors='raise').to_numpy(float)
        except (ValueError,TypeError) as exc:
            raise ValueError('Nonfinite numeric input') from exc
        if not np.isfinite(values).all(): raise ValueError('Nonfinite numeric input')
        if column in ints:
            if np.any(values!=np.floor(values)): raise ValueError('Integer clocks and groups required')
            values=values.astype(np.int64)
        result[column]=values
    if (not set(result['rank']).issubset(RANKS) or (result.event<=0).any()
            or (result.start_ms<MIN_START_MS).any() or (result.end_ms<=result.start_ms).any()):
        raise ValueError('Invalid rank or dense-period clock')
    if ((result.progress<0).any() or (result.progress>1).any()
            or np.any(result.phase.to_numpy()!=np.searchsorted([.25,.9],result.progress,side='right'))):
        raise ValueError('Phase does not match progress')
    if ((result.current<=0).any() or (result.base_prediction<result.current-1e-6).any()
            or (result.base_prediction>result.current*np.exp(4)+1e-6).any()):
        raise ValueError('Invalid bounded base score')
    if 'split' in result and not result['split'].eq('train').all():
        raise ValueError('B accepts training queries only')
    for column in ['start_ms','end_ms','event_type','band_group']:
        if column in result and (result.groupby('event')[column].nunique(dropna=False)>1).any():
            raise ValueError('Conflicting event clock or context')
    if {'cutoff','observed_at'}<=set(result):
        if (not np.isfinite(result[['cutoff','observed_at']].to_numpy(float)).all()
                or (result.observed_at<result.start_ms).any()
                or (result.observed_at>result.cutoff).any()
                or (result.cutoff>=result.end_ms).any()):
            raise ValueError('Invalid blind prefix clock')
    return result


def _frames(history,current):
    past=_validated(history,True);target=_validated(current)
    if target.empty or target.event.nunique()!=1 or target.start_ms.nunique()!=1:
        raise ValueError('One nonempty query event required')
    clock=int(target.start_ms.iloc[0])
    if ((past.end_ms>=clock).any() or set(past.event)&set(target.event)
            or set(past.case_id)&set(target.case_id)):
        raise ValueError('Only strictly earlier completed events may be released')
    return past,target


def _means(history):
    return history.groupby(['event','rank','phase'],sort=True,as_index=False).agg(
        z=('log_error','mean'),start_ms=('start_ms','first'),end_ms=('end_ms','first')
    ).sort_values(['end_ms','start_ms','event','rank','phase'])


def _filter(values):
    state=0.;n=0
    for value in values:
        state=.7*state+.3*float(value) if n else float(value)
        n+=1
    return float(np.clip(state*n/(n+10.),-.3,.3)),n


def _group_table(means,group,phase):
    part=means[(means.phase==phase)&means['rank'].isin(group)]
    table=part.pivot(index='event',columns='rank',values='z').reindex(columns=list(group))
    times=part.drop_duplicates('event').set_index('event')[['end_ms','start_ms']]
    return table.join(times).reset_index().sort_values(['end_ms','start_ms','event']).dropna(subset=list(group))


def issue_experts(history,blind_current):
    """Construct all fixed experts once per query; each row remains outcome-free."""
    past,current=_frames(history,blind_current);means=_means(past)
    own={(int(rank),int(phase)):_filter(part.z) for (rank,phase),part in
         means.groupby(['rank','phase'],sort=True)}
    tables={}
    for group in (PAIR,)+ADJACENT:
        for phase in current.phase.unique():
            tables[(group,int(phase))]=_group_table(means,group,int(phase))
    rows=[]
    for row in current.itertuples(index=False):
        value,n=own.get((int(row.rank),int(row.phase)),(0.,0))
        out={'case_id':row.case_id,'expert_ind':value,'ind_history_events':n}
        for family,groups in [('pair',(PAIR,)),('adjacent',ADJACENT)]:
            group=next((group for group in groups if row.rank in group),None)
            shared=value;matched=value;support=0
            if group and len(group)>1:
                table=tables[(group,int(row.phase))]
                support=len(table)
                if support:
                    shared,_=_filter(table[list(group)].mean(axis=1))
                    matched,_=_filter(table[int(row.rank)])
            out['expert_'+family]=shared
            out[family+'_history_events']=support
            if family=='pair':out['expert_samepanel']=matched
        rows.append(out)
    return pd.DataFrame(rows)


def _risk_value(samples,k,decay):
    m=len(samples);n_eff=0.;a=0.;c=0.
    if m:
        weights=decay**np.arange(m-1,-1,-1,dtype=float)
        values=np.asarray(samples,float)
        if values.shape!=(m,2) or not np.isfinite(values).all() or (values[:,0]<0).any():
            raise ValueError('Invalid issued risk coefficients')
        a,c=(weights@values/weights.sum()).tolist()
        n_eff=float(weights.sum()**2/(weights@weights))
    usable=m>=5 and a>1e-8
    value=float(np.clip(c/(a*(1.+k/n_eff)),0.,1.)) if usable else 0.
    return {'lambda':value,'support':m,'n_eff':n_eff,'A':a,'C':c,
            'risk_usable':bool(usable)}


def risk_lambda(samples,*,k,decay):
    """Ridge-regularized quadratic minimizer and non-probabilistic sensitivity."""
    if not math.isfinite(k) or k<0 or not 0<decay<=1:
        raise ValueError('Invalid risk shrinkage or decay')
    result=_risk_value(samples,k,decay)
    deleted=[_risk_value(samples[:i]+samples[i+1:],k,decay)['lambda']
             for i in range(len(samples))] if result['risk_usable'] else []
    result['lambda_delete_min']=min(deleted) if deleted else result['lambda']
    result['lambda_delete_max']=max(deleted) if deleted else result['lambda']
    return result


def risk_observations(history,*,rank,phase,pool):
    """One (mean d^2, mean e*d) pair per completed event, in release order."""
    if pool not in ('pair','adjacent'):raise ValueError('Unknown sharing pool')
    part=history[(history['rank']==rank)&(history.phase==phase)].copy()
    if part.empty:return []
    shared='issued_'+pool
    if {'issued_ind',shared}-set(part):
        raise ValueError('Adaptive weights require originally issued experts')
    part['a']=(part[shared]-part.issued_ind)**2
    part['c']=(part.log_error-part.issued_ind)*(part[shared]-part.issued_ind)
    grouped=part.groupby('event',as_index=False).agg(a=('a','mean'),c=('c','mean'),
        start_ms=('start_ms','first'),end_ms=('end_ms','first'))
    grouped=grouped.sort_values(['end_ms','start_ms','event'])
    return list(grouped[['a','c']].itertuples(index=False,name=None))


def gate_state(history,*,phase):
    table=_group_table(_means(history),PAIR,phase).tail(12)
    result={'gate_pairs':len(table),'rho':0.,'gap_ratio':0.,'drift_ratio':0.,'gate_open':False}
    if len(table)<8:return result
    x=table[1000].to_numpy(float);y=table[2000].to_numpy(float)
    rms=lambda z:float(np.sqrt(np.mean(z*z)))
    rho=float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-12 and np.std(y)>1e-12 else 0.
    gap=rms(x-y)/(rms(x)+rms(y)+1e-8)
    drift=rms(np.diff(x-y))/(rms(np.diff(x))+rms(np.diff(y))+1e-8)
    return result|{'rho':rho,'gap_ratio':gap,'drift_ratio':drift,
                   'gate_open':bool(rho>=.15 and gap<=.5 and drift<=.75)}


def mix_experts(history,blind_current,configuration,experts):
    """Mix cached query experts; all historical feedback is revalidated."""
    past,current=_frames(history,blind_current);cfg=dict(configuration)
    kind=cfg.get('kind');pool=cfg.get('pool','pair')
    if kind not in ('fixed','risk','gated','same_panel') or pool not in ('pair','adjacent'):
        raise ValueError('Unknown frozen candidate configuration')
    if kind=='fixed' and not 0<=cfg['lambda']<=1:raise ValueError('Invalid fixed lambda')
    if (experts.case_id.duplicated().any() or list(experts.case_id)!=list(current.case_id)):
        raise ValueError('Cached experts must match current case order')
    records=[];cache={};gates={}
    for row,expert in zip(current.itertuples(index=False),experts.itertuples(index=False)):
        group=PAIR if pool=='pair' and row.rank in PAIR else (
            next((g for g in ADJACENT if row.rank in g),()) if pool=='adjacent' else ())
        active=len(group)>1 and row.rank in cfg.get('apply_ranks',RANKS)
        ind=float(expert.expert_samepanel if kind=='same_panel' else expert.expert_ind)
        shared=float(getattr(expert,'expert_'+pool))
        detail={'lambda':0.,'support':0,'n_eff':0.,'A':0.,'C':0.,
                'lambda_delete_min':0.,'lambda_delete_max':0.,'risk_usable':False}
        gate={}
        if active and kind=='fixed':detail['lambda']=float(cfg['lambda'])
        elif active and kind in ('risk','gated'):
            key=(int(row.rank),int(row.phase))
            if key not in cache:
                samples=risk_observations(past,rank=key[0],phase=key[1],pool=pool)
                cache[key]=risk_lambda(samples,k=cfg['k'],decay=cfg['decay'])
            detail=cache[key].copy()
            if kind=='gated':
                if row.phase not in gates:gates[row.phase]=gate_state(past,phase=int(row.phase))
                gate=gates[row.phase]
                if not gate['gate_open']:detail['lambda']=0.
        lam=detail['lambda'];bias=float(np.clip((1.-lam)*ind+lam*shared,-.3,.3))
        prediction=float(np.clip(row.base_prediction*np.exp(-bias),row.current,row.current*np.exp(4)))
        records.append({'case_id':row.case_id,'event':int(row.event),'rank':int(row.rank),
            'phase':int(row.phase),'expert_ind':ind,'expert_shared':shared,
            'ind_history_events':int(expert.ind_history_events),
            'shared_history_events':int(getattr(expert,pool+'_history_events')),
            **detail,**gate,'bias':bias,'prediction':prediction})
    return pd.DataFrame(records)


def predict(history,blind_current,configuration):
    return mix_experts(history,blind_current,configuration,
                       issue_experts(history,blind_current))
