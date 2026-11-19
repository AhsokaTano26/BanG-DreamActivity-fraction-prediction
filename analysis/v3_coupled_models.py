"""Preregistered structured calibration over detached causal forecast history."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import gammaln,logsumexp
from analysis import rolling_residual_stats as stats
from analysis import v3_next_calibration as old

SEEDS=(42,123,2026)
EXPERTS=('kernel','gap','covariate','warm')
PRIOR=np.array([.35,.35,.15,.15])
LAGS=tuple('lag_'+v for v in ('anchor','gap_fast','gap_slow','gap_latest','gap_sd5','gap_n','days_since'))
NUMERIC=('progress','log_remaining_h','log_duration_h','log_base_growth','peer_log_ratio','rate_6h_relative','rate_24h_relative',*LAGS)
GATE_NUMERIC=NUMERIC+('rank_flag',)+tuple('ex_bias_'+e for e in EXPERTS)
CATEGORIES=('phase','event_type','band_group')
RECIPES={
    'ewma_warm':{'reference':'sealed ewma_warm'},
    'kernel_reference':{'reference':'sealed mape_kernel'},
    'coupled_gap':{'reference':'sealed coupled_gap'},
    **{f'blend_gap{n}':{'gap_weight':n/100,'mixture':'arithmetic final scores'} for n in (25,50,75)},
    'gap_changepoint':{'hazard':.125,'mu0':0.,'kappa0':1.,'alpha0':2.,'beta0':.01,'shrinkage':5.,'reset':'before observation; new run contains it'},
    **{f'gap_covariate{a}':{'alpha':float(a),'minimum_events':8,'irls_rounds':4,'huber_delta':.1,'blend':.5,'shrinkage':20.,'weights':'one unit per event after each robust update',
        'bias_blend':'(1-lambda)*log(base/bounded_gap_score)+lambda*(lag_anchor+ridge_contrast);lambda=.5*n/(n+20);then same bounds'} for a in (5,50)},
    'prior_expert_mix':{'experts':list(EXPERTS),'prior':PRIOR.tolist()},
    'fixed_share':{'experts':list(EXPERTS),'prior':PRIOR.tolist(),'eta':5.,'share':.05,'loss_clip':[0.,1.],'conditioning':'rank/phase; one event mean APE'},
    'context_gate':{'experts':list(EXPERTS),'prior':PRIOR.tolist(),'expert_bias_formula':'log(base/expert_final_score)','hidden':8,'dropout':.1,'minimum_events':12,'shrinkage':20.,'epochs':80,'learning_rate':.001,'weight_decay':.05,'threads':1,'seeds':list(SEEDS),'loss':'event-equal final-score MAPE','ensemble':'arithmetic final scores'},
}

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def weights(frame):
    return frame.event.map(1./frame.event.value_counts()).to_numpy(float)

def category(value):return '__missing__' if pd.isna(value) else str(value)

@dataclass
class Preprocessor:
    numeric:tuple
    medians:np.ndarray
    centers:np.ndarray
    scales:np.ndarray
    categories:dict

    @classmethod
    def fit(cls,frame,w,numeric=NUMERIC):
        raw=frame[list(numeric)].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        med=[]
        for col in raw.T:
            ok=np.isfinite(col)
            if not ok.any():med.append(0.);continue
            idx=np.argsort(col[ok],kind='stable');v=col[ok][idx];cw=np.cumsum(w[ok][idx])
            med.append(v[min(np.searchsorted(cw,cw[-1]/2),len(v)-1)])
        med=np.asarray(med);filled=np.where(np.isfinite(raw),raw,med)
        center=np.average(filled,axis=0,weights=w)
        scale=np.sqrt(np.average((filled-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
        cats={k:sorted({category(v) for v in frame[k]}) for k in CATEGORIES}
        return cls(tuple(numeric),med,center,scale,cats)

    def transform(self,frame):
        raw=frame[list(self.numeric)].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        x=np.clip((np.where(np.isfinite(raw),raw,self.medians)-self.centers)/self.scales,-10,10)
        cols=[x]
        for k,values in self.categories.items():
            arr=np.array([category(v) for v in frame[k]])
            cols.append(np.column_stack([arr==v for v in values]).astype(float))
        return np.column_stack(cols)

    def state(self):
        return {'numeric':list(self.numeric),'medians':self.medians.tolist(),'centers':self.centers.tolist(),
                'scales':self.scales.tolist(),'categories':self.categories}

def final(target,bias):
    return old._final(target,np.clip(np.asarray(bias,float),-.3,.3))

def bounded(target,prediction):
    """Protect exact score floors against floating-point mixture roundoff."""
    lo=np.maximum(target.current.to_numpy(),target.base_prediction.to_numpy()*np.exp(-.3))
    hi=np.minimum(target.current.to_numpy()*np.exp(4),target.base_prediction.to_numpy()*np.exp(.3))
    return np.clip(np.asarray(prediction,float),lo,hi)

def lag_features(past,target):
    current=target.copy(deep=True)
    means=stats._event_means(past)
    lookup={}
    for phase in current.phase.unique():
        group=means[means.phase==phase]
        anchor=group[group['rank']==2000]
        raw,na=stats._ewma(anchor.error.to_numpy(),.2)
        table=group.pivot(index='event',columns='rank',values='error').reindex(columns=[1000,2000]).dropna()
        times=group.drop_duplicates('event').set_index('event')[['end_ms','start_ms']]
        pair=table.join(times).sort_values(['end_ms','start_ms']).reset_index().sort_values(['end_ms','start_ms','event'])
        gap=(pair[1000]-pair[2000]).to_numpy();n=len(gap)
        fast,_=stats._ewma(gap,.6);slow,_=stats._ewma(gap,.2)
        lookup[phase]=dict(zip(LAGS,[raw*na/(na+10),fast*n/(n+5),slow*n/(n+5),
             gap[-1] if n else 0.,np.std(gap[-5:]) if n else 0.,n,
             (current.start_ms.iloc[0]-pair.end_ms.iloc[-1])/86400000 if n else 0.]))
    for col in LAGS:current[col]=[lookup[p][col] for p in current.phase]
    current['log_remaining_h']=np.log1p(np.maximum(0,current.remaining_h))
    current['log_duration_h']=np.log1p(np.maximum(0,current.duration_h))
    return current

def changepoint_mean(errors):
    """NIG mixture with a changepoint before an observation (run length starts1)."""
    mu=np.array([0.]);k=np.array([1.]);a=np.array([2.]);b=np.array([.01]);prob=np.array([1.])
    def logpdf(y,mu,k,a,b):
        dof=2*a;scale2=b*(k+1)/(a*k)
        return gammaln((dof+1)/2)-gammaln(dof/2)-.5*np.log(dof*np.pi*scale2)-(dof+1)/2*np.log1p((y-mu)**2/(dof*scale2))
    for y in np.asarray(errors,float):
        if not np.isfinite(y):raise ValueError('Finite completed errors required')
        growth=np.log(np.maximum(prob,np.finfo(float).tiny))+np.log(.875)+logpdf(y,mu,k,a,b)
        reset=np.log(.125)+logpdf(y,np.array([0.]),np.array([1.]),np.array([2.]),np.array([.01]))[0]
        logs=np.r_[reset,growth];prob=np.exp(logs-logsumexp(logs))
        mu0=np.r_[0.,mu];k0=np.r_[1.,k];a0=np.r_[2.,a];b0=np.r_[.01,b]
        d=y-mu0;k=k0+1;mu=mu0+d/k;a=a0+.5;b=b0+.5*k0*d*d/k
    return float(prob@mu),{'probabilities':prob.tolist(),'means':mu.tolist(),'reset_probability':float(prob[0])}

def gap_regression(name,past,target,directory):
    cfg=RECIPES[name];p=past[past['rank']==1000].copy();n=p.event.nunique()
    directory.mkdir();w=weights(p) if len(p) else np.array([])
    diagnostics={'events':int(n),'status':'coupled_fallback','missing_anchor_cases':0}
    pred=target['ref_gap'].to_numpy().copy()
    if n>=cfg['minimum_events']:
        means=stats._event_means(past)
        amap=means[means['rank']==2000].set_index(['event','phase']).error.to_dict()
        anchors=np.array([amap.get((r.event,r.phase),r.lag_anchor) for r in p.itertuples()])
        y=p.log_error.to_numpy()-anchors
        diagnostics['missing_anchor_cases']=sum((r.event,r.phase) not in amap for r in p.itertuples())
        prep=Preprocessor.fit(p,w);x=np.column_stack([np.ones(len(p)),prep.transform(p)])
        penalty=np.eye(x.shape[1])*cfg['alpha'];penalty[0,0]=0.
        for iteration in range(cfg['irls_rounds']):
            coef=np.linalg.solve(x.T@(w[:,None]*x)+penalty,x.T@(w*y))
            if iteration+1<cfg['irls_rounds']:
                robust=np.minimum(1.,cfg['huber_delta']/np.maximum(np.abs(y-x@coef),1e-12))
                scale=pd.Series(robust,index=p.index).groupby(p.event).transform('sum').to_numpy()
                w=robust/scale
        own=target['rank'].to_numpy()==1000
        q=np.column_stack([np.ones(own.sum()),prep.transform(target.loc[own])])
        contrast=q@coef
        priorbias=np.log(target.base_prediction.to_numpy()[own]/pred[own])
        fast=priorbias-target.lag_anchor.to_numpy()[own]
        newbias=priorbias+cfg['blend']*n/(n+cfg['shrinkage'])*(contrast-fast)
        pred[own]=final(target.loc[own],newbias)
        joblib.dump({'preprocessor':prep,'coefficients':coef,'recipe':cfg},directory/'model.joblib',compress=3)
        write(directory/'preprocessor.json',prep.state())
        diagnostics.update(status='fitted',features=x.shape[1])
    pd.DataFrame({'case_id':p.case_id,'event':p.event,'weight':w}).to_csv(directory/'training_weights.csv',index=False)
    write(directory/'diagnostics.json',diagnostics)
    return pred,diagnostics

def fixed_share(past,target,pool):
    groups={};detail=[]
    for rank,phase in target[['rank','phase']].drop_duplicates().itertuples(index=False,name=None):
        p=past[(past['rank']==rank)&(past.phase==phase)]
        events=sorted([(g.end_ms.iloc[0],g.start_ms.iloc[0],e,g) for e,g in p.groupby('event')],key=lambda v:v[:3])
        w=PRIOR.copy()
        for _,_,_,g in events:
            actual=np.exp(np.log(g.base_prediction.to_numpy())-g.log_error.to_numpy())
            scores=g[['expert__'+e for e in EXPERTS]].to_numpy()
            loss=np.clip(np.mean(np.abs(scores/actual[:,None]-1),axis=0),0,1)
            w=w*np.exp(-5*loss);w/=w.sum();w=.95*w+.05*PRIOR
        groups[(rank,phase)]=w
        detail.append({'rank':int(rank),'phase':int(phase),'events':len(events),'weights':dict(zip(EXPERTS,map(float,w)))})
    return np.array([pool[i]@groups[(r.rank,r.phase)] for i,r in enumerate(target.itertuples())]),{'groups':detail}

def network(features):
    import torch.nn as nn
    net=nn.Sequential(nn.Linear(features,8),nn.SiLU(),nn.Dropout(.1),nn.Linear(8,4))
    nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    return net

def neural(past,target,pool,directory):
    cfg=RECIPES['context_gate'];n=past.event.nunique();shrink=n/(n+20)
    directory.mkdir();w=weights(past) if len(past) else np.array([])
    info={'status':'fitted' if n>=12 else 'prior_fallback','events':int(n),'cases':len(past),'shrinkage':shrink}
    write(directory/'config.json',{'recipe':cfg,'query_start_ms':int(target.start_ms.iloc[0]),'query_event':int(target.event.iloc[0]),
         'case_ids':list(target.case_id),'expert_predictions':pool.tolist(),'shrinkage':shrink,'status':info['status']})
    pd.DataFrame({'case_id':past.case_id,'event':past.event,'weight':w}).to_csv(directory/'training_weights.csv',index=False)
    if n<12:
        result={f'context_gate_seed{s}':pool@PRIOR for s in SEEDS}
    else:
        import torch
        torch.set_num_threads(1)
        prep=Preprocessor.fit(past,w,GATE_NUMERIC);joblib.dump(prep,directory/'preprocessor.joblib',compress=3)
        write(directory/'preprocessor.json',prep.state())
        x=torch.tensor(prep.transform(past),dtype=torch.float32);qx=torch.tensor(prep.transform(target),dtype=torch.float32)
        px=torch.tensor(past[['expert__'+e for e in EXPERTS]].to_numpy(),dtype=torch.float32)
        actual=torch.tensor(np.exp(np.log(past.base_prediction.to_numpy())-past.log_error.to_numpy()),dtype=torch.float32)
        wt=torch.tensor(w,dtype=torch.float32);prior=torch.tensor(PRIOR,dtype=torch.float32)
        result={};training={}
        with torch.random.fork_rng(devices=[]):
            for seed in SEEDS:
                torch.manual_seed(seed);model=network(x.shape[1]);optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
                losses=[];model.train()
                for _ in range(80):
                    optim.zero_grad(set_to_none=True)
                    gate=shrink*torch.softmax(model(x)+torch.log(prior),dim=1)+(1-shrink)*prior
                    loss=(torch.abs((px*gate).sum(dim=1)/actual-1)*wt).sum()/wt.sum()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite gate loss')
                    loss.backward();optim.step();losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():gate=shrink*torch.softmax(model(qx)+torch.log(prior),dim=1)+(1-shrink)*prior
                result[f'context_gate_seed{seed}']=np.sum(pool*gate.numpy().astype(float),axis=1)
                torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':optim.state_dict(),'input_features':x.shape[1],'seed':seed,'losses':losses},directory/f'seed{seed}.pt')
                training[str(seed)]={'first_loss':losses[0],'last_loss':losses[-1],'epochs':80}
        info['training']=training
    result={key:bounded(target,value) for key,value in result.items()}
    result['context_gate']=np.mean([result[f'context_gate_seed{s}'] for s in SEEDS],axis=0)
    write(directory/'diagnostics.json',info)
    return result,info

def replay_neural(directory,features):
    directory=Path(directory);cfg=json.loads((directory/'config.json').read_text())
    if {'actual','target','final','log_error'}&set(features):raise ValueError('Replay cannot accept outcomes')
    if list(features.case_id)!=cfg['case_ids'] or not (features.start_ms==cfg['query_start_ms']).all() or set(features.event)!={cfg['query_event']}:
        raise ValueError('Different replay event/case clock')
    pool=np.asarray(cfg['expert_predictions']);shrink=cfg['shrinkage']
    if cfg['status']=='prior_fallback':out={f'context_gate_seed{s}':pool@PRIOR for s in SEEDS}
    else:
        import torch
        torch.set_num_threads(1);prep=joblib.load(directory/'preprocessor.joblib')
        x=torch.tensor(prep.transform(features),dtype=torch.float32);prior=torch.tensor(PRIOR,dtype=torch.float32);out={}
        with torch.random.fork_rng(devices=[]):
            for seed in SEEDS:
                saved=torch.load(directory/f'seed{seed}.pt',map_location='cpu',weights_only=False)
                model=network(saved['input_features']);model.load_state_dict(saved['state_dict']);model.eval()
                with torch.no_grad():gate=shrink*torch.softmax(model(x)+torch.log(prior),dim=1)+(1-shrink)*prior
                out[f'context_gate_seed{seed}']=np.sum(pool*gate.numpy().astype(float),axis=1)
    out={key:bounded(features,value) for key,value in out.items()}
    out['context_gate']=np.mean([out[f'context_gate_seed{s}'] for s in SEEDS],axis=0)
    return out

def predict_all(history,target,references,*,query_start_ms,model_directory):
    past,current,query=old._validate(history,target,query_start_ms)
    if set(references)!={'kernel','gap','warm'}:raise ValueError('Exact blind reference set required')
    refs={k:np.asarray(v,float) for k,v in references.items()}
    for x in refs.values():
        if x.shape!=(len(current),) or not np.isfinite(x).all() or (x<current.current-1e-6).any() or (x>current.current*np.exp(4)+1e-6).any() or (np.abs(np.log(x/current.base_prediction))>.3+1e-10).any():raise ValueError('Invalid reference forecast')
    for name in EXPERTS:
        if len(past):
            if 'expert__'+name not in past:raise ValueError('Missing sealed historical expert panel')
            values=pd.to_numeric(past['expert__'+name],errors='raise').to_numpy(float)
            if not np.isfinite(values).all() or (values<past.current-1e-6).any() or (values>past.current*np.exp(4)+1e-6).any():raise ValueError('Invalid historical expert scores')
    directory=Path(model_directory)
    if directory.exists():raise FileExistsError('Preserve existing model artifacts')
    directory.mkdir(parents=True)
    current=lag_features(past,current)
    current['rank_flag']=(current['rank']==1000).astype(float)
    if 'rank_flag' not in past:past['rank_flag']=(past['rank']==1000).astype(float)
    current['ref_gap']=refs['gap']
    # Saved historical lag values are never refitted using this later query.
    for col in ('log_remaining_h','log_duration_h'):
        if col not in past:
            raw=past.remaining_h if col=='log_remaining_h' else past.duration_h
            past[col]=np.log1p(np.maximum(raw,0))
    scores={'ewma_warm':refs['warm'],'kernel_reference':refs['kernel'],'coupled_gap':refs['gap']};details={}
    for n in (25,50,75):scores[f'blend_gap{n}']=(1-n/100)*refs['kernel']+n/100*refs['gap']
    means=stats._event_means(past);cp=[];states=[]
    for i,row in enumerate(current.itertuples()):
        if row.rank==2000:cp.append(refs['gap'][i]);continue
        part=means[means.phase==row.phase]
        table=part.pivot(index='event',columns='rank',values='error').reindex(columns=[1000,2000]).dropna()
        table=table.join(part.drop_duplicates('event').set_index('event')[['end_ms','start_ms']]).sort_values(['end_ms','start_ms']).reset_index().sort_values(['end_ms','start_ms','event'])
        if len(table):
            value,state=changepoint_mean((table[1000]-table[2000]).to_numpy());value=row.lag_anchor+value*len(table)/(len(table)+5)
            cp.append(final(current.iloc[[i]],np.array([value]))[0]);states.append({'case_id':row.case_id,**state})
        else:cp.append(refs['gap'][i])
    scores['gap_changepoint']=np.array(cp);details['gap_changepoint']={'states':states}
    for a in (5,50):
        name=f'gap_covariate{a}';scores[name],details[name]=gap_regression(name,past,current,directory/name)
    pool=np.column_stack([refs['kernel'],refs['gap'],scores['gap_covariate5'],refs['warm']])
    for e,arr in zip(EXPERTS,pool.T):current['ex_bias_'+e]=np.log(current.base_prediction/arr)
    scores['prior_expert_mix']=pool@PRIOR
    scores['fixed_share'],details['fixed_share']=fixed_share(past,current,pool)
    neural_scores,details['context_gate']=neural(past,current,pool,directory/'context_gate');scores.update(neural_scores)
    for e,arr in zip(EXPERTS,pool.T):current['expert__'+e]=arr
    current=current.drop(columns='ref_gap')
    scores={name:bounded(current,value) for name,value in scores.items()}
    for name,arr in scores.items():
        if not np.isfinite(arr).all() or (arr<current.current-1e-5).any() or (np.abs(np.log(arr/current.base_prediction))>.3+1e-6).any():raise ValueError('Invalid final score '+name)
    write(directory/'config.json',{'recipes':RECIPES,'query_start_ms':query,'history_events':int(past.event.nunique()),'history_cases':len(past),'current_outcome_used':False})
    write(directory/'diagnostics.json',details)
    return scores,current,details
