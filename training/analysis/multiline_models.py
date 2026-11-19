"""Causal global/group/rank, graph and factor calibration for19ranking lines."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from analysis import rolling_residual_stats as stats
from analysis import v3_next_calibration as paired
from analysis.v3_coupled_models import Preprocessor,network,bounded,weights
from download_ranks import RANKS

SEEDS=(42,123,2026)
EXPERTS=('independent','legacy','hierarchy','factor')
PRIOR=np.array([.4,.2,.2,.2])
NUMERIC=('log_rank','context_family','progress','log_growth','log_remaining_h','log_duration_h','rate_6h_relative','rate_24h_relative',
         'own_slow','own_fast','own_n','group_anchor','global_anchor',*['bias_'+e for e in EXPERTS])
RECIPES={
 'uncorrected':{'correction':'none'},'ewma_reference':{'alpha':.3,'shrinkage':10.,'history':'all prior sealed forecasts'},
 'ewma_cold':{'alpha':.3,'shrinkage':10.,'history':'validation/test prior sealed forecasts only'},
 'legacy_pair':{'ranks1000_2000':'exact original coupled_gap','others':'independent warm EWMA'},
 'hierarchy':{'core':[1000,2000,3000,4000,5000],'rank_band_upper_edges':[50,500,5000],
    'global_alpha':.2,'group_alpha':.4,'rank_alpha':.6,'global_group_shrinkage':10.,'rank_shrinkage':5.,'unseen_rank':'within-band logrank interpolation of rank deviations'},
 'graph025':{'lambda':.25,'observed_diagonal':1.,'unseen_diagonal':.1,'edges':'adjacent logrank,1/(1+distance)','quantity':'own EWMA deviations around historical rank means'},
 'graph1':{'lambda':1.,'observed_diagonal':1.,'unseen_diagonal':.1,'edges':'adjacent logrank,1/(1+distance)','quantity':'own EWMA deviations around historical rank means'},
 'factor2':{'components':2,'scale_floor':.02,'factor_alpha':.5,'idiosyncratic_alpha':.6,'shrinkage':10.,'missing':'standardized0; unseen mean/scale/loadings logrank interpolation'},
 'factor4':{'components':4,'scale_floor':.02,'factor_alpha':.5,'idiosyncratic_alpha':.6,'shrinkage':10.,'missing':'standardized0; unseen mean/scale/loadings logrank interpolation'},
 'share_gate':{'experts':list(EXPERTS),'prior':PRIOR.tolist(),'eta':5.,'share':.05,'loss':'one event mean APE per rank/phase,clipped[0,1]'},
 'neural_gate':{'experts':list(EXPERTS),'prior':PRIOR.tolist(),'numeric':list(NUMERIC),'hidden':8,'dropout':.1,'minimum_events':12,
    'epochs':80,'learning_rate':.001,'weight_decay':.05,'shrinkage':20.,'threads':1,'seeds':list(SEEDS),'loss':'event-equal actual final score MAPE'},
 'prior_mix':{'experts':list(EXPERTS),'weights':PRIOR.tolist()},
}

def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def group(rank):return int(np.searchsorted([50,500,5000],rank,side='left'))

def ewma(x,alpha):
    x=np.asarray(x,float)
    if not len(x):return 0.
    result=x[0].copy()
    for row in x[1:]:result=(1-alpha)*result+alpha*row
    return result

def interpolate(values,known,ranks):
    values=np.asarray(values,float).copy();known=np.asarray(known,bool)
    if not known.any():return np.zeros_like(values)
    x=np.log(ranks)
    if values.ndim==1:values[~known]=np.interp(x[~known],x[known],values[known])
    else:
        for j in range(values.shape[1]):values[~known,j]=np.interp(x[~known],x[known],values[known,j])
    return values

def phase_state(means,phase,ranks):
    p=means[means.phase==phase].sort_values(['end_ms','start_ms','event'])
    order=p.drop_duplicates('event').event.tolist()
    matrix=p.pivot(index='event',columns='rank',values='error').reindex(index=order,columns=ranks).to_numpy(float)
    count=np.isfinite(matrix).sum(axis=0);N=len(matrix);known=count>0
    mu=np.array([np.mean(matrix[np.isfinite(matrix[:,j]),j]) if count[j] else 0. for j in range(len(ranks))])
    mu=interpolate(mu,known,ranks)
    slow=np.array([ewma(matrix[np.isfinite(matrix[:,j]),j],.3) for j in range(len(ranks))])
    fast=np.array([ewma(matrix[np.isfinite(matrix[:,j]),j],.6) for j in range(len(ranks))])
    warm=slow*count/(count+10)
    if not N:return {'matrix':matrix,'counts':count,'mu':mu,'warm':warm,'fast':fast,'global':0.,'group':np.zeros(4),'hierarchy':np.zeros(len(ranks)),'events':[]}
    core=np.array([r in [1000,2000,3000,4000,5000] for r in ranks])
    global_rows=np.array([np.median(row[core&np.isfinite(row)]) if (core&np.isfinite(row)).any() else np.median(row[np.isfinite(row)]) for row in matrix])
    global_level=float(ewma(global_rows,.2))*N/(N+10)
    group_level=np.zeros(4);group_rows=np.full((N,4),np.nan)
    for g in range(4):
        columns=np.array([group(r)==g for r in ranks])
        for i,row in enumerate(matrix):
            observed=columns&np.isfinite(row)
            if observed.any():group_rows[i,g]=np.median(row[observed])
        valid=np.isfinite(group_rows[:,g]);ng=int(valid.sum())
        group_level[g]=global_level+float(ewma(group_rows[valid,g]-global_rows[valid],.4))*ng/(ng+10)
    deviations=np.zeros(len(ranks))
    for j,r in enumerate(ranks):
        valid=np.isfinite(matrix[:,j]);n=count[j]
        deviations[j]=float(ewma(matrix[valid,j]-group_rows[valid,group(r)],.6))*n/(n+5)
    for g in range(4):
        indices=np.array([j for j,r in enumerate(ranks) if group(r)==g],int)
        if len(indices):deviations[indices]=interpolate(deviations[indices],known[indices],np.asarray(ranks)[indices])
    hierarchy=np.array([group_level[group(r)]+deviations[j] for j,r in enumerate(ranks)])
    return {'matrix':matrix,'counts':count,'mu':mu,'warm':warm,'fast':fast,'global':global_level,'group':group_level,'hierarchy':hierarchy,'events':order}

def graph_bias(state,ranks,penalty):
    n=state['counts'];known=n>0;mu=state['mu'];shrink=n/(n+10)
    z=np.array([ewma(state['matrix'][np.isfinite(state['matrix'][:,j]),j],.3) for j in range(len(ranks))])
    lap=np.zeros((len(ranks),len(ranks)))
    for j in range(len(ranks)-1):
        w=1/(1+np.log(ranks[j+1]/ranks[j]));lap[j,j]+=w;lap[j+1,j+1]+=w;lap[j,j+1]-=w;lap[j+1,j]-=w
    smooth=np.linalg.solve(np.diag(np.where(known,1.,.1))+penalty*lap,(z-mu)*shrink)
    global_shrink=len(state['events'])/(len(state['events'])+10)
    return mu*np.where(known,shrink,global_shrink)+smooth

def factor_bias(state,ranks,k):
    matrix=state['matrix'];counts=state['counts'];active=counts>0;N=len(matrix)
    if N<2 or active.sum()<2:return state['warm'].copy(),{'status':'independent_fallback'}
    mu=state['mu'];scale=np.array([np.std(matrix[np.isfinite(matrix[:,j]),j]) if active[j] else 0. for j in range(len(ranks))])
    scale=interpolate(np.maximum(scale,.02),active,ranks)
    z=np.where(np.isfinite(matrix),(matrix-mu)/scale,0.)
    _,singular,vt=np.linalg.svd(z[:,active],full_matrices=False)
    k=min(k,len(singular),N-1);loading=np.zeros((len(ranks),k));loading[active]=vt[:k].T
    loading=interpolate(loading,active,ranks);scores=z[:,active]@loading[active]
    level=ewma(scores,.5);factor=scale*(loading@level);idio=np.zeros(len(ranks))
    for j in range(len(ranks)):
        valid=np.isfinite(matrix[:,j]);n=counts[j]
        errors=matrix[valid,j]-mu[j]-scale[j]*(scores[valid]@loading[j])
        idio[j]=float(ewma(errors,.6))*n/(n+10)
    shrink=np.where(active,counts/(counts+10),N/(N+10))
    return (mu+factor+idio)*shrink,{'status':'fitted','components':k,'means':mu.tolist(),'scales':scale.tolist(),
        'loadings':loading.tolist(),'singular_values':singular[:k].tolist(),'factor_level':level.tolist(),'rank_counts':counts.tolist()}

def final(frame,bias):return bounded(frame,frame.base_prediction.to_numpy()*np.exp(-np.clip(bias,-.3,.3)))

def fixed_share(past,current,pool):
    out=[];states={}
    for rank,phase in current[['rank','phase']].drop_duplicates().itertuples(index=False,name=None):
        p=past[(past['rank']==rank)&(past.phase==phase)];w=PRIOR.copy()
        sequence=sorted([(g.end_ms.iloc[0],g.start_ms.iloc[0],e,g) for e,g in p.groupby('event')],key=lambda v:v[:3])
        for _,_,_,g in sequence:
            actual=np.exp(np.log(g.base_prediction.to_numpy())-g.log_error.to_numpy())
            errors=np.clip(np.abs(g[['expert__'+e for e in EXPERTS]].to_numpy()/actual[:,None]-1).mean(axis=0),0,1)
            w*=np.exp(-5*errors);w/=w.sum();w=.95*w+.05*PRIOR
        states[(rank,phase)]=w
    for i,row in enumerate(current.itertuples()):out.append(pool[i]@states[(row.rank,row.phase)])
    return bounded(current,out),{'weights':[{'rank':int(r),'phase':int(p),'values':w.tolist()} for (r,p),w in states.items()]}

def fit_neural(past,current,pool,directory):
    directory.mkdir();n=past.event.nunique();shrink=n/(n+20);w=weights(past) if len(past) else np.array([])
    status='fitted' if n>=12 else 'prior_fallback'
    write(directory/'config.json',{'recipe':RECIPES['neural_gate'],'status':status,'query_start_ms':int(current.start_ms.iloc[0]),
        'event':int(current.event.iloc[0]),'case_ids':list(current.case_id),'pool':pool.tolist(),'shrinkage':shrink})
    pd.DataFrame({'case_id':past.case_id,'event':past.event,'weight':w}).to_csv(directory/'training_weights.csv',index=False)
    details={'status':status,'history_events':int(n),'history_cases':len(past),'shrinkage':shrink}
    if status=='prior_fallback':pred={f'neural_gate_seed{s}':bounded(current,pool@PRIOR) for s in SEEDS}
    else:
        import torch
        torch.set_num_threads(1);prep=Preprocessor.fit(past,w,NUMERIC)
        joblib.dump(prep,directory/'preprocessor.joblib',compress=3);write(directory/'preprocessor.json',prep.state())
        x=torch.tensor(prep.transform(past),dtype=torch.float32);qx=torch.tensor(prep.transform(current),dtype=torch.float32)
        experts=torch.tensor(past[['expert__'+e for e in EXPERTS]].to_numpy(),dtype=torch.float32)
        actual=torch.tensor(np.exp(np.log(past.base_prediction.to_numpy())-past.log_error.to_numpy()),dtype=torch.float32)
        wt=torch.tensor(w,dtype=torch.float32);prior=torch.tensor(PRIOR,dtype=torch.float32);pred={};losses_by_seed={}
        with torch.random.fork_rng(devices=[]):
            for seed in SEEDS:
                torch.manual_seed(seed);model=network(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
                losses=[];model.train()
                for _ in range(80):
                    optimizer.zero_grad(set_to_none=True)
                    gate=shrink*torch.softmax(model(x)+torch.log(prior),dim=1)+(1-shrink)*prior
                    loss=(torch.abs((experts*gate).sum(dim=1)/actual-1)*wt).sum()/wt.sum()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite neural mixture loss')
                    loss.backward();optimizer.step();losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():gate=shrink*torch.softmax(model(qx)+torch.log(prior),dim=1)+(1-shrink)*prior
                pred[f'neural_gate_seed{seed}']=bounded(current,np.sum(pool*gate.numpy().astype(float),axis=1))
                torch.save({'seed':seed,'input_features':x.shape[1],'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'losses':losses},directory/f'seed{seed}.pt')
                losses_by_seed[str(seed)]={'first':losses[0],'last':losses[-1],'epochs':80}
        details['training']=losses_by_seed
    pred['neural_gate']=np.mean([pred[f'neural_gate_seed{s}'] for s in SEEDS],axis=0)
    write(directory/'diagnostics.json',details)
    return pred,details

def replay_neural(directory,features):
    directory=Path(directory);cfg=json.loads((directory/'config.json').read_text())
    if {'actual','target','final','log_error'}&set(features) or list(features.case_id)!=cfg['case_ids'] or set(features.event)!={cfg['event']} or not (features.start_ms==cfg['query_start_ms']).all():raise ValueError('Wrong replay target/clock')
    pool=np.asarray(cfg['pool']);shrink=cfg['shrinkage']
    if cfg['status']=='prior_fallback':out={f'neural_gate_seed{s}':bounded(features,pool@PRIOR) for s in SEEDS}
    else:
        import torch
        torch.set_num_threads(1);prep=joblib.load(directory/'preprocessor.joblib');x=torch.tensor(prep.transform(features),dtype=torch.float32);prior=torch.tensor(PRIOR,dtype=torch.float32);out={}
        with torch.random.fork_rng(devices=[]):
            for seed in SEEDS:
                saved=torch.load(directory/f'seed{seed}.pt',map_location='cpu',weights_only=False)
                model=network(saved['input_features']);model.load_state_dict(saved['state_dict']);model.eval()
                with torch.no_grad():gate=shrink*torch.softmax(model(x)+torch.log(prior),dim=1)+(1-shrink)*prior
                out[f'neural_gate_seed{seed}']=bounded(features,np.sum(pool*gate.numpy().astype(float),axis=1))
    out['neural_gate']=np.mean([out[f'neural_gate_seed{s}'] for s in SEEDS],axis=0)
    return out

def predict(history,target,*,query_start_ms,model_directory,ranks=RANKS):
    ranks=sorted(set(map(int,ranks)));past,current,clock=stats._frames(history,target,query_start_ms,context=True)
    if not set(past['rank']).issubset(ranks) or not set(current['rank']).issubset(ranks) or current.empty:raise ValueError('Unknown rank or empty target')
    if any(str(c).startswith('expert__') for c in current):raise ValueError('Current target cannot carry expert feedback')
    for frame in [past,current]:
        if not {'current','base_prediction'}<=set(frame):raise ValueError('Missing score inputs')
        if not np.isfinite(frame[['current','base_prediction']]).all().all() or (frame.current<=0).any() or (frame.base_prediction<frame.current-1e-6).any() or (frame.base_prediction>frame.current*np.exp(4)+1e-6).any():raise ValueError('Invalid base score bounds')
        if 'observed_at' in frame and ((frame.observed_at>frame.cutoff).any() or (frame.observed_at<frame.start_ms).any() or (frame.cutoff>frame.end_ms).any()):raise ValueError('Invalid own prefix clock')
        frame['log_rank']=np.log(frame['rank']);frame['context_family']=frame['rank'].isin([1000,2000]).astype(float)
        frame['log_remaining_h']=np.log1p(np.maximum(frame.remaining_h,0));frame['log_duration_h']=np.log1p(np.maximum(frame.duration_h,0))
        frame['log_growth']=np.log(frame.base_prediction/frame.current)
    for e in EXPERTS:
        if len(past):
            if 'expert__'+e not in past:raise ValueError('Missing sealed historical expert')
            values=pd.to_numeric(past['expert__'+e],errors='raise').to_numpy(float)
            if not np.isfinite(values).all() or (values<past.current-1e-6).any() or (abs(np.log(values/past.base_prediction))>.3+1e-10).any():raise ValueError('Invalid sealed expert scores')
    directory=Path(model_directory)
    if directory.exists():raise FileExistsError('Preserve models; use new directory')
    directory.mkdir(parents=True)
    means=stats._event_means(past);states={p:phase_state(means,p,ranks) for p in current.phase.unique()};pos={r:i for i,r in enumerate(ranks)}
    for name in ['own_slow','own_fast','own_n','group_anchor','global_anchor']:current[name]=0.
    bias=np.zeros(len(current))
    for i,row in enumerate(current.itertuples()):
        state=states[row.phase];j=pos[row.rank];n=state['counts'][j];bias[i]=state['warm'][j]
        for name,value in [('own_slow',bias[i]),('own_fast',state['fast'][j]*n/(n+10)),('own_n',n),('group_anchor',state['group'][group(row.rank)]),('global_anchor',state['global'])]:current.loc[current.index[i],name]=value
    scores={'uncorrected':current.base_prediction.to_numpy(),'ewma_reference':final(current,bias)};details={}
    cold=past[past['split']!='train'] if 'split' in past else past.iloc[:0]
    coldmeans=stats._event_means(cold);coldstates={p:phase_state(coldmeans,p,ranks) for p in current.phase.unique()}
    scores['ewma_cold']=final(current,np.array([coldstates[r.phase]['warm'][pos[r.rank]] for r in current.itertuples()]))
    scores['legacy_pair']=scores['ewma_reference'].copy();mask=current['rank'].isin([1000,2000])
    if mask.any():
        legacy,_=paired._coupled(past,current.loc[mask],bias[mask.to_numpy()],kalman=False)
        scores['legacy_pair'][mask.to_numpy()]=final(current.loc[mask],legacy)
    scores['hierarchy']=final(current,np.array([states[r.phase]['hierarchy'][pos[r.rank]] for r in current.itertuples()]))
    for name,penalty in [('graph025',.25),('graph1',1.)]:
        values={p:graph_bias(state,ranks,penalty) for p,state in states.items()}
        scores[name]=final(current,np.array([values[r.phase][pos[r.rank]] for r in current.itertuples()]))
    for k in [2,4]:
        values={};infos={}
        for p,state in states.items():values[p],infos[str(int(p))]=factor_bias(state,ranks,k)
        scores[f'factor{k}']=final(current,np.array([values[r.phase][pos[r.rank]] for r in current.itertuples()]));details[f'factor{k}']=infos
    pool=np.column_stack([scores['ewma_reference'],scores['legacy_pair'],scores['hierarchy'],scores['factor2']])
    for e,value in zip(EXPERTS,pool.T):current['bias_'+e]=np.log(current.base_prediction/value)
    scores['prior_mix']=bounded(current,pool@PRIOR)
    scores['share_gate'],details['share_gate']=fixed_share(past,current,pool)
    nn,details['neural_gate']=fit_neural(past,current,pool,directory/'neural_gate');scores.update(nn)
    for e,value in zip(EXPERTS,pool.T):current['expert__'+e]=value
    scores={name:bounded(current,value) for name,value in scores.items()}
    details['phase_states']={str(int(p)):{'counts':state['counts'].tolist(),'global_anchor':state['global'],'group_anchors':state['group'].tolist(),'history_events':state['events']} for p,state in states.items()}
    write(directory/'config.json',{'recipes':RECIPES,'ranks':ranks,'query_start_ms':clock,'history_events':int(past.event.nunique()),'history_cases':len(past),'current_outcome_used':False})
    write(directory/'state.json',details)
    return scores,current,details
