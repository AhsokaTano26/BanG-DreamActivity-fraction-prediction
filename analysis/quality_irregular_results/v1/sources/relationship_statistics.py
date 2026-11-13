"""Training-only association analysis with event-cluster robust uncertainty."""
from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import SplineTransformer
from analysis.holdout_dataset import write_json


def clustered_ols(frame, outcome, groups, weights):
    x=frame.astype(float).to_numpy();y=np.asarray(outcome);w=np.asarray(weights)
    bread=np.linalg.pinv(x.T@(w[:,None]*x));beta=bread@(x.T@(w*y))
    residual=y-x@beta;unique=np.unique(groups);k=np.linalg.matrix_rank(x);n=len(x)
    scores=np.stack([x[groups==g].T@(w[groups==g]*residual[groups==g]) for g in unique])
    covariance=bread@(scores.T@scores)@bread*len(unique)/(len(unique)-1)*(n-1)/(n-k)
    se=np.sqrt(np.maximum(0.,np.diag(covariance)));critical=stats.t.ppf(.975,len(unique)-1)
    p=2*stats.t.sf(np.abs(beta)/np.maximum(se,1e-15),len(unique)-1)
    coefficients=pd.DataFrame({'term':frame.columns,'beta':beta,'cluster_se':se,
        'lower95':beta-critical*se,'upper95':beta+critical*se,'p':p,
        'q_bh':stats.false_discovery_control(p)})
    return coefficients,covariance,{'rows':n,'events':len(unique),'matrix_rank':int(k),'columns':len(beta),
        'weighted_r2':float(1-np.sum(w*residual**2)/np.sum(w*(y-np.average(y,weights=w))**2))}


def design(frame,numerical,categorical):
    numeric=frame[numerical].copy()
    numeric=numeric.fillna(numeric.median()).fillna(0.)
    numeric=(numeric-numeric.mean())/numeric.std().replace(0,1)
    category=pd.get_dummies(frame[categorical].astype(str),drop_first=True,dtype=float)
    x=pd.concat([numeric,category],axis=1)
    x=x.loc[:,x.std()>1e-12];x.insert(0,'intercept',1.)
    return x


def association_tests(coef,covariance,prefixes,events):
    output=[]
    for name,patterns in prefixes.items():
        positions=[i for i,t in enumerate(coef.term) if any(t.startswith(p) for p in patterns)]
        if not positions:continue
        beta=coef.beta.to_numpy()[positions];cov=covariance[np.ix_(positions,positions)]
        df=np.linalg.matrix_rank(cov)
        if not df:continue
        f=float(beta@np.linalg.pinv(cov)@beta/df)
        output.append({'factor':name,'df':int(df),'cluster_wald_f':f,'p':float(stats.f.sf(f,df,events-1))})
    adjusted=stats.false_discovery_control([r['p'] for r in output])
    for r,q in zip(output,adjusted):r['q_bh']=float(q)
    return output


def peer_interactions(frame):
    frame=frame.copy()
    for main,interaction in [('gap_log_1000_2000','rank_gap'),('gap_change_24h','rank_gapchange')]:
        median=frame[main].median()
        frame[main]=frame[main].fillna(median if np.isfinite(median) else 0.)
        frame[interaction]=frame[main]*frame['rank1000']
    return frame


def analyze(data,features,samples,metadata,directory):
    curves=data.train_curves();labels=data.labels('train');pairs={};rows=[]
    for c in curves:
        pairs.setdefault(c.event,{})[c.rank]=c
        rows.append({'event':c.event,'rank':str(c.rank),'type':c.kind,
            'band':metadata.band_only(c)['band_group'],'year':datetime.fromtimestamp(c.start/1000).year,
            'years_since_2019':(c.start-1546272000000)/86400000/365.25,
            'duration_h':c.duration/3600000,'log_duration':np.log(c.duration/3600000),
            'log_final':np.log(c.final),'points':len(c.times)})
    event=pd.DataFrame(rows);counts=event.groupby('event').event.transform('size').to_numpy();weights=1/counts
    x=design(event,['years_since_2019','log_duration'],['rank','type','band'])
    coef,cov,info=clustered_ols(x,event.log_final,event.event.to_numpy(),weights)
    coef.to_csv(directory/'final_score_adjusted_coefficients.csv',index=False)
    finaltests=association_tests(coef,cov,{'era':['years_since'],'duration':['log_duration'],
        'rank':['rank_'],'type':['type_'],'band':['band_']},info['events'])
    complete=[p for p in pairs.values() if len(p)==2]
    logs=np.array([[np.log(p[1000].final),np.log(p[2000].final)] for p in complete])
    rng=np.random.default_rng(42);cors=[]
    for _ in range(2000):
        a=logs[rng.integers(0,len(logs),len(logs))];cors.append(np.corrcoef(a.T)[0,1])
    # Pairwise correlation can reflect common era/type effects; residualize both ranks.
    pairframe=event[event.event.isin([p[1000].event for p in complete])].copy()
    residuals=[]
    for rank in ['1000','2000']:
        f=pairframe[pairframe['rank']==rank].sort_values('event')
        a=design(f,['years_since_2019','log_duration'],['type','band']).to_numpy()
        residuals.append(f.log_final.to_numpy()-a@np.linalg.lstsq(a,f.log_final,rcond=None)[0])
    prefix=features.copy();prefix['years_since_2019']=[(s['curve'].start-1546272000000)/86400000/365.25 for s in samples]
    prefix['log_duration']=np.log(prefix.duration_h)
    prefix['rank1000']=(prefix['rank']=='1000').astype(float)
    spline=SplineTransformer(n_knots=5,degree=3,include_bias=False).fit_transform(prefix[['progress']])
    for i in range(spline.shape[1]):
        prefix[f'phase_{i}']=spline[:,i];prefix[f'rankphase_{i}']=spline[:,i]*prefix.rank1000
    prefix=peer_interactions(prefix)
    numeric=['log_current','log_duration','years_since_2019','rate_6h_relative','rate_24h_relative',
             'quality_median_gap_h','quality_max_gap_h','peer_available','gap_log_1000_2000',
             'gap_change_24h','peer_rate_24h','rank_gap','rank_gapchange']+[c for c in prefix if c.startswith(('phase_','rankphase_'))]
    x=design(prefix,numeric,['rank','event_type','band_group'])
    target=np.array([np.log(labels[s['case_id']]/s['score'])+np.log(s['progress']) for s in samples])
    groups=np.array([s['curve'].event for s in samples]);count=Counter(groups);w=np.array([1/count[g] for g in groups])
    coef,cov,growthinfo=clustered_ols(x,target,groups,w)
    coef.to_csv(directory/'growth_adjusted_coefficients.csv',index=False)
    growthtests=association_tests(coef,cov,{'phase':['phase_','rankphase_'],'current_scale':['log_current'],
        'duration':['log_duration'],'era':['years_since'],'own_velocity':['rate_'],
        'quality':['quality_'],'peer':['peer_','gap_','rank_gap'],'type':['event_type_'],'band':['band_group_']},len(count))
    event.to_csv(directory/'training_event_statistics.csv',index=False)
    summary=event.groupby(['rank','year']).agg(curves=('event','size'),median_log_final=('log_final','median'),
        median_points=('points','median'),median_duration_h=('duration_h','median')).reset_index()
    summary.to_csv(directory/'era_summary.csv',index=False)
    result={'scope':'training only; observational conditional associations, not causal effects',
        'paired_events':len(complete),'log_final_correlation':float(np.corrcoef(logs.T)[0,1]),
        'correlation_event_bootstrap_95ci':np.quantile(cors,[.025,.975]).tolist(),
        'correlation_after_era_type_duration_band':float(np.corrcoef(residuals)[0,1]),
        'final_regression':info,'final_factor_tests':finaltests,'growth_regression':growthinfo,'growth_factor_tests':growthtests,
        'inference':'WLS equal event weight; event-cluster sandwich CR1; t/F with G-1 df; BH within each table; exploratory',
        'limits':['retrospective metadata','era and sample density covary','band and type may covary','many tested hypotheses','serial event dependence not modeled']}
    write_json(directory/'statistics.json',result)
    return result
