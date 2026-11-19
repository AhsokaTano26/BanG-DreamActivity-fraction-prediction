"""Independent scoring only. Never imported by predictors; writes new E bundles."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta, chi2, nct, t, norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ROUND = ROOT / 'analysis/paper_upgrade_round2/20261118'
OLD = ROOT / 'analysis/paper_upgrade_records/20261118'
BLOCKS = (1,3,5,7)
INPUTS = {}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path, **kwargs):
    path=Path(path).resolve()
    INPUTS[str(Path(path).relative_to(ROOT))] = sha(path)
    return pd.read_csv(path, **kwargs)

def read_json(path):
    path=Path(path).resolve()
    INPUTS[str(Path(path).relative_to(ROOT))] = sha(path)
    return json.loads(Path(path).read_text())

def write_json(path, value):
    def default(x):
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=default)+'\n')

def verify_seal(root, seal_name, field='files'):
    seal = read_json(root/seal_name)
    for name, expected in seal[field].items():
        path=root/name
        if sha(path) != expected: raise ValueError(f'Seal mismatch: {path}')
    return seal

def loss_metrics(data):
    df=data.copy()
    if df.case_id.duplicated().any(): raise ValueError('Duplicate physical case within method')
    if (df.actual<=0).any() or not np.isfinite(df[['predicted','actual']]).all().all():
        raise ValueError('Nonpositive target or nonfinite forecast')
    df['ape']=100*np.abs(df.predicted/df.actual-1)
    df['ae']=np.abs(df.predicted-df.actual)
    df['se']=(df.predicted-df.actual)**2
    er=df.groupby(['event','rank']).ape.mean().groupby('event').mean()
    ev=df.groupby('event').agg(mape=('ape','mean'),mae=('ae','mean'),mse=('se','mean'),
                  p90=('ape',lambda x:np.quantile(x,.9)),cases=('case_id','size'),ranks=('rank','nunique'))
    ev['rank_mape']=er
    if 'start_ms' in df: ev['start_ms']=df.groupby('event').start_ms.first()
    else: ev['start_ms']=ev.index
    ev=ev.reset_index().sort_values(['start_ms','event'])
    metrics={'events':len(ev),'cases':len(df),'ranks':int(df['rank'].nunique()),
        'event_macro_mape_pct':ev.mape.mean(),'event_rank_macro_mape_pct':ev.rank_mape.mean(),
        'case_mape_pct':df.ape.mean(),'case_mae':df.ae.mean(),'case_rmse':np.sqrt(df.se.mean()),
        'case_p90_ape_pct':np.quantile(df.ape,.9),'event_equal_mae':ev.mae.mean(),
        'event_equal_rmse':np.sqrt(ev.mse.mean()),'event_mean_p90_ape_pct':ev.p90.mean()}
    return metrics,ev

def draw_indices(n, draws=5000, block=1, seed=42):
    if block>n: return None
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(draws,math.ceil(n/block)))
    return ((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]

def mean_bootstrap(values,draws=5000,block=1):
    x=np.asarray(values,float)
    ix=draw_indices(len(x),draws,block)
    if ix is None:return {'status':'insufficient_events','low95':None,'high95':None}
    sample=x[ix].mean(axis=1)
    return {'status':'fixed_loss_exploratory','low95':float(np.quantile(sample,.025)),
            'high95':float(np.quantile(sample,.975))}

def paired_rows(left,right,meta,draws=5000):
    a=left.set_index('event');b=right.set_index('event')
    if set(a.index)!=set(b.index): raise ValueError('Unmatched paired activities')
    a=a.sort_values(['start_ms']);b=b.loc[a.index]
    rows=[];differences=[]
    for metric in ['mape','rank_mape','mae','mse','p90']:
        x=a[metric].to_numpy();y=b[metric].to_numpy()
        transform=np.sqrt if metric=='mse' else lambda z:z
        delta=float(transform(x.mean())-transform(y.mean()))
        d=x-y
        loo=np.array([transform(np.delete(x,i).mean())-transform(np.delete(y,i).mean()) for i in range(len(x))]) if len(x)>1 else np.array([np.nan])
        for block in BLOCKS:
            ix=draw_indices(len(a),draws,block)
            ci={'status':'insufficient_events','low95':None,'high95':None}
            if ix is not None:
                samples=transform(x[ix].mean(axis=1))-transform(y[ix].mean(axis=1))
                ci={'status':'fixed_loss_exploratory','low95':np.quantile(samples,.025),'high95':np.quantile(samples,.975)}
            rows.append({**meta,'metric':{'mape':'event_macro_MAPE_pct','rank_mape':'event_rank_macro_MAPE_pct',
                'mae':'event_equal_MAE','mse':'event_equal_RMSE','p90':'event_mean_p90_ape_pct'}[metric],
                'events':len(a),'difference':delta,'block':block,'draws':draws,
                'improved_events':int((d<0).sum()),'ties':int((d==0).sum()),
                'leave_one_low':np.nanmin(loo),'leave_one_high':np.nanmax(loo),**ci})
        for event,dx in zip(a.index,d):
            differences.append({**meta,'metric':metric,'event':event,'start_ms':int(a.loc[event,'start_ms']),
                                'left_loss':float(a.loc[event,metric]),'right_loss':float(b.loc[event,metric]),'difference':dx})
    return rows,differences

def interval_score(lower,upper,actual,alpha):
    if not 0<alpha<1:raise ValueError('Invalid alpha')
    lo,hi,y=np.broadcast_arrays(np.asarray(lower,float),np.asarray(upper,float),np.asarray(actual,float))
    return hi-lo+(2/alpha)*np.maximum(lo-y,0)+(2/alpha)*np.maximum(y-hi,0)

def full_joint_groups(data,dimension=None):
    keys=['event','rank','phase']
    groups=data.groupby(keys,dropna=False).agg(covered=('covered','all'),cases=('covered','size')).reset_index()
    if dimension:
        conditions=data.groupby(keys,dropna=False)[dimension].agg(lambda x:str(x.iloc[0]) if x.nunique(dropna=False)==1 else 'mixed')
        groups=groups.merge(conditions.rename('condition').reset_index(),on=keys,validate='one_to_one')
    return groups

def interval_metrics(data,draws=5000,nominal=.8):
    df=data.copy()
    if df.case_id.duplicated().any():raise ValueError('Duplicate intervals at nominal level')
    df['covered']=(df.lower<=df.actual)&(df.actual<=df.upper)
    finite=np.isfinite(df.lower)&np.isfinite(df.upper)
    width=df.upper-df.lower
    rel=100*width/df.actual
    score=interval_score(df.lower,df.upper,df.actual,1-nominal)
    df['width']=width;df['relative_width']=rel;df['score']=score
    groups=full_joint_groups(df)
    evpoint=df.groupby('event').covered.mean()
    evjoint=groups.groupby('event').covered.mean()
    whole=df.groupby('event').covered.all()
    n=len(whole);success=int(whole.sum())
    def extended_p90(values):
        # Avoid inf-inf interpolation turning an unbounded quantile into NaN.
        x=np.sort(np.asarray(values,float));position=.9*(len(x)-1)
        low=int(np.floor(position));high=int(np.ceil(position))
        if np.isinf(x[high]):return float(x[high])
        return float(x[low]+(position-low)*(x[high]-x[low]))
    result={'n_events':n,'n_cases':len(df),'n_groups':len(groups),'case_coverage':df.covered.mean(),
       'event_equal_point_coverage':evpoint.mean(),'event_equal_group_joint_coverage':evjoint.mean(),
       'whole_event_joint_coverage':whole.mean(),'whole_event_successes':success,
       'whole_event_cp_low95':0. if success==0 else beta.ppf(.025,success,n-success+1),
       'whole_event_cp_high95':1. if success==n else beta.ppf(.975,success+1,n-success),
       'finite_interval_rate':finite.mean(),'unbounded_cases':int((~finite).sum()),
       'unavailable_cases':int((~df.available.astype(bool)).sum()),
       'own_support_zero_cases':int((df.own_support==0).sum()),
       'mean_width':width.mean(),'p90_width':extended_p90(width),
       'mean_relative_width_pct':rel.mean(),'p90_relative_width_pct':extended_p90(rel),
       'winkler_score':np.mean(score),'relative_score_pct':np.mean(100*score/df.actual),
       'event_equal_width':df.groupby('event').width.mean().mean(),
       'event_equal_relative_width_pct':df.groupby('event').relative_width.mean().mean(),
       'event_mean_p90_relative_width_pct':df.groupby('event').relative_width.apply(extended_p90).mean(),
       'event_equal_winkler_score':df.groupby('event').score.mean().mean(),
       'finite_case_coverage':df.loc[finite,'covered'].mean() if finite.any() else None,
       'finite_mean_width':width[finite].mean() if finite.any() else None}
    ci_columns={'event_equal_point_coverage':evpoint,'event_equal_group_joint_coverage':evjoint,
                'whole_event_joint_coverage':whole.astype(float),
                'event_equal_width':df.groupby('event').width.mean(),
                'event_equal_relative_width_pct':df.groupby('event').relative_width.mean(),
                'event_mean_p90_relative_width_pct':df.groupby('event').relative_width.apply(extended_p90),
                'event_equal_winkler_score':df.groupby('event').score.mean()}
    for key,values in ci_columns.items():
        if np.isfinite(values).all():
            ci=mean_bootstrap(values,draws)
            result[key+'_low95']=ci['low95'];result[key+'_high95']=ci['high95']
        else:result[key+'_low95']=result[key+'_high95']=None
    return result,groups

def hac_interval(values):
    """Predeclared future primary: Bartlett/Newey-West, n-denominator autocovariances."""
    x=np.asarray(values,float);n=len(x)
    if n<2 or not np.isfinite(x).all():
        return {'status':'insufficient_events','n_events':n,'low95':None,'high95':None,'p_value':None}
    bandwidth=min(n-1,max(1,math.ceil(n**(1/3))))
    mean=float(x.mean());centered=x-mean;gamma0=float(np.dot(centered,centered)/n)
    if gamma0<=np.finfo(float).eps*max(1,float(np.mean(x*x))):
        return {'status':'degenerate_variance_no_decision','n_events':n,'mean':mean,'bandwidth':bandwidth,
                'long_run_variance':0.,'se_mean':0.,'low95':None,'high95':None,'p_value':None}
    lrv=gamma0+2*sum((1-h/(bandwidth+1))*float(np.dot(centered[h:],centered[:-h])/n) for h in range(1,bandwidth+1))
    lrv=max(0.,lrv);se=float(np.sqrt(lrv/n))
    if se==0:
        return {'status':'degenerate_variance_no_decision','n_events':n,'mean':mean,'bandwidth':bandwidth,
                'long_run_variance':lrv,'se_mean':se,'low95':None,'high95':None,'p_value':None}
    return {'status':'asymptotic_hac_reference','n_events':n,'mean':mean,'bandwidth':bandwidth,
            'long_run_variance':lrv,'se_mean':se,'low95':mean-norm.ppf(.975)*se,
            'high95':mean+norm.ppf(.975)*se,'p_value':float(2*norm.sf(abs(mean/se)))}

def paired_t_power(n,sigma,effect,alpha=.05):
    critical=t.ppf(1-alpha/2,n-1)
    nc=abs(effect)*np.sqrt(n)/sigma
    # Symmetry: F(-t;nc)=SF(t;-nc). The direct lower-tail Boost CDF can return
    # NaN for large positive noncentrality even when the survival tails are finite.
    result=float(nct.sf(critical,n-1,-nc)+nct.sf(critical,n-1,nc))
    if not np.isfinite(result):raise ArithmeticError('Nonfinite noncentral-t power')
    return float(np.clip(result,0,1))

def power_n(sigma,effect,power,rho=0.,alpha=.05):
    if sigma<=0 or effect<=0 or not 0<=rho<1: raise ValueError('Invalid power parameters')
    adjusted=sigma*np.sqrt((1+rho)/(1-rho))
    lo=2;hi=4
    while paired_t_power(hi,adjusted,effect,alpha)<power:
        hi*=2
        if hi>10000000:raise ValueError('Unbounded power search')
    while lo<hi:
        mid=(lo+hi)//2
        if paired_t_power(mid,adjusted,effect,alpha)>=power:hi=mid
        else:lo=mid+1
    return lo

def seal_bundle(output,stage,started):
    write_json(output/'analysis_manifest.json',{'stage':stage,'elapsed_seconds':time.monotonic()-started,
       'created_utc':datetime.now(timezone.utc).isoformat(),'inputs':INPUTS,
       'source_sha256':sha(__file__),'preregistration_sha256':sha(HERE/'preregistration.json'),
       'threads':{x:os.environ.get(x) for x in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
       'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},
       'limits':['Exploratory historical cohort; reused validation; retrospective metadata',
                 'Fixed-loss activity resampling; no online retraining or candidate-search correction']})
    (output/'scorer_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(output/'output_seal.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})

def score_collection(df,method_col,period_col,panels,draws):
    summaries=[];events=[];paired=[];deltas=[];strata=[]
    for panel,sub in panels.items():
        for period,part in sub.groupby(period_col,sort=False):
            store={}
            expected=set(part.iloc[:0].case_id)
            for _,one in part.groupby(method_col,sort=False):
                if not expected:expected=set(one.case_id)
                elif set(one.case_id)!=expected:raise ValueError('Methods have different physical cases')
            for method,one in part.groupby(method_col,sort=False):
                metrics,ev=loss_metrics(one)
                meta={'panel':panel,'period':str(period),'method':method}
                summaries.append({**meta,**metrics})
                events.extend(ev.assign(**meta).to_dict('records'));store[method]=ev
                for dimension in ['rank','phase','event_type','own_support_bin']:
                    if dimension not in one:continue
                    for value,group in one.groupby(dimension,dropna=False):
                        m,_=loss_metrics(group)
                        strata.append({**meta,'dimension':dimension,'value':str(value),**m})
            reference='independent_fixed_recipe' if method_col=='candidate' else 'independent'
            for method,ev in store.items():
                if method==reference:continue
                rows,ds=paired_rows(ev,store[reference],{'panel':panel,'period':str(period),'left':method,'right':reference},draws)
                paired.extend(rows);deltas.extend(ds)
    return summaries,events,paired,deltas,strata

def stage_predictions(output,draws):
    inputs=ROUND/'inputs'
    answers=read_csv(inputs/'train_residuals.csv.gz',usecols=['case_id','actual'])
    bpath=ROUND/'workspaces/B/run_v2_001'
    bm=read_json(bpath/'run_manifest.json')
    for name,expected in bm['files'].items():
        if sha(bpath/name)!=expected:raise ValueError(f'B seal mismatch {name}')
    issued=pd.concat([read_csv(p) for p in sorted((bpath/'issued').glob('*.csv.gz'))],ignore_index=True)
    metadata=read_csv(inputs/'train_residuals.csv.gz',dtype={'event_type':str,'band_group':str})
    needed=['case_id']+[x for x in ['start_ms','end_ms','event_type','progress','actual'] if x not in issued]
    b=issued.merge(metadata[needed],on='case_id',validate='many_to_one')
    b=b.rename(columns={'prediction':'predicted'})
    baselines=[]
    for name,column in [('external_original_independent','expert__independent'),('uncorrected_base','base_prediction')]:
        one=b[b.candidate=='independent_fixed_recipe'].copy()
        forecast=metadata.set_index('case_id')[column]
        one['predicted']=one.case_id.map(forecast);one['candidate']=name;baselines.append(one)
    b=pd.concat([b,*baselines],ignore_index=True)
    allblock=b.copy();allblock['block']='all';b=pd.concat([b,allblock],ignore_index=True)
    panels={'all19':b,'main_pair':b[b['rank'].isin([1000,2000])]}
    tables=score_collection(b,'candidate','block',panels,draws)
    for name,rows in zip(['summary','events','paired','differences','strata'],tables):pd.DataFrame(rows).to_csv(output/f'B_{name}.csv',index=False)
    seal=read_json(bpath/'selection_seal.json')
    order=[x['candidate'] for x in seal['scores']]
    selected=pd.DataFrame(tables[0]).query("panel=='main_pair' and period=='selection' and method in @order").copy()
    selected['order']=selected.method.map(dict(zip(order,range(len(order)))))
    best=selected.sort_values(['event_macro_mape_pct','order']).iloc[0].method
    if best!=seal['selected_candidate']:raise ValueError('Independent selection reproduction differs')
    comparisons=[]
    old=read_csv(bpath/'summary.csv')
    for _,row in pd.DataFrame(tables[0]).iterrows():
        prev=old[(old.panel==row.panel)&(old.block==row.period)&(old.candidate==row.method)].iloc[0]
        comparisons.append({'source':'B','panel':row.panel,'period':row.period,'method':row.method,
            'mape_difference_from_original':row.event_macro_mape_pct-prev.event_macro_mape})
    for exp in ['c1','c2']:
        run=ROUND/f'workspaces/C/runs/{exp}_002'
        verify_seal(run,'blind_prediction_seal.json')
        pred=read_csv(run/'predictions.csv.gz',dtype={'seed':str,'event_type':str,'band_group':str})
        if exp=='c1':
            pred=pred.merge(answers,on='case_id',validate='many_to_one')
            panels={'global19':pred,'T1000_T2000':pred[pred['rank'].isin([1000,2000])]}
            tables=score_collection(pred,'method','period',panels,draws)
            for name,rows in zip(['summary','events','paired','differences','strata'],tables):pd.DataFrame(rows).to_csv(output/f'C1_{name}.csv',index=False)
        else:
            target=read_csv(ROOT/'data/temporal_holdout_v2_dense_t1000_t2000_20261115/train_labels.csv')
            pred=pred.merge(target,on='case_id',validate='many_to_one')
            # Verify the saved ensembles against arithmetic predictions, without selecting members.
            mean=pred[pred.seed=='mean'].set_index(['endpoint','group','structure','case_id']).predicted.sort_index()
            own=pred[pred.seed!='mean'].groupby(['endpoint','group','structure','case_id']).predicted.mean().sort_index()
            if not np.allclose(mean,own,rtol=1e-12,atol=1e-6):raise ValueError('C2 ensemble arithmetic differs')
            stage_c2(pred,output,draws)
            tables=[read_csv(output/'C2_summary.csv').to_dict('records')]
        orig=read_csv(ROUND/f'workspaces/C/runs/{exp}_002_scores/summary.csv')
        for row in tables[0]:
            period_key='period' if exp=='c1' else 'endpoint'
            if exp=='c2' and str(row[period_key])=='all12':continue
            match=(orig.panel==row['panel'])&(orig.method==row['method'])&(orig[period_key].astype(str)==str(row[period_key]))
            prev=orig[match].iloc[0]
            comparisons.append({'source':exp.upper(),'panel':row['panel'],'period':row[period_key],'method':row['method'],
                'mape_difference_from_original':row['event_macro_mape_pct']-prev.event_macro_mape_pct})
    pd.DataFrame(comparisons).to_csv(output/'independent_reproduction.csv',index=False)
    if max(abs(x['mape_difference_from_original']) for x in comparisons)>1e-10:raise ValueError('Original scores differ')
    write_json(output/'selection_reproduction.json',{'B_original_selected':seal['selected_candidate'],'E_reproduced_selected':best,
             'new_selection_performed':False,'rule':'recompute frozen selection block and tie order only',
             'max_abs_original_mape_error':max(abs(x['mape_difference_from_original']) for x in comparisons)})

def stage_c2(df,output,draws):
    summaries=[];events=[];contrasts=[];deltas=[];strata=[]
    subsets=[(str(k),v) for k,v in df.groupby('endpoint')]+[('all12',df)]
    for endpoint,part in subsets:
        store={}
        for method,one in part.groupby('method',sort=False):
            m,ev=loss_metrics(one)
            meta={'panel':'T1000_T2000','endpoint':endpoint,'method':method,
                  'seed':str(one.seed.iloc[0]),'group':one.group.iloc[0],'structure':one.structure.iloc[0]}
            summaries.append({**meta,**m});events.extend(ev.assign(**meta).to_dict('records'));store[method]=ev
            for dimension in ['rank','phase','event_type']:
                for value,group in one.groupby(dimension,dropna=False):
                    sm,_=loss_metrics(group);strata.append({**meta,'dimension':dimension,'value':value,**sm})
        for seed in ['mean','42','123','2026']:
            suffix='_mean' if seed=='mean' else '_seed'+seed
            pairs=[]
            for group in ['N32full','N16span','N16recent']:
                for left,right,kind in [('independent60','shared60','exact_parameter_match'),
                                         ('independent60','shared30','approx_forward_mac_match'),
                                         ('shared30','shared60','shared_update_budget')]:
                    pairs.append((group+'_'+left+suffix,group+'_'+right+suffix,kind))
            for structure in ['shared60','shared30','independent60']:
                for left,right,kind in [('N16span','N32full','quantity_same_span_composition_changes'),
                                       ('N16recent','N32full','quantity_and_span'),
                                       ('N16recent','N16span','same_N_different_span_composition_changes')]:
                    pairs.append((left+'_'+structure+suffix,right+'_'+structure+suffix,kind))
            for left,right,kind in pairs:
                rows,ds=paired_rows(store[left],store[right],{'endpoint':endpoint,'seed':seed,
                                    'kind':kind,'left':left,'right':right},draws)
                contrasts.extend(rows);deltas.extend(ds)
    for name,rows in [('summary',summaries),('events',events),('paired',contrasts),('differences',deltas),('strata',strata)]:
        pd.DataFrame(rows).to_csv(output/f'C2_{name}.csv',index=False)

def support_bin(x):
    return pd.cut(x,[-.5,.5,9.5,29.5,59.5,np.inf],labels=['0','1–9','10–29','30–59','60+']).astype(object).fillna('missing')

def stage_reliability(output,draws,passed_manifest):
    run=OLD/'runs/fixed_replay_v2';dataset=ROOT/'data/rolling_holdout_v3_dense_19rank_20261117'
    blind=read_json(run/'blind_manifest.json');frozen=read_json(dataset/'manifest.json')
    for root,manifest,names in [(run,blind,['test_intervals.csv']),
                                (dataset,frozen,['test_cases.csv','test_labels.csv'])]:
        for name in names:
            if sha(root/name)!=manifest['artifacts'][name]:raise ValueError(f'Interval input seal mismatch {name}')
    intervals=read_csv(run/'test_intervals.csv')
    cases=read_csv(dataset/'test_cases.csv',dtype={'event_type':str,'band_group':str})
    labels=read_csv(dataset/'test_labels.csv')
    joined=intervals.merge(cases.drop(columns=['event','rank']),on='case_id',validate='many_to_one').merge(labels,on='case_id',validate='many_to_one')
    if len(joined)!=10440 or len(cases)!=5220:raise ValueError('Changed frozen interval panel')
    if joined['actual'].isna().any():raise ValueError('Missing interval answers')
    joined['actual_progress']=(joined.observed_at-joined.start_ms)/(joined.end_ms-joined.start_ms)
    joined['actual_progress_bin']=pd.cut(joined.actual_progress,[0,.25,.9,1.00000000001],right=False,
                                  labels=['[0,.25)','[.25,.9)','[.9,1]']).astype(object).fillna('missing_or_outside')
    joined['own_support_bin']=support_bin(joined.own_support)
    joined['covered']=(joined.lower<=joined.actual)&(joined.actual<=joined.upper)
    joined['width']=joined.upper-joined.lower
    joined['relative_width_pct']=100*joined.width/joined.actual
    joined['winkler_score']=[0.]*len(joined)
    summaries=[];conditionals=[];groups_all=[];conditional_joint=[];activity_rows=[]
    for (method,nominal),part in joined.groupby(['method','nominal_coverage']):
        meta={'method':method,'nominal_coverage':nominal}
        score=interval_score(part.lower,part.upper,part.actual,1-nominal)
        joined.loc[part.index,'winkler_score']=score
        m,g=interval_metrics(part,draws,nominal);summaries.append({**meta,**m})
        groups_all.extend(g.assign(**meta).to_dict('records'))
        for event,one in part.groupby('event'):
            em,_=interval_metrics(one,draws,nominal);activity_rows.append({**meta,'event':event,**em})
        for dimension in ['rank','phase','actual_progress_bin','event_type','own_support_bin','source']:
            for value,one in part.groupby(dimension,dropna=False):
                cm,_=interval_metrics(one,draws,nominal)
                # Partial strata never stand in for the calibration full-group joint target.
                for name in list(cm):
                    if name.startswith('event_equal_group_joint') or name.startswith('whole_event'):cm.pop(name)
                conditionals.append({**meta,'dimension':dimension,'value':str(value),**cm})
            full=full_joint_groups(part,dimension)
            for value,one in full.groupby('condition',dropna=False):
                event_rates=one.groupby('event').covered.mean()
                ci=mean_bootstrap(event_rates,draws)
                conditional_joint.append({**meta,'dimension':dimension,'value':str(value),'n_events':one.event.nunique(),
                    'n_groups':len(one),'n_cases':int(one.cases.sum()),'raw_group_joint_coverage':one.covered.mean(),
                    'event_equal_full_group_joint_coverage':event_rates.mean(),**ci})
    for name,rows in [('summary',summaries),('conditions',conditionals),('full_group_conditions',conditional_joint),
                      ('groups',groups_all),('activities',activity_rows)]:
        pd.DataFrame(rows).to_csv(output/f'interval_{name}.csv',index=False)
    joined.to_csv(output/'interval_case_audit.csv.gz',index=False)
    # Temporal sensitivity of the three coverage targets, preserving issued intervals.
    temporal=[]
    for nominal,part in joined.groupby('nominal_coverage'):
        clock=part.groupby('event').start_ms.first().sort_values().index
        group=full_joint_groups(part)
        series={'point':part.groupby('event').covered.mean(),'group_joint':group.groupby('event').covered.mean(),
                'whole_event':part.groupby('event').covered.all().astype(float)}
        for name,x in series.items():
            for block in BLOCKS:temporal.append({'nominal_coverage':nominal,'target':name,'block':block,
                                 'events':len(clock),'estimate':x.mean(),**mean_bootstrap(x.loc[clock],draws,block)})
    pd.DataFrame(temporal).to_csv(output/'interval_temporal_sensitivity.csv',index=False)
    stage_power(output,draws,passed_manifest)

def passed_bundle(passed_manifest):
    first=Path(passed_manifest).resolve().parent
    seal=read_json(passed_manifest)
    for name,expected in seal.items():
        if sha(first/name)!=expected:raise ValueError(f'Prediction scoring output seal mismatch {name}')
    return first

def stage_power(output,draws,passed_manifest):
    first=passed_bundle(passed_manifest)
    events=read_csv(first/'C1_events.csv')
    rows=[];sigma_rows=[];differences=[]
    for panel,sub in events.groupby('panel'):
        a=sub[sub.method=='legacy_pair'].set_index('event').sort_values('start_ms')
        b=sub[sub.method=='independent'].set_index('event').loc[a.index]
        if len(a)!=35:raise ValueError('Power requires TRAIN35')
        d=(a.mape-b.mape).to_numpy();n=len(d);s=float(np.std(d,ddof=1))
        loo=np.array([np.std(np.delete(d,i),ddof=1) for i in range(n)])
        lag=float(np.corrcoef(d[:-1],d[1:])[0,1]) if np.std(d[:-1])>0 and np.std(d[1:])>0 else None
        sigmas={'sample':s,'leave_one_min':float(loo.min()),'leave_one_max':float(loo.max()),
            'normal_iid_chi2_low95':float(np.sqrt((n-1)*s*s/chi2.ppf(.975,n-1))),
            'normal_iid_chi2_high95':float(np.sqrt((n-1)*s*s/chi2.ppf(.025,n-1)))}
        boot=np.std(d[draw_indices(n,draws)],axis=1,ddof=1)
        sigma_rows.append({'panel':panel,'n_events':n,'mean_training_difference_pp':d.mean(),'lag1':lag,
                          'sigma_bootstrap_low95':np.quantile(boot,.025),'sigma_bootstrap_high95':np.quantile(boot,.975),**sigmas})
        for event,delta in zip(a.index,d):differences.append({'panel':panel,'event':event,'start_ms':int(a.loc[event,'start_ms']),'difference_pp':delta})
        for label,sigma in sigmas.items():
            for effect in [.25,.5,1.]:
                for power in [.8,.9]:
                    for rho in [0.,.3,.6]:
                        required=power_n(sigma,effect,power,rho)
                        rows.append({'panel':panel,'sigma_scenario':label,'sigma_pp':sigma,'effect_pp':effect,
                            'target_power':power,'rho':rho,'design_effect':(1+rho)/(1-rho),'required_events':required,
                            'power_at_20_events':paired_t_power(20,sigma*np.sqrt((1+rho)/(1-rho)),effect),
                            'interpretation':'iid noncentral-t; AR1 long-run variance inflation is approximate'})
    pd.DataFrame(rows).to_csv(output/'power_grid.csv',index=False)
    # Distinguish power against zero from power for a prespecified usefulness margin.
    margin_rows=[]
    for row in rows:
        margin_rows.append({**row,'null_difference_pp':-.5,'null_interpretation':'shifted two-sided null at -0.5pp; illustrative separate design',
                           'true_difference_pp':-(.5+row['effect_pp']),
                           'effect_pp_above_margin':row['effect_pp']})
    pd.DataFrame(margin_rows).to_csv(output/'power_usefulness_margin_grid.csv',index=False)
    pd.DataFrame(sigma_rows).to_csv(output/'power_training_sigma.csv',index=False)
    pd.DataFrame(differences).to_csv(output/'power_training_activity_differences.csv',index=False)
    main=[x for x in rows if x['panel']=='global19' and x['sigma_scenario']=='normal_iid_chi2_high95' and
          x['effect_pp']==.5 and x['target_power']==.9 and x['rho']==.3][0]
    registry=read_json(OLD/'prospective/registry_v2/registry/config.json')
    plan={'version':'new_confirmation_draft_E1','status':'draft_unactivated_no_future_scores',
       'draft_created_utc':datetime.now(timezone.utc).isoformat(),
       'supersedes_existing_registry':False,'original_registry_config_sha256':sha(OLD/'prospective/registry_v2/registry/config.json'),
       'methods_original':registry['methods'],'primary_method':'ewma_reference',
       'sole_primary_comparison':'legacy_pair minus ewma_reference event-macro MAPE percentage points',
       'B_candidates':'shadow mechanism only; frozen selected method is independent',
       'minimum_useful_effect_pp':.5,'two_sided_alpha':.05,'power':.9,
       'null':'mean activity difference equals zero; primary two-sided95%HAC normal-reference CI; paired t is iid sensitivity',
       'difference_direction':'legacy_pair loss minus ewma_reference loss; negative favors legacy_pair; positive favors EWMA',
       'superiority_rule':'two-sided95%CI upper<0 supports lower legacy_pair error; CI lower>0 supports lower EWMA error; practical size checked separately against +/-0.5pp',
       'usefulness_rule':'report point estimate and95%CI relative to -0.5pp; zero-null power is not power for crossing this usefulness margin',
       'planned_events':main['required_events'],'sample_size_basis':main,
       'activation':'New independent plan seal and UTC timestamp required before cohort begins; no retrospective activation',
       'eligible_population':'All actual CN starts strictly after new seal timestamp, minimum1706504400000; fixed first N by actual start',
       'population_inventory':'Trusted collector append-only inventory with raw source SHA, retrieval time, actual start/end, all eligible events even with no predictions',
       'prediction_grid':registry['plan']['stages'],'ranks':registry['plan']['ranks'],
       'physical_dedup':'event/rank/cutoff; stage memberships reference same physical case',
       'primary_loss':'within-activity unique-case mean APE, then equal activity; rank-balanced sensitivity separately',
       'arrival_and_issue':{'minimum_prefix_points':registry['plan']['minimum_prefix_points'],
            'maximum_observation_age_ms':registry['plan']['maximum_observation_age_ms'],
            'maximum_issue_lag_ms':registry['plan']['maximum_issue_lag_ms'],
            'all_methods_same_cases':True,'issue_seal_before_outcome':True,
            'fallback':'preissued deterministic current-score fallback for pipeline failures; capture timestamp and SHA; late reconstructions invalid'},
       'feedback':'end_ms<next.start_ms and actual released_ms<next.start_ms; one shared start-of-activity history pool',
       'outcomes':registry['plan']['terminal_window'],
       'missing_rule':'Retain every eligible activity and planned slot; log no prefix, stale observation, no forecast, late forecast, missing terminal capture separately. No new substitute events. Unscorable activity stays missing; primary success assessment remains incomplete until fixed N fully scorable activities from original first-N population; show observed-subset descriptive metrics with denominators.',
       'reporting':'20-event pilot descriptive; final report at fixed N with all missing/failure inventory. No score-based early stop or method replacement; additional follow-up requires new plan',
       'fixed_inference':{
            'primary':'Bartlett/Newey-West HAC of chronological activity differences',
            'bandwidth':'L=min(n-1,max(1,ceil(n^(1/3))))',
            'autocovariance':'gamma_h=sum((d_t-dbar)*(d_(t-h)-dbar))/n',
            'long_run_variance':'LRV=gamma0+2*sum_h(1-h/(L+1))*gamma_h',
            'interval':'dbar +/- z_.975*sqrt(LRV/n)',
            'zero_variance':'degenerate_variance_no_decision; no significance or zero-width future precision claim',
            'assumptions':'weak stationarity, finite required moments and weak dependence; asymptotic approximation; AR1 power inflation supplies approximate design conditions, no finite-sample power guarantee',
            'sensitivity':'iid paired t and activity iid/circular block1/3/5/7 fixed-loss bootstrap; same paired chronology',
            'source_sha256':sha(__file__),
            'implementation':'statistics_audit.py:hac_interval; pure fixtures cover manual ordered AR path and degenerate constant'},
       'reliability':'freeze interval method/levels before activation; report80/90 point, full rank-phase joint, whole-activity joint, strata/support/finite rate/width/Winkler separately; no widening from future outcomes',
       'zero_future_events':{'n_events':0,'mape':None,'status':'waiting; no future outcome fabricated'},
       'artifacts':'sealed source/config/dependencies/commitments, inventory, issued predictions, interval rows, actual release log, independent score bundle, hash manifest'}
    write_json(output/'confirmation_plan.json',plan)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['predictions','reliability'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--draws',type=int,default=5000)
    parser.add_argument('--passed-manifest',type=Path,help='Required sealed prediction-scoring bundle for TRAIN-only power')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(HERE):raise ValueError('E workspace outputs only')
    args.output.mkdir(exist_ok=False,parents=True)
    started=time.monotonic();read_json(HERE/'preregistration.json')
    if args.stage=='predictions':stage_predictions(args.output,args.draws)
    else:
        if args.passed_manifest is None:raise ValueError('Reliability requires explicit --passed-manifest')
        stage_reliability(args.output,args.draws,args.passed_manifest)
    if time.monotonic()-started>120:raise RuntimeError('Stage exceeded120second preregistered budget; output remains incomplete')
    seal_bundle(args.output,args.stage,started)

if __name__=='__main__':main()
