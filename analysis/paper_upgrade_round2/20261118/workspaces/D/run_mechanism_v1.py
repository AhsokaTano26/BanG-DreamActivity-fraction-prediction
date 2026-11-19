"""Trusted train-only coordinator. Real numeric execution requires PI allocation.

All model kernels are in mechanism.py and accept detached arrays only.
Run with single-thread BLAS/OMP and nice19; separate 30s/90s CPU allocations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd
import mechanism as M

HERE=Path(__file__).resolve().parent
INPUTS=HERE.parent.parent/'inputs'
BROOT=HERE.parent/'B'
MIN_START=1706504400000
BLIND=('case_id','event','rank','phase','progress','start_ms','end_ms','cutoff','observed_at',
       'current','event_type','band_group','split','duration_h','remaining_h',
       'rate_6h_relative','rate_24h_relative','base_prediction')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    if not rows:raise ValueError('Empty required output: '+str(path))
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def blind_frame(frame):return frame[[c for c in BLIND if c in frame]].copy(deep=True)


def prepare_frame(frame,queries):
    if not frame['split'].eq('train').all() or any(q['split']!='train' for q in queries):
        raise ValueError('Only train inputs accepted')
    ordered=sorted(queries,key=lambda q:(q['start_ms'],q['event']))
    ids=[q['event'] for q in ordered]
    if len(ids)!=len(set(ids)) or set(frame.event)!=set(ids) or frame.case_id.duplicated().any():
        raise ValueError('Unique fixed activities and cases required')
    if not np.isfinite(frame[['progress','log_error','actual','base_prediction','current']].to_numpy(float)).all():
        raise ValueError('Finite case values required')
    if not np.array_equal(frame.phase.to_numpy(),np.searchsorted([.25,.9],frame.progress,side='right')):
        raise ValueError('Incorrect phase at fixed .25/.9 boundaries')
    if (frame.start_ms<MIN_START).any() or (frame.end_ms<=frame.start_ms).any():raise ValueError('Invalid actual activity clocks')
    if (frame.actual<=0).any() or not np.allclose(frame.log_error,np.log(frame.base_prediction/frame.actual),atol=1e-10,rtol=1e-10):
        raise ValueError('Residual sign/target consistency failure')
    for q in ordered:
        block=frame[frame.event==q['event']]
        if not block.start_ms.eq(q['start_ms']).all() or not block.end_ms.eq(q['end_ms']).all():
            raise ValueError('Conflicting query clocks')
    result=frame.copy();result['ordinal']=result.event.map(dict(zip(ids,range(len(ids)))))
    means=result.groupby(['event','rank','phase'],as_index=False).agg(
        z=('log_error','mean'),case_n=('case_id','size'),within_case_sd=('log_error','std'),
        start_ms=('start_ms','first'),end_ms=('end_ms','first'),ordinal=('ordinal','first'))
    return {'frame':result,'means':means,'queries':ordered,'ordered_events':ids}


def pair_panel(means,event_order,phase):
    part=means[(means.phase==phase)&means['rank'].isin([1000,2000])]
    panel=part.pivot(index='event',columns='rank',values='z').reindex(index=event_order,columns=[1000,2000])
    valid=panel.notna().all(axis=1)
    return panel[valid].to_numpy(float),np.flatnonzero(valid),panel.index[valid].tolist()


def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>=3 and np.std(a)>1e-12 and np.std(b)>1e-12 else None


def ranks(a):
    _,inverse,count=np.unique(a,return_inverse=True,return_counts=True)
    return (np.cumsum(count)-(count+1)/2)[inverse]


def paired_statistics(y,ordinals):
    y=np.asarray(y,float);ordinals=np.asarray(ordinals,int);n=len(y)
    adjacent=np.diff(ordinals)==1;d=np.diff(y,axis=0);delta=d[adjacent]
    lagvalid=np.flatnonzero(adjacent[:-1]&adjacent[1:])
    out={'complete_pairs':n,'adjacent_differences':int(adjacent.sum()),'difference_lag_pairs':len(lagvalid),
         'activity_span':int(ordinals[-1]-ordinals[0]+1) if n else 0,
         'maximum_activity_gap':int(np.diff(ordinals).max()) if n>1 else 0,
         'level_pearson':corr(y[:,0],y[:,1]),'level_spearman':corr(ranks(y[:,0]),ranks(y[:,1])),
         'detrended_pearson':None,'difference_pearson':corr(delta[:,0],delta[:,1]),
         'own_lag_1000':None,'own_lag_2000':None,'cross_lag_1000_to2000':None,'cross_lag_2000_to1000':None,
         'gap_mean':float(np.mean(y[:,0]-y[:,1])) if n else None,
         'gap_sd':float(np.std(y[:,0]-y[:,1],ddof=1)) if n>1 else None}
    if n>=3:
        design=np.c_[np.ones(n),ordinals]
        centered=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
        out['detrended_pearson']=corr(centered[:,0],centered[:,1])
        for name,a,b in [('own_lag_1000',0,0),('own_lag_2000',1,1),('cross_lag_1000_to2000',0,1),('cross_lag_2000_to1000',1,0)]:
            out[name]=corr(y[:-1,a][adjacent],y[1:,b][adjacent])
    moments=None
    if len(delta)>1 and len(lagvalid)>1:
        center=delta.mean(axis=0)
        gamma0=np.cov(delta,rowvar=False,ddof=1)
        gamma1=(d[lagvalid+1]-center).T@(d[lagvalid]-center)/len(lagvalid)
        moments=M.moment_decomposition(gamma0,gamma1)
        for name,matrix in [('Gamma_delta0',gamma0),('Gamma_delta1',gamma1)]:
            for i,j in np.ndindex(2,2):out[f'{name}_{i}{j}']=float(matrix[i,j])
    for name in ('Q_raw','R_raw','Q_psd','R_psd'):
        for i,j in np.ndindex(2,2):out[f'{name}_{i}{j}']=float(moments[name][i,j]) if moments else None
    for name in ('Q','R'):
        out[name+'_minimum_eigenvalue']=float(moments[name+'_eigenvalues'].min()) if moments else None
        out[name+'_projection_frobenius']=moments[name+'_projection_frobenius'] if moments else None
    return out


def paired_bootstrap(y,ordinals,rng):
    base=paired_statistics(y,ordinals);n=len(y)
    if n<8:return base
    keys=[k for k,v in base.items() if v is not None and k not in ('complete_pairs','adjacent_differences','difference_lag_pairs','activity_span','maximum_activity_gap')]
    for block in (1,3,5):
        results=[]
        for _ in range(500):
            starts=rng.integers(0,n,math.ceil(n/block))
            ix=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
            results.append(paired_statistics(y[ix],np.arange(n)))
        for k in keys:
            values=[row[k] for row in results if row.get(k) is not None]
            if values:
                low,high=np.quantile(values,[.025,.975])
                base[k+f'_block{block}_low']=float(low);base[k+f'_block{block}_high']=float(high)
    return base


def source_check():
    seal=json.loads((HERE/'code_seal.json').read_text())
    for path,expected in seal['sha256'].items():
        if sha(HERE/path)!=expected:raise ValueError('Frozen source/input/artifact changed: '+path)
    return seal


def diagnostics(prepared):
    means=prepared['means'];ids=prepared['ordered_events'];rows=[];pairs=[]
    for segment,eventids in [('calibration',ids[:25]),('diagnostic',ids[25:])]:
        sub=means[means.event.isin(eventids)]
        for (rank,phase),g in sub.groupby(['rank','phase']):
            g=g.sort_values('ordinal');x=g.z.to_numpy();ordinal=g.ordinal.to_numpy()
            for representation,values in [('level',x),('difference',np.diff(x)[np.diff(ordinal)==1])]:
                row={'segment':segment,'rank':int(rank),'phase':int(phase),'representation':representation,
                     'activity_span':int(ordinal[-1]-ordinal[0]+1),'complete_activity_n':len(x),
                     'case_n':int(g.case_n.sum()),'within_case_sd_mean':float(g.within_case_sd.mean()),
                     **M.distribution_summary(values)}
                first=g[g.ordinal<12].z.to_numpy();last=g[(g.ordinal>=12)&(g.ordinal<25)].z.to_numpy()
                row['calibration_last13_first12_variance_ratio']=float(np.var(last,ddof=1)/np.var(first,ddof=1)) if segment=='calibration' and len(first)>1 and len(last)>1 and np.var(first)>0 else None
                rows.append(row)
        family=[r for r in rows if r['segment']==segment and r['JB_asymptotic_p'] is not None]
        for row,q in zip(family,M.bh_values([r['JB_asymptotic_p'] for r in family])):row['BH_q']=float(q);row['BH_family_size']=len(family)
        for phase in range(3):
            y,ordinal,events=pair_panel(means,eventids,phase)
            pairs.append({'segment':segment,'phase':phase,'event_ids':json.dumps(events),
                          'activity_ordinals':json.dumps(ordinal.tolist()),
                          **paired_bootstrap(y,ordinal,np.random.default_rng(20261118+phase+(100 if segment=='diagnostic' else 0)))})
    return rows,pairs


def next_benefits(prepared):
    frame=prepared['frame'];means=prepared['means'];queries=prepared['queries'];result=[]
    run=BROOT/'run_v2_001'
    for index,q in enumerate(queries[25:],26):
        current=frame[frame.event==q['event']].copy()
        # Read outcome-free experts issued by B's already sealed coordinator.
        expert=pd.read_csv(run/'experts'/f'{index:02d}_event{q["event"]}.csv.gz')
        issued=pd.read_csv(run/'issued'/f'{index:02d}_event{q["event"]}.csv.gz')
        issued=issued[issued.candidate=='risk_recent_k8'][['case_id','bias']].rename(columns={'bias':'risk_bias'})
        if set(expert.case_id)!=set(current.case_id) or set(issued.case_id)!=set(current.case_id):raise ValueError('Issued expert case mismatch')
        current=current.merge(expert,on='case_id',validate='one_to_one').merge(issued,on='case_id',validate='one_to_one')
        pastmeans=means[(means.end_ms<q['start_ms'])&(means.ordinal<index-1)]
        pastids=[p['event'] for p in queries[:index-1] if p['end_ms']<q['start_ms']]
        for phase in range(3):
            cases=current[(current.phase==phase)&current['rank'].isin([1000,2000])]
            if cases.empty:continue
            y,ordinals,events=pair_panel(pastmeans,pastids,phase);recent=y[-12:]
            rms=lambda a:float(np.sqrt(np.mean(a*a)))
            signal={'past_pairs':len(y),'recent_pairs':len(recent),'past_activity_span':int(ordinals[-1]-ordinals[0]+1) if len(ordinals) else 0,
                    'past_max_activity_gap':int(np.diff(ordinals).max()) if len(ordinals)>1 else 0,
                    'past_rho':corr(recent[:,0],recent[:,1]) if len(recent) else None,
                    'past_gap_ratio':rms(recent[:,0]-recent[:,1])/(rms(recent[:,0])+rms(recent[:,1])+1e-8) if len(recent) else None,
                    'past_drift_ratio':rms(np.diff(recent[:,0]-recent[:,1]))/(rms(np.diff(recent[:,0]))+rms(np.diff(recent[:,1]))+1e-8) if len(recent)>1 else None}
            methods={'I':cases.expert_ind.to_numpy(),'S':cases.expert_pair.to_numpy(),
                     'fixed_half':(cases.expert_ind+cases.expert_pair).to_numpy()/2,'risk_recent_k8':cases.risk_bias.to_numpy()}
            base_row={'event':q['event'],'query_index':index,'phase':phase,'cases':len(cases),
                      'current_rank_count':cases['rank'].nunique(),'current_start_ms':q['start_ms'],**signal}
            reference=None
            for method,bias in methods.items():
                prediction=np.clip(cases.base_prediction.to_numpy()*np.exp(-bias),cases.current.to_numpy(),cases.current.to_numpy()*np.exp(4))
                ape=100*np.abs(prediction/cases.actual.to_numpy()-1)
                score=pd.DataFrame({'rank':cases['rank'].to_numpy(),'ape':ape})
                values={'case_mape':float(ape.mean()),'rank_equal_mape':float(score.groupby('rank').ape.mean().mean()),
                        'log_mse':float(np.mean((cases.log_error.to_numpy()-bias)**2))}
                if method=='I':reference=values
                result.append({**base_row,'method':method,**values,**{k+'_minus_I':v-reference[k] for k,v in values.items()}})
    return result


def manifest(start,stage,extra=None):
    return {'stage':stage,'cpu_seconds':time.process_time()-start,'python':platform.python_version(),
            'numpy':np.__version__,'pandas':pd.__version__,'threads':1,'code_seal_sha256':sha(HERE/'code_seal.json'),
            'source_seal':json.loads((HERE/'code_seal.json').read_text()),**(extra or {})}


def run_calibration(output):
    start=time.process_time();source_check();config=json.loads((HERE/'preregistration.json').read_text())
    queries=json.loads((INPUTS/'train_queries.json').read_text())
    for file,digest in config['input_sha256'].items():
        if sha(INPUTS/file)!=digest:raise ValueError('Input fingerprint mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    frame=pd.read_csv(INPUTS/'train_residuals.csv.gz');prepared=prepare_frame(frame,queries)
    ids=prepared['ordered_events']
    if ids[:25]!=config['calibration_event_ids'] or ids[25:]!=config['diagnostic_event_ids']:raise ValueError('Fixed activity order changed')
    y,ordinals,events=pair_panel(prepared['means'],ids[:25],1)
    calibration=M.calibrate(y,ordinals)
    write_csv(output/'calibration_profile.csv',calibration['grid'])
    ranges=[]
    supported=[g for g in calibration['grid'] if g['support']]
    for parameter in ('qc','qd','r','rho','m','q_r_ratio'):
        ranges.append({'parameter':parameter,'best':calibration['best'][parameter],
                       'support_low':min(g[parameter] for g in supported),'support_high':max(g[parameter] for g in supported),
                       'delete_one_low':min(g[parameter] for g in calibration['delete_one_best']),
                       'delete_one_high':max(g[parameter] for g in calibration['delete_one_best']),
                       'support_grid_n':len(supported),'complete_pairs':len(y),'activity_span':int(ordinals[-1]-ordinals[0]+1)})
    write_csv(output/'calibration_parameters.csv',ranges)
    write_csv(output/'delete_one_profile.csv',calibration['delete_one_best'])
    calibration.update(complete_pair_events=events,activity_ordinals=ordinals.tolist(),complete_pairs=len(y),
                       source_config_sha256=sha(HERE/'preregistration.json'),input_sha256=config['input_sha256'],
                       support_band_identity='descriptive, no nominal coverage',oracle_extra_information='true generation Q/R and initial state')
    # Parameters are sealed before any last10 calculations or simulated draws.
    write_json(output/'calibration_seal.json',calibration)
    diag,pairs=diagnostics(prepared)
    write_csv(output/'diagnostics.csv',diag);write_csv(output/'paired_diagnostics.csv',pairs)
    prepared['means'].to_csv(output/'event_rank_phase_means.csv',index=False)
    benefits=next_benefits(prepared);write_csv(output/'next_event_signal_benefit.csv',benefits)
    associations=[]
    for phase in range(3):
        part=[r for r in benefits if r['phase']==phase and r['method']=='S']
        for signal in ('past_rho','past_gap_ratio','past_drift_ratio'):
            for target in ('case_mape_minus_I','log_mse_minus_I'):
                valid=[r for r in part if r[signal] is not None]
                x=np.array([r[signal] for r in valid]);z=np.array([r[target] for r in valid])
                estimate=corr(ranks(x),ranks(z));deletions=[corr(ranks(np.delete(x,j)),ranks(np.delete(z,j))) for j in range(len(x))]
                deletions=[v for v in deletions if v is not None]
                associations.append({'phase':phase,'signal':signal,'target':target,'events':len(x),'spearman':estimate,
                                     'delete_one_low':min(deletions) if deletions else None,'delete_one_high':max(deletions) if deletions else None})
    write_csv(output/'next_event_associations.csv',associations)
    write_json(output/'run_manifest.json',manifest(start,'calibration',{'cases':len(frame),'activities':len(ids),
               'actual_ranks':sorted(int(r) for r in frame['rank'].unique()),'calibration_seal_sha256':sha(output/'calibration_seal.json'),
               'outputs_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}))


def run_simulation(calibration_path,output):
    start=time.process_time();source_check();cal=json.loads(Path(calibration_path).read_text())
    if cal['source_config_sha256']!=sha(HERE/'preregistration.json'):raise ValueError('Calibration configuration mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    qfit,rfit=M.covariances(cal['best']);mu=np.array(cal['mu'])
    path_rows=[];seed_rows=[];failure=[];link=[]
    event_file=(output/'per_event_simulation.csv').open('x',newline='');event_writer=None
    for seed in (42,123,2026):
        rng=np.random.default_rng(seed);state_units=rng.standard_normal((100,35,2));normal=rng.standard_normal((100,35,2))
        heavy=M.standardize_noise('t5',rng.standard_t(5,(100,35,2)))
        contaminated=M.standardize_noise('contamination5',normal,rng.random((100,35,2))<.05)
        for anchor_index,anchor in enumerate(cal['anchors']):
            qtrue,rtrue=M.covariances(anchor)
            for variant in M.VARIANTS:
                regime=f'a{anchor_index}_{variant}';units=heavy if variant.startswith('t5') else contaminated if variant.startswith('contamination') else normal
                state,observed=M.make_path(anchor,variant,state_units,units,mu,cal['common_shift'],cal['difference_shift'])
                delay=1 if variant=='delay1' else 3 if variant=='delay3' else 0
                issued=M.replay_arrays(observed,qfit,rfit,delay=delay,oracle_q=qtrue,oracle_r=rtrue,mu=mu)
                scores=M.score_arrays(state,observed,issued)
                averaged={key:value[:,25:].mean(axis=1) for key,value in scores.items()}
                for method_index,method in enumerate(M.METHODS):
                    for path in range(100):
                        row={'regime':regime,'anchor':anchor_index,'variant':variant,'seed':seed,'path':path,'method':method,'oracle':method.startswith('oracle')}
                        for key,value in averaged.items():row[key]=float(value[path,method_index]);row[key+'_minus_I']=float(value[path,method_index]-value[path,0])
                        row['clipped_fraction']=float(np.mean(np.abs(issued['raw_bias'][path,25:,method_index])>.3))
                        path_rows.append(row)
                        for event in range(25,35):
                            erow={'regime':regime,'seed':seed,'path':path,'event':event+1,'method':method,
                                  **{key:float(value[path,event,method_index]) for key,value in scores.items()},
                                  'lambda_mean':float(issued['lambda'][path,event].mean()) if method=='risk_recent_k8' else None}
                            if event_writer is None:event_writer=csv.DictWriter(event_file,fieldnames=list(erow));event_writer.writeheader()
                            event_writer.writerow(erow)
                    for key,value in averaged.items():
                        diff=value[:,method_index]-value[:,0]
                        seed_rows.append({'regime':regime,'seed':seed,'method':method,'metric':key,'paths':100,
                                          'mean':float(value[:,method_index].mean()),'mean_minus_I':float(diff.mean()),
                                          'paired_mc_se':float(diff.std(ddof=1)/10)})
                if seed==42:link.append({'regime':regime,'anchor':anchor_index,'variant':variant,**anchor,
                                        'real_complete_pairs':cal['complete_pairs'],'real_activity_ordinals':json.dumps(cal['activity_ordinals']),
                                        'noise_family':'t5 standardized' if variant.startswith('t5') else '5% contamination standardized' if variant.startswith('contamination') else 'Gaussian',
                                        'delay_events':delay,'shift_common':cal['common_shift'] if variant=='common_shift_at26' else 0.,
                                        'shift_difference':cal['difference_shift'] if variant=='difference_shift_at26' else 0.,
                                        'relation':'conditional profile anchor; perturbation strength fixed from first25 or preregistered constant'})
    event_file.close();write_csv(output/'per_path.csv',path_rows);write_csv(output/'per_seed.csv',seed_rows)
    frame=pd.DataFrame(path_rows)
    for (regime,method),g in frame.groupby(['regime','method'],sort=False):
        for key in ('state_mse','raw_state_mse','observed_log_mse','synthetic_unbounded_ape_pct'):
            delta=g[key+'_minus_I'].to_numpy();se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
            failure.append({'regime':regime,'method':method,'metric':key,'paths':len(g),'mean':float(g[key].mean()),
                            'mean_minus_I':float(delta.mean()),'paired_mc_se':se,'mc_95_low':float(delta.mean()-1.96*se),
                            'mc_95_high':float(delta.mean()+1.96*se),'worse_than_I_mean':bool(delta.mean()>0),
                            'path_P90':float(np.quantile(g[key],.9)),'clipped_fraction':float(g.clipped_fraction.mean())})
    write_csv(output/'failure_regions.csv',failure);write_csv(output/'simulation_real_link.csv',link)
    write_json(output/'run_manifest.json',manifest(start,'simulation',{'regimes':24,'seeds':[42,123,2026],'paths_per_seed':100,'methods':M.METHODS,
               'calibration_seal_sha256':sha(calibration_path),'outputs_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['calibration','simulation'],required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--calibration');parser.add_argument('--pi-authorized',action='store_true')
    args=parser.parse_args()
    if not args.pi_authorized:parser.error('Numerical execution requires PI allocation; fixtures use unittest only')
    limit=30 if args.stage=='calibration' else 90
    resource.setrlimit(resource.RLIMIT_CPU,(limit,limit))
    if args.stage=='calibration':run_calibration(args.output)
    else:
        if not args.calibration:parser.error('--calibration is required for simulation')
        run_simulation(args.calibration,args.output)


if __name__=='__main__':main()
