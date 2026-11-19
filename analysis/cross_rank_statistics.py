"""Activity-level cross-rank associations; read training outcomes and blind indexes only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/rolling_holdout_v3_dense_19rank_20261117'
RUN = ROOT / 'analysis/rolling_results/v3_multiline_calibration_20261117'
OUTPUT = ROOT / 'analysis/cross_rank_records/20261117/results'
RANKS = [10,20,30,40,50,100,200,300,400,500,1000,1500,2000,3000,4000,5000,10000,20000,30000]
SEED, DRAWS = 42, 2000


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def residualize(values, controls):
    return values - controls @ np.linalg.lstsq(controls,values,rcond=None)[0]


def standardized(values, axis=0):
    z = values - values.mean(axis=axis,keepdims=True)
    norm = np.sqrt(np.sum(z*z,axis=axis,keepdims=True))
    return np.divide(z,norm,out=np.full_like(z,np.nan),where=norm>1e-12)


def correlation(values, other=None, *, rank=True):
    a = np.asarray(values,dtype=float)
    b = a if other is None else np.asarray(other,dtype=float)
    if rank:
        a = stats.rankdata(a,axis=0)
        b = a if other is None else stats.rankdata(b,axis=0)
    return np.clip(standardized(a).T @ standardized(b),-1.,1.)


def resample_indices(n, draws, block, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0,n,(draws,int(np.ceil(n/block))))
    return ((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]


def common_panel(frame, ranks):
    chosen = frame.reindex(columns=ranks)
    active = [r for r in ranks if chosen[r].notna().any()]
    common = chosen[active].dropna()
    return common.to_numpy(float),list(common.index),active


def legal_lag(matrix, clock):
    ordered = sorted(matrix.index,key=lambda e:(clock[int(e)]['start_ms'],int(e)))
    pairs = [(a,b) for a,b in zip(ordered,ordered[1:]) if clock[int(a)]['end_ms'] < clock[int(b)]['start_ms']]
    return (matrix.loc[[a for a,b in pairs]].to_numpy(float),
            matrix.loc[[b for a,b in pairs]].to_numpy(float),pairs)


def control_design(context):
    numeric = np.column_stack([stats.rankdata(context.start_ms.to_numpy()),
                              stats.rankdata(context.duration_h.to_numpy())])
    numeric = numeric-numeric.mean(axis=0)
    scale = numeric.std(axis=0)
    numeric = numeric[:,scale>1e-12]/scale[scale>1e-12]
    categories = pd.get_dummies(context[['event_type','band_group']].astype(str),drop_first=True,dtype=float)
    return np.column_stack([np.ones(len(context)),numeric,categories.to_numpy()])


def normalized(values, context=None):
    z = stats.rankdata(values,axis=0)
    if context is not None:
        z = residualize(z,control_design(context))
    return standardized(z)


def bootstrap_correlations(x, y, *, context, block):
    indices = resample_indices(len(x),DRAWS,block,SEED)
    result = []
    for batch in np.array_split(indices,20):
        if context is None:
            a = standardized(stats.rankdata(x[batch],axis=1),axis=1)
            b = a if y is None else standardized(stats.rankdata(y[batch],axis=1),axis=1)
            result.extend(np.einsum('bnr,bns->brs',a,b,optimize=True))
        else:
            for idx in batch:
                a = normalized(x[idx],context.iloc[idx])
                result.append(a.T@a)
    return np.clip(np.asarray(result),-1.,1.)


def permutation_probabilities(x, y):
    a,b = normalized(x),normalized(x if y is None else y)
    observed = np.abs(a.T@b)
    rng = np.random.default_rng(SEED)
    permutations = np.stack([rng.permutation(len(x)) for _ in range(DRAWS)])
    iid = np.empty_like(observed)
    for i in range(a.shape[1]):
        null = a[permutations,i] @ b
        iid[i] = (1+(np.abs(null)>=observed[i]-1e-12).sum(axis=0))/(DRAWS+1)
    count = np.zeros_like(observed)
    for shift in range(len(x)):
        count += np.abs(a.T@np.roll(b,shift,axis=0)) >= observed-1e-12
    return iid,count/len(x)


def band(rank):
    return int(np.searchsorted([50,500,5000],rank,side='left'))


def panel(name, scope, frame, output, *, context=None, lag_clock=None):
    values,events,ranks = common_panel(frame,RANKS)
    if len(events)<8:
        raise ValueError(f'{name}: fewer than eight common activities')
    common = frame.loc[events,ranks]
    y = None
    pairs = None
    if lag_clock is not None:
        values,y,pairs = legal_lag(common,lag_clock)
        events = [int(b) for a,b in pairs]
    controls = None if context is None else context.loc[events].reset_index(drop=True)
    a = normalized(values,controls)
    b = a if y is None else normalized(y)
    c = np.clip(a.T@b,-1.,1.)
    pearson = correlation(values,y,rank=False) if controls is None else np.full_like(c,np.nan)
    iid_boot = bootstrap_correlations(values,y,context=controls,block=1)
    block_boot = bootstrap_correlations(values,y,context=controls,block=5)
    iid_ci = np.quantile(iid_boot,[.025,.975],axis=0)
    block_ci = np.quantile(block_boot,[.025,.975],axis=0)
    p_iid,p_shift = permutation_probabilities(values,y) if controls is None else (np.full_like(c,np.nan),np.full_like(c,np.nan))
    directory = output/'matrices';directory.mkdir(exist_ok=True)
    pd.DataFrame(c,index=ranks,columns=ranks).reindex(index=RANKS,columns=RANKS).to_csv(directory/f'{name}.csv')
    observations = pd.DataFrame(values,index=events,columns=ranks).rename_axis('event')
    observations.to_csv(output/'observations'/f'{name}.csv')
    if y is not None:
        pd.DataFrame(y,index=events,columns=ranks).rename_axis('event').to_csv(output/'observations'/f'{name}_target.csv')
    rows = []
    positions = {r:i for i,r in enumerate(ranks)}
    for i,r in enumerate(RANKS):
        for s in (RANKS if y is not None else RANKS[i+1:]):
            valid = r in positions and s in positions
            row = {'panel':name,'scope':scope,'rank_a':r,'rank_b':s,'n':len(events) if valid else 0,
                   'same_band':band(r)==band(s),'directed_lag':y is not None}
            if valid:
                j,k = positions[r],positions[s]
                row.update(spearman=c[j,k],pearson_log_scale=pearson[j,k],
                           iid_low=iid_ci[0,j,k],iid_high=iid_ci[1,j,k],
                           block5_low=block_ci[0,j,k],block5_high=block_ci[1,j,k],
                           p_iid=p_iid[j,k],p_circular_shift=p_shift[j,k])
            rows.append(row)
    ii,jj = np.triu_indices(len(ranks),1) if y is None else np.where(~np.eye(len(ranks),dtype=bool))
    within = np.array([band(ranks[i])==band(ranks[j]) for i,j in zip(ii,jj)])
    group_info = {}
    for label,mask in [('within_band',within),('between_band',~within)]:
        group_info[label+'_median'] = float(np.median(c[ii[mask],jj[mask]]))
        estimates = np.median(block_boot[:,ii[mask],jj[mask]],axis=1)
        group_info[label+'_block5_95ci'] = np.quantile(estimates,[.025,.975]).tolist()
    difference = np.median(block_boot[:,ii[within],jj[within]],axis=1)-np.median(block_boot[:,ii[~within],jj[~within]],axis=1)
    group_info['within_minus_between_block5_95ci'] = np.quantile(difference,[.025,.975]).tolist()
    info = {'panel':name,'scope':scope,'common_events':len(events),'available_events':len(frame),
            'active_ranks':ranks,'unsupported_ranks':[r for r in RANKS if r not in ranks],
            'event_ids':events,'lag_pairs':pairs,'groups':group_info}
    if controls is not None:
        info['control_design_columns'] = control_design(controls).shape[1]
        info['control_design_rank'] = int(np.linalg.matrix_rank(control_design(controls)))
    pca,loading,clusters = [],[],[]
    if y is None:
        eigenvectors = np.linalg.eigh(c)
        order = np.argsort(eigenvectors.eigenvalues)[::-1]
        eig = np.maximum(0.,eigenvectors.eigenvalues[order]);vec = eigenvectors.eigenvectors[:,order]
        for j,ev in enumerate(eig):
            pca.append({'panel':name,'component':j+1,'variance_fraction':ev/eig.sum(),'cumulative_fraction':eig[:j+1].sum()/eig.sum(),'n':len(events),'ranks':len(ranks)})
        for i,r in enumerate(ranks):
            loading.append({'panel':name,'rank':r,**{f'PC{j+1}':vec[i,j]*np.sqrt(eig[j]) for j in range(4)}})
        distance = np.sqrt(np.maximum(0.,(1-c)/2));np.fill_diagonal(distance,0.)
        tree = linkage(squareform(distance,checks=False),method='average')
        labels = fcluster(tree,t=4,criterion='maxclust')
        consensus = np.zeros_like(c)
        # A subset of already generated activity-block resamples; no additional fitting or data.
        for boot in block_boot[:200]:
            d = np.sqrt(np.maximum(0.,(1-boot)/2));np.fill_diagonal(d,0.)
            lab = fcluster(linkage(squareform(d,checks=False),method='average'),t=4,criterion='maxclust')
            consensus += lab[:,None]==lab[None,:]
        pd.DataFrame(consensus/200,index=ranks,columns=ranks).reindex(index=RANKS,columns=RANKS).to_csv(directory/f'{name}_cluster_stability.csv')
        for r,lab in zip(ranks,labels):clusters.append({'panel':name,'rank':r,'cluster':int(lab),'fixed_clusters':4,'n':len(events)})
        info['rank_PCA_variance_first2'] = float(eig[:2].sum()/eig.sum())
        info['rank_PCA_variance_first4'] = float(eig[:4].sum()/eig.sum())
    print(json.dumps({'completed_panel':name,'n':len(events),'ranks':len(ranks)},ensure_ascii=False),flush=True)
    return rows,info,pca,loading,clusters


def stage_frames(cases, members, values):
    joined = members.merge(cases[['case_id','event','rank']],on='case_id',validate='many_to_one')
    joined['value'] = joined.case_id.map(values)
    result = {}
    for p in [25,50,75,90,95]:
        part = joined[joined.stage_id==f'progress_{p}pct']
        if part.duplicated(['event','rank']).any():raise ValueError('Repeated activity/rank at a stage')
        result[p] = part.pivot(index='event',columns='rank',values='value')
    return result


def run(destination=OUTPUT):
    from analysis.multiline_dataset import MultilineDataset
    from analysis.rolling_benchmark import verify_seal
    destination = Path(destination)
    if destination.exists():raise FileExistsError('Preserve previous statistical analysis')
    destination.mkdir(parents=True);(destination/'observations').mkdir()
    data = MultilineDataset(DATA)
    blind = verify_seal(RUN/'blind_manifest.json')
    assert data.manifest['dataset_id']==blind['dataset_id']=='db4e9b9c84bd64a6de5ee30a090cfac83efbe85c2f6dd899304516e7d031cdf1'
    cases = data.case_index('train');members = data.memberships('train')
    train_ids = set(data.manifest['splits']['train']['event_ids'])
    context = cases.drop_duplicates('event').set_index('event')[['start_ms','end_ms','event_type','band_group']]
    assert len(context)==len(train_ids)==67
    context['duration_h'] = (context.end_ms-context.start_ms)/3600000
    curves = list(data.train_curves())
    curve_rows = [{'event':c.event,'rank':c.rank,'final':c.final,'log_final':np.log(c.final),
                   'observations':len(c.times),'start_ms':c.start,'end_ms':c.end} for c in curves]
    curve_frame = pd.DataFrame(curve_rows)
    assert set(curve_frame.event)==train_ids and (curve_frame.start_ms>=1706504400000).all()
    assert not curve_frame.duplicated(['event','rank']).any()
    curve_frame.to_csv(destination/'training_curves.csv',index=False)
    order = sorted(train_ids,key=lambda e:(data.event_index[e]['start_ms'],e))
    finals = curve_frame.pivot(index='event',columns='rank',values='log_final').reindex(order)
    final_map = {(c.event,c.rank):c.final for c in curves}
    actual = cases.apply(lambda r:final_map[int(r.event),int(r['rank'])],axis=1)
    growth_values = dict(zip(cases.case_id,np.log(actual.to_numpy()/cases.current.to_numpy())))
    growth = stage_frames(cases,members,growth_values)
    growth = {p:f.reindex(order) for p,f in growth.items()}
    configs = [('train_log_final','train_67',finals,{}),
               ('train_final_partial','train_67_conditional',finals,{'context':context}),
               ('train_final_changes','train_66_changes',finals.diff().iloc[1:],{})]
    configs += [(f'train_remaining_growth_p{p}','train_frozen_stage',f,{}) for p,f in growth.items()]
    first_val = min((r for r in data.event_index.values() if r['split']=='validation'),key=lambda r:(r['start_ms'],r['event']))
    residual_path = RUN/'events'/str(first_val['event'])/'released_residuals.csv'
    past = pd.read_csv(residual_path,float_precision='round_trip')
    assert sha(residual_path)==blind['artifacts'][str(residual_path.relative_to(RUN))]
    assert past.event.nunique()==35 and set(past.event)<=train_ids and set(past.split)=={'train'}
    assert past.case_id.is_unique and (past.end_ms<first_val['start_ms']).all()
    targets = np.array([final_map[int(e),int(r)] for e,r in zip(past.event,past['rank'])])
    np.testing.assert_allclose(past.base_prediction*np.exp(-past.log_error),targets,rtol=1e-12)
    past['corrected_error'] = np.log(past.expert__independent/targets)
    warm_order = [e for e in order if e in set(past.event)]
    clock = {e:data.event_index[e] for e in warm_order}
    error_counts = []
    # Sensitivity: match nominal cutoffs across all 18 ranks before event/phase averaging.
    supported = [r for r in RANKS if r!=1500]
    count = past.groupby(['event','cutoff'])['rank'].transform('nunique')
    matched = past[count==len(supported)]
    for phase,part in past.groupby('phase'):
        for column,label in [('log_error','raw'),('corrected_error','independent')]:
            means = part.groupby(['event','rank'])[column].mean().unstack().reindex(warm_order)
            name = f'train_error_{label}_phase{phase}'
            configs.append((name,'train_35_causal_forecasts',means,{}))
            configs.append((name+'_lag1','train_previous_completed_to_next',means,{'lag_clock':clock}))
            error_counts.extend({'panel':name,'event':int(e),'rank':int(r),'prefixes':int(n)} for (e,r),n in part.groupby(['event','rank']).size().items())
        m = matched[matched.phase==phase].groupby(['event','rank']).log_error.mean().unstack().reindex(warm_order)
        configs.append((f'train_error_matched_phase{phase}','train_common_cutoff_sensitivity',m,{}))
    pd.DataFrame(error_counts).to_csv(destination/'error_aggregation_counts.csv',index=False)
    # A separate descriptive panel of *blind current values*: no heldout outcomes loaded.
    heldout = data.case_index('test');heldout_members = data.memberships('test')
    current = stage_frames(heldout,heldout_members,dict(zip(heldout.case_id,np.log(heldout.current))))
    heldout_order = sorted(set(heldout.event),key=lambda e:(data.event_index[int(e)]['start_ms'],int(e)))
    current = {p:f.reindex(heldout_order) for p,f in current.items()}
    configs += [(f'blind_current_p{p}','heldout_blind_current_only',f,{}) for p,f in current.items()]
    configs += [(f'blind_increment_p{a}_p{b}','heldout_blind_observed_growth_only',current[b]-current[a],{}) for a,b in [(25,50),(50,75),(75,90)]]
    all_rows,infos,pcs,loads,cluster_rows,available = [],[],[],[],[],[]
    for name,scope,frame,options in configs:
        rows,info,pca,loading,clusters = panel(name,scope,frame,destination,**options)
        all_rows.extend(rows);infos.append(info);pcs.extend(pca);loads.extend(loading);cluster_rows.extend(clusters)
        if 'lag_clock' not in options and 'context' not in options:
            for i,r in enumerate(RANKS):
                for s in RANKS[i+1:]:
                    f = frame.reindex(columns=[r,s]).dropna()
                    c = correlation(f.to_numpy())[0,1] if len(f)>=4 else np.nan
                    available.append({'panel':name,'rank_a':r,'rank_b':s,'n':len(f),'spearman':c})
    pairs = pd.DataFrame(all_rows)
    for field in ['p_iid','p_circular_shift']:
        valid = pairs[field].notna()
        for method in ['bh','by']:
            pairs.loc[valid,f'{field}_q_{method}_global'] = stats.false_discovery_control(pairs.loc[valid,field],method=method)
    pairs.to_csv(destination/'pairwise_correlations.csv',index=False)
    pd.DataFrame(available).to_csv(destination/'pairwise_available.csv',index=False)
    pd.DataFrame(pcs).to_csv(destination/'pca_variance.csv',index=False)
    pd.DataFrame(loads).to_csv(destination/'pca_loadings.csv',index=False)
    pd.DataFrame(cluster_rows).to_csv(destination/'clusters.csv',index=False)
    pd.DataFrame([{'panel':r['panel'],'scope':r['scope'],'common_events':r['common_events'],'supported_ranks':len(r['active_ranks']),**r['groups']} for r in infos]).to_csv(destination/'panel_summary.csv',index=False)
    pd.read_csv(DATA/'partitions.csv').groupby(['rank','split']).size().unstack(fill_value=0).reindex(RANKS).to_csv(destination/'rank_coverage.csv')
    source_files = [Path(__file__),ROOT/'tests/test_cross_rank_statistics.py',destination.parent/'analysis_plan.json',DATA/'manifest.json',
                    DATA/'train_curves.jsonl.gz',DATA/'train_cases.csv',DATA/'train_memberships.csv',DATA/'test_cases.csv',DATA/'test_memberships.csv',
                    RUN/'blind_manifest.json',residual_path,ROOT/'analysis/multiline_records/20261117/delivery_manifest.json']
    provenance = {'scope':'No validation or test labels/errors read; blind heldout features form a separate descriptive supplement. No forecasting models trained or selected.',
                  'seed':SEED,'draws':DRAWS,'bootstrap_blocks':[1,5],'partial_controls':['rank(start_ms)','rank(duration_h)','event_type','band_group'],
                  'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source_files},
                  'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','matplotlib']},
                  'limitations':['Exposed test/reused validation/retrospective metadata inherited. Blind supplement uses no outcomes but is heldout feature exploration.',
                                'Common panels drop only missing rows; pairwise-available counts and results exported separately.',
                                'Same-event correlation is not predictive causation; lag panels are separate.',
                                'Pairing permutation assumes exchangeable activities; circular shifts and block bootstrap assume approximate stationarity. Trends and structural breaks weaken these assumptions.',
                                '95% intervals are marginal, not simultaneous. Global BH/BY retained; BY does not repair invalid time-series null assumptions.',
                                'Rank PCA measures variance of standardized ranks, not explained forecast accuracy; four clusters are descriptive, not a fitted forecasting recipe.',
                                'T1000/T2000 use Context raw forecasts and other ranks use shared-rank raw forecasts; error differences can reflect base-model family.'],
                  'training_events':67,'warm_events':35,'all_rank_pairs':171,'panels':infos}
    (destination/'results.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (destination/'source.py').write_text(Path(__file__).read_text())
    print(json.dumps({'finished_panels':len(infos),'rows':len(pairs),'destination':str(destination)}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
