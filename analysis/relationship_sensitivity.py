"""Train-only robustness checks; auxiliary reports do not change model selection."""
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import SplineTransformer
from analysis.holdout_dataset import HoldoutDataset,write_json,sha256
from analysis.event_features import EventFeatures
from analysis.relationship_features import relationship_frame
from analysis.relationship_statistics import design,clustered_ols,association_tests,peer_interactions


def main():
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run
    destination=root/'sensitivity_and_ablation.json'
    if destination.exists():raise FileExistsError(destination)
    data=HoldoutDataset();meta=EventFeatures.load(data.metadata_directory)
    f=pd.read_csv(root/'training_event_statistics.csv');result={}
    for name,frame in [('year_fixed_effect',f),('exclude_rare_bands_year_fixed',f[f.band.map(f.groupby('band').event.nunique())>=5])]:
        x=design(frame,['log_duration'],['rank','type','band','year']);g=frame.event.to_numpy()
        coef,cov,info=clustered_ols(x,frame.log_final,g,1/frame.groupby('event').event.transform('size').to_numpy())
        result[name]=info|{'tests':association_tests(coef,cov,{'band':['band_'],'type':['type_'],'duration':['log_duration'],'year':['year_']},info['events'])}
    coverage={}
    for split in ['train','validation']:
        samples=data.inputs(split);frame=relationship_frame(samples,meta)
        coverage[split]={'peer_available_fraction':float(frame.peer_available.mean()),
            'median_prefix_points':float(np.expm1(frame.quality_log_points).median()),
            'median_gap_h':float(frame.quality_median_gap_h.median()),
            'events_with_peer':len({s['curve'].event for s,a in zip(samples,frame.peer_available) if a})}
        if split=='train':train=samples;prefix=frame.copy()
    prefix['year']=[datetime.fromtimestamp(s['curve'].start/1000).year for s in train]
    prefix['log_duration']=np.log(prefix.duration_h);prefix['rank1000']=(prefix['rank']=='1000').astype(float)
    basis=SplineTransformer(n_knots=5,degree=3,include_bias=False).fit_transform(prefix[['progress']])
    for i in range(basis.shape[1]):prefix[f'phase_{i}']=basis[:,i];prefix[f'rankphase_{i}']=basis[:,i]*prefix.rank1000
    prefix=peer_interactions(prefix)
    numerical=['log_current','log_duration','rate_6h_relative','rate_24h_relative','quality_median_gap_h','quality_max_gap_h',
               'peer_available','gap_log_1000_2000','gap_change_24h','peer_rate_24h','rank_gap','rank_gapchange']+[c for c in prefix if c.startswith(('phase_','rankphase_'))]
    x=design(prefix,numerical,['rank','event_type','band_group','year']);labels=data.labels('train')
    target=np.array([np.log(labels[s['case_id']]/s['score'])+np.log(s['progress']) for s in train]);groups=np.array([s['curve'].event for s in train])
    counts=pd.Series(groups).value_counts();w=np.array([1/counts[g] for g in groups])
    coef,cov,info=clustered_ols(x,target,groups,w)
    result['growth_year_fixed_effect']=info|{'tests':association_tests(coef,cov,{'band':['band_group_'],'type':['event_type_'],
        'duration':['log_duration'],'year':['year_'],'quality':['quality_'],'peer':['peer_','gap_','rank_gap'],
        'phase':['phase_','rankphase_'],'own_velocity':['rate_'],'scale':['log_current']},info['events'])}
    cv=pd.read_csv(root/'training_cv_predictions.csv');cv['ape']=abs(cv.predicted/cv.actual-1)*100
    errors=cv.groupby(['method','event']).ape.mean().unstack('method');rng=np.random.default_rng(42);draw=rng.integers(0,len(errors),(5000,len(errors)))
    ablation={}
    for method in errors:
        diff=(errors[method]-errors['full']).to_numpy()
        ablation[method]={'training_oof_event_mape_pct':float(errors[method].mean()),'minus_full_mape_points':float(diff.mean()),
                          'paired_event_bootstrap95':np.quantile(diff[draw].mean(axis=1),[.025,.975]).tolist()}
    paired=f.pivot(index='event',columns='rank',values='log_final').dropna()
    result.update({'training_time_cv_ablation':ablation,'coverage':coverage,
        'paired_spearman':float(stats.spearmanr(paired.iloc[:,0],paired.iloc[:,1]).statistic),
        'band_training_event_counts':f.groupby('band').event.nunique().to_dict(),
        'source_sha256':sha256(__file__),'dataset_id':data.manifest['dataset_id']})
    write_json(destination,result)
    (root/'sensitivity_source.py').write_text(Path(__file__).read_text())
    print('Sensitivity and ablation evidence saved; test labels not accessed.')


if __name__=='__main__':main()
