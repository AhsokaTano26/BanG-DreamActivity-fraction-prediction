"""Descriptive event and rank analysis from an immutable local source snapshot."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import gzip
import json
import math
import numpy as np
import pandas as pd
from scipy import stats

P = Path(__file__).resolve().parents[1]
D = P / 'data'
HOUR = 3600000
DAY = 24 * HOUR
TZ = ZoneInfo('Asia/Shanghai')
RANKS = [10,20,30,40,50,100,200,300,400,500,1000,1500,2000,3000,4000,5000,10000,20000,30000]
CORE = [r for r in RANKS if r != 1500]
PROGRESS = [.10,.25,.50,.75,.90,.95]
REMAINING = [48,24,12,6,3,1,.5,.25]


def save(name, frame):
    frame.to_csv(D / name, index=False)


def serializable(x):
    if isinstance(x, dict): return {str(k):serializable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [serializable(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x) if math.isfinite(x) else None
    return x


def at(times, scores, when, max_age=HOUR):
    j = int(np.searchsorted(times, when, side='right') - 1)
    if j < 3 or when - times[j] > max_age or scores[j] <= 0:
        return None
    return float(scores[j]), int(times[j]), float((when-times[j])/60000)


def block_ci(values, seed=42, count=2000, block=5):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2: return [None,None]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0,len(a),(count,math.ceil(len(a)/block)))
    ix = ((starts[:,:,None]+np.arange(block))%len(a)).reshape(count,-1)[:,:len(a)]
    return np.quantile(np.median(a[ix],axis=1),[.025,.975]).tolist()


def design(frame, numeric, categorical):
    x = pd.DataFrame({'intercept':np.ones(len(frame))},index=frame.index)
    refs = {}
    for col in numeric:
        a = frame[col].astype(float)
        if a.std() > 1e-10: x[col] = a - a.mean()
    for col, preferred in categorical.items():
        labels = sorted(frame[col].astype(str).unique())
        ref = preferred if preferred in labels else frame[col].astype(str).value_counts().index[0]
        refs[col] = ref
        for label in labels:
            if label != ref: x[col+'='+label] = (frame[col].astype(str)==label).astype(float)
    return x, refs


def fit_model(frame, outcome, name, numeric, categorical):
    frame = frame.sort_values(['start_ms','event','rank'] if 'rank' in frame else ['start_ms','event']).copy()
    x, refs = design(frame, numeric, categorical)
    a=x.to_numpy();y=frame[outcome].to_numpy(float)
    counts=frame.groupby('event').event.transform('size').to_numpy();w=1/counts
    bread=np.linalg.pinv(a.T@(w[:,None]*a));beta=bread@(a.T@(w*y));res=y-a@beta
    order=frame.drop_duplicates('event').event.to_numpy();g=len(order);n=len(y);k=np.linalg.matrix_rank(a)
    assert k==a.shape[1] and g>k, (name,n,g,k,a.shape[1])
    sums=np.stack([a[frame.event.to_numpy()==e].T@(w[frame.event.to_numpy()==e]*res[frame.event.to_numpy()==e]) for e in order])
    scale=g/(g-1)*(n-1)/(n-k)
    meat=sums.T@sums
    cov=bread@meat@bread*scale
    hacmeat=meat.copy()
    for lag in range(1,6):
        cross=sums[lag:].T@sums[:-lag]
        hacmeat+=(1-lag/6)*(cross+cross.T)
    hac=bread@hacmeat@bread*scale
    critical=stats.t.ppf(.975,g-1)
    se=np.sqrt(np.maximum(0,np.diag(cov)));hse=np.sqrt(np.maximum(0,np.diag(hac)))
    coef=pd.DataFrame({'model':name,'term':x.columns,'beta':beta,'cluster_se':se,
        'cluster_lower95':beta-critical*se,'cluster_upper95':beta+critical*se,'hac5_se':hse,
        'hac5_lower95':beta-critical*hse,'hac5_upper95':beta+critical*hse})
    sse=float(np.sum(w*res**2));sst=float(np.sum(w*(y-np.average(y,weights=w))**2))
    groups={col:[j for j,t in enumerate(x.columns) if t==col or t.startswith(col+'=')] for col in list(numeric)+list(categorical)}
    tests=[]
    for factor, ix in groups.items():
        if not ix:continue
        c=hac[np.ix_(ix,ix)];df=np.linalg.matrix_rank(c)
        f=float(beta[ix]@np.linalg.pinv(c)@beta[ix]/df)
        keep=[j for j in range(a.shape[1]) if j not in ix]
        ar=a[:,keep];b=np.linalg.lstsq(ar*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        reduced=float(np.sum(w*(y-ar@b)**2))
        tests.append({'model':name,'factor':factor,'df':int(df),'hac5_wald_f':f,
            'p':float(stats.f.sf(f,df,g-1)),'partial_r2':float((reduced-sse)/reduced)})
    info={'model':name,'rows':n,'events':g,'columns':len(beta),'matrix_rank':int(k),'r2':1-sse/sst,
          'references':refs,'event_ids':[int(e) for e in order],'condition_number':float(np.linalg.cond(a*np.sqrt(w[:,None]))),
          'uncertainty':'Event cluster CR1; event score Bartlett HAC with lag 5, same finite sample scale; t/F G-1 df; exploratory.'}
    return coef, tests, info


def corr_panels(final, anchors, event):
    table=final.pivot(index='event',columns='rank',values='final').reindex(columns=RANKS)
    table=table.reindex(event.event)
    rho=table.corr(method='spearman',min_periods=8)
    rho.to_csv(D/'final_spearman_pairwise.csv')
    support=table.notna().astype(int).T@table.notna().astype(int)
    support.to_csv(D/'final_pairwise_support.csv')
    core=table[CORE].dropna()
    core.to_csv(D/'common18_final_scores.csv')
    z=stats.rankdata(core.to_numpy(),axis=0);z=(z-z.mean(0))/z.std(0,ddof=1)
    _,s,v=np.linalg.svd(z,full_matrices=False);variance=s*s/np.sum(s*s)
    save('pca_variance.csv',pd.DataFrame({'component':np.arange(1,len(s)+1),'variance_fraction':variance,'cumulative_fraction':variance.cumsum()}))
    save('pca_loadings.csv',pd.DataFrame({'rank':CORE,'PC1':v[0]*np.sign(np.median(v[0])),'PC2':v[1]}))
    a=anchors[(anchors.axis=='progress') & (anchors.value==.5)].pivot(index='event',columns='rank',values='log_remaining').reindex(columns=CORE)
    a=a.reindex(event.event).dropna();common=core.index.intersection(a.index)
    matched=core.loc[common];growth=a.loc[common]
    r1=matched.corr(method='spearman');r2=growth.corr(method='spearman')
    r1.to_csv(D/'matched_final_spearman.csv');r2.to_csv(D/'matched_growth_spearman.csv')
    growth.to_csv(D/'common18_growth_p50.csv')
    e=event.set_index('event').loc[core.index]
    x,_=design(e,['time_years','log_duration'],{'kind':'mission_live','band_model':'mixed'})
    ranks=stats.rankdata(core.to_numpy(),axis=0)
    partial=ranks-x.to_numpy()@np.linalg.lstsq(x.to_numpy(),ranks,rcond=None)[0]
    pd.DataFrame(np.corrcoef(partial.T),index=CORE,columns=CORE).to_csv(D/'partial_spearman_common18.csv')
    def med(r):return float(np.median(r.to_numpy()[np.triu_indices(len(r),1)]))
    summaries=[{'panel':'all_pairwise_final','events':len(table.dropna(how='all')),'ranks':19,'median_pair_correlation':med(rho)},
        {'panel':'common18_final','events':len(core),'ranks':18,'median_pair_correlation':med(core.corr(method='spearman'))},
        {'panel':'matched_final','events':len(common),'ranks':18,'median_pair_correlation':med(r1)},
        {'panel':'matched_growth_p50','events':len(common),'ranks':18,'median_pair_correlation':med(r2)},
        {'panel':'partial_final_common18','events':len(core),'ranks':18,'median_pair_correlation':float(np.median(np.corrcoef(partial.T)[np.triu_indices(18,1)]))}]
    save('correlation_summary.csv',pd.DataFrame(summaries))
    pairs=[]
    for left,right in [(10,20),(40,50),(200,300),(400,500),(500,1000),(1000,1500),(1000,2000),(1500,2000),(4000,5000),(10000,20000)]:
        f=table[[left,right]].dropna();r=stats.spearmanr(f.iloc[:,0],f.iloc[:,1]).statistic
        pairs.append({'left':left,'right':right,'events':len(f),'final_spearman':r})
    save('representative_pairs.csv',pd.DataFrame(pairs))
    return {'common18_events':len(core),'matched_growth_events':len(common),'rank_pca_first':float(variance[0]),'rank_pca_first_two':float(variance[:2].sum()),'summaries':summaries}


def main():
    meta=json.load(gzip.open(D/'source_metadata.json.gz','rt'))
    contexts={int(x['event_id']):x for x in meta['event_contexts']}
    cards=[]
    for cid,c in meta['cards'].items():
        values=c.get('releasedAt',[]);release=values[3] if len(values)>3 else None
        if release:
            character=meta['characters'].get(str(c.get('characterId')), {})
            cards.append((int(cid),int(release),int(c.get('characterId',0)),int(character.get('bandId',0)),int(c.get('rarity',0))))
    curves=[];quality=[];events={};anchors=[];cleaned=[];raw_duplicates=0;invalid_points=0;late_points=0;characters=[]
    for line in gzip.open(D/'source_curves.jsonl.gz','rt'):
        c=json.loads(line);eid=c['event'];r=c['rank'];start=c['start_ms'];end=c['end_ms']
        if eid not in events:
            context=contexts.get(eid,{});chars=context.get('characters',[])
            shares=context.get('band_shares',{})
            band_key=next(iter(shares)) if len(shares)==1 else None
            band_name=context.get('band_names',{}).get(band_key,'unknown') if band_key is not None else 'mixed'
            dt=datetime.fromtimestamp(start/1000,TZ)
            available=[a for a in cards if a[1]<=start]
            recent=[a for a in available if start-30*DAY<=a[1] and a[2] in chars]
            events[eid]={'event':eid,'start_ms':start,'end_ms':end,'start_date':dt.date().isoformat(),'year':dt.year,'month':dt.month,
                'duration_h':(end-start)/HOUR,'log_duration':np.log((end-start)/HOUR),
                'time_years':(start-1706504400000)/(365.25*DAY),'kind':c['kind'],
                'band':band_name,'composition':context.get('band_group','unknown'),'context_available':bool(context),'bonus_characters':len(chars),
                'available_cn_cards':len(available),'available_cn_five_star':sum(a[4]==5 for a in available),
                'recent30_matched_cards':len(recent),'recent30_matched_five_star':sum(a[4]==5 for a in recent)}
            for char in chars:
                character=meta['characters'].get(str(char),{});names=character.get('characterName',[])
                characters.append({'event':eid,'character':int(char),'name':names[3] if len(names)>3 and names[3] else str(char),
                    'band_id':character.get('bandId'),'fractional_event_weight':1/len(chars)})
        seen={};bad=0;late=0
        for point in c['points']:
            try:
                t=int(point['time']);v=float(point['ep'])
                if not np.isfinite(v) or v<0 or t<start:raise ValueError()
                if t>end+300000:
                    late+=1
                    continue
                seen[t]=v
            except (ValueError,TypeError,KeyError,OverflowError):bad+=1
        raw_duplicates+=len(c['points'])-bad-late-len(seen);invalid_points+=bad;late_points+=late
        times=np.array(sorted(seen),dtype=np.int64);scores=np.array([seen[t] for t in times])
        reason='valid'
        if end<=start:reason='invalid_duration'
        elif len(times)<5:reason='fewer_than_5_points'
        elif scores[-1]<=0:reason='nonpositive_terminal'
        elif not -300000<=end-int(times[-1])<=60000:reason='terminal_outside_window'
        row={'event':eid,'rank':r,'points_raw':len(c['points']),'points_clean':len(times),'invalid_points':bad,
             'after_end_window_points':late,'duplicate_times':len(c['points'])-bad-late-len(times),'score_decreases':int((np.diff(scores)<0).sum()),
             'median_gap_min':float(np.median(np.diff(times))/60000) if len(times)>1 else np.nan,
             'last_offset_min':float((times[-1]-end)/60000) if len(times) else np.nan,'quality':reason}
        quality.append(row)
        if reason!='valid':continue
        f=float(scores[-1]);row={**events[eid],**row,'final':f,'log_final':np.log(f)}
        curves.append(row)
        cleaned.append({'event':eid,'rank':r,'times':times.tolist(),'scores':scores.tolist(),'final':f})
        for axis, grid in [('progress',PROGRESS),('remaining_h',REMAINING)]:
            for v in grid:
                when=start+v*(end-start) if axis=='progress' else end-v*HOUR
                if when<start:continue
                value=at(times,scores,when)
                if value is None:continue
                score,u,age=value
                anchors.append({'event':eid,'rank':r,'axis':axis,'value':v,'nominal_ms':int(when),'observed_ms':u,
                    'lag_min':age,'score':score,'final':f,'fraction_final':score/f,'remaining_share':1-score/f,
                    'log_remaining':np.log(f/score),'year':events[eid]['year']})
    e=pd.DataFrame(events.values()).sort_values('start_ms');q=pd.DataFrame(quality);f=pd.DataFrame(curves);a=pd.DataFrame(anchors)
    supports=e.band.value_counts();e['band_model']=e.band.where(e.band.map(supports)>=5,'other')
    f=f.merge(e[['event','band_model']],on='event',validate='many_to_one')
    partition=pd.read_csv(D/'original_partitions.csv').drop_duplicates('event').set_index('event')['split'].to_dict()
    e['original_partition']=e.event.map(partition).fillna('descriptive_extension')
    e['valid_curves']=e.event.map(f.groupby('event').size()).fillna(0).astype(int)
    save('event_metadata.csv',e);save('curve_quality.csv',q);save('qualified_curves.csv',f);save('aligned_observations.csv',a)
    with gzip.open(D/'cleaned_qualified_curves.jsonl.gz','wt') as stream:
        for c in cleaned:stream.write(json.dumps(c,separators=(',',':'))+'\n')
    coverage=[]
    for rank in RANKS:
        x=f[f['rank']==rank];raw=q[q['rank']==rank]
        coverage.append({'rank':rank,'source_curves':len(raw),'valid_events':x.event.nunique(),'median_final':x.final.median(),
            'q25_final':x.final.quantile(.25),'q75_final':x.final.quantile(.75),'min_final':x.final.min(),'max_final':x.final.max(),
            'median_points':x.points_clean.median(),'median_gap_min':x.median_gap_min.median(),'decreasing_curves':int((x.score_decreases>0).sum())})
    save('rank_summary.csv',pd.DataFrame(coverage))
    profile=a.groupby(['axis','value','rank']).agg(events=('event','nunique'),median_fraction=('fraction_final','median'),
        q25_fraction=('fraction_final',lambda x:x.quantile(.25)),q75_fraction=('fraction_final',lambda x:x.quantile(.75)),
        median_remaining_share=('remaining_share','median'),median_lag_min=('lag_min','median')).reset_index()
    save('growth_profiles.csv',profile)
    # Equal event weight within each time condition, then summarize across events.
    macro=a.groupby(['axis','value','event']).agg(mean_fraction=('fraction_final','mean'),mean_remaining_share=('remaining_share','mean'),ranks=('rank','nunique')).reset_index()
    summary=macro.groupby(['axis','value']).agg(events=('event','nunique'),median_event_fraction=('mean_fraction','median'),
        median_event_remaining_share=('mean_remaining_share','median'),min_ranks=('ranks','min'),max_ranks=('ranks','max')).reset_index()
    save('event_macro_growth.csv',summary)
    fy=f.merge(e[['event','original_partition']],on='event',validate='many_to_one')
    aligned=fy[fy.month.between(2,10)]
    yearly=aligned.groupby(['rank','year']).agg(events=('event','nunique'),median_final=('final','median'),q25_final=('final',lambda x:x.quantile(.25)),q75_final=('final',lambda x:x.quantile(.75))).reset_index()
    save('calendar_aligned_years.csv',yearly)
    yearly_all=fy.groupby(['rank','year']).agg(events=('event','nunique'),median_final=('final','median')).reset_index()
    save('all_years_unbalanced.csv',yearly_all)
    wide=f.pivot(index='event',columns='rank',values='final').reindex(columns=RANKS)
    ratio=[]
    for left,right in [(10,1000),(500,1000),(1000,2000),(2000,10000),(10000,30000)]:
        subset=wide[[left,right]].dropna()
        for eid,row in subset.iterrows():ratio.append({'event':eid,'numerator':left,'denominator':right,'ratio':row[left]/row[right],'log_ratio':np.log(row[left]/row[right])})
    ratios=pd.DataFrame(ratio).merge(e,on='event',validate='many_to_one');save('event_score_ratios.csv',ratios)
    rs=[]
    for (l,r,year),x in ratios[ratios.month.between(2,10)].groupby(['numerator','denominator','year']):
        x=x.sort_values('start_ms');lo,hi=block_ci(x.ratio)
        rs.append({'numerator':l,'denominator':r,'year':year,'events':len(x),'median_ratio':x.ratio.median(),'q25_ratio':x.ratio.quantile(.25),'q75_ratio':x.ratio.quantile(.75),'block5_lower95':lo,'block5_upper95':hi})
    save('calendar_ratio_summary.csv',pd.DataFrame(rs))
    inverted=[]
    for eid,row in wide.iterrows():
        have=row.dropna().sort_index()
        for (l,v1),(r,v2) in zip(have.items(),list(have.items())[1:]):
            if v1<v2:inverted.append({'event':eid,'higher_rank':l,'lower_rank':r,'higher_score':v1,'lower_score':v2})
    save('terminal_rank_inversions.csv',pd.DataFrame(inverted,columns=['event','higher_rank','lower_rank','higher_score','lower_score']))
    for col,filename in [('kind','activity_type_summary.csv'),('band','band_summary.csv')]:
        rows=[]
        for value,x in e.groupby(col):
            ids=x.event;z=f[f.event.isin(ids)]
            rows.append({col:value,'events':len(x),'qualified_events':int((x.valid_curves>0).sum()),'duration_median_h':x.duration_h.median(),
                't1000_events':z[z['rank']==1000].event.nunique(),'t1000_median':z.loc[z['rank']==1000,'final'].median(),
                't2000_median':z.loc[z['rank']==2000,'final'].median(),'first_date':x.start_date.min(),'last_date':x.start_date.max()})
        save(filename,pd.DataFrame(rows))
    ch=pd.DataFrame(characters);save('event_characters.csv',ch)
    save('character_exposure.csv',ch.groupby(['character','name','band_id']).agg(events=('event','nunique'),fractional_events=('fractional_event_weight','sum')).reset_index().sort_values('events',ascending=False))
    # Association models share all included rows; no predictive model selection.
    f['rank_label']=f['rank'].astype(str);f['time_squared']=f.time_years**2
    cats={'rank_label':'2000','kind':'mission_live','band_model':'mixed'}
    coefs=[];tests=[];models=[]
    specifications=[(f,'log_final','pooled_final',['time_years','log_duration'],cats),
        (f[f.event.isin(wide.reindex(columns=CORE).dropna().index)&f['rank'].isin(CORE)],'log_final','common18_final',['time_years','log_duration'],cats),
        (f,'log_final','quadratic_time',['time_years','time_squared','log_duration'],cats),
        (f,'log_final','card_context',['time_years','log_duration','recent30_matched_cards'],cats)]
    rat=ratios[(ratios.numerator==1000)&(ratios.denominator==2000)].copy()
    specifications.append((rat,'log_ratio','ratio1000_2000',['time_years','log_duration'],{'kind':'mission_live','band_model':'mixed'}))
    for frame,outcome,name,numeric,categorical in specifications:
        coef,test,info=fit_model(frame,outcome,name,numeric,categorical);coefs.append(coef);tests+=test;models.append(info)
    tests=pd.DataFrame(tests);tests['q_by_global']=stats.false_discovery_control(tests.p.to_numpy(),method='by')
    save('association_coefficients.csv',pd.concat(coefs,ignore_index=True));save('association_factor_tests.csv',tests)
    corr=corr_panels(f,a,e)
    # Primary effect rescaling retains the original log coefficient and uncertainty.
    (D/'models.json').write_text(json.dumps(serializable(models),ensure_ascii=False,indent=2)+'\n')
    source=json.loads((D/'source_manifest.json').read_text())
    result={'source_events':len(e),'source_curves':len(q),'qualified_curves':len(f),'qualified_events':f.event.nunique(),
        'points_raw':int(q.points_raw.sum()),'qualified_points':int(f.points_clean.sum()),'duplicate_timestamp_points':raw_duplicates,'invalid_points':invalid_points,'after_end_window_points':late_points,
        'exclusions':q.quality.value_counts().to_dict(),'score_decrease_curves':int((f.score_decreases>0).sum()),'score_decrease_points':int(f.score_decreases.sum()),
        'terminal_rank_inversions':len(inverted),'rank_inversion_events':len(set(x['event'] for x in inverted)),
        'first_start':e.start_date.min(),'last_start':e.start_date.max(),'last_end':datetime.fromtimestamp(e.end_ms.max()/1000,TZ).isoformat(),
        'duration_median_h':e.duration_h.median(),'duration_range_h':[e.duration_h.min(),e.duration_h.max()],
        'activity_types':e.kind.value_counts().to_dict(),'bands':e.band.value_counts().to_dict(),'compositions':e.composition.value_counts().to_dict(),'all_contexts_present':bool(e.context_available.all()),
        'band_model_counts':e.band_model.value_counts().to_dict(),'original_partition_counts':e.original_partition.value_counts().to_dict(),
        'descriptive_extension_events':e.loc[e.original_partition=='descriptive_extension','event'].tolist(),
        'calendar_matched_event_counts':e[e.month.between(2,10)].groupby('year').size().to_dict(),
        'correlations':corr,'model_families':models,'factor_tests_global_family':len(tests),
        'negative_remaining_shares':int((a.remaining_share<0).sum()),'anchors':len(a),
        'metadata_file_times':source.get('metadata_file_mtimes',{}),'metadata_refresh_records':source.get('metadata_state',[]),
        'metadata_dictionary_counts':{k:len(meta[k]) for k in ['events','characters','bands','cards']},
        'observed_bonus_characters':ch.character.nunique(),'character_event_memberships':len(ch),'fractional_character_events':ch.fractional_event_weight.sum(),
        'cards_cn_released_by_last_start':int(e.available_cn_cards.iloc[-1]),'cards_five_star_by_last_start':int(e.available_cn_five_star.iloc[-1]),
        'source_row_fingerprint':source['canonical_selected_rows_sha256'],
        'scope':'Descriptive retrospective local CN ranking score study; no DAU, payer, revenue or individual effort estimates; no prediction benchmark; cohorts remain unchanged.'}
    (D/'results.json').write_text(json.dumps(serializable(result),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:serializable(result[k]) for k in ['source_events','source_curves','qualified_curves','qualified_events','exclusions','activity_types','original_partition_counts','correlations']},ensure_ascii=False))


if __name__ == '__main__':
    main()
