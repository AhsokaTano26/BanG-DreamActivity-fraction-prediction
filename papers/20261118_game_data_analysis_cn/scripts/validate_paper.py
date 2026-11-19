"""Independent source, statistical and manuscript checks; imports no analysis helpers."""
from pathlib import Path
from bisect import bisect_right
from collections import Counter
import gzip
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy import stats

P = Path(__file__).resolve().parents[1]
D = P / 'data'
CHECKS = []
RANKS = [10,20,30,40,50,100,200,300,400,500,1000,1500,2000,3000,4000,5000,10000,20000,30000]
CORE = [r for r in RANKS if r != 1500]


def check(name, condition):
    CHECKS.append({'check':name,'passed':bool(condition)})
    if not condition:
        raise AssertionError(name)


def near(name, a, b, atol=1e-9, rtol=1e-8):
    check(name, np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True))


def frame(name):
    return pd.read_csv(D/name)


def matrix(name):
    return pd.read_csv(D/name,index_col=0).to_numpy()


def main():
    manifest=json.loads((D/'source_manifest.json').read_text())
    for name,expected in manifest['snapshot_hashes'].items():
        check('Immutable source '+name,hashlib.sha256((D/name).read_bytes()).hexdigest()==expected)
    raw=gzip.open(D/'source_curves.jsonl.gz','rt').read()
    check('Canonical source record digest',hashlib.sha256(raw.encode()).hexdigest()==manifest['canonical_selected_rows_sha256'])
    records=[json.loads(line) for line in raw.splitlines()]
    meta=json.load(gzip.open(D/'source_metadata.json.gz','rt'))
    res=json.loads((D/'results.json').read_text())
    check('Source counts',len(records)==1871 and len({c['event'] for c in records})==103)
    check('Unique event and rank',len({(c['event'],c['rank']) for c in records})==len(records))
    check('Actual time policy',all(c['start_ms']>=1706504400000 and c['end_ms']<manifest['cutoff_ms'] for c in records))
    check('Actual CN metadata time alignment',all(int(meta['events'][str(c['event'])]['startAt'][3])==c['start_ms'] and int(meta['events'][str(c['event'])]['endAt'][3])==c['end_ms'] for c in records))
    check('All 19 ranks',sorted({c['rank'] for c in records})==RANKS)
    valid=[];anchors=[];quality=[];npoints=0
    for c in records:
        npoints+=len(c['points'])
        points={}
        for p in c['points']:
            t=int(p['time']);y=float(p['ep'])
            if np.isfinite(y) and y>=0 and c['start_ms']<=t<=c['end_ms']+300000:
                points[t]=y
        tt=sorted(points);yy=[points[t] for t in tt]
        ok=(c['end_ms']>c['start_ms'] and len(tt)>=5 and yy[-1]>0 and tt[-1]>=c['end_ms']-60000)
        quality.append((c['event'],c['rank'],len(tt),ok))
        if not ok:
            continue
        valid.append({'event':c['event'],'rank':c['rank'],'final':yy[-1],'points':len(tt),'decreases':sum(v<u for u,v in zip(yy,yy[1:])),
                      'median_gap_min':np.median(np.diff(tt))/60000})
        for axis,grid in [('progress',[.1,.25,.5,.75,.9,.95]),('remaining_h',[48,24,12,6,3,1,.5,.25])]:
            for v in grid:
                target=c['start_ms']+v*(c['end_ms']-c['start_ms']) if axis=='progress' else c['end_ms']-v*3600000
                if target<c['start_ms']:
                    continue
                j=bisect_right(tt,target)-1
                if j<3 or target-tt[j]>3600000 or yy[j]<=0:
                    continue
                anchors.append((c['event'],c['rank'],axis,v,int(target),tt[j],yy[j],yy[-1],1-yy[j]/yy[-1]))
    v=pd.DataFrame(valid).sort_values(['event','rank']).reset_index(drop=True)
    f=frame('qualified_curves.csv').sort_values(['event','rank']).reset_index(drop=True)
    check('Qualified curves and events',len(v)==1780 and v.event.nunique()==103)
    near('All terminal scores reconstructed from source',v[['event','rank','final']],f[['event','rank','final']],atol=0,rtol=0)
    near('All curve point counts',v.points,f.points_clean,atol=0,rtol=0)
    near('Sampling intervals independently reconstructed',v.median_gap_min,f.median_gap_min)
    check('Source and qualified point totals',npoints==1796085==res['points_raw'] and int(v.points.sum())==1718910==res['qualified_points'])
    check('Declines preserved',int((v.decreases>0).sum())==48 and int(v.decreases.sum())==50)
    a=pd.DataFrame(anchors,columns=['event','rank','axis','value','nominal_ms','observed_ms','score','final','remaining_share'])
    keys=['event','rank','axis','value'];a=a.sort_values(keys).reset_index(drop=True)
    saved=frame('aligned_observations.csv').sort_values(keys).reset_index(drop=True)
    check('All anchor keys and count',a[keys].equals(saved[keys]) and len(a)==23987)
    near('Anchor values and actual timestamps',a.drop(columns=['axis']),saved[a.columns.drop('axis')],atol=1e-7,rtol=1e-12)
    check('Negative observed growth retained',int((a.remaining_share<0).sum())==125)
    summary=frame('rank_summary.csv').set_index('rank')
    for rank,rows in v.groupby('rank'):
        row=summary.loc[rank]
        near(f'T{rank} counts and distribution',[len(rows),rows.final.median(),rows.final.quantile(.25),rows.final.quantile(.75),rows.median_gap_min.median()],
             [row.valid_events,row.median_final,row.q25_final,row.q75_final,row.median_gap_min])
    wide=v.pivot(index='event',columns='rank',values='final').reindex(columns=RANKS)
    check('No terminal adjacent rank inversion',all(np.all(np.diff(row.dropna().to_numpy())<=0) for _,row in wide.iterrows()))
    supports=np.zeros((19,19),dtype=int);rho=np.zeros((19,19))
    for i,l in enumerate(RANKS):
        for j,r in enumerate(RANKS):
            available=wide[[l,r]].dropna().to_numpy()
            supports[i,j]=len(available)
            rho[i,j]=stats.spearmanr(available[:,0],available[:,1]).statistic
    near('All 19 rank pair support counts',supports,matrix('final_pairwise_support.csv'),atol=0,rtol=0)
    near('All 19 rank correlations independently computed',rho,matrix('final_spearman_pairwise.csv'))
    core=wide[CORE].dropna()
    ranked=stats.rankdata(core,axis=0)
    eig=np.linalg.eigvalsh(np.corrcoef(ranked.T))[::-1];eig=eig/eig.sum()
    near('PCA eigenvalue versus SVD calculation',eig,frame('pca_variance.csv').variance_fraction)
    check('18 rank terminal panel count',len(core)==94)
    halfway=a[(a.axis=='progress')&(a.value==.5)].copy()
    halfway['growth']=np.log(halfway.final/halfway.score)
    growth=halfway.pivot(index='event',columns='rank',values='growth').reindex(columns=CORE).dropna()
    ids=core.index.intersection(growth.index)
    check('Identical 92 events for final versus growth',len(ids)==92)
    for values,name in [(core.loc[ids],'matched_final_spearman.csv'),(growth.loc[ids],'matched_growth_spearman.csv')]:
        near('Matched '+name,stats.spearmanr(values.to_numpy(),axis=0).statistic,matrix(name))
    em=frame('event_metadata.csv').sort_values('start_ms')
    check('Frozen cohort preserved',em.original_partition.value_counts().to_dict()=={'train':67,'test':16,'validation':14,'descriptive_extension':6})
    check('Additional descriptive events',set(em.loc[em.original_partition=='descriptive_extension','event'])=={238,280,282,310,327,328})
    check('All contexts and type composition',em.context_available.all() and em.kind.value_counts().to_dict()==res['activity_types'])
    check('Single and mixed band composition',int((em.composition=='single').sum())==75 and int((em.composition=='mixed').sum())==28)
    ee=em.set_index('event').loc[core.index]
    controls=np.column_stack([np.ones(len(ee)),ee.time_years,ee.log_duration,
        pd.get_dummies(ee.kind,drop_first=True,dtype=float),
        pd.get_dummies(ee.band,drop_first=True,dtype=float)])
    basis=np.linalg.qr(controls,mode='reduced')[0]
    ranked=stats.rankdata(core.to_numpy(),axis=0)
    adjusted=ranked-basis@(basis.T@ranked)
    near('Conditional rank correlation via QR residuals',np.corrcoef(adjusted.T),matrix('partial_spearman_common18.csv'))
    profiles=frame('matched_four_rank_growth.csv');panel_ids=json.loads((D/'matched_growth_event_ids.json').read_text())
    for axis,expected in [('progress',74),('remaining_h',86)]:
        aa=a[(a.axis==axis)&a['rank'].isin([10,1000,2000,30000])].copy()
        aa['fraction']=aa.score/aa.final
        complete=aa.pivot(index='event',columns=['rank','value'],values='fraction').dropna()
        check(axis+' complete event panel',len(complete)==expected and set(complete.index)==set(panel_ids[axis]))
        for row in profiles[profiles.axis==axis].itertuples():
            near(f'{axis} T{row.rank} at {row.value}',np.median(complete[(row.rank,row.value)]),row.median_fraction)
    yearly=frame('calendar_aligned_years.csv');rs=frame('calendar_ratio_summary.csv')
    for row in yearly.itertuples():
        z=f[(f.year==row.year)&(f['rank']==row.rank)&f.month.between(2,10)]
        near(f'Calendar matched T{row.rank} {row.year}',[len(z),z.final.median()],[row.events,row.median_final])
    for year in [2024,2025,2026]:
        ee=em[(em.year==year)&em.month.between(2,10)].sort_values('start_ms').event
        matched=wide.reindex(ee)[[1000,2000]].dropna();rat=(matched[1000]/matched[2000]).to_numpy()
        rr=rs[(rs.year==year)&(rs.numerator==1000)&(rs.denominator==2000)].iloc[0]
        rng=np.random.default_rng(42);starts=rng.integers(0,len(rat),(2000,int(np.ceil(len(rat)/5))))
        boot=[np.median([rat[(int(s)+d)%len(rat)] for s in row for d in range(5)][:len(rat)]) for row in starts]
        near(str(year)+' ratio and circular block interval',[np.median(rat),*np.quantile(boot,[.025,.975])],[rr.median_ratio,rr.block5_lower95,rr.block5_upper95])
    # Different numerical solution and a full Bartlett kernel matrix check the stored regression.
    coefs=frame('association_coefficients.csv');tests=frame('association_factor_tests.csv')
    ff=f.copy();ff['rank_label']=ff['rank'].astype(str);ff['time_squared']=ff.time_years**2
    ratioframe=frame('event_score_ratios.csv');ratioframe=ratioframe[(ratioframe.numerator==1000)&(ratioframe.denominator==2000)]
    specs=[('pooled_final',ff,'log_final',['time_years','log_duration'],True),
           ('common18_final',ff[ff.event.isin(core.index)&ff['rank'].isin(CORE)],'log_final',['time_years','log_duration'],True),
           ('quadratic_time',ff,'log_final',['time_years','time_squared','log_duration'],True),
           ('card_context',ff,'log_final',['time_years','log_duration','recent30_matched_cards'],True),
           ('ratio1000_2000',ratioframe,'log_ratio',['time_years','log_duration'],False)]
    for name,rows,response,numeric,rank_fe in specs:
        rows=rows.sort_values(['start_ms','event']+(['rank'] if rank_fe else []))
        terms=['intercept']+numeric
        cols=[np.ones(len(rows))]+[(rows[c]-rows[c].mean()).to_numpy() for c in numeric]
        categories=([('rank_label','2000')] if rank_fe else [])+[('kind','mission_live'),('band_model','mixed')]
        for col,base in categories:
            for label in sorted(rows[col].astype(str).unique()):
                if label!=base:
                    terms.append(col+'='+label);cols.append((rows[col].astype(str)==label).to_numpy(float))
        x=np.column_stack(cols);y=rows[response].to_numpy()
        w=rows.event.map(rows.event.value_counts()).to_numpy(float)**-1
        beta=np.linalg.lstsq(np.sqrt(w)[:,None]*x,np.sqrt(w)*y,rcond=None)[0]
        residual=y-x@beta
        u=pd.DataFrame(x*(w*residual)[:,None]);u['event']=rows.event.to_numpy()
        event_order=rows.event.drop_duplicates().to_numpy()
        scores=u.groupby('event').sum().loc[event_order].to_numpy()
        g,n,k=len(event_order),len(y),len(terms)
        lag=np.abs(np.subtract.outer(np.arange(g),np.arange(g)))
        kernel=np.maximum(1-lag/6,0)
        bread=np.linalg.inv(x.T@(w[:,None]*x));scale=g/(g-1)*(n-1)/(n-k)
        vc=bread@(scores.T@scores)@bread*scale
        vh=bread@(scores.T@kernel@scores)@bread*scale
        z=coefs[coefs.model==name].set_index('term').loc[terms]
        near(name+' WLS coefficients',beta,z.beta,atol=2e-9)
        near(name+' cluster covariance diagonals',np.sqrt(np.diag(vc)),z.cluster_se)
        near(name+' HAC kernel covariance diagonals',np.sqrt(np.diag(vh)),z.hac5_se)
        near(name+' marginal intervals',beta-stats.t.ppf(.975,g-1)*np.sqrt(np.diag(vh)),z.hac5_lower95)
        for test in tests[tests.model==name].itertuples():
            ix=[i for i,t in enumerate(terms) if t==test.factor or t.startswith(test.factor+'=')]
            xr=np.delete(x,ix,axis=1);br=np.linalg.lstsq(np.sqrt(w)[:,None]*xr,np.sqrt(w)*y,rcond=None)[0]
            sr=np.dot(w,(y-xr@br)**2);sf=np.dot(w,residual**2)
            near(name+' '+test.factor+' partial R2',(sr-sf)/sr,test.partial_r2)
            statistic=beta[ix]@np.linalg.solve(vh[np.ix_(ix,ix)],beta[ix])/len(ix)
            near(name+' '+test.factor+' omnibus F',statistic,test.hac5_wald_f,rtol=2e-7)
            near(name+' '+test.factor+' omnibus p',stats.f.sf(statistic,len(ix),g-1),test.p,atol=1e-140,rtol=2e-6)
    p=tests.p.to_numpy();order=np.argsort(p);m=len(p);h=sum(1/i for i in range(1,m+1))
    q=np.clip(np.minimum.accumulate((p[order]*m*h/np.arange(1,m+1))[::-1])[::-1],0,1)
    near('BY correction across all 26 factor tests',q,tests.q_by_global.to_numpy()[order],atol=1e-140)
    check('Global family size',m==26)
    # Metadata fields are reconstructed without trusting the precomputed contexts.
    chars=[];cards=[]
    for row in em.itertuples():
        cc={int(z['characterId']) for z in meta['events'][str(row.event)].get('characters',[])}
        bid={int(meta['characters'][str(c)]['bandId']) for c in cc}
        band=meta['bands'][str(next(iter(bid)))]['bandName'][3] if len(bid)==1 else 'mixed'
        check('Event '+str(row.event)+' band and membership',band==row.band and len(cc)==row.bonus_characters)
        chars.extend((row.event,c,1/len(cc)) for c in cc)
    check('Character memberships and unique characters',len(chars)==527 and len({c for e,c,w in chars})==40)
    near('Fractional character activity total',sum(w for e,c,w in chars),103)
    charcounts=Counter(c for e,c,w in chars)
    exposure=frame('character_exposure.csv').set_index('character')
    check('All 40 heatmap character counts',all(exposure.loc[c,'events']==n for c,n in charcounts.items()))
    for cid,c in meta['cards'].items():
        times=c.get('releasedAt',[])
        if len(times)>3 and times[3] and int(times[3])<=int(em.start_ms.max()):
            cards.append((int(cid),int(times[3]),int(c.get('rarity',0))))
    check('CN published card counts at last event start',len(cards)==2179 and sum(c[2]==5 for c in cards)==346)
    check('All card IDs and release timestamps',set(cards)==set(map(tuple,frame('cn_released_cards.csv')[['card','release_ms','rarity']].to_numpy())))
    published=frame('cn_released_cards.csv');monthly=frame('card_releases_complete_months.csv')
    check('All 33 complete calendar months',len(monthly)==33 and monthly.year_month.tolist()==pd.period_range('2024-02','2026-10',freq='M').astype(str).tolist())
    for row in monthly.itertuples():
        cc=published[published.year_month==row.year_month]
        check('Card release counts '+row.year_month,len(cc)==row.cards and int((cc.rarity==5).sum())==row.five_star)
    for rank,row in frame('coverage_median_comparison.csv').set_index('rank').iterrows():
        zz=wide[[1000,1500,2000]].dropna()
        near('T'+str(rank)+' own versus matched coverage',[wide[rank].count(),wide[rank].median(),len(zz),zz[rank].median()],
             [row.own_events,row.own_median,row.matched_events,row.matched_median])
    legacy=frame('legacy_terminal_scores.csv')
    near('Legacy cache marginal medians',[legacy[legacy['rank']==r].final.median()/10000 for r in [1000,1500]],[451.791,495.86485],atol=.000001)
    # Editorial checks apply to all manuscript prose, tables and number bindings.
    main=(P/'main.tex').read_text();tables='\n'.join(p.read_text() for p in (P/'tables').glob('*.tex'));text=main+'\n'+tables
    banned=['不是','而是','不能','不等于','更准确地说','并非','不等同于','不代表','不足以','不意味着','不构成','未必','尚不能','归根结底','深入探讨','促进','借力','值得注意的是','更重要的是','真正地','简而言之','最简单的思维模型']
    check('Requested professional Chinese style',not any(term in text for term in banned))
    check('Author Tano',r'\author{Tano}' in main)
    check('Native Elsevier class',r'\documentclass[preprint,12pt]{elsarticle}' in main)
    check('Seven native tables and seven vector figures',main.count(r'\input{tables/')==7 and main.count(r'\includegraphics')==7 and tables.count(r'\begin{table}')==7)
    check('Seven numbered statistical equations',main.count(r'\begin{equation}')==7)
    cited={s.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',main) for s in group.split(',')}
    bib=set(re.findall(r'@\w+\{([^,]+),',(P/'references.bib').read_text()))
    check('All eleven references cited and resolved',cited==bib and len(bib)==11)
    numbers=dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}',(P/'numbers.tex').read_text()))
    check('Fifty machine bound manuscript numbers',len(numbers)==50 and numbers==json.loads((P/'qa/manuscript_number_bindings.json').read_text()))
    bindings={
        'NumEvents':103,'NumSourceCurves':1871,'NumValidCurves':1780,'NumPoints':1796085,'NumValidPoints':1718910,'NumExcluded':91,
        'NumCoreEvents':94,'NumCorrelationEvents':92,'NumCards':2179,'NumFiveStar':346,'NumCharacterMemberships':527,'NumTests':26,
        'NumNegativeAnchors':125,'NumAnchors':23987,'NumDecliningCurves':48,'NumDeclines':50,'ProgressEvents':74,'RemainingEvents':86,'MatchedRareEvents':15,
        'PcaFirst':eig[0]*100,'PcaTwo':eig[:2].sum()*100}
    summaries=frame('correlation_summary.csv').set_index('panel')
    for panel,key in [('matched_final','RhoFinal'),('matched_growth_p50','RhoGrowth'),('partial_final_common18','RhoPartial')]:bindings[key]=summaries.loc[panel,'median_pair_correlation']
    for model,term,prefix in [('pooled_final','time_years','Time'),('pooled_final','log_duration','Duration'),('pooled_final','band_model=MyGO!!!!!','Mygo'),('card_context','recent30_matched_cards','Card')]:
        row=coefs[(coefs.model==model)&(coefs.term==term)].iloc[0]
        for suffix,col in [('Beta','beta'),('Lower','hac5_lower95'),('Upper','hac5_upper95')]:bindings[prefix+suffix]=row[col]
    for rank,suffix in [(10,'Top'),(1000,'Thousand'),(2000,'TwoThousand'),(30000,'ThirtyThousand')]:
        bindings['Half'+suffix]=float(profiles[(profiles['rank']==rank)&(profiles.axis=='progress')&(profiles.value==.5)].median_fraction.iloc[0])*100
        bindings['LateSix'+suffix]=float(profiles[(profiles['rank']==rank)&(profiles.axis=='remaining_h')&(profiles.value==6)].median_remaining_share.iloc[0])*100
    for yr,suffix in [(2024,'First'),(2025,'Second'),(2026,'Third')]:bindings['RatioYear'+suffix]=rs[(rs.year==yr)&(rs.numerator==1000)&(rs.denominator==2000)].median_ratio.iloc[0]
    sens=frame('measurement_sensitivity.csv')
    for value,key in [(6,'InterpolationSix'),(.25,'InterpolationQuarter')]:bindings[key]=sens[(sens.axis=='remaining_h')&(sens.value==value)].median_step_excess_growth_pp.iloc[0]
    for name,value in bindings.items():
        near('Number macro '+name,float(numbers[name].replace(r'\,','').replace('−','-')),float(value),atol=.000501,rtol=0)
    check('Source fingerprint bound in prose',numbers['SourceFingerprint']==manifest['canonical_selected_rows_sha256'][:12])
    abstract=main.split(r'\section*{Abstract}',1)[1].split(r'\noindent\textbf{Keywords:}',1)[0]
    abstract=abstract.split('We analyze',1)[1]
    for key,value in numbers.items():abstract=abstract.replace('\\'+key+'{}',value)
    wc=len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?",abstract))+2
    check('English abstract between 150 and 250 words',150<=wc<=250)
    result={'status':'passed','checks':len(CHECKS),'english_abstract_words':wc,'source_sha256':manifest['canonical_selected_rows_sha256'],
            'implementation':'Independent source reconstruction, SciPy pair correlations, eigenvalue PCA, weighted least squares and Bartlett kernel covariance; no analysis helper imports.',
            'results':CHECKS}
    (P/'qa/independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','checks','english_abstract_words','source_sha256']}))


if __name__=='__main__':
    main()
