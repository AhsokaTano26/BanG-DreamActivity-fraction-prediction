"""Measurement sensitivity and association influence checks, without prediction fitting."""
from pathlib import Path
import gzip,json
import numpy as np
import pandas as pd
from analyze_game import design

P=Path(__file__).resolve().parents[1];D=P/'data'


def main():
    final=pd.read_csv(D/'qualified_curves.csv');anchors=pd.read_csv(D/'aligned_observations.csv')
    wide=final.pivot(index='event',columns='rank',values='final')
    matched=wide[[1000,1500,2000]].dropna()
    rows=[]
    for r in [1000,1500,2000]:
        x=wide[r].dropna()
        rows.append({'rank':r,'own_events':len(x),'own_median':x.median(),
                     'matched_events':len(matched),'matched_median':matched[r].median()})
    pd.DataFrame(rows).to_csv(D/'coverage_median_comparison.csv',index=False)
    matched.to_csv(D/'t1500_matched_scores.csv')
    final.groupby('rank').start_date.agg(['min','max']).reset_index().to_csv(D/'rank_date_coverage.csv',index=False)
    # Keep the same events and four ranks across the entire grid of each time axis.
    profiles=[];grid_events={}
    for axis,x in anchors[anchors['rank'].isin([10,1000,2000,30000])].groupby('axis'):
        panel=x.pivot(index='event',columns=['rank','value'],values='fraction_final').dropna()
        grid_events[axis]=[int(e) for e in panel.index]
        for (rank,value),values in panel.items():
            profiles.append({'axis':axis,'rank':rank,'value':value,'events':len(panel),
                'median_fraction':values.median(),'q25_fraction':values.quantile(.25),'q75_fraction':values.quantile(.75),
                'median_remaining_share':1-values.median()})
    pd.DataFrame(profiles).to_csv(D/'matched_four_rank_growth.csv',index=False)
    (D/'matched_growth_event_ids.json').write_text(json.dumps(grid_events,indent=2)+'\n')
    curves={}
    for line in gzip.open(D/'cleaned_qualified_curves.jsonl.gz','rt'):
        c=json.loads(line);curves[(c['event'],c['rank'])]=(np.array(c['times']),np.array(c['scores']))
    values=[]
    for row in anchors.itertuples():
        t,y=curves[(row.event,row.rank)];when=row.nominal_ms;j=int(np.searchsorted(t,when,side='right')-1)
        val=np.nan;span=np.nan
        if j>=0 and j+1<len(t) and t[j+1]-t[j]<=3600000:
            val=y[j]+(y[j+1]-y[j])*(when-t[j])/(t[j+1]-t[j]);span=(t[j+1]-t[j])/60000
        values.append({'event':row.event,'rank':row.rank,'axis':row.axis,'value':row.value,'step_fraction':row.fraction_final,
                       'linear_fraction':val/row.final,'span_min':span,'lag_min':row.lag_min})
    comp=pd.DataFrame(values);comp['growth_difference_pp']=(comp.linear_fraction-comp.step_fraction)*100
    comp.to_csv(D/'measurement_interpolation.csv',index=False)
    summaries=[]
    for (axis,value),x in comp.dropna(subset=['linear_fraction']).groupby(['axis','value']):
        per=x.groupby('event').growth_difference_pp.mean()
        summaries.append({'axis':axis,'value':value,'events':len(per),'cases':len(x),
            'median_step_excess_growth_pp':per.median(),'q25_step_excess_growth_pp':per.quantile(.25),
            'q75_step_excess_growth_pp':per.quantile(.75),'median_lag_min':x.lag_min.median()})
    pd.DataFrame(summaries).to_csv(D/'measurement_sensitivity.csv',index=False)
    f=final.sort_values(['start_ms','event','rank']).copy();f['rank_label']=f['rank'].astype(str)
    x,_=design(f,['time_years','log_duration'],{'rank_label':'2000','kind':'mission_live','band_model':'mixed'})
    a=x.to_numpy();y=f.log_final.to_numpy();groups=f.event.to_numpy();w=1/f.groupby('event').event.transform('size').to_numpy()
    gram=a.T@(w[:,None]*a);rhs=a.T@(w*y)
    out=[]
    for e in f.event.drop_duplicates():
        mask=groups==e;g=gram-a[mask].T@(w[mask,None]*a[mask]);h=rhs-a[mask].T@(w[mask]*y[mask])
        assert np.linalg.matrix_rank(g)==g.shape[0]
        b=np.linalg.solve(g,h)
        for term,value in zip(x.columns,b):
            if term in ['time_years','log_duration'] or term.startswith(('kind=','band_model=')):
                out.append({'omitted_event':e,'term':term,'beta':value})
    loo=pd.DataFrame(out);loo.to_csv(D/'leave_one_event_out.csv',index=False)
    loo.groupby('term').beta.agg(['min','max','median']).reset_index().to_csv(D/'coefficient_influence_ranges.csv',index=False)
    # Observable terminal digit patterns are supplementary descriptions only.
    digits=[]
    for rank,x in final.groupby('rank'):
        integer=x.final.to_numpy().astype(np.int64)
        digits.append({'rank':rank,'events':len(x),'ends_00_fraction':float((integer%100==0).mean()),
                       'ends_000_fraction':float((integer%1000==0).mean()),'unique_finals':x.final.nunique()})
    pd.DataFrame(digits).to_csv(D/'terminal_digit_descriptions.csv',index=False)
    # Card release counts are metadata availability, not collection or purchase records.
    meta=json.load(gzip.open(D/'source_metadata.json.gz','rt'))
    from datetime import datetime
    from zoneinfo import ZoneInfo
    lower=1706504400000;upper=int(pd.read_csv(D/'event_metadata.csv').start_ms.max());cardrows=[]
    for cid,c in meta['cards'].items():
        t=c.get('releasedAt',[]);release=t[3] if len(t)>3 else None
        if release and int(release)<=upper:
            dt=datetime.fromtimestamp(int(release)/1000,ZoneInfo('Asia/Shanghai'))
            cardrows.append({'card':int(cid),'release_ms':int(release),'release_date':dt.date().isoformat(),
                'year':dt.year,'month':dt.month,'year_month':dt.strftime('%Y-%m'),'rarity':c.get('rarity'),
                'character':c.get('characterId'),'during_study':int(release)>=lower})
    cardframe=pd.DataFrame(cardrows);cardframe.to_csv(D/'cn_released_cards.csv',index=False)
    monthly=cardframe[cardframe.year_month.between('2024-02','2026-10')].copy()
    monthly['five_star']=(monthly.rarity==5).astype(int)
    monthly.groupby('year_month').agg(cards=('card','size'),five_star=('five_star','sum')).reset_index().to_csv(D/'card_releases_complete_months.csv',index=False)
    print('Coverage comparison:',rows)
    print('Last 6h interpolation sensitivity:',[x for x in summaries if x['axis']=='remaining_h' and x['value']==6])


if __name__=='__main__':main()
