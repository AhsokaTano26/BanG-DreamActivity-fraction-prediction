"""Dimensionless dynamics, clock-to-end and sampling-quality features; prefixes only."""
from datetime import datetime
import numpy as np
import pandas as pd
from analysis.backtest import TZ, snapshot


def relationship_frame(samples, metadata):
    rows=[]
    for s in samples:
        c=s['curve'];f=s['features'].copy();t=s['times'];v=s['scores'];now=s['now']
        f.update(metadata.band_only(c))
        f.pop('type_band',None)
        end=datetime.fromtimestamp(c.end/1000,TZ)
        f.update({'log_remaining_h':np.log1p(max(0.,f['remaining_h'])),
                  'end_weekday':str(end.weekday()),'end_hour':end.hour,
                  'quality_log_points':np.log1p(len(t)),
                  'quality_median_gap_h':np.median(np.diff(t))/3600000,
                  'quality_max_gap_h':np.max(np.diff(t))/3600000,
                  'quality_coverage':(now-t[0])/max(1,now-c.start),
                  'quality_age_h':(s['cutoff']-now)/3600000})
        rates={}
        for h in [1,3,6,12,24,48]:
            earlier=now-h*3600000
            rates[h]=max(0.,(s['score']-np.interp(earlier,t,v))/h) if earlier>=t[0] else np.nan
        for a,b in [(1,6),(3,12),(6,24),(12,48)]:
            f[f'momentum_{a}_{b}']=np.log((rates[a]+1)/(rates[b]+1)) if np.isfinite(rates[a]+rates[b]) else np.nan
        for h in [6,24]:
            f[f'velocity_growth_{h}h']=np.log1p(rates[h]*max(0.,f['remaining_h'])/s['score'])
        peer=s.get('peer_curve');ps=snapshot(peer,now) if peer is not None else None
        f['peer_available']=float(ps is not None)
        for name in ['gap_log_1000_2000','gap_change_6h','gap_change_24h','peer_rate_6h','peer_rate_24h','peer_progress']:
            f[name]=np.nan
        if ps is not None:
            sign=1 if c.rank==1000 else -1
            gap=sign*np.log(s['score']/ps['score'])
            f['gap_log_1000_2000']=gap
            f['peer_progress']=ps['progress']
            for h in [6,24]:
                f[f'peer_rate_{h}h']=ps['features'][f'rate_{h}h_relative']
                earlier=ps['now']-h*3600000
                if earlier>=max(t[0],ps['times'][0]):
                    oldgap=sign*np.log(max(1.,np.interp(earlier,t,v))/max(1.,np.interp(earlier,ps['times'],ps['scores'])))
                    # Compare contemporaneous curves at peer observation time.
                    currentgap=sign*np.log(max(1.,np.interp(ps['now'],t,v))/ps['score'])
                    f[f'gap_change_{h}h']=currentgap-oldgap
        rows.append(f)
    return pd.DataFrame(rows)


def select_features(frame, variant):
    drop=[]
    for column in frame:
        band=column.startswith('band_') or column in ['character_count','mixed_band']
        peer=column.startswith(('peer_','gap_'))
        quality=column.startswith('quality_')
        if variant in ['no_band','invariant','invariant_rank'] and band:drop.append(column)
        if variant=='no_peer' and peer:drop.append(column)
        if variant=='no_type' and column=='event_type':drop.append(column)
        if variant=='no_quality' and quality:drop.append(column)
        if variant in ['invariant','invariant_rank'] and column=='log_current':drop.append(column)
    return frame.drop(columns=drop)
