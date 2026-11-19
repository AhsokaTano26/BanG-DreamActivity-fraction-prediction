"""Causal functional priors: progress matching versus hours-to-deadline matching."""
import numpy as np
import pandas as pd
from analysis.backtest import weighted_quantile
from analysis.deadline_diagnostics import at_time

SETTINGS={'recency_days':720.,'different_type_weight':.15,'log_duration_scale':.5,'shape_bandwidth':.5}


def log_multiplier(curve,time,current):
    p=(time-curve.start)/curve.duration
    return float(np.clip(np.log(max(curve.final/current-1,1e-4)*p/(1-p)),-6,4))


def historical_rate(curve,time,current,h):
    earlier=time-h*3600000
    value,span=at_time(curve,earlier)
    if not np.isfinite(value):return np.nan,np.nan
    return max(0.,current-value)/current*(time-curve.start)/(h*3600000),span


def history_features(samples,history):
    pools={};rows=[]
    for sample in samples:
        target=sample['curve'];key=(target.event,target.rank,target.start)
        if key not in pools:
            pools[key]=[c for c in history if c.rank==target.rank and c.event!=target.event and c.end+900000<target.start]
        progress=[];deadline=[];shape=[];spans=[];same=[]
        for past in pools[key]:
            weight=np.exp(-(target.start-past.end)/86400000/SETTINGS['recency_days']
                          -abs(np.log(past.duration/target.duration))/SETTINGS['log_duration_scale'])
            weight*=1. if past.kind==target.kind else SETTINGS['different_type_weight']
            time=past.start+sample['progress']*past.duration
            current,_=at_time(past,time)
            if np.isfinite(current) and current>0 and past.start<time<past.end:
                progress.append((log_multiplier(past,time,current),weight))
            time=past.end-(target.end-sample['now'])
            current,span=at_time(past,time)
            if not np.isfinite(current) or current<=0 or time<=past.start or time>=past.end:continue
            logk=log_multiplier(past,time,current);deadline.append((logk,weight));distance=[];support=[span]
            for h in [6,24]:
                own=sample['features'][f'rate_{h}h_relative'];old,lookback_span=historical_rate(past,time,current,h)
                if np.isfinite(own+old):
                    distance.append((np.log(own+1e-8)-np.log(old+1e-8))**2);support.append(lookback_span)
            # Unknown shape remains eligible for the pooled deadline prior, not for a shape match.
            if not distance:continue
            similarity=np.exp(-min(40.,np.mean(distance)/SETTINGS['shape_bandwidth']))
            shape.append((logk,weight*similarity));spans.append(max(support));same.append(float(past.kind==target.kind))
        row={k:np.nan for k in ['hist_progress_logk','hist_deadline_logk','hist_shape_logk','hist_shape_iqr',
                               'hist_effective_n','hist_span_h','hist_same_type_fraction']}
        row['hist_count']=len(shape)
        for name,values in [('hist_progress_logk',progress),('hist_deadline_logk',deadline),('hist_shape_logk',shape)]:
            if values:
                values=np.asarray(values);row[name]=weighted_quantile(values[:,0],values[:,1])
        if shape:
            values=np.asarray(shape);w=values[:,1];w=w/w.sum()
            row['hist_shape_iqr']=weighted_quantile(values[:,0],w,.75)-weighted_quantile(values[:,0],w,.25)
            row['hist_effective_n']=1/np.sum(w*w)
            row['hist_span_h']=float(np.dot(w,spans));row['hist_same_type_fraction']=float(np.dot(w,same))
        rows.append(row)
    return pd.DataFrame(rows)


def prior_offset(frame):
    return frame.hist_shape_logk.fillna(frame.hist_deadline_logk).fillna(frame.hist_progress_logk).fillna(0.).to_numpy()
