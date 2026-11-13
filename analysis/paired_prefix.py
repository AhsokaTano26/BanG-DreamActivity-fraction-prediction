"""Canonical rank slots, shared time grid, and explicit sampling masks."""
from dataclasses import replace
import numpy as np
from analysis.backtest import snapshot


def paired_sequence(sample,length=64,window_hours=None):
    target=sample['curve'];now=sample['now']
    start=target.start if window_hours is None else max(target.start,now-window_hours*3600000)
    grid=np.linspace(start,now,length);output=np.zeros((2,length,8),dtype=np.float32)
    curves={target.rank:target};peer=sample.get('peer_curve')
    if peer is not None:curves[peer.rank]=peer
    for slot,rank in enumerate([1000,2000]):
        curve=curves.get(rank)
        # Use the same current-time peer validity rule as the frozen protocol.
        s=snapshot(curve,now) if curve is not None else None
        if s is None:continue
        t,v=s['times'],s['scores'];valid=(grid>=t[0])&(grid<=t[-1])
        cumulative=np.interp(grid,t,v)/s['score']
        rate=np.r_[0.,np.diff(cumulative)/np.diff(grid)*(now-target.start)]
        rate[1:][~(valid[1:]&valid[:-1])]=0.;rate[~valid]=0.
        previous=np.clip(np.searchsorted(t,grid,side='right')-1,0,len(t)-1)
        following=np.minimum(previous+1,len(t)-1)
        age=np.maximum(0.,grid-t[previous])/target.duration
        span=(t[following]-t[previous])/target.duration
        hour=(grid/3600000+8)%24
        channels=np.column_stack([cumulative,np.clip(rate,-8,8),valid.astype(float),
            np.clip(age,0,1),np.clip(span,0,1),(grid-target.start)/target.duration,
            np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)])
        channels[~valid]=0.
        output[slot]=channels.astype(np.float32)
    return output


def thin_sample(sample,stride=4):
    def thin(curve):
        if curve is None:return None
        stop=np.searchsorted(curve.times,sample['now'],side='right')
        if stop==0:return None
        count=min(stop,max(4,stop//stride))
        indices=np.unique(np.linspace(0,stop-1,count,dtype=int))
        return replace(curve,times=curve.times[indices],scores=curve.scores[indices])
    curve=thin(sample['curve']);value=snapshot(curve,sample['cutoff'])
    if value is None:raise ValueError('Thinning changed a valid target prefix into an invalid one')
    return sample|value|{'peer_curve':thin(sample.get('peer_curve'))}
