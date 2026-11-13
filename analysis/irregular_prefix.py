"""Observed-node paths; no uniform-grid interpolation and no future observations."""
import numpy as np
from analysis.backtest import snapshot


def observed_path(sample,length=32):
    if length<2:raise ValueError('At least two nodes required')
    path=np.zeros((2,length,4),np.float32);mask=np.zeros((2,length),bool)
    target=sample['curve'];curves={target.rank:target};peer=sample.get('peer_curve')
    if peer is not None:curves[peer.rank]=peer
    for slot,rank in enumerate([1000,2000]):
        curve=curves.get(rank);s=snapshot(curve,sample['now']) if curve is not None else None
        if s is None:continue
        t,v=s['times'],s['scores'];count=min(length,len(t))
        indices=np.linspace(0,len(t)-1,count,dtype=int)
        nodes=np.column_stack([(t[indices]-target.start)/target.duration,v[indices]/s['score'],
                               np.log1p(indices)/np.log(1000),(t[indices]-target.start)/3600000/168])
        path[slot,:count]=nodes;mask[slot,:count]=True;path[slot,count:]=nodes[-1]
    return path,mask
