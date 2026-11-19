"""Pure numeric kernels. No file readers and no access to real outcomes."""
from __future__ import annotations

import itertools
import math
import numpy as np

METHODS=('independent_ewma','fully_shared_pair','fixed_half','risk_recent_k8',
         'fitted_fullcov_DLM','oracle_fullcov_DLM')
VARIANTS=('gaussian','t5_variance_standardized','contamination5_variance_standardized',
          'common_shift_at26','difference_shift_at26','delay1','delay3','heteroskedastic')


def psd_projection(matrix):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    return (vectors*np.maximum(values,0.))@vectors.T


def moment_decomposition(diff0,diff1):
    zero=np.asarray(diff0,float);lag=np.asarray(diff1,float)
    r=-(lag+lag.T)/2;q=zero-2*r
    result={}
    for name,raw in [('Q',q),('R',r)]:
        projected=psd_projection(raw)
        result.update({name+'_raw':raw,name+'_psd':projected,
                       name+'_eigenvalues':np.linalg.eigvalsh(raw),
                       name+'_projection_frobenius':float(np.linalg.norm(raw-projected))})
    return result


def standardize_noise(kind,draws,contamination_flags=None):
    values=np.asarray(draws,float)
    if kind=='t5':return values*np.sqrt(3/5)
    if kind=='contamination5':
        return values*np.where(np.asarray(contamination_flags,bool),5.,1.)/np.sqrt(2.2)
    if kind=='gaussian':return values.copy()
    raise ValueError('Unknown noise family')


def covariances(cfg):
    loading=np.array([[1.,1.-cfg['m']],[1.,-cfg['m']]])
    q=loading@np.diag([cfg['qc'],cfg['qd']])@loading.T
    r=cfg['r']*np.array([[1.,cfg['rho']],[cfg['rho'],1.]])
    return q,r


def conditional_nll(y,ordinals,q,r):
    return float(batch_nll(y,ordinals,np.asarray(q)[None],np.asarray(r)[None])[0])


def batch_nll(y,ordinals,qs,rs):
    """Conditional flat-initial-level likelihood, vectorized over fixed grid."""
    y=np.asarray(y,float);ordinals=np.asarray(ordinals,int)
    qs=np.asarray(qs,float);rs=np.asarray(rs,float)
    if len(y)<2 or not np.isfinite(y).all() or np.any(np.diff(ordinals)<=0):
        raise ValueError('Finite complete observations in strictly increasing activity order required')
    mean=np.broadcast_to(y[0],(len(qs),2)).copy();p=rs.copy()
    loss=np.zeros(len(qs));identity=np.eye(2)
    for i in range(1,len(y)):
        prior=p+(ordinals[i]-ordinals[i-1])*qs;s=prior+rs
        innovation=y[i]-mean
        sign,logdet=np.linalg.slogdet(s)
        solved=np.linalg.solve(s,innovation[...,None])[...,0]
        loss+=.5*(2*np.log(2*np.pi)+logdet+np.sum(innovation*solved,axis=-1))
        loss=np.where(sign>0,loss,np.inf)
        gain=np.linalg.solve(s,np.swapaxes(prior,-1,-2)).swapaxes(-1,-2)
        mean+=np.einsum('gij,gj->gi',gain,innovation)
        rem=identity-gain
        p=rem@prior@rem.swapaxes(-1,-2)+gain@rs@gain.swapaxes(-1,-2)
        p=(p+p.swapaxes(-1,-2))/2
    return loss


def calibrate(y,ordinals,*,delete_one=True):
    y=np.asarray(y,float);ordinals=np.asarray(ordinals,int)
    adjacent=np.diff(ordinals)==1
    if len(y)<12 or adjacent.sum()<8:raise ValueError('Insufficient complete pair or adjacent difference support')
    delta=np.diff(y,axis=0)[adjacent]
    scale=max(float(np.trace(np.cov(delta,rowvar=False,ddof=1))/2),1e-8)
    grid=[]
    for a,b,c,rho,m in itertools.product([0,.125,.5,1,2],[0,.125,.5,1,2],
                                       [.025,.125,.25,.5],[-.5,0,.5],[0,.5,1.]):
        grid.append({'qc':a*scale,'qd':b*scale,'r':c*scale,'rho':rho,'m':m})
    qs,rs=map(np.array,zip(*(covariances(g) for g in grid)))
    nll=batch_nll(y,ordinals,qs,rs);best=int(np.argmin(nll))
    for i,g in enumerate(grid):
        g.update(grid_index=i,nll=float(nll[i]),two_delta_nll=float(2*(nll[i]-nll[best])),
                 support=bool(2*(nll[i]-nll[best])<=10),
                 q_r_ratio=float(np.trace(qs[i])/np.trace(rs[i])),complete_pairs=len(y),
                 adjacent_differences=int(adjacent.sum()),activity_span=int(ordinals[-1]-ordinals[0]+1))
    supported=sorted((g for g in grid if g['support']),key=lambda g:(g['q_r_ratio'],g['nll'],g['grid_index']))
    anchors=[supported[int(round(p*(len(supported)-1)))] for p in (0,.5,1)]
    deletion=[]
    if delete_one:
        for j in range(len(y)):
            keep=np.arange(len(y))!=j
            losses=batch_nll(y[keep],ordinals[keep],qs,rs)
            deletion.append({'deleted_ordinal':int(ordinals[j]),**grid[int(np.argmin(losses))]})
    return {'scale':scale,'grid':grid,'best':grid[best],'anchors':anchors,
            'delete_one_best':deletion,'mu':y.mean(axis=0),
            'common_shift':2*float(np.std(y.mean(axis=1),ddof=1)),
            'difference_shift':2*float(np.std(y[:,0]-y[:,1],ddof=1))}


def filter_mean(observations,q,r,prior_mean=None,prior_cov=None):
    """Complete batched paths or one sequence with missing observed lines."""
    y=np.asarray(observations,float);q=np.asarray(q,float);r=np.asarray(r,float)
    m=np.zeros(y.shape[:-2]+(2,)) if prior_mean is None else np.broadcast_to(prior_mean,y.shape[:-2]+(2,)).copy()
    p=np.eye(2)*.1 if prior_cov is None else np.asarray(prior_cov,float).copy()
    for i in range(y.shape[-2]):
        prior=p+q
        mask=np.isfinite(y[...,i,:])
        if y.ndim==3 and not mask.all():raise ValueError('Batched missing masks require separate paths')
        h=np.eye(2) if y.ndim==3 else np.eye(2)[mask]
        if len(h)==0:p=prior;continue
        noise=h@r@h.T;s=h@prior@h.T+noise
        gain=np.linalg.solve(s,(prior@h.T).T).T
        observed=y[...,i,:] if y.ndim==3 else y[i,mask]
        m+=(observed-m@h.T)@gain.T
        rem=np.eye(2)-gain@h
        p=rem@prior@rem.T+gain@noise@gain.T
        p=(p+p.T)/2
    return m


def ewma_bias(history):
    n=history.shape[-2]
    if not n:return np.zeros(history.shape[:-2]+(2,))
    weights=.3*.7**np.arange(n-1,-1,-1,dtype=float);weights[0]=.7**(n-1)
    return np.clip(np.tensordot(history,weights,axes=([-2],[0]))*n/(n+10.),-.3,.3)


def replay_arrays(y,q,r,*,delay=0,oracle_q=None,oracle_r=None,mu=None):
    """Prequential complete-pair predictions. Labels release in order after issue.

    At event e (zero based), available feedback indexes are < e-delay. The
    risk coefficients always retain experts issued at their original events.
    """
    y=np.asarray(y,float)
    if y.ndim!=3 or y.shape[-1]!=2 or delay not in (0,1,3):raise ValueError('Invalid replay shape/delay')
    paths,events,_=y.shape
    raw=np.zeros((paths,events,len(METHODS),2));issued_i=np.zeros_like(y);issued_s=np.zeros_like(y);lambdas=np.zeros_like(y)
    for e in range(events):
        n=max(e-delay,0);history=y[:,:n]
        ind=ewma_bias(history);shared=np.repeat(ewma_bias(history.mean(axis=-1,keepdims=True).repeat(2,axis=-1))[...,:1],2,axis=-1)
        issued_i[:,e]=ind;issued_s[:,e]=shared
        lam=np.zeros((paths,2))
        if n>=5:
            weights=.85**np.arange(n-1,-1,-1,dtype=float);neff=weights.sum()**2/(weights@weights)
            d=issued_s[:,:n]-issued_i[:,:n]
            a=np.einsum('n,pnr->pr',weights,d*d)/weights.sum()
            c=np.einsum('n,pnr->pr',weights,(history-issued_i[:,:n])*d)/weights.sum()
            lam=np.where(a>1e-8,np.clip(c/np.maximum(a*(1+8/neff),1e-30),0,1),0.)
        lambdas[:,e]=lam
        raw[:,e,0]=ind;raw[:,e,1]=shared;raw[:,e,2]=(ind+shared)/2;raw[:,e,3]=(1-lam)*ind+lam*shared
        raw[:,e,4]=filter_mean(history,q,r,mu,np.eye(2)*.1)
        raw[:,e,5]=filter_mean(history,q if oracle_q is None else oracle_q,
                              r if oracle_r is None else oracle_r,mu,np.zeros((2,2)))
    return {'bias':np.clip(raw,-.3,.3),'raw_bias':raw,'issued_I':issued_i,'issued_S':issued_s,'lambda':lambdas}


def make_path(cfg,variant,state_units,noise_units,mu,common_shift,difference_shift):
    """External draws permit common random numbers and deterministic fixtures."""
    q,r=covariances(cfg)
    def root(matrix):
        eigen,vectors=np.linalg.eigh(matrix)
        return (vectors*np.sqrt(np.maximum(eigen,0)))@vectors.T
    state=np.asarray(mu)+np.cumsum(np.asarray(state_units)@root(q).T,axis=1)
    if variant=='common_shift_at26':state[:,25:]+=common_shift
    if variant=='difference_shift_at26':
        state[:,25:]+=np.array([(1-cfg['m'])*difference_shift,-cfg['m']*difference_shift])
    noise=np.asarray(noise_units)@root(r).T
    if variant=='heteroskedastic':
        noise=noise*np.sqrt(np.r_[np.full(25,.5),np.full(10,2.25)])[None,:,None]
    return state,state+noise


def complete_pairs(records,phase):
    values={}
    for event,rank,stage,z in records:
        if stage==phase and rank in (1000,2000):values.setdefault(event,{})[rank]=float(z)
    events=sorted(e for e,v in values.items() if len(v)==2)
    return {'events':events,'y':np.array([[values[e][1000],values[e][2000]] for e in events]).reshape(-1,2)}


def distribution_summary(x):
    x=np.asarray(x,float);n=len(x)
    out={'n':n,'mean':float(np.mean(x)) if n else None,'sd':float(np.std(x,ddof=1)) if n>1 else None,
         'median':float(np.median(x)) if n else None,'skew':None,'excess_kurtosis':None,'JB':None,'JB_asymptotic_p':None}
    if n>=8:
        variance=float(np.mean((x-x.mean())**2))
        if variance>1e-16:
            skew=float(np.mean((x-x.mean())**3)/variance**1.5)
            kurt=float(np.mean((x-x.mean())**4)/variance**2-3)
            jb=n/6*(skew*skew+kurt*kurt/4)
            out.update(skew=skew,excess_kurtosis=kurt,JB=jb,JB_asymptotic_p=math.exp(-jb/2))
    return out


def bh_values(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.zeros(len(p));out[order]=np.minimum(q,1);return out


def score_arrays(state,observed,issued):
    state=np.asarray(state)[:,:,None,:];observed=np.asarray(observed)[:,:,None,:]
    bias=issued['bias'];raw=issued['raw_bias']
    return {'state_mse':np.mean((state-bias)**2,axis=-1),
            'raw_state_mse':np.mean((state-raw)**2,axis=-1),
            'observed_log_mse':np.mean((observed-bias)**2,axis=-1),
            'synthetic_unbounded_ape_pct':100*np.mean(np.abs(np.expm1(observed-bias)),axis=-1)}
