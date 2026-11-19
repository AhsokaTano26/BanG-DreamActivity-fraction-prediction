"""Small pure shared/rank/peer residual learners over detached causal batches."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.data_policy import MIN_EVENT_START_MS
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import write_new_json
from analysis.holdout_dataset import sha256

MODES=('shared_single','rank_heads','related_static','related_phase')
SEEDS=(42,123,2026)
RECIPE={'hidden':16,'head_hidden':8,'epochs':60,'minimum_events':12,'learning_rate':.002,
        'weight_decay':.05,'dropout':.1,'total_shrinkage':20.,'rank_shrinkage':10.,
        'log_bound':.2,'delta_penalty':.05,'graph_minimum_pairs':8,'graph_topk':3,'graph_shrinkage':10.,
        'phase_edges':[.25,.9],'threads':1,'seeds':list(SEEDS)}
FORBIDDEN={'actual','target','final','final_score','log_error','label','labels'}
CTX_NAMES=['log_rank','progress','log_current','reference_growth','log_duration','log_remaining','context_family','log_support']


def event_weights(frame):
    return 1./frame.groupby('event').event.transform('size').to_numpy(float)


def check_history(h,query_start_ms,ranks):
    needed={'case_id','event','rank','phase','progress','start_ms','end_ms','reference_prediction','reference_log_error'}
    if needed-set(h) or FORBIDDEN&set(h):raise ValueError('Explicit historical reference errors required')
    if h.case_id.duplicated().any() or not set(h['rank'])<=set(ranks):raise ValueError('Duplicate or unknown history rank')
    if ((h.start_ms<MIN_EVENT_START_MS).any() or (h.end_ms>=query_start_ms).any() or (h.start_ms>=h.end_ms).any()):raise ValueError('History has not strictly ended')
    if len(h) and (h.groupby('event')[['start_ms','end_ms']].nunique()>1).any().any():raise ValueError('Event history clock differs')
    if not np.isfinite(h[['reference_prediction','reference_log_error','progress']].to_numpy(float)).all() or (h.reference_prediction<=0).any():raise ValueError('Invalid history values')
    if (np.searchsorted([.25,.9],h.progress.to_numpy(),side='right')!=h.phase.to_numpy()).any():raise ValueError('Wrong historical phase')


def relation_state(history,*,query_start_ms,ranks):
    ranks=list(map(int,ranks));check_history(history,query_start_ms,ranks)
    prior=np.exp(-np.abs(np.log(ranks)[:,None]-np.log(ranks)[None,:]));np.fill_diagonal(prior,0.)
    def graph(part):
        if len(part):
            matrix=part.groupby(['event','rank']).reference_log_error.mean().unstack().reindex(columns=ranks)
        else:matrix=pd.DataFrame(columns=ranks,dtype=float)
        known=matrix.notna().to_numpy(np.int64);counts=known.T@known
        corr=matrix.corr(method='spearman',min_periods=8).to_numpy(float)
        mix=prior.copy()
        valid=np.isfinite(corr)&(counts>=8);shrink=counts/(counts+10.)
        mix[valid]=shrink[valid]*np.maximum(corr[valid],0.)**2+(1-shrink[valid])*.1*prior[valid]
        np.fill_diagonal(mix,0.)
        for i in range(len(ranks)):
            chosen=np.argsort(-mix[i],kind='stable')[:min(3,len(ranks)-1)]
            mask=np.ones(len(ranks),bool);mask[chosen]=False;mix[i,mask]=0.
        totals=mix.sum(axis=1,keepdims=True);mix=np.divide(mix,totals,out=np.zeros_like(mix),where=totals>0)
        return {'weights':mix.tolist(),'pair_counts':counts.tolist(),
                'correlations':[[None if not np.isfinite(x) else float(x) for x in row] for row in corr]}
    return {'ranks':ranks,'query_start_ms':int(query_start_ms),'history_events':int(history.event.nunique()),
            'released_events':sorted(map(int,history.event.unique())),
            'max_history_end_ms':int(history.end_ms.max()) if len(history) else None,
            'rank_counts':[int(history[history['rank']==r].event.nunique()) for r in ranks],
            'static':graph(history),'phases':{str(p):graph(history[history.phase==p]) for p in range(3)}}


def issue_features(frame,states,ranks):
    static=[];phase=[];support=[]
    for row in frame.itertuples():
        if int(row.event) not in states:raise ValueError('Missing originally issued graph')
        state=states[int(row.event)]
        if (state['ranks']!=ranks or state['query_start_ms']!=row.start_ms or row.event in state['released_events']
            or (state['max_history_end_ms'] is not None and state['max_history_end_ms']>=row.start_ms)):
            raise ValueError('Historical relation state is not its own issuance state')
        i=ranks.index(int(row.rank));static.append(state['static']['weights'][i]);phase.append(state['phases'][str(int(row.phase))]['weights'][i]);support.append(state['rank_counts'][i])
    return np.asarray(static,float).reshape(len(frame),len(ranks)),np.asarray(phase,float).reshape(len(frame),len(ranks)),np.asarray(support,float)


def context_numeric(frame,support):
    return np.column_stack([np.log(frame['rank']),frame.progress,np.log1p(frame.current),np.log(frame.reference_prediction/frame.current),
                            np.log1p(frame.duration_h),np.log1p(frame.remaining_h),frame['rank'].isin([1000,2000]).astype(float),np.log1p(support)])


def weighted_scale(values,weights):
    known=np.isfinite(values);safe=np.where(known,values,0.);mass=(weights[:,None]*known).sum(axis=0)
    mean=np.divide((safe*weights[:,None]).sum(axis=0),mass,out=np.zeros(values.shape[1]),where=mass>0)
    centered=np.where(known,values-mean,0.)
    var=np.divide((centered**2*weights[:,None]).sum(axis=0),mass,out=np.zeros(values.shape[1]),where=mass>0)
    return mean,np.maximum(np.sqrt(var),1e-4)


def numeric_transform(values,mean,std):
    missing=~np.isfinite(values);z=(np.where(missing,mean,values)-mean)/std
    return np.column_stack([np.clip(z,-8,8),missing.astype(float)]).astype(np.float32)


def fit_preprocessor(batch,history,support):
    w=event_weights(history)
    own=np.array([batch.links[i,batch.ranks.index(int(r))] for i,r in enumerate(history['rank'])])
    mean,std=weighted_scale(batch.nodes[own],w)
    cmean,cstd=weighted_scale(context_numeric(history,support),w)
    return {'node_columns':batch.columns,'node_mean':mean.tolist(),'node_std':std.tolist(),
            'context_names':CTX_NAMES,'context_mean':cmean.tolist(),'context_std':cstd.tolist(),
            'categories':{k:sorted(set(history[k].map(str))) for k in ['event_type','band_group']},
            'fit_case_ids':history.case_id.tolist(),'fit_events':sorted(map(int,history.event.unique())),
            'fit_scope':'historical OWN nodes and rows only, equal activity mass; no peer-only/current fitting'}


def transform(prep,batch,frame,support):
    if prep['node_columns']!=batch.columns:raise ValueError('Node column schema differs')
    nodes=numeric_transform(batch.nodes,np.array(prep['node_mean']),np.array(prep['node_std']))
    context=[numeric_transform(context_numeric(frame,support),np.array(prep['context_mean']),np.array(prep['context_std']))]
    for key,vocab in prep['categories'].items():
        values=frame[key].map(str).to_numpy();context.append(np.column_stack([*[values==v for v in vocab],~np.isin(values,vocab)]).astype(np.float32))
    context.append(np.eye(3,dtype=np.float32)[frame.phase.to_numpy(int)])
    return nodes,np.column_stack(context).astype(np.float32)


def make_network(node_dim,context_dim,rank_count,mode):
    import torch
    from torch import nn
    if mode not in MODES:raise ValueError('Unknown neural mode')
    class Residual(nn.Module):
        def __init__(self):
            super().__init__();self.mode=mode
            self.encoder=nn.Sequential(nn.Linear(node_dim,16),nn.SiLU(),nn.Dropout(.1),nn.Linear(16,16),nn.SiLU())
            self.head=nn.Sequential(nn.Linear(16+context_dim,8),nn.SiLU(),nn.Dropout(.1),nn.Linear(8,1))
            self.private_weight=nn.Parameter(torch.zeros(rank_count,16+context_dim));self.private_bias=nn.Parameter(torch.zeros(rank_count))
            self.peer_head=nn.Linear(16,1);self.static_amplitude=nn.Parameter(torch.tensor(-2.))
            self.query=nn.Linear(16,16,bias=False);self.key=nn.Linear(16,16,bias=False)
            self.dynamic_amplitude=nn.Linear(16+context_dim,1);self.age_penalty=nn.Parameter(torch.tensor(0.))
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
            nn.init.zeros_(self.peer_head.weight);nn.init.zeros_(self.peer_head.bias)
            nn.init.zeros_(self.dynamic_amplitude.weight);nn.init.constant_(self.dynamic_amplitude.bias,-2.)
        def components(self,nodes,links,ages,context,rank_index,prior,support):
            z=self.encoder(nodes);own=z[links[torch.arange(len(links)),rank_index]];x=torch.cat([own,context],dim=1)
            shared=self.head(x).squeeze(1);private=torch.zeros_like(shared);peer=torch.zeros_like(shared)
            if self.mode!='shared_single':private=((x*self.private_weight[rank_index]).sum(1)+self.private_bias[rank_index])*support
            if self.mode in ['related_static','related_phase']:
                available=(links>=0)&(prior>0)
                available[torch.arange(len(links)),rank_index]=False
                peers=z[links.clamp(min=0)]
                if self.mode=='related_phase':
                    keys=self.key(z)[links.clamp(min=0)]
                    logits=(self.query(own)[:,None,:]*keys).sum(2)/4+torch.log(prior.clamp(min=1e-12))
                    logits=logits-torch.nn.functional.softplus(self.age_penalty)*ages
                    weight=torch.softmax(logits.masked_fill(~available,-1e4),dim=1)*available
                    amplitude=torch.sigmoid(self.dynamic_amplitude(x)).squeeze(1)
                else:
                    weight=prior*available;amplitude=torch.sigmoid(self.static_amplitude)
                weight=weight/weight.sum(1,keepdim=True).clamp(min=1e-12)
                message=(weight[:,:,None]*peers).sum(1)
                peer=self.peer_head(message).squeeze(1)*amplitude*available.any(1)
            return {'shared':shared,'private':private,'peer':peer}
        def forward(self,*args):
            c=self.components(*args);return c['shared']+c['private']+c['peer']
    return Residual()


def check_batch(batch,frame):
    if list(batch.frame.case_id)!=list(frame.case_id):raise ValueError('Detached batch case identity differs')
    if not np.isfinite(batch.ages).all() or (batch.ages<0).any() or (batch.ages>1).any():raise ValueError('Invalid peer ages')
    for i,row in enumerate(frame.itertuples()):
        indices=batch.links[i];live=indices>=0;nodes=batch.node_table.iloc[indices[live]]
        if (nodes.event!=row.event).any() or (nodes.observed_at>row.observed_at).any():raise ValueError('Future peer clock')
        own=indices[batch.ranks.index(int(row.rank))]
        if own<0 or batch.node_table.iloc[own].observed_at!=row.observed_at:raise ValueError('Missing exact own prefix')


def check_current(current,query_start_ms,ranks):
    if FORBIDDEN&set(current) or 'reference_log_error' in current:raise ValueError('Current outcomes forbidden')
    if (current.start_ms!=query_start_ms).any() or current.event.nunique()!=1 or current.case_id.duplicated().any():raise ValueError('Wrong current event clock/cases')
    if not set(current['rank'])<=set(ranks) or not np.isfinite(current[['current','reference_prediction','progress']].to_numpy()).all():raise ValueError('Invalid current rank/value')
    if (current.current<=0).any() or (current.reference_prediction<current.current).any():raise ValueError('Invalid anchor prediction')
    if (np.searchsorted([.25,.9],current.progress.to_numpy(),side='right')!=current.phase.to_numpy()).any():raise ValueError('Wrong current phase')


def tensor_input(batch,nodes,context,prior,support):
    import torch
    return (torch.tensor(nodes),torch.tensor(batch.links,dtype=torch.long),torch.tensor(batch.ages,dtype=torch.float32),
            torch.tensor(context),torch.tensor([batch.ranks.index(int(r)) for r in batch.frame['rank']],dtype=torch.long),
            torch.tensor(prior,dtype=torch.float32),torch.tensor(support/(support+10.),dtype=torch.float32))


def decode(raw,current,shrink):
    delta=shrink*.2*np.tanh(np.asarray(raw,dtype=float)/.2)
    return np.clip(current.reference_prediction.to_numpy(float)*np.exp(delta),current.current.to_numpy(float),current.current.to_numpy(float)*np.exp(4))


def fit_predict(past_batch,target_batch,history,current,issue_states,*,query_start_ms,model_directory,epochs=60,minimum_events=12):
    ranks=target_batch.ranks;check_history(history,query_start_ms,ranks);check_current(current,query_start_ms,ranks)
    if set(history.event)&set(current.event):raise ValueError('Current event in fitting history')
    check_batch(past_batch,history);check_batch(target_batch,current)
    hs,hp,hn=issue_features(history,issue_states,ranks);ts,tp,tn=issue_features(current,issue_states,ranks)
    directory=Path(model_directory)
    if directory.exists():raise FileExistsError('Preserve previous network fit')
    directory.mkdir(parents=True)
    n=history.event.nunique();shrink=n/(n+20.);status='fitted' if n>=minimum_events else 'reference_fallback'
    config={'status':status,'query_start_ms':int(query_start_ms),'case_ids':current.case_id.tolist(),'ranks':ranks,
            'epochs':epochs,'minimum_events':minimum_events,'recipe':RECIPE,'shrinkage':shrink,
            'current_graph_sha256':identity(issue_states[int(current.event.iloc[0])]),
            'training_graph_hashes':{str(e):identity(issue_states[int(e)]) for e in history.event.unique()},'history_events':int(n)}
    scores={};details={'status':status,'history_events':int(n),'target_rank_counts':{str(r):int(issue_states[int(current.event.iloc[0])]['rank_counts'][i]) for i,r in enumerate(ranks)}}
    if status=='reference_fallback':
        for mode in MODES:
            for seed in SEEDS:scores[f'{mode}_seed{seed}']=current.reference_prediction.to_numpy(float).copy()
    else:
        import torch
        torch.set_num_threads(1);prep=fit_preprocessor(past_batch,history,hn)
        write_new_json(directory/'preprocessor.json',prep)
        w=event_weights(history);pd.DataFrame({'case_id':history.case_id,'event':history.event,'weight':w}).to_csv(directory/'training_weights.csv',index=False)
        x,c=transform(prep,past_batch,history,hn);qx,qc=transform(prep,target_batch,current,tn)
        config.update(node_dim=x.shape[1],context_dim=c.shape[1],preprocessor_sha256=sha256(directory/'preprocessor.json'))
        wt=torch.tensor(w/w.sum(),dtype=torch.float32);error=torch.tensor(history.reference_log_error.to_numpy(),dtype=torch.float32)
        floor=torch.tensor(np.log(history.current/history.reference_prediction).to_numpy(),dtype=torch.float32)
        details['training']={}
        with torch.random.fork_rng(devices=[]):
            for mode in MODES:
                path=directory/mode;path.mkdir();prior=hp if mode=='related_phase' else hs;qprior=tp if mode=='related_phase' else ts
                args=tensor_input(past_batch,x,c,prior,hn);qargs=tensor_input(target_batch,qx,qc,qprior,tn)
                for seed in SEEDS:
                    torch.manual_seed(seed);model=make_network(x.shape[1],c.shape[1],len(ranks),mode)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.05);losses=[];model.train()
                    for _ in range(epochs):
                        optimizer.zero_grad(set_to_none=True);raw=model(*args);delta=shrink*.2*torch.tanh(raw/.2)
                        implied=torch.minimum(torch.maximum(delta,floor),floor+4)
                        loss=(wt*(torch.abs(torch.exp(implied+error)-1)+.05*delta.square())).sum()
                        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                        loss.backward();optimizer.step();losses.append(float(loss.detach()))
                    model.eval()
                    with torch.no_grad():raw=model(*qargs).numpy()
                    scores[f'{mode}_seed{seed}']=decode(raw,current,shrink)
                    torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'seed':seed,'losses':losses},path/f'seed{seed}.pt')
                    details['training'][f'{mode}_seed{seed}']={'first':losses[0],'last':losses[-1],'epochs':epochs,'parameters':sum(p.numel() for p in model.parameters())}
    for mode in MODES:scores[mode]=np.mean([scores[f'{mode}_seed{s}'] for s in SEEDS],axis=0)
    write_new_json(directory/'config.json',config);write_new_json(directory/'diagnostics.json',details)
    return scores,details


def replay(directory,batch,current,state):
    directory=Path(directory);cfg=json.loads((directory/'config.json').read_text());check_current(current,cfg['query_start_ms'],cfg['ranks']);check_batch(batch,current)
    if list(current.case_id)!=cfg['case_ids'] or state['ranks']!=cfg['ranks'] or identity(state)!=cfg['current_graph_sha256']:raise ValueError('Replay graph/case identity differs')
    out={}
    if cfg['status']=='reference_fallback':
        for mode in MODES:
            for seed in SEEDS:out[f'{mode}_seed{seed}']=current.reference_prediction.to_numpy(float).copy()
    else:
        import torch
        torch.set_num_threads(1)
        if sha256(directory/'preprocessor.json')!=cfg['preprocessor_sha256']:raise ValueError('Preprocessing changed')
        prep=json.loads((directory/'preprocessor.json').read_text());s,p,n=issue_features(current,{int(current.event.iloc[0]):state},batch.ranks)
        nodes,ctx=transform(prep,batch,current,n)
        for mode in MODES:
            args=tensor_input(batch,nodes,ctx,p if mode=='related_phase' else s,n)
            for seed in SEEDS:
                model=make_network(cfg['node_dim'],cfg['context_dim'],len(cfg['ranks']),mode)
                saved=torch.load(directory/mode/f'seed{seed}.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict']);model.eval()
                with torch.no_grad():raw=model(*args).numpy()
                out[f'{mode}_seed{seed}']=decode(raw,current,cfg['shrinkage'])
    for mode in MODES:out[mode]=np.mean([out[f'{mode}_seed{s}'] for s in SEEDS],axis=0)
    return out
