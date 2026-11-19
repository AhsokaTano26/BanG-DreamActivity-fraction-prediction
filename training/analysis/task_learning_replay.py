"""Read frozen task-learning weights using their original categorical layout.

The historical fit used event_type then band_group. Sorted JSON reverses these
mapping keys. Keep sealed training sources, weights and issued scores untouched;
this reader explicitly restores the layout before applying the original model.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from analysis.holdout_dataset import sha256
from analysis.multirank_dataset import identity
from analysis import task_learning_models as m

CATEGORY_ORDER=('event_type','band_group')

def replay(directory,batch,current,state):
    directory=Path(directory);cfg=json.loads((directory/'config.json').read_text())
    m.check_current(current,cfg['query_start_ms'],cfg['ranks']);m.check_batch(batch,current)
    if (list(current.case_id)!=cfg['case_ids'] or state['ranks']!=cfg['ranks']
        or identity(state)!=cfg['current_graph_sha256']):raise ValueError('Replay graph/case identity differs')
    if cfg['status']=='reference_fallback':return m.replay(directory,batch,current,state)
    if cfg['status']!='fitted':raise ValueError('Unknown model status')
    if sha256(directory/'preprocessor.json')!=cfg['preprocessor_sha256']:raise ValueError('Preprocessing changed')
    prep=json.loads((directory/'preprocessor.json').read_text())
    if set(prep['categories'])!=set(CATEGORY_ORDER) or prep['context_names']!=m.CTX_NAMES:
        raise ValueError('Unsupported historical context layout')
    prep['categories']={k:prep['categories'][k] for k in CATEGORY_ORDER}
    s,p,n=m.issue_features(current,{int(current.event.iloc[0]):state},batch.ranks)
    nodes,ctx=m.transform(prep,batch,current,n)
    if nodes.shape[1]!=cfg['node_dim'] or ctx.shape[1]!=cfg['context_dim']:raise ValueError('Replay dimensions differ')
    import torch
    torch.set_num_threads(1);out={}
    for mode in m.MODES:
        args=m.tensor_input(batch,nodes,ctx,p if mode=='related_phase' else s,n)
        for seed in m.SEEDS:
            model=m.make_network(cfg['node_dim'],cfg['context_dim'],len(cfg['ranks']),mode)
            saved=torch.load(directory/mode/f'seed{seed}.pt',map_location='cpu',weights_only=True)
            if saved['seed']!=seed:raise ValueError('Checkpoint seed differs')
            model.load_state_dict(saved['state_dict']);model.eval()
            with torch.no_grad():raw=model(*args).numpy()
            out[f'{mode}_seed{seed}']=m.decode(raw,current,cfg['shrinkage'])
    for mode in m.MODES:out[mode]=np.mean([out[f'{mode}_seed{s}'] for s in m.SEEDS],axis=0)
    return out
