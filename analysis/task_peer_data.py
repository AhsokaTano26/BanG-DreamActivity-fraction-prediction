"""Trusted as-of peer feature preparation; detached batches contain blind values only."""
from __future__ import annotations
import argparse,gzip,hashlib,json,platform,shutil
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from analysis.backtest import ROOT,snapshot
from analysis.holdout_dataset import from_record,sha256
from analysis.multirank_dataset import identity
from analysis.multirank_model import feature_row
from analysis.rolling_benchmark import seal_directory,verify_seal,write_new_json
from analysis.data_policy import MIN_EVENT_START_MS

DEFAULT=ROOT/'data/peer_prefix_v3_19rank_20261118'
FORBIDDEN={'actual','target','final','final_score','log_error','reference_log_error','label','labels'}

@dataclass
class PeerBatch:
    nodes: np.ndarray
    node_table: pd.DataFrame
    links: np.ndarray
    ages: np.ndarray
    frame: pd.DataFrame
    ranks: list
    columns: list


def prefix_hash(times,scores):
    h=hashlib.sha256(np.asarray(times,dtype=np.int64).tobytes())
    h.update(np.asarray(scores,dtype=np.float64).tobytes());return h.hexdigest()


def numeric_features(sample):
    f=feature_row(sample);f.pop('event_type',None)
    for key in ['elapsed_h','remaining_h','duration_h']:
        f['log_'+key]=float(np.log1p(max(0.,f.pop(key))))
    f['calendar_years']=(sample['now']-MIN_EVENT_START_MS)/(365.25*86400000)
    return {k:float(v) if v is not None else np.nan for k,v in f.items()}


def prepare_records(records,cases,ranks):
    cases=cases.copy(deep=True).reset_index(drop=True);ranks=list(map(int,ranks))
    if cases.case_id.duplicated().any() or ranks!=sorted(set(ranks)) or not ranks or min(ranks)<=0:
        raise ValueError('Unique frozen cases and rank scope required')
    if FORBIDDEN&set(cases) or (cases.start_ms<MIN_EVENT_START_MS).any():raise ValueError('Only dense blind cases accepted')
    lookup=cases.set_index('case_id');latest={};own={};clocks={}
    for rec in records:
        if FORBIDDEN&set(rec) or rec['case_id'] not in lookup.index or rec['case_id'] in own:raise ValueError('Blind source case set differs')
        c=from_record(rec);row=lookup.loc[rec['case_id']]
        if (c.event!=int(row.event) or c.rank!=int(row['rank']) or c.start!=int(row.start_ms) or c.end!=int(row.end_ms)
            or c.rank not in ranks or int(rec['cutoff'])!=int(row.cutoff) or len(c.times)<4
            or np.any(np.diff(c.times)<=0) or not np.isfinite(c.scores).all() or (c.scores<0).any()
            or int(c.times[-1])!=int(row.observed_at) or c.times[-1]>row.cutoff
            or row.cutoff-c.times[-1]>3600000 or float(c.scores[-1])!=float(row.current)
            or len(c.times)!=int(row.prefix_points) or c.start>=c.end or not c.start<c.times[-1]<c.end):
            raise ValueError('Own blind prefix disagrees with frozen case')
        key=(c.event,c.rank);dates=(c.start,c.end)
        if clocks.setdefault(c.event,dates)!=dates:raise ValueError('Peer event dates disagree')
        own[rec['case_id']]=(len(c.times),prefix_hash(c.times,c.scores))
        if key not in latest or len(c.times)>len(latest[key].times):latest[key]=c
    if set(own)!=set(lookup.index):raise ValueError('Missing source prefix')
    values=[];node_rows=[];node_lookup={};columns=None
    links=np.full((len(cases),len(ranks)),-1,dtype=np.int32);ages=np.zeros(links.shape,dtype=np.float32)

    def node(event,rank,asof):
        nonlocal columns
        curve=latest.get((event,rank))
        if curve is None:return -1
        stop=int(np.searchsorted(curve.times,asof,side='right'))
        if stop<4 or curve.scores[stop-1]<=0 or asof-curve.times[stop-1]>3600000:return -1
        key=(event,rank,int(curve.times[stop-1]))
        if key in node_lookup:return node_lookup[key]
        # crop before snapshot: later source-container points cannot reach the model features.
        detached=type(curve)(curve.event,curve.rank,curve.kind,curve.start,curve.end,curve.times[:stop].copy(),curve.scores[:stop].copy())
        sample=snapshot(detached,int(asof))
        if sample is None:raise ValueError('Declared peer snapshot unavailable')
        f=numeric_features(sample)
        if columns is None:columns=sorted(f)
        if sorted(f)!=columns:raise ValueError('Node feature schema drift')
        idx=len(values);node_lookup[key]=idx;values.append([f[k] for k in columns])
        node_rows.append({'node_id':idx,'event':event,'rank':rank,'start_ms':curve.start,'end_ms':curve.end,
                          'observed_at':sample['now'],'prefix_points':stop,'prefix_sha256':prefix_hash(detached.times,detached.scores)})
        return idx

    for i,row in enumerate(cases.itertuples()):
        for j,rank in enumerate(ranks):
            idx=node(int(row.event),rank,int(row.observed_at));links[i,j]=idx
            if idx>=0:ages[i,j]=(row.observed_at-node_rows[idx]['observed_at'])/3600000
        j=ranks.index(int(row.rank));idx=links[i,j]
        if idx<0:raise ValueError('Own node missing')
        n,h=own[row.case_id];stored=node_rows[idx]
        if stored['prefix_points']!=n or stored['prefix_sha256']!=h:raise ValueError('Earlier source prefix was changed by later source')
    return {'values':np.asarray(values,dtype=np.float64),'node_table':pd.DataFrame(node_rows),
            'links':links,'ages':ages,'cases':cases,'ranks':ranks,'columns':columns}


def save_cache(arrays,destination,*,source_identity,dataset_id):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Preserve prior peer cache')
    destination.mkdir(parents=True)
    arrays['cases'].to_csv(destination/'cases.csv',index=False)
    arrays['node_table'].to_csv(destination/'nodes.csv',index=False)
    joblib.dump({k:arrays[k] for k in ['values','links','ages']},destination/'arrays.joblib',compress=3)
    write_new_json(destination/'schema.json',{'columns':arrays['columns'],'ranks':arrays['ranks'],
        'clock':'every peer observed_at <= target OWN observed_at; age<=1h; no later source-container points exposed'})
    shutil.copyfile(__file__,destination/'source.py')
    return seal_directory(destination,'manifest.json',{'protocol':'blind_peer_prefix_cache_v1','dataset_id':dataset_id,
        'source_identity':source_identity,'cases':len(arrays['cases']),'nodes':len(arrays['node_table']),
        'ranks':arrays['ranks'],'columns':arrays['columns'],'python':platform.python_version(),
        'preprocessing_fit':False,'outcomes_parsed':False})


class PeerCache:
    def __init__(self,root=DEFAULT):
        self.root=Path(root);self.manifest=verify_seal(self.root/'manifest.json')
        if self.manifest['protocol']!='blind_peer_prefix_cache_v1':raise ValueError('Wrong peer cache protocol')
        self.arrays=joblib.load(self.root/'arrays.joblib')
        self.cases=pd.read_csv(self.root/'cases.csv');self.nodes=pd.read_csv(self.root/'nodes.csv')
        self.ranks=list(self.manifest['ranks']);self.columns=list(self.manifest['columns'])
        self.positions={c:i for i,c in enumerate(self.cases.case_id)}
        if len(self.positions)!=len(self.cases):raise ValueError('Duplicate frozen cache cases')

    def batch(self,case_ids):
        ids=list(case_ids)
        if len(ids)!=len(set(ids)) or not set(ids)<=set(self.positions):raise ValueError('Exact unique cached cases required')
        idx=np.array([self.positions[c] for c in ids],dtype=int)
        links=self.arrays['links'][idx].copy();chosen=np.unique(links[links>=0])
        node_table=self.nodes.iloc[chosen].copy(deep=True).reset_index(drop=True)
        mapping=np.full(len(self.nodes),-1,dtype=np.int32);mapping[chosen]=np.arange(len(chosen))
        links=np.where(links>=0,mapping[np.maximum(links,0)],-1)
        frame=self.cases.iloc[idx].copy(deep=True).reset_index(drop=True)
        if FORBIDDEN&set(frame):raise ValueError('Outcome field in blind cache')
        for i,row in enumerate(frame.itertuples()):
            peers=links[i][links[i]>=0];n=node_table.iloc[peers]
            if (n.event!=row.event).any() or (n.observed_at>row.observed_at).any():raise ValueError('Peer clock/event violation')
            if (row.observed_at-n.observed_at>3600000).any():raise ValueError('Stale peer violation')
            own=links[i,self.ranks.index(int(row.rank))]
            if own<0 or node_table.iloc[own].observed_at!=row.observed_at:raise ValueError('Own cache identity differs')
        return PeerBatch(self.arrays['values'][chosen].copy(),node_table,links,
                         self.arrays['ages'][idx].copy(),frame,self.ranks.copy(),self.columns.copy())


def build(data,destination=DEFAULT):
    cases=pd.concat([data.case_index(s).assign(split=s) for s in ['train','validation','test']],ignore_index=True)
    def records():
        for split in ['train','validation','test']:
            with gzip.open(data.root/f'{split}_inputs.jsonl.gz','rt') as f:
                for line in f:yield json.loads(line)
    source={name:sha256(data.root/name) for name in ['manifest.json',*[f'{s}_{kind}' for s in ['train','validation','test'] for kind in ['inputs.jsonl.gz','cases.csv']]]}
    arrays=prepare_records(records(),cases,data.manifest['protocol']['ranks'])
    return save_cache(arrays,destination,source_identity=source,dataset_id=data.manifest['dataset_id'])


if __name__=='__main__':
    from analysis.multiline_dataset import MultilineDataset,DEFAULT as DATA
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DATA);p.add_argument('--destination',type=Path,default=DEFAULT);a=p.parse_args()
    print(json.dumps(build(MultilineDataset(a.dataset),a.destination),ensure_ascii=False)[:400])
