"""Trusted rank-scope adapter; legacy two-rank dataset contracts stay unchanged."""
from __future__ import annotations
import argparse,json,shutil,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.data_policy import MIN_EVENT_START_MS
from analysis.holdout_dataset import HoldoutDataset,SPLITS,sha256,write_json
from analysis.multirank_dataset import MultirankDataset,identity
from analysis.temporal_dataset import TemporalDataset,trusted_curves
from analysis import rolling_dataset as legacy

PARENT=ROOT/'data/temporal_holdout_v2_dense_20261115'
DEFAULT=ROOT/'data/rolling_holdout_v3_dense_19rank_20261117'

def cohort(parent):
    ranks=sorted(set(parent.manifest['protocol']['ranks']))
    if not ranks or any(int(r)!=r or r<=0 for r in ranks):raise ValueError('Positive frozen ranks required')
    p=pd.read_csv(parent.root/'partitions.csv')
    if p[['event','rank']].duplicated().any():raise ValueError('Duplicate curve')
    groups={}
    for split in SPLITS:
        for event in parent.manifest['splits'][split]['event_ids']:
            if event in groups:raise ValueError('Overlapping activity groups')
            groups[event]=split
    events={}
    for row in p.to_dict('records'):
        event,rank=int(row['event']),int(row['rank']);start,end=int(row['start_ms']),int(row['end_ms'])
        if rank not in ranks or groups.get(event)!=row['split'] or start<MIN_EVENT_START_MS or end<=start:raise ValueError('Invalid frozen cohort row')
        obj=events.setdefault(event,{'event':event,'split':row['split'],'start_ms':start,'end_ms':end,'ranks':[]})
        if obj['start_ms']!=start or obj['end_ms']!=end:raise ValueError('Rank dates disagree')
        obj['ranks'].append(rank)
    if set(events)!=set(groups):raise ValueError('Activity cohort mismatch')
    ordered=sorted(events.values(),key=lambda r:(r['start_ms'],r['event']))
    if [SPLITS.index(r['split']) for r in ordered]!=sorted(SPLITS.index(r['split']) for r in ordered):raise ValueError('Nonchronological groups')
    return p,ordered

def build(parent,destination,curves,*,source_identity):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    cohort(parent);destination.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.multiline-prepare-',dir=destination.parent));staged=temp/'data'
    original=legacy._cohort
    try:
        # Only trusted preparation uses this adapter. Legacy source bytes and
        # its public rank guard are restored before exposing the new dataset.
        legacy._cohort=cohort
        try:legacy.build_rolling(parent,staged,curves,source_identity=source_identity)
        finally:legacy._cohort=original
        body=json.loads((staged/'manifest.json').read_text());body.pop('dataset_id')
        shutil.copyfile(staged/'manifest.json',staged/'legacy_adapter_manifest.json')
        body['protocol'].update(version='rolling_multiline_v3',ranks=parent.manifest['protocol']['ranks'],
            scope='Exact dense19rank V2 cases with strictly completed-before-start feedback')
        body['source_identity']=source_identity|{'rank_scope_adapter_sha256':sha256(Path(__file__))}
        (staged/'builder_sources').mkdir(exist_ok=True)
        shutil.copyfile(__file__,staged/'builder_sources/multiline_dataset.py')
        body['build_environment']['source_hashes']['training/analysis/multiline_dataset.py']=sha256(Path(__file__))
        body['artifacts']={str(p.relative_to(staged)):sha256(p) for p in sorted(staged.rglob('*')) if p.is_file() and p!=staged/'manifest.json'}
        body['dataset_id']=identity(body);(staged/'manifest.json').write_text(json.dumps(body,sort_keys=True,indent=2))
        MultilineDataset(staged);staged.rename(destination)
        return body
    finally:
        legacy._cohort=original
        if temp.exists():shutil.rmtree(temp)

class MultilineDataset(TemporalDataset):
    def __init__(self,root=DEFAULT):
        super().__init__(root)
        if self.manifest['protocol'].get('version')!='rolling_multiline_v3':raise ValueError('Expected generalized rolling dataset')
        _,expected=cohort(self);rows=self.manifest['history_index'];self.event_index={int(r['event']):dict(r) for r in rows}
        if len(rows)!=len(self.event_index) or set(self.event_index)!={r['event'] for r in expected}:raise ValueError('History cohort mismatch')
        for row in expected:
            saved=self.event_index[row['event']]
            if any(saved[k]!=row[k] for k in ['event','split','start_ms','end_ms']) or sorted(saved['ranks'])!=sorted(row['ranks']):raise ValueError('History clock/rank mismatch')
            if saved['sha256']!=self.manifest['artifacts'].get(saved['file']):raise ValueError('Missing history identity')
    reveal_event=legacy.RollingDataset.reveal_event
    history_before=legacy.RollingDataset.history_before

def prepare(destination=DEFAULT):
    parent=TemporalDataset(PARENT);base=MultirankDataset(ROOT/'data/multirank_v2');old=HoldoutDataset();archive=ROOT/'data/downloads/cn_19_ranks_v1'
    wanted={(int(r.event),int(r['rank'])) for _,r in pd.read_csv(parent.root/'partitions.csv').iterrows()}
    curves=(c for c in trusted_curves(base,old,archive) if (c.event,c.rank) in wanted)
    return build(parent,destination,curves,source_identity={'original_snapshot_sha256':sha256(old.root/'snapshot.sqlite3'),
        'download_manifest_sha256':sha256(archive/'download_manifest.json'),'parent_manifest_sha256':sha256(parent.root/'manifest.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,default=DEFAULT);a=p.parse_args()
    j=prepare(a.destination);print(json.dumps({'dataset_id':j['dataset_id'],'splits':j['splits']},indent=2))
