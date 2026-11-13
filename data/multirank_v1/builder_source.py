"""Supplementary 19-rank dataset retaining holdout_v1 event membership exactly."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from analysis.backtest import ROOT,snapshot
from analysis.holdout_dataset import (HoldoutDataset,PROTOCOL,SPLITS,curve_record,from_record,
    candidate_cutoffs,prefix_curve,sha256,write_json,write_csv)
from download_ranks import RANKS,verify_download,latest_records

DEFAULT=ROOT/'data/multirank_v1'

def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def eligible(curve,as_of):
    if curve.duration<=0 or len(curve.times)<5:return 'invalid duration or fewer than 5 observations'
    if np.any(np.diff(curve.times)<=0) or not np.isfinite(curve.scores).all() or np.any(curve.scores<0):return 'invalid times/scores'
    if curve.end>=as_of:return 'ongoing'
    if curve.final<=0:return 'nonpositive final'
    if not -300000<=curve.end-int(curve.times[-1])<=60000:return 'no near-end final'
    return None

def archived_curves(archive):
    archive=Path(archive);verify_download(archive)
    for (event,rank),record in sorted(latest_records(archive).items()):
        if record['status']!='ok':continue
        path=archive/record['curve']
        d=json.loads(path.read_text())
        if (d['event'],d['rank'])!=(event,rank):raise ValueError('Archive curve identity mismatch')
        yield from_record({'event':d['event'],'rank':d['rank'],'kind':'', 'start':d['start'],'end':d['end'],
            'times':[p['time'] for p in d['points']],'scores':[p['ep'] for p in d['points']]})

def build_extension(destination,base,additional_curves,*,source_identity):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='.multirank-',dir=destination.parent))
    try:
        shutil.copytree(base.metadata_directory,work/'metadata')
        metadata=json.loads((work/'metadata/events.json').read_text())
        membership={e:s for s in SPLITS for e in base.manifest['splits'][s]['event_ids']}
        partitions=[];excluded=[];unavailable=[];counts=Counter();curve_counts=Counter();seen=set()
        with (base.root/'partitions.csv').open() as f:partitions.extend(csv.DictReader(f))
        with ExitStack() as stack:
            inputs={s:stack.enter_context(gzip.open(work/f'{s}_inputs.jsonl.gz','wt',compresslevel=1)) for s in SPLITS}
            history=stack.enter_context(gzip.open(work/'train_curves.jsonl.gz','wt',compresslevel=1))
            labels={s:csv.writer(stack.enter_context((work/f'{s}_labels.csv').open('w',newline=''))) for s in SPLITS}
            for s in SPLITS:
                labels[s].writerow(['case_id','actual'])
                with (base.root/f'{s}_inputs.jsonl').open() as f:
                    for line in f:
                        inputs[s].write(line);r=json.loads(line);counts[s]+=1;counts[(s,r['rank'])]+=1
                # Dataset construction is the trusted label-writing stage.
                with (base.root/f'{s}_labels.csv').open() as f:
                    labels[s].writerows((r['case_id'],r['actual']) for r in csv.DictReader(f))
            with (base.root/'train_curves.jsonl').open() as f:shutil.copyfileobj(f,history)
            for c in additional_curves:
                if c.rank in (1000,2000):continue
                if (c.event,c.rank) in seen:raise ValueError('Duplicate additional event/rank')
                seen.add((c.event,c.rank))
                if c.rank not in RANKS:raise ValueError('Unexpected rank')
                reason='event outside frozen v1 membership' if c.event not in membership else eligible(c,base.manifest['as_of_ms'])
                if reason:excluded.append({'event':c.event,'rank':c.rank,'reason':reason});continue
                if str(c.event) not in metadata:raise ValueError('Missing frozen metadata')
                c.kind=metadata[str(c.event)].get('eventType') or c.kind
                split=membership[c.event];curve_counts[(split,c.rank)]+=1
                partitions.append({'split':split,'event':c.event,'rank':c.rank,'start_ms':c.start,'end_ms':c.end,
                    'observations':len(c.times),'score_decreases':int(np.sum(np.diff(c.scores)<0))})
                if split=='train':history.write(json.dumps(curve_record(c),separators=(',',':'))+'\n')
                for stage,cutoff in candidate_cutoffs(c,split):
                    sample=snapshot(c,cutoff);case=f'{split}:{c.event}:{c.rank}:{stage}'
                    if sample is None:
                        unavailable.append({'split':split,'event':c.event,'rank':c.rank,'stage':stage,'cutoff_ms':cutoff,'reason':'same prefix eligibility as v1'});continue
                    record=curve_record(prefix_curve(c,cutoff))|{'case_id':case,'stage':stage,'cutoff':cutoff,'peer':None}
                    inputs[split].write(json.dumps(record,separators=(',',':'))+'\n')
                    labels[split].writerow([case,c.final]);counts[split]+=1;counts[(split,c.rank)]+=1
        write_csv(work/'partitions.csv',partitions,['split','event','rank','start_ms','end_ms','observations','score_decreases'])
        write_csv(work/'excluded_curves.csv',excluded,['event','rank','reason'])
        write_csv(work/'unavailable_cases.csv',unavailable,['split','event','rank','stage','cutoff_ms','reason'])
        coverage=[{'rank':r,**{s+'_cases':counts[(s,r)] for s in SPLITS},
            **{s+'_events':len({int(p['event']) for p in partitions if p['split']==s and int(p['rank'])==r}) for s in SPLITS}} for r in RANKS]
        write_csv(work/'coverage.csv',coverage,list(coverage[0]))
        shutil.copyfile(__file__,work/'builder_source.py')
        manifest={'protocol':PROTOCOL|{'version':'multirank_v1','ranks':RANKS,'scope':'supplementary; not a replacement of holdout_v1',
            'group':'exact inherited holdout_v1 event memberships; no new split'},
            'base_dataset_id':base.manifest['dataset_id'],'source_identity':source_identity,
            'splits':{s:{'event_ids':base.manifest['splits'][s]['event_ids'],'cases':counts[s]} for s in SPLITS},
            'artifacts':{str(p.relative_to(work)):sha256(p) for p in sorted(work.rglob('*')) if p.is_file()}}
        manifest['dataset_id']=identity(manifest);write_json(work/'manifest.json',manifest);work.rename(destination)
        return manifest
    finally:
        if work.exists():shutil.rmtree(work)

class MultirankDataset:
    def __init__(self,root=DEFAULT):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if identity({k:v for k,v in self.manifest.items() if k!='dataset_id'})!=self.manifest['dataset_id']:raise ValueError('Dataset manifest hash mismatch')
        for name,digest in self.manifest['artifacts'].items():
            if sha256(self.root/name)!=digest:raise ValueError('Dataset artifact hash mismatch: '+name)
    @property
    def metadata_directory(self):return self.root/'metadata'
    def inputs(self,split):
        if split not in SPLITS:raise ValueError(split)
        with gzip.open(self.root/f'{split}_inputs.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line);c=from_record(r)
                if c.times[-1]>r['cutoff']:raise ValueError('Future observation in blind prefix')
                s=snapshot(c,r['cutoff'])
                if s is None:raise ValueError('Invalid frozen case')
                s.update({'case_id':r['case_id'],'stage':r['stage']});yield s
    def labels(self,split,*,allow_test=False):
        if split not in SPLITS or (split=='test' and not allow_test):raise ValueError('Test labels restricted to final scorer')
        with (self.root/f'{split}_labels.csv').open() as f:return {r['case_id']:float(r['actual']) for r in csv.DictReader(f)}
    def train_curves(self):
        with gzip.open(self.root/'train_curves.jsonl.gz','rt') as f:
            for line in f:yield from_record(json.loads(line))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,default=DEFAULT)
    p.add_argument('--archive',type=Path,default=ROOT/'data/downloads/cn_19_ranks_v1');a=p.parse_args()
    verify_download(a.archive);base=HoldoutDataset()
    result=build_extension(a.destination,base,archived_curves(a.archive),source_identity={'download_manifest_sha256':sha256(a.archive/'download_manifest.json')})
    print(json.dumps({'dataset_id':result['dataset_id'],'splits':result['splits']},ensure_ascii=False))
