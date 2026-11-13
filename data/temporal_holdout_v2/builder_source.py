"""Temporal V2: fixed activity cohort, matched time grid, blind prefix inputs."""
from __future__ import annotations
import argparse,ast,csv,gzip,importlib.metadata,json,platform,shutil,sqlite3,tempfile
from collections import Counter,defaultdict
from contextlib import ExitStack,closing
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT,TZ,Curve,snapshot
from analysis.holdout_dataset import HoldoutDataset,SPLITS,curve_record,from_record,prefix_curve,sha256,write_json,write_csv
from analysis.multirank_dataset import MultirankDataset,archived_curves,identity
from analysis.event_features import EventFeatures
from download_ranks import RANKS

DEFAULT=ROOT/'data/temporal_holdout_v2'
HOUR=3600000
BUILD_SOURCES=['analysis/temporal_dataset.py','analysis/multirank_dataset.py','analysis/backtest.py',
    'analysis/holdout_dataset.py','analysis/event_features.py','download_ranks.py']
STAGES=[{'stage_id':f'elapsed_{h:g}h','axis':'elapsed','value':h} for h in (1,3,6,12,24,48,72)]
STAGES += [{'stage_id':f'progress_{p}pct','axis':'progress','value':p/100} for p in (10,25,50,75,90,95)]
STAGES += [{'stage_id':f'remaining_{h:g}h','axis':'remaining','value':h} for h in (48,24,12,6,3,1,.5,.25)]


def origin_groups(curve):
    groups=defaultdict(list)
    for stage in STAGES:
        if stage['axis']=='elapsed':cutoff=curve.start+round(stage['value']*HOUR)
        elif stage['axis']=='remaining':cutoff=curve.end-round(stage['value']*HOUR)
        else:cutoff=curve.start+round(stage['value']*curve.duration)
        groups[int(cutoff)].append(stage)
    return [{'cutoff':t,'stages':stages} for t,stages in sorted(groups.items())]


def trusted_curves(base,original,archive):
    """Builder-only source path; never called from prediction or scoring."""
    archive=Path(archive)
    if original.manifest['dataset_id']!=base.manifest['base_dataset_id']:
        raise ValueError('Original frozen source dataset identity differs')
    expected_archive=base.manifest['source_identity'].get('download_manifest_sha256')
    if expected_archive is None or sha256(archive/'download_manifest.json')!=expected_archive:
        raise ValueError('Download archive differs from frozen cohort source')
    rows=pd.read_csv(base.root/'partitions.csv')
    wanted={(int(r.event),int(r['rank'])) for _,r in rows.iterrows()}
    with closing(sqlite3.connect((original.root/'snapshot.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)) as con:
        con.row_factory=sqlite3.Row
        for row in con.execute("SELECT * FROM event WHERE Country='中国大陆' AND Rank IN (1000,2000)"):
            if (row['EventID'],row['Rank']) not in wanted:continue
            points=ast.literal_eval(row['PointRank'])
            yield Curve(row['EventID'],row['Rank'],row['EventType'],row['StartAt'],row['EndAt'],
                np.array([p['time'] for p in points],dtype=np.int64),np.array([p['ep'] for p in points],dtype=float))
    metadata=json.loads((base.metadata_directory/'events.json').read_text())
    for c in archived_curves(archive):
        if c.rank in (1000,2000) or (c.event,c.rank) not in wanted:continue
        stop=int(np.searchsorted(c.times,c.end+300000,side='right'))
        c.times=c.times[:stop];c.scores=c.scores[:stop];c.kind=metadata[str(c.event)]['eventType']
        yield c


def build_temporal(destination,base,curves,*,source_identity):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='.temporal-v2-',dir=destination.parent))
    try:
        partitions=pd.read_csv(base.root/'partitions.csv')
        expected={(int(r.event),int(r['rank'])):r for _,r in partitions.iterrows()}
        memberships={int(e):s for s in SPLITS for e in base.manifest['splits'][s]['event_ids']}
        if len(memberships)!=sum(len(base.manifest['splits'][s]['event_ids']) for s in SPLITS):raise ValueError('Overlapping event groups')
        shutil.copytree(base.metadata_directory,work/'metadata');shutil.copyfile(base.root/'partitions.csv',work/'partitions.csv')
        metadata=EventFeatures.load(base.metadata_directory)
        indexes={s:[] for s in SPLITS};stage_rows={s:[] for s in SPLITS};unavailable=[];seen=set();counts=Counter()
        with ExitStack() as stack:
            inputs={s:stack.enter_context(gzip.open(work/f'{s}_inputs.jsonl.gz','wt',compresslevel=1)) for s in SPLITS}
            history=stack.enter_context(gzip.open(work/'train_curves.jsonl.gz','wt',compresslevel=1))
            labels={s:csv.writer(stack.enter_context((work/f'{s}_labels.csv').open('w',newline=''))) for s in SPLITS}
            for writer in labels.values():writer.writerow(['case_id','actual'])
            for c in curves:
                key=(c.event,c.rank)
                if key not in expected or key in seen:raise ValueError('Unexpected/duplicate curve cohort member')
                seen.add(key);partition=expected[key];split=memberships[c.event]
                if (partition['split']!=split or int(partition.start_ms)!=c.start or int(partition.end_ms)!=c.end
                    or int(partition.observations)!=len(c.times)):raise ValueError('Frozen cohort timing/observation count differs')
                if (c.duration<=0 or len(c.times)<5 or np.any(np.diff(c.times)<=0) or not np.isfinite(c.scores).all()
                    or np.any(c.scores<0) or c.final<=0 or not -300000<=c.end-int(c.times[-1])<=60000):raise ValueError('Invalid frozen cohort curve')
                if split=='train':history.write(json.dumps(curve_record(c),separators=(',',':'))+'\n')
                date=datetime.fromtimestamp(c.start/1000,TZ)
                for group in origin_groups(c):
                    cutoff=group['cutoff'];reason=None
                    if not c.start<cutoff<c.end:reason='origin outside event'
                    sample=snapshot(c,cutoff) if reason is None else None
                    if sample is None and reason is None:reason='fewer than 4 points, nonpositive score or age >60min'
                    if sample is None:
                        for stage in group['stages']:
                            unavailable.append({'split':split,'event':c.event,'rank':c.rank,'stage_id':stage['stage_id'],
                                'cutoff':cutoff,'reason':reason})
                        continue
                    case=f'{split}:{c.event}:{c.rank}:{cutoff}'
                    rec=curve_record(prefix_curve(c,cutoff))|{'case_id':case,'stage':group['stages'][0]['stage_id'],
                        'stage_memberships':[s['stage_id'] for s in group['stages']],'cutoff':cutoff,'peer':None}
                    inputs[split].write(json.dumps(rec,separators=(',',':'))+'\n');labels[split].writerow([case,c.final])
                    indexes[split].append({'case_id':case,'event':c.event,'rank':c.rank,'event_type':c.kind,
                        'band_group':metadata.band_only(c)['band_group'],
                        'start_ms':c.start,'end_ms':c.end,'cutoff':cutoff,'observed_at':sample['now'],
                        'current':sample['score'],'prefix_points':len(sample['times']),
                        'requested_elapsed_h':(cutoff-c.start)/HOUR,'requested_remaining_h':(c.end-cutoff)/HOUR,
                        'observed_remaining_h':(c.end-sample['now'])/HOUR,
                        'observation_age_minutes':(cutoff-sample['now'])/60000,
                        'calendar_quarter':f'{date.year}Q{(date.month-1)//3+1}'})
                    for stage in group['stages']:
                        stage_rows[split].append({'case_id':case,**stage});counts[(split,c.rank,stage['stage_id'])]+=1
        if seen!=set(expected):raise ValueError('Missing cohort curves')
        for split in SPLITS:
            if not indexes[split]:raise ValueError('No usable cases in split '+split)
            write_csv(work/f'{split}_cases.csv',indexes[split],list(indexes[split][0]))
            write_csv(work/f'{split}_memberships.csv',stage_rows[split],['case_id','stage_id','axis','value'])
        write_csv(work/'unavailable_cases.csv',unavailable,['split','event','rank','stage_id','cutoff','reason'])
        coverage=[]
        for split in SPLITS:
            events=len(base.manifest['splits'][split]['event_ids'])
            for rank in RANKS:
                eligible=int(((partitions['split']==split)&(partitions['rank']==rank)).sum())
                for stage in STAGES:
                    n=counts[(split,rank,stage['stage_id'])]
                    coverage.append({'split':split,'rank':rank,**stage,'group_events':events,'eligible_curves':eligible,
                        'missing_curves':events-eligible,'available_cases':n,'missing_prefixes':eligible-n})
        write_csv(work/'coverage.csv',coverage,list(coverage[0]))
        shutil.copyfile(__file__,work/'builder_source.py')
        (work/'builder_sources').mkdir()
        for name in BUILD_SOURCES:shutil.copyfile(ROOT/name,work/'builder_sources'/Path(name).name)
        protocol=base.manifest['protocol']|{'version':'temporal_holdout_v2','scope':'Temporal protocol V2; preserve V1 as historical reference',
            'stages':STAGES,'evaluation_stages':[s['stage_id'] for s in STAGES],
            'training_sampling':'same prescribed origin grid for all three groups',
            'case_key':'split/event/rank/requested cutoff; alias stages do not duplicate physical cases',
            'maximum_observation_age_minutes':60,'fresh_subset_maximum_age_minutes':15,
            'cohort':'exact inherited event and eligible curve lists','first_experiment':'frozen existing models, no refit or new selection'}
        manifest={'protocol':protocol,'base_dataset_id':base.manifest['dataset_id'],'original_v1_dataset_id':base.manifest['base_dataset_id'],
            'source_identity':source_identity,'python':platform.python_version(),
            'build_environment':{'source_hashes':{n:sha256(ROOT/n) for n in BUILD_SOURCES},
                'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','httpx','scikit-learn','xgboost']}},
            'splits':{s:{'event_ids':base.manifest['splits'][s]['event_ids'],'cases':len(indexes[s]),
                'stage_memberships':len(stage_rows[s])} for s in SPLITS},
            'artifacts':{str(p.relative_to(work)):sha256(p) for p in sorted(work.rglob('*')) if p.is_file()}}
        manifest['dataset_id']=identity(manifest);write_json(work/'manifest.json',manifest);work.rename(destination)
        return manifest
    finally:
        if work.exists():shutil.rmtree(work)


class TemporalDataset(MultirankDataset):
    def __init__(self,root=DEFAULT):super().__init__(root)
    def case_index(self,split):
        if split not in SPLITS:raise ValueError(split)
        return pd.read_csv(self.root/f'{split}_cases.csv')
    def memberships(self,split):
        if split not in SPLITS:raise ValueError(split)
        return pd.read_csv(self.root/f'{split}_memberships.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,default=DEFAULT)
    p.add_argument('--base',type=Path,default=ROOT/'data/multirank_v2')
    p.add_argument('--archive',type=Path,default=ROOT/'data/downloads/cn_19_ranks_v1');a=p.parse_args()
    base=MultirankDataset(a.base);original=HoldoutDataset()
    result=build_temporal(a.destination,base,trusted_curves(base,original,a.archive),source_identity={
        'old_two_rank_snapshot_sha256':sha256(original.root/'snapshot.sqlite3'),
        'download_manifest_sha256':sha256(a.archive/'download_manifest.json')})
    print(json.dumps({'dataset_id':result['dataset_id'],'splits':{s:{k:v for k,v in d.items() if k!='event_ids'} for s,d in result['splits'].items()}}))
