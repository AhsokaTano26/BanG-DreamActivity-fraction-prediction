"""New dense cohorts exclude manual-era history without rewriting old experiments."""
import gzip
import importlib.util
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.backtest import Curve
from analysis.holdout_dataset import HoldoutDataset, build_dataset, sha256
from analysis.multirank_dataset import MultirankDataset, build_extension
from analysis.temporal_dataset import TemporalDataset, build_temporal

HOUR = 3600000
CUTOFF = 1706504400000


class DenseDatasetTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.dense_dataset'),
                             'Dense cohort builder must be implemented')
        from analysis import dense_dataset
        return dense_dataset

    def fixture(self, temporal=False):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name); meta = root/'metadata'; meta.mkdir()
        curves = []; source = root/'source.sqlite3'
        with closing(sqlite3.connect(source)) as con:
            con.execute('CREATE TABLE event (EventID INTEGER,EventType TEXT,StartAt INTEGER,EndAt INTEGER,Rank INTEGER,Country TEXT,PointRank TEXT)')
            for i in range(20):
                event = 1000-i; start = CUTOFF+(i-5)*300*HOUR; end = start+200*HOUR
                times = np.arange(start, end+1, HOUR, dtype=np.int64)
                for rank in (10,1000,2000):
                    scores = np.arange(len(times), dtype=float)*100 + rank
                    c = Curve(event, rank, 'normal', start, end, times, scores)
                    curves.append(c)
                    if rank != 10:
                        points = [{'time':int(t),'ep':float(s)} for t,s in zip(times,scores)]
                        con.execute('INSERT INTO event VALUES (?,?,?,?,?,?,?)',
                                    (event,'normal',start,end,rank,'中国大陆',str(points)))
            con.commit()
        metadata = {'events': {str(1000-i): {'eventType':'normal','characters':[{'characterId':1}],
                     'members':[], 'attributes':[]} for i in range(20)},
                    'characters': {'1':{'bandId':1}},'bands': {'1':{'bandName':['A']*5}},'cards':{},
                    'event_contexts': [{'event_id':1000-i,'band_shares':{'1':1.}} for i in range(20)]}
        for name,value in metadata.items(): (meta/f'{name}.json').write_text(json.dumps(value))
        build_dataset(root/'original',source,meta,as_of_ms=CUTOFF+30*300*HOUR)
        original = HoldoutDataset(root/'original')
        if temporal:
            build_extension(root/'multirank',original,[c for c in curves if c.rank==10],source_identity={'fixture':True})
            base = MultirankDataset(root/'multirank')
            build_temporal(root/'parent',base,curves,source_identity={'fixture':True})
        return root, original, curves

    def test_date_cutoff_resplits_all_ranks_and_removes_old_history_and_metadata(self):
        m=self.module(); root,old,_=self.fixture()
        hashes={str(p.relative_to(old.root)):sha256(p) for p in old.root.rglob('*') if p.is_file()}
        result=m.build_dense_v1(old,root/'dense')
        data=HoldoutDataset(root/'dense')
        self.assertEqual(result['dataset_id'],data.manifest['dataset_id'])
        groups=data.manifest['splits']
        self.assertEqual(groups['train']['event_ids'],list(range(995,985,-1)))
        self.assertEqual(groups['validation']['event_ids'],[985,984])
        self.assertEqual(groups['test']['event_ids'],[983,982,981])
        self.assertEqual(set(c.event for c in data.train_curves()),set(range(995,985,-1)))
        self.assertTrue(all(c.start>=CUTOFF for c in data.train_curves()))
        self.assertEqual(set(json.loads((data.metadata_directory/'events.json').read_text())),
                         {str(i) for i in range(981,996)})
        self.assertEqual(len(json.loads((data.metadata_directory/'event_contexts.json').read_text())),15)
        with closing(sqlite3.connect(data.root/'snapshot.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT MIN(StartAt),COUNT(DISTINCT EventID) FROM event').fetchone(),(CUTOFF,15))
        for split in ('train','validation','test'):
            for sample in data.inputs(split):
                self.assertGreaterEqual(sample['curve'].start,CUTOFF)
                self.assertLessEqual(sample['times'][-1],sample['cutoff'])
                self.assertLessEqual(sample['peer_curve'].times[-1],sample['now'])
        self.assertEqual(hashes,{str(p.relative_to(old.root)):sha256(p) for p in old.root.rglob('*') if p.is_file()})
        with self.assertRaises(ValueError): data.labels('test')
        with self.assertRaises(FileExistsError): m.build_dense_v1(old,data.root)

    def test_temporal_migration_preserves_physical_cases_labels_and_stage_aliases(self):
        m=self.module();root,old,curves=self.fixture(temporal=True)
        m.build_dense_v1(old,root/'dense');dense=HoldoutDataset(root/'dense')
        parent=TemporalDataset(root/'parent');old_id=parent.manifest['dataset_id']
        m.build_dense_temporal(parent,dense,root/'dense_temporal',curves,source_identity={'fixture':True})
        new=TemporalDataset(root/'dense_temporal')
        reference={}
        for split in ('train','validation','test'):
            with gzip.open(parent.root/f'{split}_inputs.jsonl.gz','rt') as f:
                for line in f:
                    row=json.loads(line)
                    if row['start']>=CUTOFF: reference[row['case_id'].split(':',1)[1]]=row
        seen=set()
        for split in ('train','validation','test'):
            self.assertEqual(new.manifest['splits'][split]['event_ids'],dense.manifest['splits'][split]['event_ids'])
            cases=new.case_index(split);members=new.memberships(split)
            self.assertEqual(set(cases.case_id),set(members.case_id))
            self.assertFalse(cases.duplicated(['event','rank','cutoff']).any())
            with gzip.open(new.root/f'{split}_inputs.jsonl.gz','rt') as f:
                for line in f:
                    row=json.loads(line);suffix=row['case_id'].split(':',1)[1]
                    before=reference[suffix];seen.add(suffix)
                    self.assertEqual(row,{**before,'case_id':split+':'+suffix})
            labels=new.labels(split,allow_test=True)
            finals={(c.event,c.rank):c.final for c in curves}
            self.assertEqual(labels,{r.case_id:finals[(r.event,r['rank'])] for _,r in cases.iterrows()})
        self.assertEqual(seen,set(reference))
        history=list(new.train_curves())
        self.assertEqual(len(history),30)
        self.assertTrue(all(c.start>=CUTOFF and c.event in range(986,996) for c in history))
        self.assertEqual(TemporalDataset(parent.root).manifest['dataset_id'],old_id)
        coverage=pd.read_csv(new.root/'coverage.csv')
        row=coverage[(coverage.split=='test')&(coverage['rank']==1000)&(coverage.stage_id=='elapsed_1h')].iloc[0]
        self.assertEqual((row.group_events,row.eligible_curves,row.available_cases,row.missing_prefixes),(3,3,0,3))

    def test_temporal_builder_rejects_missing_history_and_keeps_destination_absent(self):
        m=self.module();root,old,curves=self.fixture(temporal=True)
        m.build_dense_v1(old,root/'dense');dense=HoldoutDataset(root/'dense')
        parent=TemporalDataset(root/'parent')
        # Drop a newly-training activity's full curve while retaining every prefix.
        incomplete=[c for c in curves if not(c.event==986 and c.rank==10)]
        with self.assertRaisesRegex(ValueError,'[Mm]issing|[Hh]istory|[Cc]ohort'):
            m.build_dense_temporal(parent,dense,root/'bad',incomplete,source_identity={})
        self.assertFalse((root/'bad').exists())

    def test_temporal_builder_rejects_changed_frozen_curve_dates(self):
        m=self.module();root,old,curves=self.fixture(temporal=True)
        m.build_dense_v1(old,root/'dense');dense=HoldoutDataset(root/'dense')
        from dataclasses import replace
        changed=[replace(c,start=c.start+1) if c.event==986 and c.rank==10 else c for c in curves]
        with self.assertRaisesRegex(ValueError,'[Dd]iffer|[Mm]ismatch|[Cc]ohort'):
            m.build_dense_temporal(TemporalDataset(root/'parent'),dense,root/'bad',changed,source_identity={})
        self.assertFalse((root/'bad').exists())

    def test_temporal_builder_rejects_changed_intermediate_history_scores(self):
        m=self.module();root,old,curves=self.fixture(temporal=True)
        m.build_dense_v1(old,root/'dense');dense=HoldoutDataset(root/'dense')
        from dataclasses import replace
        changed=[]
        for c in curves:
            if c.event==986 and c.rank==10:
                scores=c.scores.copy();scores[12]+=50
                c=replace(c,scores=scores)
            changed.append(c)
        with self.assertRaisesRegex(ValueError,'[Pp]refix|[Cc]urve'):
            m.build_dense_temporal(TemporalDataset(root/'parent'),dense,root/'bad',changed,source_identity={})
        self.assertFalse((root/'bad').exists())

    def test_temporal_builder_rejects_unpaired_time_and_score_lengths(self):
        m=self.module();root,old,curves=self.fixture(temporal=True)
        m.build_dense_v1(old,root/'dense');dense=HoldoutDataset(root/'dense')
        from dataclasses import replace
        changed=[replace(c,scores=np.append(c.scores,c.final)) if c.event==986 and c.rank==10 else c
                 for c in curves]
        with self.assertRaisesRegex(ValueError,'[Cc]urve|[Cc]ohort'):
            m.build_dense_temporal(TemporalDataset(root/'parent'),dense,root/'bad',changed,source_identity={})
        self.assertFalse((root/'bad').exists())


if __name__=='__main__': unittest.main()
