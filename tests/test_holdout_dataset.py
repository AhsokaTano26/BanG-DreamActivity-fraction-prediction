"""Fixed holdout must survive future DB changes without moving test cases."""
import ast
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

if importlib.util.find_spec('analysis.holdout_dataset'):
    from analysis.holdout_dataset import build_dataset,HoldoutDataset
else:
    build_dataset=HoldoutDataset=None

HOUR=3600000


class HoldoutDatasetTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(build_dataset,'standard holdout builder is not implemented')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.db=self.root/'source.sqlite3';self.meta=self.root/'metadata';self.meta.mkdir()
        base=1500000000000;self.now=base+30*300*HOUR
        with sqlite3.connect(self.db) as c:
            c.execute('CREATE TABLE event (EventID INTEGER,EventType TEXT,StartAt INTEGER,EndAt INTEGER,Rank INTEGER,Country TEXT,PointRank TEXT)')
            for index in range(20):
                event=100-index;start=base+index*300*HOUR;end=start+200*HOUR
                for rank in [1000,2000]:
                    hours=list(range(0,201,1 if index else 25))
                    points=[{'time':start+h*HOUR,'ep':h*100+1} for h in hours]
                    points[-1]['time']=end+1000
                    c.execute('INSERT INTO event VALUES (?,?,?,?,?,?,?)',(event,'live_try',start,end,rank,'中国大陆',str(points)))
        metadata={'events':{str(100-i):{'eventType':'live_try','characters':[{'characterId':1}], 'members':[], 'attributes':[]} for i in range(20)},
                  'characters':{'1':{'bandId':1}},'bands':{'1':{'bandName':['A']*5}},'cards':{}}
        for name,value in metadata.items():(self.meta/f'{name}.json').write_text(json.dumps(value))

    def freeze(self):
        destination=self.root/'holdout'
        build_dataset(destination,self.db,self.meta,as_of_ms=self.now)
        return HoldoutDataset(destination)

    def test_chronology_and_activity_grouping_are_fixed(self):
        data=self.freeze();splits=data.manifest['splits']
        self.assertEqual([len(splits[s]['event_ids']) for s in ['train','validation','test']],[14,3,3])
        self.assertEqual(splits['train']['event_ids'][0],100)
        groups=[set(splits[s]['event_ids']) for s in ['train','validation','test']]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertLess(splits['train']['latest_end_ms'],splits['validation']['earliest_start_ms'])
        self.assertLess(splits['validation']['latest_end_ms'],splits['test']['earliest_start_ms'])
        self.assertIn(100,{s['curve'].event for s in data.inputs('train')})

    def test_test_labels_require_explicit_final_evaluation_access(self):
        data=self.freeze()
        with self.assertRaisesRegex(ValueError,'test|测试'):data.labels('test')
        self.assertTrue(data.labels('test',allow_test=True))

    def test_prefix_and_peer_inputs_do_not_contain_future_scores(self):
        data=self.freeze()
        for s in data.inputs('test'):
            self.assertLessEqual(int(s['times'][-1]),s['cutoff'])
            self.assertNotIn('target',s)
            peer=s.get('peer_curve')
            if peer is not None:self.assertLessEqual(int(peer.times[-1]),s['now'])

    def test_frozen_snapshot_does_not_follow_later_source_updates(self):
        data=self.freeze();before=data.manifest['dataset_id'];cases=[s['case_id'] for s in data.inputs('test')]
        with sqlite3.connect(self.db) as c:c.execute('DELETE FROM event')
        again=HoldoutDataset(data.root)
        self.assertEqual(again.manifest['dataset_id'],before)
        self.assertEqual([s['case_id'] for s in again.inputs('test')],cases)

    def test_existing_benchmark_is_never_overwritten(self):
        data=self.freeze()
        with self.assertRaises(FileExistsError):build_dataset(data.root,self.db,self.meta,as_of_ms=self.now)

    def test_input_tampering_is_detected(self):
        data=self.freeze();path=data.root/'test_inputs.jsonl';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'hash|指纹|校验'):HoldoutDataset(data.root)


if __name__=='__main__':unittest.main()
