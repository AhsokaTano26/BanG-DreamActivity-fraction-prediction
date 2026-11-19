import importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd

START=1706504400000;H=3600000
RANKS=[1000,1500,2000]

def peer_fixture():
    records=[];rows=[]
    for rank,delay,scale in [(1000,0,1.),(2000,5*60000,.5)]:
        for hours in [12,24]:
            alltimes=START+np.arange(1,25)*H+delay
            times=alltimes[alltimes<=START+hours*H]
            scores=np.arange(1,len(times)+1,dtype=float)*100*scale
            cid=f'train:91:{rank}:{START+hours*H}'
            records.append(dict(case_id=cid,event=91,rank=rank,kind='normal',start=START,end=START+48*H,
                                times=times.tolist(),scores=scores.tolist(),cutoff=START+hours*H))
            rows.append(dict(case_id=cid,event=91,rank=rank,event_type='normal',band_group='1',split='train',
                             start_ms=START,end_ms=START+48*H,cutoff=START+hours*H,observed_at=int(times[-1]),
                             current=float(scores[-1]),prefix_points=len(times)))
    return records,pd.DataFrame(rows)

class TaskPeerDataTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.task_peer_data'),'causal peer preparation missing')
        return __import__('analysis.task_peer_data',fromlist=['prepare_records'])

    def test_peer_clock_is_own_observation_and_future_values_cannot_change_it(self):
        m=self.module();records,cases=peer_fixture()
        a=m.prepare_records(records,cases,RANKS)
        poisoned=[dict(r,scores=[x*1000 if t>START+12*H else x for t,x in zip(r['times'],r['scores'])]) for r in records]
        later=cases.copy();later['current']=later.case_id.map({r['case_id']:r['scores'][-1] for r in poisoned})
        b=m.prepare_records(poisoned,later,RANKS)
        first=cases.case_id.iloc[0];ia=list(a['cases'].case_id).index(first);ib=list(b['cases'].case_id).index(first)
        for j in [0,2]:
            na,nb=int(a['links'][ia,j]),int(b['links'][ib,j])
            np.testing.assert_allclose(a['values'][na],b['values'][nb],equal_nan=True)
            self.assertLessEqual(a['node_table'].iloc[na].observed_at,cases.observed_at.iloc[0])
        self.assertEqual(a['node_table'].iloc[a['links'][ia,2]].observed_at,START+11*H+5*60000)
        self.assertEqual(a['links'][ia,1],-1)

    def test_missing_or_stale_peers_are_masked_without_removing_target_cases(self):
        m=self.module();records,cases=peer_fixture()
        peer=dict(next(r for r in records if r['rank']==2000))
        peer.update(times=peer['times'][:4],scores=peer['scores'][:4],cutoff=START+4*H+5*60000,
                    case_id=f'train:91:2000:{START+4*H+5*60000}')
        records=[r for r in records if r['rank']==1000]+[peer]
        row=cases[cases['rank']==2000].iloc[0].copy()
        row['case_id']=peer['case_id'];row['cutoff']=peer['cutoff'];row['observed_at']=peer['cutoff'];row['current']=200.;row['prefix_points']=4
        cases=pd.concat([cases[cases['rank']==1000],row.to_frame().T],ignore_index=True)
        a=m.prepare_records(records,cases,RANKS)
        self.assertEqual(set(a['cases'].case_id),set(cases.case_id))
        for i in np.flatnonzero(cases['rank']==1000):self.assertEqual(a['links'][i,2],-1)

    def test_own_record_and_case_identity_disagreement_fails(self):
        m=self.module();records,cases=peer_fixture()
        for invalid in [cases.assign(observed_at=cases.observed_at+1),cases.assign(current=1.),pd.concat([cases,cases.iloc[:1]])]:
            with self.subTest(),self.assertRaises(ValueError):m.prepare_records(records,invalid,RANKS)
        broken=[dict(r) for r in records];broken[0]=dict(broken[0],scores=[1.]*len(broken[0]['scores']))
        with self.assertRaises(ValueError):m.prepare_records(broken,cases,RANKS)
        with self.assertRaises(ValueError):m.prepare_records(records,cases,[1000,1000,2000])

    def test_saved_cache_returns_detached_exact_batches_and_checks_every_hash(self):
        m=self.module();records,cases=peer_fixture();arrays=m.prepare_records(records,cases,RANKS)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cache';m.save_cache(arrays,root,source_identity={'fixture':True},dataset_id='fixture')
            cache=m.PeerCache(root);batch=cache.batch([cases.case_id.iloc[0]])
            self.assertEqual(batch.frame.case_id.tolist(),[cases.case_id.iloc[0]])
            self.assertEqual(batch.links.shape,(1,3));self.assertEqual(batch.links[0,1],-1)
            self.assertTrue((batch.node_table.observed_at<=batch.frame.observed_at.iloc[0]).all())
            batch.nodes[:]=12345.
            self.assertFalse(np.all(cache.batch([cases.case_id.iloc[0]]).nodes==12345.))
            with self.assertRaises(ValueError):cache.batch([cases.case_id.iloc[0]]*2)
            with (root/'arrays.joblib').open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):m.PeerCache(root)

if __name__=='__main__':unittest.main()
