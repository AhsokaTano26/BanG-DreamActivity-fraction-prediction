import gzip,importlib.util,json,shutil,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from test_rolling_v3 import fixture
from analysis.holdout_dataset import curve_record,sha256,write_json
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import TemporalDataset
from analysis import rolling_dataset,rolling_benchmark,v3_next_runner

def expanded_parent(parent,curves,destination):
    shutil.copytree(parent.root,destination)
    extras=[replace(c,rank=5000,times=c.times.copy(),scores=c.scores*.4) for c in curves if c.rank==2000]
    extras += [replace(c,rank=1500,times=c.times.copy(),scores=c.scores*1.2) for c in curves if c.rank==2000 and c.event in parent.manifest['splits']['test']['event_ids']]
    parts=pd.read_csv(destination/'partitions.csv');new=[]
    for c in extras:
        split=next(s for s,x in parent.manifest['splits'].items() if c.event in x['event_ids'])
        new.append({'event':c.event,'rank':c.rank,'split':split,'start_ms':c.start,'end_ms':c.end,'observations':len(c.times)})
    pd.concat([parts,pd.DataFrame(new)],ignore_index=True).to_csv(destination/'partitions.csv',index=False)
    for split in ['train','validation','test']:
        mapping=[(5000,.4)]+([(1500,1.2)] if split=='test' else [])
        for suffix in ['cases','memberships','labels']:
            p=destination/f'{split}_{suffix}.csv';frame=pd.read_csv(p);add=[]
            for rank,scale in mapping:
                x=frame[frame.case_id.str.contains(':2000:')].copy()
                x['case_id']=x.case_id.str.replace(':2000:',f':{rank}:',regex=False)
                if 'rank' in x:x['rank']=rank
                if 'current' in x:x['current']*=scale
                if 'actual' in x:x['actual']*=scale
                add.append(x)
            pd.concat([frame,*add],ignore_index=True).to_csv(p,index=False)
        p=destination/f'{split}_inputs.jsonl.gz'
        with gzip.open(p,'rt') as f:records=[json.loads(s) for s in f]
        added=[]
        for r in records:
            if r['rank']!=2000:continue
            for rank,scale in mapping:
                v=dict(r);v['rank']=rank;v['scores']=[x*scale for x in r['scores']];v['case_id']=r['case_id'].replace(':2000:',f':{rank}:');added.append(v)
        with gzip.open(p,'wt') as f:
            for r in records+added:f.write(json.dumps(r)+'\n')
    with gzip.open(destination/'train_curves.jsonl.gz','at') as f:
        for c in extras:
            if c.event in parent.manifest['splits']['train']['event_ids']:f.write(json.dumps(curve_record(c))+'\n')
    manifest=json.loads((destination/'manifest.json').read_text());manifest.pop('dataset_id')
    manifest['protocol']['ranks']=[1000,1500,2000,5000]
    for split in ['train','validation','test']:
        manifest['splits'][split]['cases']=len(pd.read_csv(destination/f'{split}_cases.csv'))
        manifest['splits'][split]['stage_memberships']=len(pd.read_csv(destination/f'{split}_memberships.csv'))
    manifest['artifacts']={str(p.relative_to(destination)):sha256(p) for p in destination.rglob('*') if p.is_file() and p.name!='manifest.json'}
    manifest['dataset_id']=identity(manifest);(destination/'manifest.json').write_text(json.dumps(manifest))
    return TemporalDataset(destination),curves+extras

class MultilineDataBaseTests(unittest.TestCase):
    def module(self,name):
        self.assertIsNotNone(importlib.util.find_spec('analysis.'+name),name+' missing')
        return __import__('analysis.'+name,fromlist=['build'])

    def test_builder_preserves_cases_cold_rank_and_legacy_guard(self):
        m=self.module('multiline_dataset')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old,curves=fixture(root);parent,allcurves=expanded_parent(old,curves,root/'parent19')
            guard=rolling_dataset._cohort
            m.build(parent,root/'data',allcurves,source_identity={'fixture':True})
            self.assertIs(rolling_dataset._cohort,guard)
            data=m.MultilineDataset(root/'data')
            for split in ['train','validation','test']:
                self.assertEqual((data.root/f'{split}_inputs.jsonl.gz').read_bytes(),(parent.root/f'{split}_inputs.jsonl.gz').read_bytes())
                self.assertEqual(set(data.case_index(split).case_id),set(parent.case_index(split).case_id))
            self.assertFalse((data.case_index('train')['rank']==1500).any())
            event=data.event_index[900]
            with self.assertRaises(ValueError):data.reveal_event(900,before_ms=event['end_ms'])
            self.assertEqual({c.rank for c in data.reveal_event(900,before_ms=event['end_ms']+1)},{1000,2000,5000})
            with self.assertRaises(ValueError):rolling_dataset._cohort(parent)

    def test_builder_restores_guard_after_bad_full_curve(self):
        m=self.module('multiline_dataset')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old,curves=fixture(root);parent,allcurves=expanded_parent(old,curves,root/'parent19')
            guard=rolling_dataset._cohort
            bad=list(allcurves);bad[-1]=replace(bad[-1],times=bad[-1].times[:-1])
            with self.assertRaises(ValueError):m.build(parent,root/'bad',bad,source_identity={'fixture':True})
            self.assertIs(rolling_dataset._cohort,guard)

    def test_raw_cache_has_causal_weights_and_exact_context_controls(self):
        m=self.module('multiline_dataset');b=self.module('multiline_base')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old,curves=fixture(root)
            rolling_dataset.build_rolling(old,root/'pair',curves,source_identity={'fixture':True});pair=rolling_dataset.RollingDataset(root/'pair')
            rolling_benchmark.run_experiment(pair,root/'oldbase',iterations=2,allow_fixture=True)
            v3_next_runner.run_experiment(pair,root/'context',base_run=root/'oldbase',warmup_events=2,iterations=2,allow_fixture=True)
            parent,allcurves=expanded_parent(old,curves,root/'parent19');m.build(parent,root/'data',allcurves,source_identity={'fixture':True});data=m.MultilineDataset(root/'data')
            with patch.object(type(data),'labels',side_effect=AssertionError('Raw coordinator cannot open answers')):
                b.run(data,root/'raw',context_run=root/'context',pair_dataset=pair,iterations=2,allow_fixture=True)
            for event in [800,3,700,4,600,5]:
                audit=json.loads((root/f'raw/events/{event}/audit.json').read_text())
                self.assertTrue(all(r['end_ms']<audit['query_start_ms'] for r in audit['released_history']))
                weights=pd.read_csv(root/f'raw/events/{event}/training_weights.csv')
                mass=weights.groupby('event').weight.sum();np.testing.assert_allclose(mass,mass.iloc[0])
                pred=pd.read_csv(root/f'raw/events/{event}/predictions.csv')
                source=pd.read_csv(root/f'context/events/{event}/predictions.csv').query("method=='uncorrected'")
                np.testing.assert_allclose(pred.query('rank in [1000,2000]').set_index('case_id').loc[source.case_id].predicted,source.predicted,rtol=1e-12)
            cold=pd.read_csv(root/'raw/events/600/predictions.csv').query('rank==1500')
            self.assertTrue(np.isfinite(cold.predicted).all());self.assertGreater(len(cold),0)
            original=data.reveal_event
            def poison(event,*,before_ms):
                curves=original(event,before_ms=before_ms)
                if event==800:
                    for c in curves:c.scores[-1]*=2
                return curves
            with patch.object(data,'reveal_event',side_effect=poison):
                b.run(data,root/'poison',context_run=root/'context',pair_dataset=pair,iterations=2,allow_fixture=True)
            self.assertEqual((root/'raw/events/800/predictions.csv').read_bytes(),(root/'poison/events/800/predictions.csv').read_bytes())
