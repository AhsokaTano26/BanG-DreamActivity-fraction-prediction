"""Time boundaries, immutable event grouping and all-rank prediction contracts."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
from analysis.backtest import Curve,snapshot
from analysis.holdout_dataset import build_dataset,HoldoutDataset

H=3600000
class MultirankTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.multirank_dataset'),'19-rank dataset missing')
        from analysis import multirank_dataset
        return multirank_dataset

    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        meta=root/'metadata';meta.mkdir();db=root/'db.sqlite3';curves=[]
        with sqlite3.connect(db) as con:
            con.execute('CREATE TABLE event (EventID INTEGER,EventType TEXT,StartAt INTEGER,EndAt INTEGER,Rank INTEGER,Country TEXT,PointRank TEXT)')
            for i in range(20):
                event=100-i;start=1500000000000+i*300*H;end=start+200*H
                times=np.arange(start,end+1,H,dtype=np.int64);scores=np.arange(len(times))*100+1
                for rank in (1000,2000):
                    points=[{'time':int(t),'ep':int(s)} for t,s in zip(times,scores)]
                    con.execute('INSERT INTO event VALUES (?,?,?,?,?,?,?)',(event,'normal',start,end,rank,'中国大陆',str(points)))
                curves.append(Curve(event,10,'normal',start,end,times,scores.astype(float)))
        metadata={'events':{str(c.event):{'characters':[{'characterId':1}], 'members':[], 'attributes':[]} for c in curves},'characters':{'1':{'bandId':1}},'bands':{'1':{'bandName':['A']*5}},'cards':{}}
        for name,value in metadata.items():(meta/f'{name}.json').write_text(json.dumps(value))
        build_dataset(root/'base',db,meta,as_of_ms=1600000000000)
        return root,HoldoutDataset(root/'base'),curves

    def test_extension_keeps_base_cases_and_grouping_without_future_suffixes(self):
        m=self.module();root,base,curves=self.fixture()
        m.build_extension(root/'extended',base,curves,source_identity={'fixture':True})
        data=m.MultirankDataset(root/'extended')
        self.assertEqual(data.manifest['splits']['train']['event_ids'],list(range(100,86,-1)))
        for split in ('train','validation','test'):
            old=list(base.inputs(split));new=list(data.inputs(split))
            self.assertEqual({s['case_id'] for s in old},{s['case_id'] for s in new if s['curve'].rank in (1000,2000)})
            for s in new:
                self.assertLessEqual(s['curve'].times[-1],s['cutoff']);self.assertNotIn('target',s)
        with self.assertRaises(ValueError):data.labels('test')
        with self.assertRaises(FileExistsError):m.build_extension(root/'extended',base,curves,source_identity={})
        path=root/'extended'/'test_inputs.jsonl.gz';path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaises(ValueError):m.MultirankDataset(root/'extended')

    def test_unlisted_archive_curves_cannot_enter_dataset(self):
        m=self.module()
        from analysis.holdout_dataset import sha256,write_json
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);(root/'curves').mkdir()
        good={'event':1,'rank':10,'start':1000,'end':2000,'points':[{'time':1000,'ep':1}]}
        write_json(root/'curves/1_10.json',good)
        write_json(root/'config.json',{'events':[1],'ranks':[10]});write_json(root/'summary.json',{})
        (root/'records.jsonl').write_text(json.dumps({'event':1,'rank':10,'status':'ok','curve':'curves/1_10.json','curve_sha256':sha256(root/'curves/1_10.json')})+'\n')
        write_json(root/'download_manifest.json',{'artifacts':{str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}})
        write_json(root/'curves/999_10.json',good|{'event':999})
        self.assertEqual([c.event for c in m.archived_curves(root)],[1])

    def test_post_end_polling_does_not_hide_valid_final_observation(self):
        m=self.module();root,base,curves=self.fixture();c=curves[0]
        curves[0]=Curve(c.event,10,c.kind,c.start,c.end,np.append(c.times,c.end+H),np.append(c.scores,99999999.))
        m.build_extension(root/'extended',base,curves,source_identity={})
        data=m.MultirankDataset(root/'extended')
        extra=[s for s in data.inputs('train') if s['curve'].event==100 and s['curve'].rank==10]
        self.assertTrue(extra,'Complete near-end observation must survive later polling')
        labels=data.labels('train')
        self.assertEqual({labels[s['case_id']] for s in extra},{20001.})
        saved=next(c for c in data.train_curves() if c.event==100 and c.rank==10)
        self.assertEqual(int(saved.times[-1]),c.end)

    def test_only_eligible_extra_ranks_and_frozen_events_enter_dataset(self):
        m=self.module();root,base,curves=self.fixture();c=curves[0]
        curves[0]=Curve(c.event,10,c.kind,c.start,c.end,c.times[:-2],c.scores[:-2])
        curves.append(Curve(999,10,c.kind,c.start,c.end,c.times,c.scores))
        m.build_extension(root/'extended',base,curves,source_identity={})
        data=m.MultirankDataset(root/'extended')
        keys={(s['curve'].event,s['curve'].rank) for s in data.inputs('train')}
        self.assertNotIn((100,10),keys);self.assertNotIn((999,10),keys)
        self.assertIn((100,1000),keys)

    def test_prefix_features_ignore_future_suffix_and_training_events_have_equal_weight(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.multirank_model'),'19-rank model missing')
        from analysis.multirank_model import feature_row,event_weights
        times=np.arange(101)*H+1500000000000;scores=np.arange(101,dtype=float)+10
        c=Curve(1,1500,'normal',int(times[0]),int(times[-1]),times,scores)
        a=snapshot(c,int(times[50]));scores2=scores.copy();scores2[51:]=1e12
        b=snapshot(Curve(1,1500,'normal',c.start,c.end,times,scores2),int(times[50]))
        self.assertEqual(feature_row(a),feature_row(b))
        self.assertAlmostEqual(feature_row(a)['log_rank'],np.log(1500))
        w=event_weights([1,1,1,2]);self.assertAlmostEqual(sum(w[:3]),w[3])

    def test_replay_time_is_explicit_and_cannot_be_in_future(self):
        import predict_multirank as live
        self.assertTrue(hasattr(live,'parse_replay_time'),'CLI needs explicit replay cutoff')
        self.assertEqual(live.parse_replay_time('2026-11-11T22:00:00+08:00'),1794405600000)
        import asyncio
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);out=Path(tmp.name)/'future'
        with self.assertRaises(ValueError):asyncio.run(live.fetch_prefixes(out,as_of=9999999999999))
        self.assertFalse(out.exists())

    def test_live_fetch_clips_future_points_and_records_missing_ranks(self):
        import asyncio,httpx
        from predict_multirank import fetch_prefixes
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);out=Path(tmp.name)/'live'
        start=1500000000000;cutoff=start+100000
        times=[start+i*1000 for i in range(1,5)]+[cutoff+1]
        def handler(request):
            if request.url.path.endswith('all.5.json'):
                return httpx.Response(200,json={'100':{'startAt':[None,None,None,start],'endAt':[None,None,None,start+200000],'eventType':'normal'}})
            if request.url.path.endswith('eventtop/data'):
                return httpx.Response(200,json={'points':[{'time':t,'uid':uid,'value':uid*100+i} for i,t in enumerate(times) for uid in range(1,11)]})
            rank=int(request.url.params['tier'])
            return httpx.Response(200,json={'result':True,'cutoffs':[] if rank==20000 else [{'time':t,'ep':i+1} for i,t in enumerate(times)]})
        curves,actual=asyncio.run(fetch_prefixes(out,as_of=cutoff,transport=httpx.MockTransport(handler)))
        self.assertEqual(actual,cutoff);self.assertEqual(len(curves),18)
        self.assertEqual({int(c.times[-1]) for c in curves},{start+4000})
        self.assertEqual(next(c for c in curves if c.rank==10).final,103.)
        self.assertEqual(json.loads((out/'fetch.json').read_text())['failures'][0]['rank'],20000)

    def test_real_small_pipeline_is_sealed_and_scores_exact_cases(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.multirank_model'),'19-rank model missing')
        m=self.module();root,base,curves=self.fixture();m.build_extension(root/'extended',base,curves,source_identity={})
        from analysis.multirank_model import run_experiment,predict_samples
        from analysis.multirank_evaluate import evaluate
        data=m.MultirankDataset(root/'extended');run=root/'run'
        run_experiment(data,run,iterations=5)
        occupied=root/'occupied';occupied.mkdir();(occupied/'test_summary.csv').write_text('preserve')
        with self.assertRaises(FileExistsError):evaluate(run,occupied)
        self.assertEqual((occupied/'test_summary.csv').read_text(),'preserve')
        self.assertFalse((occupied/'evaluation.json').exists())
        result=evaluate(run);self.assertEqual(result['selected']['events'],3)
        self.assertEqual(evaluate(run),result)
        samples=list(data.inputs('test'))[:2]
        output=predict_samples(run,samples)
        self.assertEqual(len(output),2);self.assertTrue(np.isfinite(output).all())
        with self.assertRaises(FileExistsError):run_experiment(data,run,iterations=5)
        from predict_multirank import forecast_rows
        curves=[samples[0]['curve']]
        rows=forecast_rows(run,curves,samples[0]['cutoff'])
        self.assertEqual(len(rows),19)
        self.assertTrue(any(r['status']=='unavailable' for r in rows))
        self.assertTrue(any(r['status']=='ok' for r in rows))
        c=curves[0];unseen=Curve(c.event,1500,c.kind,c.start,c.end,c.times,c.scores)
        sparse=forecast_rows(run,[unseen],samples[0]['cutoff'])
        row=next(r for r in sparse if r['rank']==1500)
        self.assertTrue(row.get('provisional'), 'Unseen rank should be explicit')
        self.assertEqual(row['training_events'],0)
        from predict_multirank import runtime_identity
        self.assertIn('httpx',runtime_identity()['software'])
        self.assertIn('python',runtime_identity())
        past=next(data.inputs('train'))
        with self.assertRaisesRegex(ValueError,'training'):
            forecast_rows(run,[past['curve']],past['cutoff'])
        path=run/'test_predictions.csv';path.write_text(path.read_text()+'\n')
        with self.assertRaises(ValueError):evaluate(run)

if __name__=='__main__':unittest.main()
