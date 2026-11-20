import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from analysis.backtest import Curve
from analysis.data_policy import MIN_EVENT_START_MS as S
H=3600000

class FakeStore:
    curves=[];index=[]
    def __init__(self,root):self.root=Path(root);self.metadata_directory=self.root/'metadata';self.metadata_directory.mkdir(parents=True,exist_ok=True)
    def bootstrap(self,**kwargs):return {'status':'ok'}
    def events(self,now_ms):return [r for r in self.index if r['start_ms']<=now_ms]
    def load_curves(self,before_ms,ranks=None,complete_only=True,*,events=None):
        return [c for c in self.curves if c.rank in ranks and c.start<before_ms
                and (events is None or c.event in events) and (not complete_only or c.end<before_ms)]

class ProductionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'live'
        self.runtime=importlib.import_module('production.runtime');self.fit_ids=[]
        FakeStore.curves=[];FakeStore.index=[]
        for event,start in [(900,S),(9,S+3*H),(2,S+5*H)]:
            end=start+H;FakeStore.index.append({'event':event,'start_ms':start,'end_ms':end,'kind':'normal','name':str(event)})
            FakeStore.curves.append(Curve(event,1000,'normal',start,end,np.array([start+1,start+H//6,start+H//3,start+H//2,end]),np.array([1.,20.,40.,50.,100.])))
        self.config={'root':str(self.root),'ranks':[1000],'interval_seconds':900,'bootstrap_database':'unused','bootstrap_metadata':'unused','seed_run':None,'feedback_grace_ms':0}
        def train(curves,metadata_directory,versions_directory,*,before_ms,ranks):
            assert all(c.end<before_ms for c in curves)
            key=hashlib.sha256(str([(c.event,c.final) for c in curves]).encode()).hexdigest()[:12]
            p=Path(versions_directory)/key
            if not p.exists():
                p.mkdir(parents=True);self.fit_ids.append(key)
                (p/'manifest.json').write_text(json.dumps({'model_id':key,'history_max_end_ms':max(c.end for c in curves),'production_ready':True,'training_events':len(curves),'ranks':ranks}))
            return p
        def predict(path,samples,**kwargs):
            return [{'event':s['curve'].event,'rank':s['curve'].rank,'phase':1,'observed_at':s['now'],'current_ep':s['score'],'raw_predicted_final_ep':110.,'predicted_final_ep':100.,'available':True} for s in samples]
        async def sync(*args,**kwargs):return {'status':'ok','errors':[]}
        for name,value in [('Store',FakeStore),('train_pack',train),('verify_pack',lambda p:json.loads((Path(p)/'manifest.json').read_text())),('predict_pack',predict),('sync_data',sync),('build_bias',lambda rows,**kw:{'feedback_groups':len(rows)})]:
            patcher=patch.object(self.runtime,name,value);patcher.start();self.addCleanup(patcher.stop)

    def test_predict_then_completed_feedback_trains_new_version_and_pins_are_stable(self):
        first=self.runtime.run_cycle(self.config,now_ms=S+3*H+H//2)
        self.assertTrue(first['predictions']);self.assertEqual(first['ledger']['forecasts'],1)
        pin=self.runtime.Ledger(self.root).pin(9)
        repeated=self.runtime.run_cycle(self.config,now_ms=S+3*H+H//2)
        self.assertEqual(repeated['ledger']['forecasts'],1);self.assertEqual(len(self.fit_ids),1)
        following=self.runtime.run_cycle(self.config,now_ms=S+5*H+H//2)
        self.assertEqual(len(self.fit_ids),2);self.assertEqual(following['ledger']['live_feedback_groups'],1)
        self.assertEqual(self.runtime.Ledger(self.root).pin(9),pin)
        self.assertNotEqual(self.runtime.Ledger(self.root).pin(2)['model_id'],pin['model_id'])

    def test_failed_update_keeps_previous_valid_model_and_predictions(self):
        self.runtime.run_cycle(self.config,now_ms=S+3*H+H//2)
        with patch.object(self.runtime,'train_pack',side_effect=RuntimeError('training failed')):
            next_cycle=self.runtime.run_cycle(self.config,now_ms=S+5*H+H//2)
        self.assertEqual(next_cycle['status'],'degraded');self.assertTrue(next_cycle['predictions'])
        self.assertIn('training failed',str(next_cycle['errors']))

    def test_missing_new_observations_are_reported_without_fabricating_forecasts(self):
        FakeStore.curves=[c for c in FakeStore.curves if c.event!=9]
        cycle=self.runtime.run_cycle(self.config,now_ms=S+3*H+H//2)
        self.assertFalse(any(r.get('available',False) for r in cycle['predictions']))
        self.assertEqual(cycle['ledger']['forecasts'],0)

    def test_does_not_allow_runtime_to_replace_benchmark_storage(self):
        with self.assertRaises(ValueError):self.runtime.validate_root(self.runtime.ROOT/'data/holdout_v1')

    def test_corrupt_pointer_falls_back_to_valid_immutable_version(self):
        self.runtime.run_cycle(self.config,now_ms=S+3*H+H//2)
        (self.root/'models/current.json').write_text('{')
        with patch.object(self.runtime,'train_pack',side_effect=RuntimeError('training failed')):
            cycle=self.runtime.run_cycle(self.config,now_ms=S+5*H+H//2)
        self.assertEqual(cycle['status'],'degraded');self.assertTrue(cycle['predictions'])

    def test_corrupt_event_pin_is_archived_and_replaced_by_causal_valid_version(self):
        self.runtime.run_cycle(self.config,now_ms=S+3*H+H//2)
        pin=self.runtime.Ledger(self.root).pin(9)
        original=Path(pin['model_path']);meta=json.loads((original/'manifest.json').read_text())
        backup=original.parent/'previous';backup.mkdir();(backup/'manifest.json').write_text(json.dumps(meta|{'model_id':'previous'}))
        (original/'manifest.json').write_text('{')
        cycle=self.runtime.run_cycle(self.config,now_ms=S+3*H+H//2)
        self.assertEqual(cycle['status'],'degraded');self.assertTrue(cycle['predictions'])
        self.assertEqual(self.runtime.Ledger(self.root).pin(9)['model_id'],'previous')
        with self.runtime.Ledger(self.root).connect() as con:self.assertEqual(con.execute('SELECT COUNT(*) FROM pin_history').fetchone()[0],1)
