import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from analysis.data_policy import MIN_EVENT_START_MS as S
from production.state import Ledger
H=3600000

def feedback(error=.2,origin='retrospective_v3_seed'):
    return dict(event=900,rank=1000,phase=1,start_ms=S,end_ms=S+H,mean_log_error=error,cases=3,
                actual_final=100.,origin=origin,source='sealed-test-run')

class SelectedServiceTests(unittest.TestCase):
    def module(self,name):
        self.assertIsNotNone(importlib.util.find_spec('production.'+name),'selected service component missing')
        return __import__('production.'+name,fromlist=['run_cycle'])

    def test_seed_replacement_is_archived_idempotent_and_live_feedback_wins(self):
        m=self.module('selected_feedback')
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Ledger(tmp);ledger.seed([feedback(.1)],now_ms=S+3*H)
            r=m.merge_seed(ledger,[feedback(.2)],now_ms=S+3*H)
            self.assertEqual(r['replaced'],1)
            self.assertEqual(ledger.feedback_rows(before_ms=S+3*H)[0]['mean_log_error'],.2)
            with ledger.connect() as con:
                old=con.execute('SELECT old_row FROM selected_feedback_history').fetchone()
                self.assertEqual(json.loads(old[0])['mean_log_error'],.1)
            self.assertEqual(m.merge_seed(ledger,[feedback(.2)],now_ms=S+3*H)['replaced'],0)
            with ledger.connect() as con:con.execute("UPDATE feedback SET origin='live',mean_log_error=.4")
            self.assertEqual(m.merge_seed(ledger,[feedback(.9)],now_ms=S+3*H)['skipped_live'],1)
            self.assertEqual(ledger.feedback_rows(before_ms=S+3*H)[0]['mean_log_error'],.4)

    def test_actual_live_learning_archives_superseded_seed_once(self):
        m=self.module('selected_feedback')
        self.assertTrue(callable(getattr(m,'ensure_archive',None)),'live feedback archive hook missing')
        from analysis.backtest import Curve
        import math,numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Ledger(tmp);m.ensure_archive(ledger)
            ledger.seed([feedback(.2)],now_ms=S+3*H)
            event=dict(event=900,start_ms=S,end_ms=S+H)
            row=dict(rank=1000,phase=1,observed_at=S+H//2,current_ep=50.,
                raw_predicted_final_ep=110.,predicted_final_ep=105.,available=True)
            ledger.record(event,[row],cutoff_ms=S+H//2,issued_ms=S+H//2,model_id='base')
            curve=Curve(900,1000,'normal',S,S+H,np.array([S+H//2,S+H]),np.array([50.,100.]))
            self.assertEqual(ledger.learn_completed([curve],now_ms=S+3*H,grace_ms=0),1)
            actual=ledger.feedback_rows(before_ms=S+3*H)[0]
            self.assertEqual(actual['origin'],'live');self.assertAlmostEqual(actual['mean_log_error'],math.log(1.1))
            with ledger.connect() as con:
                old=con.execute('SELECT old_row FROM selected_feedback_history').fetchall()
            self.assertEqual(len(old),1);self.assertEqual(json.loads(old[0][0])['mean_log_error'],.2)
            self.assertEqual(json.loads(old[0][0])['source'],'sealed-test-run')
            m.ensure_archive(ledger)
            self.assertEqual(ledger.learn_completed([curve],now_ms=S+4*H,grace_ms=0),0)
            with ledger.connect() as con:self.assertEqual(con.execute('SELECT COUNT(*) FROM selected_feedback_history').fetchone()[0],1)

    def test_failed_seed_defers_new_pins_keeps_existing_and_retries(self):
        rt=self.module('selected_runtime')
        from test_production_runtime import FakeStore
        from analysis.backtest import Curve
        from production import selected_model
        from production.model import build_bias
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'live';ledger=Ledger(root);ledger.fact('bootstrap_complete',True)
            FakeStore.index=[];FakeStore.curves=[]
            for e,start in [(900,S),(9,S+3*H),(2,S+5*H)]:
                FakeStore.index.append(dict(event=e,start_ms=start,end_ms=start+H,kind='normal',name=str(e)))
                FakeStore.curves.append(Curve(e,1000,'normal',start,start+H,
                    np.array([start+1,start+H//6,start+H//3,start+H//2,start+H]),np.array([1.,20.,40.,50.,100.])))
            def train(curves,metadata,versions,*,before_ms,ranks):
                assert all(c.end<before_ms for c in curves)
                d=Path(versions)/str(len(curves));d.mkdir(parents=True,exist_ok=True)
                (d/'manifest.json').write_text(json.dumps(dict(model_id=d.name,history_max_end_ms=max(c.end for c in curves),production_ready=True)))
                return d
            def verify(p):return json.loads((Path(p)/'manifest.json').read_text())
            def raw(path,samples,**kw):
                return [dict(event=s['curve'].event,rank=s['curve'].rank,phase=1,start_ms=s['curve'].start,
                    end_ms=s['curve'].end,observed_at=s['now'],current_ep=s['score'],raw_predicted_final_ep=110.,predicted_final_ep=110.,available=True) for s in samples]
            cfg=dict(root=str(root),ranks=[1000],interval_seconds=900,bootstrap_database='unused',bootstrap_metadata='unused',
                forecast_method='legacy_pair',selected_seed_run='/invalid-seed',feedback_grace_ms=0)
            with patch.object(rt,'Store',FakeStore),patch.object(rt,'train_pack',train),patch.object(rt,'verify_pack',verify),patch.object(selected_model.base,'predict_pack',raw):
                with patch.object(rt,'seed_feedback',side_effect=ValueError('bad seed seal')):
                    blocked=rt.run_cycle(cfg,now_ms=S+3*H+H//2,sync=False)
                self.assertIsNone(ledger.pin(9),'failed seed must not permanently pin incomplete feedback')
                self.assertEqual(blocked['ledger']['forecasts'],0)
                self.assertFalse(any(r.get('available') for r in blocked['predictions']))
                pack=Path(json.loads((root/'models/current.json').read_text())['path'])
                ledger.create_pin(FakeStore.index[1],pack,pack.name,build_bias([],before_ms=S+3*H),now_ms=S+3*H)
                prior=ledger.pin(9)
                with patch.object(rt,'seed_feedback',side_effect=ValueError('bad seed seal')):
                    kept=rt.run_cycle(cfg,now_ms=S+3*H+H//2,sync=False)
                    pending=rt.run_cycle(cfg,now_ms=S+5*H+H//2,sync=False)
                self.assertTrue(any(r.get('available') for r in kept['predictions']));self.assertEqual(ledger.pin(9),prior)
                self.assertIsNone(ledger.pin(2));self.assertFalse(any(r.get('available') for r in pending['predictions']))
                with patch.object(rt,'seed_feedback',return_value={'status':'complete','added':0}):
                    repaired=rt.run_cycle(cfg,now_ms=S+5*H+H//2,sync=False)
                    current=ledger.pin(2)
                    repeated=rt.run_cycle(cfg,now_ms=S+5*H+H//2,sync=False)
                self.assertEqual(current['calibration']['forecast_method'],'legacy_pair')
                self.assertTrue(any(r.get('available') for r in repaired['predictions']))
                self.assertEqual(ledger.pin(2),current);self.assertEqual(repaired['ledger']['forecasts'],repeated['ledger']['forecasts'])

    def test_existing_pin_stays_old_and_next_activity_uses_selected_default(self):
        rt=self.module('selected_runtime')
        from test_production_runtime import FakeStore
        from analysis.backtest import Curve
        from production.model import build_bias
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'live';ledger=Ledger(root);ledger.fact('bootstrap_complete',True)
            FakeStore.index=[];FakeStore.curves=[]
            for event,start in [(900,S),(9,S+3*H),(2,S+5*H)]:
                FakeStore.index.append(dict(event=event,start_ms=start,end_ms=start+H,kind='normal',name=str(event)))
                for rank in [1000,2000]:
                    FakeStore.curves.append(Curve(event,rank,'normal',start,start+H,np.array([start+1,start+H//6,start+H//3,start+H//2,start+H]),np.array([1.,20.,40.,50.,100.])))
            def train(curves,metadata,versions,*,before_ms,ranks):
                assert all(c.end<before_ms for c in curves)
                d=Path(versions)/str(len(curves));d.mkdir(parents=True,exist_ok=True)
                (d/'manifest.json').write_text(json.dumps(dict(model_id=d.name,history_max_end_ms=max(c.end for c in curves),production_ready=True)))
                return d
            def verify(p):return json.loads((Path(p)/'manifest.json').read_text())
            def base_predict(path,samples,**kw):
                return [dict(case_id=s['case_id'],event=s['curve'].event,rank=s['curve'].rank,phase=1,start_ms=s['curve'].start,
                    end_ms=s['curve'].end,observed_at=s['now'],current_ep=s['score'],raw_predicted_final_ep=110.,predicted_final_ep=110.,available=True) for s in samples]
            config=dict(root=str(root),ranks=[1000,2000],interval_seconds=900,bootstrap_database='unused',bootstrap_metadata='unused',
                forecast_method='legacy_pair',selected_seed_run=None,feedback_grace_ms=0)
            pack=train(FakeStore.curves[:2],None,root/'models/versions',before_ms=S+3*H,ranks=config['ranks'])
            oldstate=build_bias([],before_ms=S+3*H)
            ledger.create_pin(FakeStore.index[1],pack,pack.name,oldstate,now_ms=S+3*H)
            prior=ledger.pin(9)
            from production import selected_model
            with patch.object(rt,'Store',FakeStore),patch.object(rt,'train_pack',train),patch.object(rt,'verify_pack',verify),patch.object(selected_model.base,'predict_pack',base_predict):
                first=rt.run_cycle(config,now_ms=S+3*H+H//2,sync=False)
                self.assertEqual(ledger.pin(9),prior)
                self.assertEqual({r['forecast_method'] for r in first['predictions']},{'pinned_independent_ewma'})
                again=rt.run_cycle(config,now_ms=S+3*H+H//2,sync=False)
                self.assertEqual(first['ledger']['forecasts'],again['ledger']['forecasts'])
                next_cycle=rt.run_cycle(config,now_ms=S+5*H+H//2,sync=False)
                self.assertEqual({r['forecast_method'] for r in next_cycle['predictions']},{'legacy_pair'})
                self.assertEqual(ledger.pin(2)['calibration']['forecast_method'],'legacy_pair')
                self.assertEqual(ledger.pin(9),prior)
                self.assertGreater(next_cycle['ledger']['live_feedback_groups'],0)

    def test_schedule_switch_accepts_our_old_worker_and_rejects_foreign_worker(self):
        m=self.module('selected_schedule')
        from production import launchd
        import plistlib
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(root=tmp,interval_seconds=900);cfg=Path(tmp)/'config.json';p=Path(tmp)/'task.plist'
            p.write_bytes(plistlib.dumps(launchd.plist_spec(config,cfg)))
            m.validate_existing(p,config,cfg)
            spec=m.plist_spec(config,cfg)
            self.assertIn('production.selected_cli',spec['ProgramArguments'])
            self.assertEqual(spec['EnvironmentVariables']['OMP_NUM_THREADS'],'1')
            p.write_bytes(plistlib.dumps(spec));m.validate_existing(p,config,cfg)
            spec['ProgramArguments']=['python','foreign.py'];p.write_bytes(plistlib.dumps(spec))
            with self.assertRaises(FileExistsError):m.validate_existing(p,config,cfg)

if __name__=='__main__':unittest.main()
