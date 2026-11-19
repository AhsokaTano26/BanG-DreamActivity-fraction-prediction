"""Diverse category blocks must survive sorted-JSON checkpoint serialization."""
import importlib.util,os,subprocess,sys,textwrap,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class TaskLearningReplayTests(unittest.TestCase):
    def test_diverse_category_layout_replays_all_seeds_without_mutating_artifacts(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.task_learning_replay'),'fixed category-layout reader missing')
        code=textwrap.dedent('''
        import tempfile,json,numpy as np,torch
        from pathlib import Path
        from test_task_learning_models import fixture,RANKS
        from analysis import task_learning_models as m
        from analysis.task_learning_replay import replay,CATEGORY_ORDER
        from analysis.holdout_dataset import sha256
        torch.set_num_threads(1)
        h,t,pb,tb,q=fixture()
        activity=(900-h.event)//7
        h['event_type']=['type'+str(i%2) for i in activity]
        h['band_group']=['band'+str(i%3) for i in activity]
        pb.frame=h.drop(columns='reference_log_error').copy()
        t['event_type']='type1';t['band_group']='band2';tb.frame=t.copy()
        states={}
        for event,p in h.groupby('event'):
            start=int(p.start_ms.iloc[0]);states[int(event)]=m.relation_state(h[h.end_ms<start],query_start_ms=start,ranks=RANKS)
        states[17]=m.relation_state(h,query_start_ms=q,ranks=RANKS)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fit'
            scores,_=m.fit_predict(pb,tb,h,t,states,query_start_ms=q,model_directory=path,epochs=3,minimum_events=2)
            prep=json.loads((path/'preprocessor.json').read_text())
            assert list(prep['categories'])==['band_group','event_type']
            assert list(CATEGORY_ORDER)==['event_type','band_group']
            assert [len(prep['categories'][k]) for k in CATEGORY_ORDER]==[2,3]
            old=m.replay(path,tb,t,states[17])
            assert max(float(np.max(abs(old[k]-scores[k]))) for k in scores)>1e-4, 'fixture must expose original order bug'
            before={str(p.relative_to(path)):sha256(p) for p in path.rglob('*') if p.is_file()}
            fixed=replay(path,tb,t,states[17])
            assert set(fixed)==set(scores)
            for k,v in scores.items():np.testing.assert_allclose(v,fixed[k],rtol=1e-12,atol=1e-9)
            assert before=={str(p.relative_to(path)):sha256(p) for p in path.rglob('*') if p.is_file()}
            bad=dict(states[17]);bad['query_start_ms']+=1
            try:replay(path,tb,t,bad)
            except ValueError:pass
            else:raise AssertionError('reader accepted wrong graph')
        ''')
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'tests'),str(ROOT)]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        done=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)

if __name__=='__main__':unittest.main()
