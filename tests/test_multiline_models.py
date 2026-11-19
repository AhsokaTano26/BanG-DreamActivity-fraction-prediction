import importlib.util,os,subprocess,sys,tempfile,textwrap,unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];START=1706504400000;H=3600000
RANKS=[1000,1500,2000,5000]
EXPERTS=('independent','legacy','hierarchy','factor')

def fixture(n=9):
    rows=[]
    for i in range(n):
        for rank in [1000,2000,5000]:
            for j in range(1+i%2):
                row=dict(case_id=f'h:{i}:{rank}:{j}',event=900-i*7,rank=rank,phase=1,progress=.4+.05*j,
                    start_ms=START+i*72*H,end_ms=START+(i*72+48)*H,cutoff=START+(i*72+24)*H,
                    observed_at=START+(i*72+24)*H,current=100.,base_prediction=200.,event_type='normal',band_group='1',split='train',
                    remaining_h=24.,duration_h=48.,rate_6h_relative=1.,rate_24h_relative=1.,log_error=.1,
                    log_rank=np.log(rank),log_remaining_h=np.log(25),log_growth=np.log(2),log_duration_h=np.log(49),
                    own_slow=0.,own_fast=0.,own_n=0.,group_anchor=0.,global_anchor=0.)
                for e in EXPERTS:row['expert__'+e]=200.*np.exp(-.05);row['bias_'+e]=.05
                rows.append(row)
    p=pd.DataFrame(rows);q=START+(n+1)*72*H
    t=pd.DataFrame([dict(rows[0],case_id=f't:{r}',event=17,rank=r,log_rank=np.log(r),start_ms=q,end_ms=q+48*H,cutoff=q+24*H,observed_at=q+24*H) for r in RANKS])
    return p,t.drop(columns=['log_error',*['expert__'+e for e in EXPERTS]]),q

class MultilineModelTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.multiline_models'),'multi-line models missing')
        return __import__('analysis.multiline_models',fromlist=['predict'])

    def test_cold_rank_borrows_history_without_fabricating_own_observations(self):
        m=self.module();p,t,q=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            scores,f,d=m.predict(p,t,query_start_ms=q,model_directory=Path(tmp)/'m',ranks=RANKS)
            self.assertEqual(len(scores),15)
            i=list(t['rank']).index(1500);self.assertEqual(f.own_n.iloc[i],0)
            self.assertEqual(scores['ewma_reference'][i],200.)
            for name in ['hierarchy','graph025','factor2']:self.assertLess(scores[name][i],200.)
            for v in scores.values():
                self.assertTrue(np.isfinite(v).all());self.assertTrue((v>=t.current).all())
                self.assertLessEqual(abs(np.log(v/t.base_prediction)).max(),.3+1e-12)

    def test_constant_errors_match_independent_reference_and_neural_score_mean(self):
        m=self.module();p,t,q=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            scores,f,d=m.predict(p,t,query_start_ms=q,model_directory=Path(tmp)/'m',ranks=RANKS)
            expected=200*np.exp(-.1*9/19)
            np.testing.assert_allclose(scores['ewma_reference'][t['rank']!=1500],expected)
            np.testing.assert_allclose(scores['hierarchy'],expected)
            np.testing.assert_allclose(scores['neural_gate'],np.mean([scores[f'neural_gate_seed{s}'] for s in [42,123,2026]],axis=0))

    def test_clock_outcomes_duplicates_and_unknown_ranks_fail_before_write(self):
        m=self.module();p,t,q=fixture(1)
        variants=[(p.assign(end_ms=q),t),(p.assign(start_ms=START-1),t),(p.assign(event=17),t),(pd.concat([p,p]),t),(p,t.assign(actual=1)),(p,t.assign(rank=777))]
        with tempfile.TemporaryDirectory() as tmp:
            for i,(h,c) in enumerate(variants):
                with self.subTest(i=i),self.assertRaises(ValueError):m.predict(h,c,query_start_ms=q,model_directory=Path(tmp)/str(i),ranks=RANKS)
                self.assertFalse((Path(tmp)/str(i)).exists())

    def test_snapshot_duplication_does_not_change_statistical_evidence(self):
        m=self.module();p,t,q=fixture(9);copy=p.copy();copy['case_id']+='copy'
        with tempfile.TemporaryDirectory() as tmp:
            a,_,_=m.predict(p,t,query_start_ms=q,model_directory=Path(tmp)/'a',ranks=RANKS)
            b,_,_=m.predict(pd.concat([p,copy],ignore_index=True),t,query_start_ms=q,model_directory=Path(tmp)/'b',ranks=RANKS)
            for key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-11)

    def test_three_seed_neural_replay_and_event_weights_in_subprocess(self):
        self.module()
        code=textwrap.dedent('''
        import tempfile,numpy as np,pandas as pd
        from pathlib import Path
        from test_multiline_models import fixture,RANKS
        from analysis.multiline_models import predict,replay_neural
        p,t,q=fixture(13)
        with tempfile.TemporaryDirectory() as tmp:
            scores,f,d=predict(p,t,query_start_ms=q,model_directory=Path(tmp)/'m',ranks=RANKS)
            assert d['neural_gate']['status']=='fitted'
            replay=replay_neural(Path(tmp)/'m/neural_gate',f)
            for k,v in replay.items():np.testing.assert_allclose(v,scores[k],rtol=1e-12)
            w=pd.read_csv(Path(tmp)/'m/neural_gate/training_weights.csv')
            np.testing.assert_allclose(w.groupby('event').weight.sum(),1.)
        ''')
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'tests'),str(ROOT)]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        done=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
