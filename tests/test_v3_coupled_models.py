import importlib.util,json,os,subprocess,sys,tempfile,textwrap,unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
START=1706504400000
H=3600000
EXPERTS=('kernel','gap','covariate','warm')

def fixture(n=7):
    rows=[]
    for i in range(n):
        for rank in [1000,2000]:
            for j in range(1+i%3):
                ratio=.2+.02*i+.005*j
                row=dict(case_id=f'h:{i}:{rank}:{j}',event=1000-i*7,rank=rank,phase=1,
                    progress=.4+.05*j,start_ms=START+i*72*H,end_ms=START+(i*72+48)*H,
                    current=100.,base_prediction=200.,ewma_bias=0.,event_type='normal',band_group='1',
                    duration_h=48.,remaining_h=24.,log_base_growth=np.log(2),
                    peer_log_ratio=ratio,rate_6h_relative=1.+ratio,rate_24h_relative=1.-ratio,
                    log_error=.02 if rank==2000 else .02+.3*ratio)
                for k in ['anchor','gap_fast','gap_slow','gap_latest','gap_sd5','gap_n','days_since']:
                    row['lag_'+k]=0.
                for expert,bias in zip(EXPERTS,[.03,.08,.06,.04]):
                    row['expert__'+expert]=200.*np.exp(-bias)
                    row['ex_bias_'+expert]=bias
                rows.append(row)
    past=pd.DataFrame(rows)
    query=START+(n+1)*72*H
    target=past.iloc[:2].copy().drop(columns=['log_error',*['expert__'+e for e in EXPERTS]])
    target=target.iloc[[0,0]].copy();target['case_id']=['t:1000','t:2000'];target['rank']=[1000,2000]
    target['event']=17;target['start_ms']=query;target['end_ms']=query+48*H
    refs={e:200.*np.exp(-np.array([b,.01])) for e,b in [('kernel',.03),('gap',.08),('warm',.04)]}
    return past,target,refs,query

class CoupledModelTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.v3_coupled_models'),'new coupled model is missing')
        return __import__('analysis.v3_coupled_models',fromlist=['predict_all'])

    def test_fixed_menu_bounds_fallback_and_no_mutation(self):
        m=self.module();past,target,refs,query=fixture()
        old=past.copy(deep=True)
        with tempfile.TemporaryDirectory() as tmp:
            scores,features,details=m.predict_all(past,target,refs,query_start_ms=query,model_directory=Path(tmp)/'m')
        self.assertEqual(len(scores),15) # 12 new/reference warm methods plus three seed diagnostics; runner adds two cold controls
        self.assertEqual(set(scores),set(m.RECIPES)|{f'context_gate_seed{s}' for s in [42,123,2026]})
        for x in scores.values():
            self.assertTrue(np.isfinite(x).all());self.assertTrue((x>=target.current).all())
            self.assertLessEqual(abs(np.log(x/target.base_prediction)).max(),.3+1e-12)
        self.assertEqual(details['context_gate']['status'],'prior_fallback')
        pd.testing.assert_frame_equal(past,old)
        self.assertFalse('log_error' in features)

    def test_future_self_outcome_and_current_expert_rejected_before_fallback(self):
        m=self.module();past,target,refs,query=fixture(1)
        bad=[]
        for col,val in [('end_ms',query),('start_ms',START-1),('event',17)]:
            p=past.copy();p[col]=val;bad.append((p,target))
        for col in ['actual','log_error','target','expert__gap']:
            bad.append((past,target.assign(**{col:1})))
        with tempfile.TemporaryDirectory() as tmp:
            for i,(p,t) in enumerate(bad):
                with self.subTest(i=i),self.assertRaises(ValueError):
                    m.predict_all(p,t,refs,query_start_ms=query,model_directory=Path(tmp)/str(i))
                self.assertFalse((Path(tmp)/str(i)).exists())

    def test_duplicate_snapshots_keep_event_mass_and_predictions(self):
        m=self.module();p,t,r,q=fixture(9)
        duplicate=p.copy();duplicate['case_id']=duplicate.case_id+'copy'
        with tempfile.TemporaryDirectory() as tmp:
            a,_,_=m.predict_all(p,t,r,query_start_ms=q,model_directory=Path(tmp)/'a')
            b,_,_=m.predict_all(pd.concat([p,duplicate],ignore_index=True),t,r,query_start_ms=q,model_directory=Path(tmp)/'b')
            for method in a:np.testing.assert_allclose(a[method],b[method],rtol=1e-11)
            w=pd.read_csv(Path(tmp)/'a/gap_covariate5/training_weights.csv')
            np.testing.assert_allclose(w.groupby('event').weight.sum(),1.)

    def test_history_only_category_and_numeric_state(self):
        m=self.module();p,t,r,q=fixture(9)
        with tempfile.TemporaryDirectory() as tmp:
            m.predict_all(p,t.assign(event_type='unseen',band_group='future',peer_log_ratio=np.inf),r,
                          query_start_ms=q,model_directory=Path(tmp)/'m')
            state=json.loads((Path(tmp)/'m/gap_covariate5/preprocessor.json').read_text())
            self.assertNotIn('unseen',state['categories']['event_type'])
            self.assertNotIn('future',state['categories']['band_group'])

    def test_conditional_evidence_can_relax_a_reference_score_floor(self):
        m=self.module();p,t,r,q=fixture(9)
        p['log_error']=0.
        t['base_prediction']=110.;t['log_base_growth']=np.log(1.1)
        r={'kernel':np.full(len(t),110.),'warm':np.full(len(t),110.),'gap':np.full(len(t),100.)}
        with tempfile.TemporaryDirectory() as tmp:
            scores,_,_=m.predict_all(p,t,r,query_start_ms=q,model_directory=Path(tmp)/'m')
            self.assertGreater(scores['gap_covariate5'][0],100.)
            self.assertLess(scores['gap_covariate5'][0],110.)
            self.assertEqual(scores['gap_covariate5'][1],100.)

    def test_missing_other_rank_retains_all_own_training_cases(self):
        m=self.module();p,t,r,q=fixture(9)
        event=p.event.iloc[0]
        p=p[~((p.event==event)&(p['rank']==2000))]
        with tempfile.TemporaryDirectory() as tmp:
            _,_,info=m.predict_all(p,t,r,query_start_ms=q,model_directory=Path(tmp)/'m')
            self.assertGreater(info['gap_covariate5']['missing_anchor_cases'],0)
            w=pd.read_csv(Path(tmp)/'m/gap_covariate5/training_weights.csv')
            self.assertEqual(set(w.case_id),set(p.loc[p['rank']==1000,'case_id']))
            np.testing.assert_allclose(w.groupby('event').weight.sum(),1.)

    def test_changepoint_posterior_tracks_a_sustained_jump(self):
        m=self.module()
        steady,_=m.changepoint_mean(np.zeros(20))
        jumped,state=m.changepoint_mean(np.r_[np.zeros(20),np.full(6,.3)])
        self.assertAlmostEqual(steady,0.)
        self.assertGreater(jumped,.15);self.assertAlmostEqual(sum(state['probabilities']),1.)
        self.assertTrue(np.isfinite(state['probabilities']).all())

    def test_fixed_share_uses_sealed_expert_loss_and_retains_switch_mass(self):
        m=self.module();p,t,r,q=fixture(9)
        p['log_error']=0.
        for e in EXPERTS:p['expert__'+e]=100.
        p['expert__gap']=200.
        with tempfile.TemporaryDirectory() as tmp:
            scores,_,details=m.predict_all(p,t,r,query_start_ms=q,model_directory=Path(tmp)/'m')
            w=details['fixed_share']['groups'][0]['weights']
            self.assertGreater(w['gap'],w['kernel'])
            self.assertTrue(all(v>0 for v in w.values()))
            self.assertTrue(np.isfinite(scores['fixed_share']).all())

    def test_neural_three_seed_saved_replay_is_isolated(self):
        self.module()
        code=textwrap.dedent('''
            from pathlib import Path
            import tempfile,numpy as np
            from test_v3_coupled_models import fixture
            from analysis.v3_coupled_models import predict_all,replay_neural
            p,t,r,q=fixture(13)
            with tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)/'m'
                scores,features,details=predict_all(p,t,r,query_start_ms=q,model_directory=directory)
                assert details['context_gate']['status']=='fitted'
                import json
                state=json.loads((directory/'context_gate/preprocessor.json').read_text())
                assert 'rank_flag' in state['numeric'], 'Gate must distinguish the two ranking lines explicitly'
                replay=replay_neural(directory/'context_gate',features)
                for key,value in replay.items():np.testing.assert_allclose(value,scores[key],rtol=1e-12)
                np.testing.assert_allclose(scores['context_gate'],np.mean([scores[f'context_gate_seed{s}'] for s in [42,123,2026]],axis=0),rtol=1e-12)
                t['current']=t['base_prediction'];t['log_base_growth']=0.
                r={key:t.current.to_numpy() for key in r}
                boundary,features,_=predict_all(p,t,r,query_start_ms=q,model_directory=Path(tmp)/'floor')
                for key,value in boundary.items():assert np.all(value>=t.current), (key,value)
                replay=replay_neural(Path(tmp)/'floor/context_gate',features)
                for key,value in replay.items():np.testing.assert_array_equal(value,boundary[key])
        ''')
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'tests'),str(ROOT)]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        done=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,text=True,capture_output=True,timeout=120)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
