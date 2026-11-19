import importlib.util,os,subprocess,sys,tempfile,textwrap,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.task_peer_data import PeerBatch

ROOT=Path(__file__).resolve().parents[1];START=1706504400000;H=3600000;RANKS=[1000,1500,2000,5000]

def fixture(n=13):
    rows=[];nodes=[];tables=[];links=[]
    for i in range(n):
        ids={r:len(nodes)+j for j,r in enumerate([1000,2000,5000])}
        for rank in [1000,2000,5000]:
            start=START+i*72*H;end=start+48*H;observed=start+20*H
            nodes.append([np.sin(i)+rank/10000,np.log(rank)])
            tables.append(dict(event=900-i*7,rank=rank,start_ms=start,end_ms=end,observed_at=observed))
            rows.append(dict(case_id=f'h:{i}:{rank}',event=900-i*7,rank=rank,phase=1,progress=20/48,
               start_ms=start,end_ms=end,cutoff=observed,observed_at=observed,current=100.,base_prediction=210.,reference_prediction=200.,
               duration_h=48.,remaining_h=28.,event_type='normal',band_group='1',split='train',reference_log_error=.1+.01*np.cos(i)))
            links.append([ids.get(r,-1) for r in RANKS])
    history=pd.DataFrame(rows);past=PeerBatch(np.array(nodes),pd.DataFrame(tables),np.array(links),np.zeros((len(rows),4)),history.drop(columns='reference_log_error'),RANKS,['shape','log_rank'])
    q=START+(n+1)*72*H
    current=pd.DataFrame([dict(rows[0],case_id=f'q:{r}',event=17,rank=r,start_ms=q,end_ms=q+48*H,cutoff=q+20*H,observed_at=q+20*H) for r in RANKS]).drop(columns='reference_log_error')
    target=PeerBatch(np.array([[.2+r/10000,np.log(r)] for r in RANKS]),
       pd.DataFrame([dict(event=17,rank=r,start_ms=q,end_ms=q+48*H,observed_at=q+20*H) for r in RANKS]),
       np.tile(np.arange(4),(4,1)),np.zeros((4,4)),current.copy(),RANKS,['shape','log_rank'])
    return history,current,past,target,q

class TaskLearningModelTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.task_learning_models'),'shared/rank/peer models missing')
        return __import__('analysis.task_learning_models',fromlist=['relation_state'])

    def test_graph_evidence_counts_activities_and_cold_rank_has_only_declared_prior(self):
        m=self.module();h,t,pb,tb,q=fixture()
        state=m.relation_state(h,query_start_ms=q,ranks=RANKS)
        self.assertEqual(state['rank_counts'],[13,0,13,13])
        self.assertEqual(state['static']['pair_counts'][0][1],0)
        self.assertIsNone(state['static']['correlations'][0][1])
        a=np.asarray(state['static']['weights']);np.testing.assert_allclose(a.sum(axis=1),1.);np.testing.assert_array_equal(a.diagonal(),0.)
        self.assertGreater(a[1,0],0.)
        copy=h.copy();copy['case_id']+='copy'
        again=m.relation_state(pd.concat([h,copy],ignore_index=True),query_start_ms=q,ranks=RANKS)
        self.assertEqual(state,again)

    def test_illegal_history_or_current_answers_fail_before_model_files(self):
        m=self.module();h,t,pb,tb,q=fixture()
        for bad in [h.assign(end_ms=q),h.assign(start_ms=START-1),h.assign(event=17),pd.concat([h,h])]:
            with self.subTest(),self.assertRaises(ValueError):m.relation_state(bad,query_start_ms=q,ranks=RANKS)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):m.fit_predict(pb,tb,h,t.assign(actual=1),{},query_start_ms=q,model_directory=Path(tmp)/'bad',epochs=2,minimum_events=2)
            self.assertFalse((Path(tmp)/'bad').exists())

    def test_neural_paths_learn_and_replay_without_cold_adapter_or_missing_peer_leak(self):
        self.module()
        code=textwrap.dedent('''
        import tempfile,json,numpy as np,torch
        from pathlib import Path
        from test_task_learning_models import fixture,RANKS
        from analysis.task_learning_models import relation_state,fit_predict,replay,make_network
        torch.set_num_threads(1)
        h,t,pb,tb,q=fixture();states={}
        for event,p in h.groupby('event'):
            start=int(p.start_ms.iloc[0]);states[int(event)]=relation_state(h[h.end_ms<start],query_start_ms=start,ranks=RANKS)
        states[17]=relation_state(h,query_start_ms=q,ranks=RANKS)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fit'
            scores,details=fit_predict(pb,tb,h,t,states,query_start_ms=q,model_directory=path,epochs=3,minimum_events=2)
            assert len(scores)==16 and details['status']=='fitted'
            again=replay(path,tb,t,states[17])
            for k,v in scores.items():np.testing.assert_allclose(v,again[k],rtol=1e-12)
            for name in ['shared_single','rank_heads','related_static','related_phase']:
                np.testing.assert_allclose(scores[name],np.mean([scores[f'{name}_seed{s}'] for s in [42,123,2026]],axis=0))
                assert np.isfinite(scores[name]).all() and (scores[name]>=t.current).all()
                assert np.max(abs(np.log(scores[name]/t.reference_prediction)))<=.2+1e-7
                assert np.max(abs(np.log(scores[name]/t.reference_prediction)))>1e-5, name+' failed to learn a nonzero residual'
            cfg=json.loads((path/'config.json').read_text())
            torch.manual_seed(42);initial=make_network(cfg['node_dim'],cfg['context_dim'],4,'related_phase')
            saved=torch.load(path/'related_phase/seed42.pt',map_location='cpu',weights_only=True)['state_dict']
            decay_only=(1-.002*.05)**3
            assert not torch.allclose(saved['query.weight'],initial.query.weight*decay_only,rtol=0,atol=1e-7)
            assert not torch.allclose(saved['key.weight'],initial.key.weight*decay_only,rtol=0,atol=1e-7)
            weights=np.genfromtxt(path/'training_weights.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
            import pandas as pd
            w=pd.read_csv(path/'training_weights.csv');np.testing.assert_allclose(w.groupby('event').weight.sum(),1.)
            prep=json.loads((path/'preprocessor.json').read_text());own=pb.nodes[np.array([pb.links[i,RANKS.index(int(r))] for i,r in enumerate(h['rank'])])]
            np.testing.assert_allclose(prep['node_mean'],own.mean(axis=0))
            changed=type(tb)(tb.nodes*1e6,tb.node_table.copy(),tb.links.copy(),tb.ages.copy(),tb.frame.copy(),tb.ranks,tb.columns)
            fit_predict(pb,changed,h,t,states,query_start_ms=q,model_directory=Path(tmp)/'other',epochs=3,minimum_events=2)
            assert (path/'preprocessor.json').read_bytes()==(Path(tmp)/'other/preprocessor.json').read_bytes()
        torch.manual_seed(42);net=make_network(4,3,4,'related_phase');net.eval()
        x=torch.randn(4,4);links=torch.arange(4).repeat(4,1);ages=torch.zeros(4,4);context=torch.randn(4,3);ri=torch.arange(4)
        prior=(1-torch.eye(4))/3;support=torch.tensor([.6,0.,.6,.6])
        comp=net.components(x,links,ages,context,ri,prior,support)
        assert torch.allclose(sum(comp[k] for k in ['shared','private','peer']),torch.zeros(4))
        with torch.no_grad():net.peer_head.weight.fill_(.2);net.private_weight.fill_(.2)
        comp=net.components(x,links,ages,context,ri,prior,support)
        assert comp['private'][1].item()==0.
        loss=comp['peer'].square().sum();loss.backward()
        assert net.query.weight.grad.abs().sum()>0 and net.key.weight.grad.abs().sum()>0
        solo=torch.full_like(links,-1);solo[torch.arange(4),torch.arange(4)]=torch.arange(4)
        comp=net.components(x,solo,ages,context,ri,prior,support)
        assert torch.equal(comp['peer'],torch.zeros(4))
        other=x.clone();other[1:]*=100
        comp2=net.components(other,solo,ages,context,ri,prior,support)
        assert comp['shared'][0].item()==comp2['shared'][0].item()
        assert comp['private'][0].item()==comp2['private'][0].item()
        ''')
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'tests'),str(ROOT)]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        done=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)

if __name__=='__main__':unittest.main()
