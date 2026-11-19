import importlib.util
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
import pandas as pd
from analysis.backtest import Curve, snapshot


class QualityHierarchyTests(unittest.TestCase):
    def test_quality_unknown_shape_cannot_create_a_shape_deviation(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.quality_hierarchy'))
        from analysis.quality_hierarchy import quality_features
        history=pd.DataFrame({'hist_progress_logk':[.2,.2,.2], 'hist_deadline_logk':[.3,.3,.3],
            'hist_shape_logk':[.9,np.nan,.9], 'hist_effective_n':[20,np.nan,20],
            'hist_span_h':[0,np.nan,10]})
        frame=quality_features(history)
        self.assertEqual(frame.quality_shape_reliability.iloc[1],0)
        self.assertEqual(frame.quality_shape_deviation.iloc[1],0)
        self.assertGreater(frame.quality_shape_reliability.iloc[0],frame.quality_shape_reliability.iloc[2])
        self.assertAlmostEqual(frame.quality_shape_deviation.iloc[0],.4)

    def test_type_expert_shrinkage_counts_events_and_preserves_global_weights(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.quality_hierarchy'))
        from analysis.quality_hierarchy import type_shares
        from analysis.relationship_benchmark import weights
        samples=[{'curve':type('C',(),{'event':e,'kind':k})()} for e,k in
                 [(1,'a')]*20+[(2,'a')]*4+[(3,'b')]*3]
        self.assertEqual(type_shares(samples),{'a':2/22,'b':1/21})
        w=weights(samples)
        total=[sum(v for s,v in zip(samples,w) if s['curve'].event==e) for e in [1,2,3]]
        np.testing.assert_allclose(total,total[0])


class IrregularTests(unittest.TestCase):
    def sample(self):
        t=np.array([0,1,3,8,13,22,30,47,53,72,90,100])*3600000+1
        a=Curve(1,1000,'live_try',1,100*3600000+1,t,np.linspace(100,10000,len(t)))
        b=replace(a,rank=2000,scores=a.scores*.7)
        s=snapshot(a,t[8]);s['peer_curve']=b
        return s

    def test_observed_nodes_ignore_future_preserve_endpoints_and_units(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.irregular_neural'))
        # Prefix construction is separate from torch for subprocess isolation.
        from analysis.irregular_prefix import observed_path
        s=self.sample();path,mask=observed_path(s,length=6)
        self.assertEqual(path.shape,(2,6,4));self.assertTrue(mask.all())
        np.testing.assert_allclose(path[:,0,0],0)
        np.testing.assert_allclose(path[:,-1,0],s['progress'])
        np.testing.assert_allclose(path[:,-1,1],1)
        s['curve'].scores[9:]*=100;s['peer_curve'].scores[9:]*=100
        np.testing.assert_array_equal(path,observed_path(s,length=6)[0])
        a=replace(s['curve'],scores=s['curve'].scores*10)
        b=replace(s['peer_curve'],scores=s['peer_curve'].scores*10)
        scaled=snapshot(a,s['cutoff']);scaled['peer_curve']=b
        np.testing.assert_allclose(path,observed_path(scaled,length=6)[0],atol=1e-7)
        expected=((s['times']-s['curve'].start)/s['curve'].duration).astype(np.float32)
        self.assertTrue(set(path[0,:,0]).issubset(set(expected)))

    def test_missing_peer_and_padding_are_explicit(self):
        from analysis.irregular_prefix import observed_path
        s=self.sample();s['peer_curve']=None;path,mask=observed_path(s,length=16)
        self.assertEqual(mask[1].sum(),0);self.assertEqual(mask[0].sum(),9)
        np.testing.assert_array_equal(path[1],0)
        np.testing.assert_array_equal(path[0,9:],np.repeat(path[0,8:9],7,axis=0))

    def test_neural_solver_gradient_padding_and_replay(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.irregular_neural'))
        script=r'''
import io,torch
from analysis.irregular_neural import IrregularForecaster,heun_step
torch.set_num_threads(2)
# Known differential equation dz=z dx: second-order refinement converges to exp(1).
z=torch.ones(1,1);dx=torch.ones(1,1)
def field(z):return z.unsqueeze(-1)
one=heun_step(field,z,dx,1);four=heun_step(field,z,dx,4)
assert abs(four.item()-2.718281828)<abs(one.item()-2.718281828)
for kind in ['time_gru','neural_cde']:
    torch.manual_seed(42);model=IrregularForecaster(kind,5,width=16)
    torch.nn.init.normal_(model.head[-1].weight,std=.1)
    path=torch.rand(3,2,8,4).cumsum(2)/8
    mask=torch.ones(3,2,8,dtype=torch.bool);mask[0,1]=False;mask[1,:,5:]=False
    tab=torch.randn(3,5);rank=torch.tensor([0,1,0])
    out=model(path,mask,tab,rank);((out-torch.tensor([.1,-.2,.3]))**2).mean().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith(('initial','gru','field_net')))
    model.eval();before=model(path,mask,tab,rank).detach()
    changed=path.clone();changed[~mask]=999
    torch.testing.assert_close(before,model(changed,mask,tab,rank))
    store=io.BytesIO();torch.save(model.state_dict(),store);store.seek(0)
    other=IrregularForecaster(kind,5,width=16);other.load_state_dict(torch.load(store,weights_only=True));other.eval()
    torch.testing.assert_close(before,other(path,mask,tab,rank))
'''
        env=os.environ.copy();env['DYLD_LIBRARY_PATH']='/opt/homebrew/opt/libomp/lib'
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],env=env,
                              capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_development_is_train_validation_only_and_sealed(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.quality_irregular_benchmark'))
        script=r'''
import json,sys
sys.path.insert(0,'tests')
from pathlib import Path
from unittest import mock
import numpy as np,pandas as pd
from test_holdout_dataset import HoldoutDatasetTests
from analysis.holdout_dataset import write_json,sha256
from analysis.relationship_benchmark import rows
from analysis import quality_irregular_benchmark as b
fixture=HoldoutDatasetTests();fixture.setUp()
try:
    data=fixture.freeze();validation=data.inputs('validation');ref=fixture.root/'reference';ref.mkdir()
    pd.DataFrame(rows(validation,'reference',np.array([s['score']/s['progress'] for s in validation]))).to_csv(ref/'validation_predictions.csv',index=False)
    write_json(ref/'selection.json',{'selected_recipe':'reference'})
    write_json(ref/'blind_test_manifest.json',{'dataset_id':data.manifest['dataset_id'],'run_id':'synthetic',
        'artifacts':{p.name:sha256(p) for p in ref.iterdir()}})
    original_labels,original_inputs=data.labels,data.inputs
    def labels(split,*args,**kwargs):
        assert split!='test','test answers requested'
        return original_labels(split,*args,**kwargs)
    def inputs(split,*args,**kwargs):
        assert split!='test','test inputs requested'
        return original_inputs(split,*args,**kwargs)
    data.labels,data.inputs=labels,inputs;run=fixture.root/'development'
    with mock.patch.multiple(b,FOLD_LEFT=[10],FOLD_SIZE=4,SEEDS=[42],NEURAL_SETTINGS=b.NEURAL_SETTINGS|{'epochs':2,'minimum_epochs':1,'patience':1}),mock.patch.dict(b.TREE_PARAMETERS,iterations=5):
        b.develop(run,data,ref)
    b.verify_run(run,data)
    out=pd.read_csv(run/'validation_predictions.csv')
    for _,g in out.groupby('method'):assert set(g.case_id)=={s['case_id'] for s in validation}
    assert not (run/'test_predictions.csv').exists()
    try:b.develop(run,data,ref)
    except FileExistsError:pass
    else:raise AssertionError('sealed run overwritten')
finally:fixture.doCleanups()
'''
        env=os.environ.copy();env['DYLD_LIBRARY_PATH']='/opt/homebrew/opt/libomp/lib'
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],env=env,
                              capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
