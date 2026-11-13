import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from analysis.backtest import Curve,snapshot


class PairedPrefixTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.paired_prefix'),'paired prefix module missing')
        from analysis.paired_prefix import paired_sequence,thin_sample
        self.sequence,self.thin=paired_sequence,thin_sample
        h=3600000;t=np.arange(101)*h+1
        self.a=Curve(1,1000,'live_try',1,100*h+1,t,np.arange(1,102.)*100)
        self.b=Curve(1,2000,'live_try',1,100*h+1,t,np.arange(1,102.)*75)
        self.sample=snapshot(self.a,50*h+1);self.sample.update({'peer_curve':self.b,'case_id':'train:1:1000:x','stage':'x'})

    def test_both_timescales_ignore_future_and_score_units(self):
        for window in [None,48]:
            before=self.sequence(self.sample,window_hours=window)
            self.a.scores[51:]*=100;self.b.scores[51:]*=100
            np.testing.assert_array_equal(before,self.sequence(self.sample,window_hours=window))
            self.a.scores*=10;self.b.scores*=10
            # Reconstruct prefix scalar fields in the new units.
            s=snapshot(self.a,self.sample['cutoff']);s['peer_curve']=self.b
            np.testing.assert_allclose(before,self.sequence(s,window_hours=window),atol=1e-6)
            self.sample=s

    def test_missing_peer_is_masked_and_slot_order_is_canonical(self):
        first=self.sequence(self.sample)
        s=snapshot(self.b,self.sample['cutoff']);s['peer_curve']=self.a
        np.testing.assert_allclose(first,self.sequence(s),atol=1e-6)
        s['peer_curve']=None;missing=self.sequence(s)
        self.assertEqual(missing.shape,(2,64,8));self.assertTrue(np.isfinite(missing).all())
        np.testing.assert_array_equal(missing[0],0.)
        self.assertEqual(missing[1,-1,2],1.)

    def test_thinning_preserves_forecast_origin_and_target(self):
        s=self.thin(self.sample)
        self.assertEqual(s['now'],self.sample['now']);self.assertEqual(s['score'],self.sample['score'])
        self.assertEqual(s['case_id'],self.sample['case_id']);self.assertLess(len(s['times']),len(self.sample['times']))
        self.assertGreaterEqual(len(s['times']),4);self.assertLessEqual(s['peer_curve'].times[-1],s['now'])


class StructuredNetworkTests(unittest.TestCase):
    def test_architectures_backpropagate_and_reload_with_missing_peers(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.structured_neural_models'),'neural backbones missing')
        script=r'''
import io,torch
from analysis.structured_neural_models import StructuredForecaster,ARCHITECTURES
torch.set_num_threads(2)
pair=torch.randn(4,2,64,8);pair[:,:,:,2]=1.;pair[0,1]=0.;pair[1,0]=0.
recent=pair.clone();tab=torch.randn(4,15);rank=torch.tensor([0,1,0,1])
for architecture in ARCHITECTURES:
    model=StructuredForecaster(architecture,15,width=32)
    value=model(pair,recent,tab,rank)
    assert value.shape==(4,) and torch.isfinite(value).all(),architecture
    ((value-torch.tensor([.1,-.2,.3,-.4]))**2).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),architecture
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()),architecture
    model.eval();before=model(pair,recent,tab,rank).detach()
    store=io.BytesIO();torch.save(model.state_dict(),store);store.seek(0)
    other=StructuredForecaster(architecture,15,width=32);other.load_state_dict(torch.load(store,weights_only=True));other.eval()
    torch.testing.assert_close(before,other(pair,recent,tab,rank))
'''
        env=os.environ.copy();env['DYLD_LIBRARY_PATH']='/opt/homebrew/opt/libomp/lib'
        p=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],env=env,
                         capture_output=True,text=True,timeout=90)
        self.assertEqual(p.returncode,0,p.stderr)


if __name__=='__main__':unittest.main()
