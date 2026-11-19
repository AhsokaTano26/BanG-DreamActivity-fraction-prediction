"""Small synthetic fixtures only; this file never opens the shared input data."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('D_mechanism', HERE/'mechanism.py')
if (HERE/'mechanism.py').exists():
    M = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(M)
else:
    M = None


class MechanismFixtures(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(M, 'Pure mechanism implementation is missing')

    def test_difference_moments_keep_negative_eigenvalues_before_projection(self):
        got = M.moment_decomposition(np.array([[6.,1.],[1.,10.]]),
                                     np.array([[-1.,-.5],[-.5,-2.]]))
        np.testing.assert_allclose(got['R_raw'], [[1.,.5],[.5,2.]])
        np.testing.assert_allclose(got['Q_raw'], [[4.,0.],[0.,6.]])
        bad = M.moment_decomposition(np.eye(2), np.eye(2))
        np.testing.assert_allclose(bad['R_raw'], -np.eye(2))
        np.testing.assert_allclose(bad['R_psd'], np.zeros((2,2)))
        self.assertAlmostEqual(bad['R_projection_frobenius'], np.sqrt(2))

    def test_noise_transform_uses_predeclared_variance_not_sample_variance(self):
        np.testing.assert_allclose(M.standardize_noise('t5', np.sqrt(5/3)*np.ones((2,2))), 1.)
        draws = np.ones(20); flags = np.zeros(20, bool); flags[0] = True
        got = M.standardize_noise('contamination5', draws, flags)
        self.assertAlmostEqual(float(np.mean(got**2)), 1.)
        self.assertAlmostEqual(got[0], 5/np.sqrt(2.2))

    def test_conditional_likelihood_uses_first_pair_and_activity_gap(self):
        y = np.array([[0.,0.],[2.,0.]])
        # Q=0, R=I -> second innovation covariance 2I.
        got = M.conditional_nll(y, np.array([0,3]), np.zeros((2,2)), np.eye(2))
        self.assertAlmostEqual(got, np.log(2*np.pi)+np.log(2)+1.)
        larger = M.conditional_nll(y, np.array([0,3]), np.eye(2), np.eye(2))
        self.assertAlmostEqual(larger, np.log(2*np.pi)+np.log(5)+.4)
        batch=M.batch_nll(y,np.array([0,3]),np.array([np.zeros((2,2)),np.eye(2)]),np.array([np.eye(2),np.eye(2)]))
        np.testing.assert_allclose(batch,[np.log(2*np.pi)+np.log(2)+1.,np.log(2*np.pi)+np.log(5)+.4])

    def test_local_level_filter_updates_only_present_line_and_is_coordinate_invariant(self):
        q=np.array([[.2,.1],[.1,.3]]);r=np.array([[1.,.2],[.2,2.]])
        y=np.array([[.2,.7],[-.1,.5],[.4,.9]])
        t=np.array([[0.,1.],[1.,-1.]])
        original=M.filter_mean(y,q,r,np.zeros(2),np.eye(2))
        changed=M.filter_mean(y@t.T,t@q@t.T,t@r@t.T,np.zeros(2),t@t.T)
        np.testing.assert_allclose(changed@np.linalg.inv(t).T,original,atol=1e-12)
        partial=M.filter_mean(np.array([[2.,np.nan]]),np.zeros((2,2)),np.eye(2),np.zeros(2),np.eye(2))
        np.testing.assert_allclose(partial,[1.,0.])

    def test_release_delay_and_future_mutation_preserve_issued_predictions(self):
        y=np.ones((1,8,2))
        for delay,first in [(0,1),(1,2),(3,4)]:
            got=M.replay_arrays(y,np.eye(2)*.01,np.eye(2),delay=delay)
            np.testing.assert_allclose(got['bias'][:,:first,0,:],0.)
            np.testing.assert_allclose(got['bias'][:,first,0,:],1/11)
            altered=y.copy();altered[:,6:,:]=100.
            other=M.replay_arrays(altered,np.eye(2)*.01,np.eye(2),delay=delay)
            np.testing.assert_array_equal(got['bias'][:,:6],other['bias'][:,:6])

    def test_vectorized_issued_risk_matches_B_on_nontrivial_fixture(self):
        spec=importlib.util.spec_from_file_location('B_fixture',HERE.parent/'B'/'adaptive_sharing.py')
        b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
        y=np.array([[[.1,.3],[.25,.3],[.3,.1],[.15,.4],[.2,.35],
                     [.1,.2],[.4,.3],[.3,.2],[.1,.2],[.25,.4]]])
        got=M.replay_arrays(y,np.eye(2)*.01,np.eye(2))
        for e in range(len(y[0])):
            for rank in range(2):
                ind=b._filter(y[0,:e,rank])[0]
                shared=b._filter(y[0,:e].mean(axis=1))[0]
                self.assertAlmostEqual(got['issued_I'][0,e,rank],ind)
                self.assertAlmostEqual(got['issued_S'][0,e,rank],shared)
                samples=[]
                for j in range(e):
                    d=got['issued_S'][0,j,rank]-got['issued_I'][0,j,rank]
                    samples.append((d*d,(y[0,j,rank]-got['issued_I'][0,j,rank])*d))
                lam=b.risk_lambda(samples,k=8.,decay=.85)['lambda']
                self.assertAlmostEqual(got['lambda'][0,e,rank],lam)
                self.assertAlmostEqual(got['bias'][0,e,3,rank],np.clip((1-lam)*ind+lam*shared,-.3,.3))

    def test_calibration_profile_stops_on_short_or_gapped_support(self):
        with self.assertRaisesRegex(ValueError,'support'):
            M.calibrate(np.zeros((11,2)),np.arange(11))
        with self.assertRaisesRegex(ValueError,'support'):
            M.calibrate(np.zeros((12,2)),2*np.arange(12))

    def test_simulation_shift_and_heteroskedasticity_use_fixed_event26(self):
        unit=np.ones((1,35,2));zeros=np.zeros_like(unit)
        cfg={'qc':0.,'qd':0.,'r':1.,'rho':0.,'m':.5}
        state,obs=M.make_path(cfg,'difference_shift_at26',zeros,unit,np.zeros(2),2.,4.)
        np.testing.assert_allclose(state[:,:25],0.)
        np.testing.assert_allclose(state[:,25:],[np.tile([2.,-2.],(10,1))])
        state,obs=M.make_path(cfg,'heteroskedastic',zeros,unit,np.zeros(2),2.,4.)
        self.assertAlmostEqual(float(np.mean(obs**2)),1.)

    def test_profile_identity_and_missing_phase_never_create_new_rank(self):
        q,r=M.covariances({'qc':2.,'qd':4.,'r':3.,'rho':.5,'m':.5})
        np.testing.assert_allclose(q,[[3.,1.],[1.,3.]])
        np.testing.assert_allclose(r,[[3.,1.5],[1.5,3.]])
        pair=M.complete_pairs([(3,1000,1,.1),(3,2000,1,.2),(4,1000,1,.5)],1)
        self.assertEqual(pair['events'],[3]);np.testing.assert_allclose(pair['y'],[[.1,.2]])

    def test_score_keeps_path_event_method_axes_and_exponential_ape(self):
        state=np.zeros((2,35,2));observed=np.ones_like(state)*np.log(2)
        issued={'bias':np.zeros((2,35,6,2)),'raw_bias':np.zeros((2,35,6,2))}
        scores=M.score_arrays(state,observed,issued)
        self.assertEqual(scores['state_mse'].shape,(2,35,6))
        np.testing.assert_allclose(scores['state_mse'],0.)
        np.testing.assert_allclose(scores['synthetic_unbounded_ape_pct'],100.)

    def test_same_linear_filter_coordinate_control_before_and_after_common_cap(self):
        history=np.array([[[.1,.3],[-.2,.1],[.05,-.1]]])
        t=np.array([[0.,1.],[1.,-1.]])
        # This low-amplitude fixture keeps all component clips inactive.
        independent=M.ewma_bias(history)
        transformed=M.ewma_bias(history@t.T)@np.linalg.inv(t).T
        np.testing.assert_allclose(independent,transformed,atol=1e-15)
        np.testing.assert_allclose(np.clip(independent,-.01,.01),np.clip(transformed,-.01,.01),atol=1e-15)


if __name__ == '__main__':
    unittest.main(verbosity=2)
