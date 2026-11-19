"""Behavioral contracts for the detached paper replay rules (synthetic only)."""
import importlib
import importlib.util
import unittest
import numpy as np
import pandas as pd

MODULE = 'analysis.paper_upgrade_replay_models'
MINIMUM = 1706504400000


def frame(event, start, errors=None):
    rows=[]
    for rank in [1000,2000]:
        row=dict(case_id=f'{event}:{rank}',event=event,rank=rank,phase=1,progress=.5,
                 start_ms=start,end_ms=start+100,cutoff=start+50,observed_at=start+50,
                 current=100.,base_prediction=200.,event_type='normal',band_group='A',
                 split='train',duration_h=100/3600000,remaining_h=50/3600000,
                 rate_6h_relative=1.,rate_24h_relative=1.)
        if errors is not None:row.update(log_error=errors[rank],center_log_error=errors[rank])
        rows.append(row)
    return pd.DataFrame(rows)


class ReplayRules(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE),'detached replay rules must exist')
        self.m=importlib.import_module(MODULE)
        self.h=pd.concat([frame(1,MINIMUM,{1000:.2,2000:.1}),
                          frame(2,MINIMUM+200,{1000:.4,2000:.2})],ignore_index=True)
        self.t=frame(3,MINIMUM+1000)

    def test_reference_and_pair_have_hand_computed_event_support(self):
        out,_=self.m.predict_all(self.h,self.t,query_start_ms=MINIMUM+1000)
        np.testing.assert_allclose(out['ewma_reference'],[200*np.exp(-.26/6),200*np.exp(-.13/6)])
        np.testing.assert_allclose(out['legacy_pair'],[200*np.exp(-(.02+.16*2/7)),200*np.exp(-.02)])

    def test_same_filter_pair_is_algebraically_identical_on_same_panel(self):
        out,_=self.m.predict_all(self.h,self.t,query_start_ms=MINIMUM+1000)
        np.testing.assert_allclose(out['same_filter_pair'],out['same_panel_independent'],rtol=0,atol=1e-12)

    def test_missing_pair_falls_back_to_own_rank_and_does_not_invent_support(self):
        past=self.h[self.h['rank']==1000]
        out,diag=self.m.predict_all(past,self.t,query_start_ms=MINIMUM+1000)
        np.testing.assert_allclose(out['legacy_pair'],out['ewma_reference'])
        self.assertEqual(diag['legacy_pair'][0]['paired_events'],0)

    def test_no_history_simple_baseline_uses_time_and_growth_falls_back(self):
        h=self.h.iloc[:0]
        target=self.t.copy();target['progress']=.1;target['phase']=0
        out,_=self.m.predict_all(h,target,query_start_ms=MINIMUM+1000)
        np.testing.assert_allclose(out['time_proportion'],[1000,1000])
        np.testing.assert_allclose(out['history_growth'],[1000,1000])
        np.testing.assert_allclose(out['ewma_reference'],[200,200])

    def test_equal_end_boundary_and_current_outcome_fail_closed(self):
        past=self.h.copy();past.loc[0,'end_ms']=MINIMUM+1000
        with self.assertRaises(ValueError):self.m.predict_all(past,self.t,query_start_ms=MINIMUM+1000)
        target=self.t.assign(actual=250)
        with self.assertRaises(ValueError):self.m.predict_all(self.h,target,query_start_ms=MINIMUM+1000)

    def test_phase_uses_actual_progress_and_rejects_requested_phase(self):
        target=self.t.copy();target['progress']=.9;target['phase']=1
        with self.assertRaises(ValueError):self.m.predict_all(self.h,target,query_start_ms=MINIMUM+1000)
        target['phase']=2
        out,_=self.m.predict_all(self.h,target,query_start_ms=MINIMUM+1000)
        np.testing.assert_allclose(out['ewma_reference'],[200,200])

    def test_complete_feedback_and_delays_only_remove_existing_rows(self):
        got=self.m.feedback_history(self.h,query_start_ms=MINIMUM+1000,policy={'delay_ms':1000})
        self.assertTrue(got.empty)
        own=self.m.feedback_history(self.h,query_start_ms=MINIMUM+1000,policy={'drop_rank':2000})
        self.assertEqual(set(own['rank']),{1000})
        self.assertEqual(set(own.case_id),{'1:1000','2:1000'})

    def test_dlm_coordinate_change_retains_covariances(self):
        errors=np.array([[.2,.1],[.4,-.1],[.3,.1]])
        params={'Q':[[.001,.0004],[.0004,.002]],'R':[[.01,.002],[.002,.02]],
                'P0':[[.1,.03],[.03,.2]]}
        raw,state=self.m.dlm_filter(errors,params)
        transformed,other=self.m.dlm_filter(errors,params,anchor_coordinates=True)
        np.testing.assert_allclose(raw,transformed,atol=1e-12,rtol=0)
        self.assertGreaterEqual(np.linalg.eigvalsh(state['covariance']).min(),-1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(other['covariance']).min(),-1e-12)

    def test_dlm_missing_coordinate_can_still_update_other_and_stays_finite(self):
        params={'Q':np.eye(2)*.001,'R':np.eye(2)*.01,'P0':np.eye(2)*.1}
        value,state=self.m.dlm_filter(np.array([[.2,np.nan],[np.nan,.1]]),params)
        self.assertTrue(np.isfinite(value).all())
        self.assertGreater(value[0],0)
        self.assertGreater(value[1],0)
        self.assertGreaterEqual(np.linalg.eigvalsh(state['covariance']).min(),-1e-12)

    def test_empirical_intervals_use_event_maxima_and_explicit_unavailable(self):
        one=self.h.iloc[:2]
        intervals=self.m.empirical_intervals(one,self.t,np.array([200.,200.]))
        self.assertFalse(intervals.available.any())
        self.assertTrue(np.isinf(intervals.upper).all())
        records=[]
        for i,e in enumerate([.1,.2,.3,.4],1):records.append(frame(i,MINIMUM+i*100,{1000:e,2000:e/2}))
        history=pd.concat(records,ignore_index=True)
        intervals=self.m.empirical_intervals(history,self.t,np.array([200.,200.]))
        row=intervals[(intervals['rank']==1000)&(intervals.level==.8)].iloc[0]
        self.assertAlmostEqual(row.q_log,.4)
        self.assertAlmostEqual(row.lower,200*np.exp(-.4))
        self.assertAlmostEqual(row.upper,200*np.exp(.4))

    def test_fixed_dlm_parameters_do_not_read_nontraining_rows(self):
        params=self.m.dlm_parameters(self.h)
        injected=pd.concat([self.h,self.h.assign(event=lambda x:x.event+10,case_id=lambda x:x.case_id+'v',split='validation',log_error=99)],ignore_index=True)
        self.assertEqual(params,self.m.dlm_parameters(injected))

if __name__=='__main__':unittest.main()
