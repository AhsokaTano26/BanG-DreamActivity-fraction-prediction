import unittest
from types import SimpleNamespace
import pandas as pd
from sharing_input_guard import GuardedSharingBackend

class Backend:
    MIN_START_MS=1706504400000
    def __init__(self):self.calls=0
    def predict(self,h,c,cfg):self.calls+=1;return c.copy()
    def issue_experts(self,h,c):self.calls+=1;return c.copy()
    def mix_experts(self,h,c,cfg,experts):self.calls+=1;return c.copy()

class PhysicalGuardTests(unittest.TestCase):
    def setUp(self):
        self.backend=Backend();self.adapter=GuardedSharingBackend(self.backend)
        self.current=pd.DataFrame([dict(case_id='first',event=259,rank=1000,cutoff=1733137200000)])
        self.history=self.current.iloc[:0].copy()
    def test_alias_and_different_phase_do_not_hide_physical_duplicate(self):
        alias=self.current.assign(case_id='alias',phase=2)
        bad=pd.concat([self.current,alias],ignore_index=True)
        for method in ['predict','issue_experts','mix_experts']:
            with self.subTest(method=method),self.assertRaises(ValueError):
                args=(self.history,bad) if method=='issue_experts' else (self.history,bad,{})
                if method=='mix_experts':args+=({},)
                getattr(self.adapter,method)(*args)
        self.assertEqual(self.backend.calls,0)
    def test_history_alias_is_rejected(self):
        bad=pd.concat([self.current,self.current.assign(case_id='alias')])
        with self.assertRaises(ValueError):self.adapter.predict(bad,self.current,{})
    def test_numeric_spellings_share_identity(self):
        alias=self.current.assign(case_id='alias',event='259',rank='1000',cutoff=float(self.current.cutoff.iloc[0]))
        with self.assertRaises(ValueError):self.adapter.predict(self.history,pd.concat([self.current,alias]),{})
    def test_missing_and_nonintegral_clock_rejected(self):
        for bad in [self.current.drop(columns='cutoff'),self.current.assign(cutoff=float('nan')),self.current.assign(cutoff=1.5)]:
            with self.assertRaises(ValueError):self.adapter.predict(self.history,bad,{})
    def test_distinct_rank_same_cutoff_and_nonmutation(self):
        valid=pd.concat([self.current,self.current.assign(case_id='other',rank=2000)],ignore_index=True)
        old=valid.copy(deep=True);result=self.adapter.predict(self.history,valid,{})
        pd.testing.assert_frame_equal(valid,old);pd.testing.assert_frame_equal(result,old)
        self.assertEqual(self.backend.calls,1);self.assertEqual(self.adapter.MIN_START_MS,1706504400000)

if __name__=='__main__':unittest.main()
