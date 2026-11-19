import importlib.util,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from analysis.data_policy import MIN_EVENT_START_MS as S
H=3600000

def rows():
    return [dict(event=9,rank=r,phase=1,start_ms=S,end_ms=S+H,mean_log_error=e,cases=2,origin='live')
            for r,e in [(1000,.3),(2000,.1),(500,.2)]]

class SelectedModelTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('production.selected_model'),'selected recipe missing')
        return __import__('production.selected_model',fromlist=['build_bias'])

    def test_hand_calculated_pair_other_ranks_and_missing_pair(self):
        m=self.module();state=m.build_bias(rows(),before_ms=S+3*H)
        self.assertAlmostEqual(m.bias_for(state,1000,1),.1/11+.2/6)
        self.assertAlmostEqual(m.bias_for(state,2000,1),.1/11)
        self.assertAlmostEqual(m.bias_for(state,500,1),.2/11)
        self.assertEqual(m.bias_for(state,1500,1),0.)
        solo=m.build_bias(rows()[:1],before_ms=S+3*H)
        self.assertAlmostEqual(m.bias_for(solo,1000,1),.3/11)
        self.assertEqual(state,m.build_bias(rows()+rows(),before_ms=S+3*H))

    def test_dense_strict_clock_and_conflicting_event_metadata(self):
        m=self.module();state=m.build_bias(rows(),before_ms=S+3*H)
        bad=[dict(rows()[0],event=100,start_ms=S-10*H,end_ms=S-H),dict(rows()[0],event=101,end_ms=S+3*H)]
        self.assertEqual(state,m.build_bias(rows()+bad,before_ms=S+3*H))
        with self.assertRaises(ValueError):m.build_bias([rows()[0],dict(rows()[1],start_ms=S+1)],before_ms=S+3*H)

    def test_new_state_predicts_bounded_scores_and_keeps_old_pin_effective(self):
        m=self.module();state=m.build_bias(rows(),before_ms=S+3*H)
        raw=[dict(case_id='x',event=10,rank=1000,phase=1,start_ms=S+3*H,current_ep=100.,raw_predicted_final_ep=200.,predicted_final_ep=200.,available=True)]
        with patch.object(m.base,'predict_pack',return_value=raw):
            got=m.predict_pack('unused',[],calibration=state)
        self.assertAlmostEqual(got[0]['predicted_final_ep'],200*math.exp(-(.1/11+.2/6)))
        self.assertEqual(got[0]['forecast_method'],'legacy_pair')
        from production.model import build_bias
        old=build_bias(rows(),before_ms=S+3*H)
        oldrows=[dict(raw[0],predicted_final_ep=190.)]
        with patch.object(m.base,'predict_pack',return_value=oldrows):
            kept=m.predict_pack('unused',[],calibration=old)
        self.assertEqual(kept[0]['predicted_final_ep'],190.)
        self.assertEqual(kept[0]['forecast_method'],'pinned_independent_ewma')
        with patch.object(m.base,'predict_pack',return_value=[dict(raw[0],start_ms=S+H)]):
            with self.assertRaises(ValueError):m.predict_pack('unused',[],calibration=state)

if __name__=='__main__':unittest.main()
