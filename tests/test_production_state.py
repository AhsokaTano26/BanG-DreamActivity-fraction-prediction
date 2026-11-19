from pathlib import Path
import tempfile
import unittest
import numpy as np
from analysis.backtest import Curve
from analysis.data_policy import MIN_EVENT_START_MS as S
from production.state import Ledger

class LiveLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.ledger=Ledger(Path(self.tmp.name))
        self.event={'event':9,'start_ms':S,'end_ms':S+10000}
        self.row={'rank':1000,'phase':1,'observed_at':S+5000,'current_ep':50.,'raw_predicted_final_ep':110.,'predicted_final_ep':100.}

    def test_duplicate_observation_is_never_reissued_or_reweighted(self):
        for expected in [1,0]:
            self.assertEqual(self.ledger.record(self.event,[self.row],cutoff_ms=S+5001,issued_ms=S+5001,model_id='m'),expected)
        self.assertEqual(self.ledger.summary()['forecasts'],1)

    def test_feedback_waits_for_verified_completion_then_uses_raw_forecast_once(self):
        self.ledger.record(self.event,[self.row],cutoff_ms=S+5001,issued_ms=S+5001,model_id='m')
        curve=Curve(9,1000,'normal',S,S+10000,np.array([S,S+9999]),np.array([1.,100.]))
        self.assertEqual(self.ledger.learn_completed([curve],now_ms=S+10000,grace_ms=0),0)
        self.assertEqual(self.ledger.learn_completed([curve],now_ms=S+10001,grace_ms=0),1)
        self.assertEqual(self.ledger.learn_completed([curve],now_ms=S+10002,grace_ms=0),0)
        rows=self.ledger.feedback_rows(before_ms=S+10001)
        self.assertAlmostEqual(rows[0]['mean_log_error'],np.log(1.1))
        self.assertEqual(rows[0]['cases'],1)

    def test_future_prediction_and_invalid_phase_are_rejected(self):
        with self.assertRaises(ValueError):self.ledger.record(self.event,[self.row],cutoff_ms=S+4000,issued_ms=S+5001,model_id='m')
        with self.assertRaises(ValueError):self.ledger.record(self.event,[self.row|{'phase':4}],cutoff_ms=S+5001,issued_ms=S+5001,model_id='m')

    def test_pin_is_immutable_and_roundtrips_snapshot(self):
        original=self.ledger.create_pin(self.event,'/tmp/model','m',{'groups':{}},now_ms=S+5)
        self.assertEqual(original['calibration'],{'groups':{}})
        with self.assertRaises(Exception):self.ledger.create_pin(self.event,'/tmp/other','o',{},now_ms=S+6)
        self.assertEqual(self.ledger.pin(9)['model_id'],'m')

    def test_corrected_observation_is_audited_but_feedback_uses_one_latest_revision(self):
        self.ledger.record(self.event,[self.row],cutoff_ms=S+5001,issued_ms=S+5001,model_id='m')
        changed=self.row|{'raw_predicted_final_ep':120.,'current_ep':55.}
        self.assertEqual(self.ledger.record(self.event,[changed],cutoff_ms=S+5002,issued_ms=S+5002,model_id='m'),1)
        self.assertEqual(self.ledger.record(self.event,[changed],cutoff_ms=S+5003,issued_ms=S+5003,model_id='m'),0)
        with self.ledger.connect() as con:
            self.assertEqual(con.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0],1)
            self.assertEqual(con.execute('SELECT COUNT(*) FROM forecast_revisions').fetchone()[0],1)
        curve=Curve(9,1000,'normal',S,S+10000,np.array([S,S+9999]),np.array([1.,100.]))
        self.ledger.learn_completed([curve],now_ms=S+10001,grace_ms=0)
        row=self.ledger.feedback_rows(before_ms=S+10001)[0]
        self.assertAlmostEqual(row['mean_log_error'],np.log(1.2));self.assertEqual(row['cases'],1)

    def test_revision_returning_to_prior_content_still_tracks_latest_issuance(self):
        first=self.row|{'raw_predicted_final_ep':100.}
        changed=self.row|{'raw_predicted_final_ep':200.}
        for i,row in enumerate([first,changed,first,changed]):
            self.assertEqual(self.ledger.record(self.event,[row],cutoff_ms=S+5001+i,issued_ms=S+5001+i,model_id='m'),1)
        curve=Curve(9,1000,'normal',S,S+10000,np.array([S,S+9999]),np.array([1.,100.]))
        self.ledger.learn_completed([curve],now_ms=S+10001,grace_ms=0)
        self.assertAlmostEqual(self.ledger.feedback_rows(before_ms=S+10001)[0]['mean_log_error'],np.log(2))
