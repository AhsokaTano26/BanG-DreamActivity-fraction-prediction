"""Display checkpoints must reflect predictions actually available at the time."""
import unittest
from analysis.data_policy import MIN_EVENT_START_MS as S
from production import web_history

H=3600000


class CheckpointTests(unittest.TestCase):
    def records(self):
        return [dict(observed_at=S+H,issued_at=S+H+1000,predicted_final_ep=120.),
                dict(observed_at=S+H,issued_at=S+3*H+1,predicted_final_ep=140.),
                dict(observed_at=S+2*H+1,issued_at=S+2*H+2,predicted_final_ep=130.)]

    def test_checkpoint_never_uses_later_prediction_or_later_revision(self):
        rows=web_history.forecast_checkpoints(self.records(),start_ms=S,end_ms=S+4*H,now_ms=S+5*H)
        rows={r['key']:r for r in rows}
        self.assertFalse(rows['progress_25']['available'])
        self.assertEqual(rows['progress_50']['predicted_final_ep'],120.)
        self.assertEqual(rows['progress_75']['predicted_final_ep'],130.)
        self.assertEqual(rows['remaining_1h']['predicted_final_ep'],130.)
        self.assertFalse(rows['progress_90']['available'])
        self.assertEqual(rows['progress_90']['reason'],'没有一小时内已签发的预测')
        self.assertFalse(rows['remaining_24h']['available'])
        self.assertEqual(rows['last']['predicted_final_ep'],130.)

    def test_future_stages_and_unissued_predictions_remain_unavailable(self):
        rows=web_history.forecast_checkpoints(self.records(),start_ms=S,end_ms=S+4*H,now_ms=S+H//2)
        self.assertFalse(any(row['available'] for row in rows))
        self.assertEqual(next(r for r in rows if r['key']=='progress_75')['reason'],'尚未到达该时点')

    def test_missing_records_are_explicit_and_rows_have_unique_stage_keys(self):
        rows=web_history.forecast_checkpoints([],start_ms=S,end_ms=S+72*H,now_ms=S+73*H)
        self.assertEqual(len(rows),8)
        self.assertEqual(len({row['key'] for row in rows}),8)
        self.assertFalse(any(row['available'] for row in rows))
        self.assertTrue(all(row['predicted_final_ep'] is None for row in rows))

    def test_last_revision_wins_when_issue_clocks_are_equal(self):
        rows=web_history.forecast_checkpoints([
            dict(observed_at=S+H,issued_at=S+H,predicted_final_ep=110.),
            dict(observed_at=S+H,issued_at=S+H,predicted_final_ep=120.)],
            start_ms=S,end_ms=S+4*H,now_ms=S+5*H)
        self.assertEqual(next(r for r in rows if r['key']=='progress_25')['predicted_final_ep'],120.)
