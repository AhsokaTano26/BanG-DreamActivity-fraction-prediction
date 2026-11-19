"""User-visible one-shot predictions must preserve live isolation and pins."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from analysis.backtest import Curve
from analysis.data_policy import MIN_EVENT_START_MS as S
from production.state import Ledger

H = 3600000
NOW = S + 5 * H
EVENT = dict(event=329, start_ms=S + 3 * H, end_ms=S + 9 * H,
             name='测试活动', kind='normal')


class StoreFixture:
    curves = []
    index = [EVENT]

    def __init__(self, root):
        self.metadata_directory = Path(root) / 'metadata'

    def events(self, now):
        return self.index

    def load_curves(self, before_ms, ranks, complete_only=True):
        return [c for c in self.curves if c.rank in ranks]


class InteractiveForecastTests(unittest.TestCase):
    def setUp(self):
        from production import interactive
        self.m = interactive
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = dict(root=str(self.root), ranks=[1000, 2000], forecast_method='legacy_pair')
        self.ledger = Ledger(self.root)
        self.ledger.fact('bootstrap_complete', True)
        self.manifest = dict(model_id='ready', history_max_end_ms=S + H, production_ready=True)
        self.ledger.create_pin(EVENT, self.root / 'model', 'ready',
                               {'schema_version': 1}, now_ms=S + 3 * H)
        StoreFixture.index = [EVENT]
        StoreFixture.curves = [Curve(329, rank, 'normal', EVENT['start_ms'], EVENT['end_ms'],
            np.array([S + 3 * H + 1, S + 4 * H, NOW - 1000, NOW]),
            np.array([10., 30., 45., 50.])) for rank in [1000, 2000]]

    def fake_predict(self, path, samples, **kw):
        return [dict(event=s['curve'].event, rank=s['curve'].rank, phase=1,
            start_ms=s['curve'].start, end_ms=s['curve'].end, cutoff=s['cutoff'],
            observed_at=s['now'], current_ep=s['score'], progress=s['progress'],
            raw_predicted_final_ep=110., predicted_final_ep=105., available=True,
            provisional=False, model_id='ready', forecast_method='pinned_independent_ewma') for s in samples]

    def patches(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(self.m, 'Store', StoreFixture))
        stack.enter_context(patch.object(self.m, 'verify_pack', return_value=self.manifest))
        stack.enter_context(patch.object(self.m, 'predict_pack', side_effect=self.fake_predict))
        return stack

    def test_requested_rank_records_real_issue_without_overwriting_all_rank_snapshot(self):
        (self.root / 'latest.json').write_text('{"all_ranks":true}')
        original = self.ledger.pin(329)
        with self.patches():
            result = self.m.predict_once(self.config, 1000, now_ms=NOW, sync=False)
        self.assertEqual([r['rank'] for r in result['predictions']], [1000])
        self.assertEqual(result['predictions'][0]['predicted_final_ep'], 105.)
        self.assertEqual(self.ledger.summary()['forecasts'], 1)
        self.assertEqual(self.ledger.pin(329), original)
        self.assertEqual(json.loads((self.root / 'latest.json').read_text()), {'all_ranks': True})
        self.assertTrue((self.root / 'requests/latest_t1000.json').exists())

    def test_network_refresh_scopes_rank_and_failed_refresh_marks_cached_result_degraded(self):
        async def refresh(store, now_ms, ranks, transport=None):
            self.assertEqual(ranks, [2000])
            return {'status': 'degraded', 'errors': [{'error': 'network unavailable'}]}
        with self.patches(), patch.object(self.m, 'sync_data', side_effect=refresh):
            result = self.m.predict_once(self.config, 2000, now_ms=NOW)
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(result['predictions'][0]['rank'], 2000)

    def test_stale_data_cannot_be_reported_as_success(self):
        for c in StoreFixture.curves:
            c.times[:] -= 2 * H
        with self.patches():
            result = self.m.predict_once(self.config, 1000, now_ms=NOW, sync=False)
        self.assertEqual(result['status'], 'unavailable')
        self.assertFalse(result['predictions'][0]['available'])
        self.assertEqual(self.ledger.summary()['forecasts'], 0)

    def test_future_model_is_rejected_before_recording(self):
        self.manifest['history_max_end_ms'] = EVENT['start_ms']
        with self.patches(), self.assertRaisesRegex(ValueError, '期初|future|causal'):
            self.m.predict_once(self.config, 1000, now_ms=NOW, sync=False)
        self.assertEqual(self.ledger.summary()['forecasts'], 0)

    def test_missing_event_returns_explicit_waiting_status(self):
        StoreFixture.index = []
        with self.patches():
            result = self.m.predict_once(self.config, 1000, now_ms=NOW, sync=False)
        self.assertEqual(result['status'], 'waiting_for_event')
        self.assertEqual(result['predictions'], [])

    def test_new_pin_uses_only_pre_event_feedback_and_does_not_fit(self):
        with self.ledger.connect() as con:
            con.execute('DELETE FROM pins')
        self.ledger.seed([dict(event=1, rank=1000, phase=1, start_ms=S, end_ms=S+H,
            mean_log_error=.1, cases=1, actual_final=100., origin='retrospective_v3_seed'),
            dict(event=2, rank=1000, phase=1, start_ms=S+2*H, end_ms=NOW-1,
            mean_log_error=.2, cases=1, actual_final=100., origin='retrospective_v3_seed')], now_ms=NOW)
        def bias(rows, *, before_ms, exclude_event_id):
            self.assertEqual(before_ms, EVENT['start_ms'])
            self.assertEqual([r['event'] for r in rows], [1])
            return {'schema_version': 2, 'forecast_method': 'legacy_pair'}
        with self.patches(), patch.object(self.m, '_previous', return_value=(self.root/'ready', self.manifest)), \
             patch.object(self.m, 'build_bias', side_effect=bias), \
             patch.object(self.m, 'seed_feedback', return_value={'status':'complete'}):
            result = self.m.predict_once(self.config, 1000, now_ms=NOW, sync=False)
        self.assertTrue(result['predictions'][0]['available'])
        self.assertEqual(self.ledger.pin(329)['calibration']['forecast_method'], 'legacy_pair')

    def test_new_activity_collects_verified_outcome_of_previously_issued_forecast(self):
        with self.ledger.connect() as con:
            con.execute('DELETE FROM pins')
        previous = dict(event=900, start_ms=S, end_ms=S+H)
        self.ledger.record(previous, [dict(rank=1000, phase=1, observed_at=S+H//2,
            current_ep=50., raw_predicted_final_ep=110., predicted_final_ep=105., available=True)],
            cutoff_ms=S+H//2, issued_ms=S+H//2, model_id='past')
        StoreFixture.curves.append(Curve(900, 1000, 'normal', S, S+H,
            np.array([S+1,S+H//4,S+H//2,S+3*H//4,S+H]), np.array([1.,25.,50.,75.,100.])))
        def bias(rows, **kw):
            live = [r for r in rows if r['origin'] == 'live']
            self.assertEqual([r['event'] for r in live], [900])
            self.assertAlmostEqual(live[0]['mean_log_error'], np.log(1.1))
            return {'schema_version': 2, 'forecast_method': 'legacy_pair'}
        with self.patches(), patch.object(self.m, '_previous', return_value=(self.root/'ready', self.manifest)), \
             patch.object(self.m, 'build_bias', side_effect=bias), \
             patch.object(self.m, 'seed_feedback', return_value={'status':'complete'}):
            self.m.predict_once(self.config, 1000, now_ms=NOW, sync=False)

    def test_rank_parser_and_default_cli_require_explicit_single_rank(self):
        self.assertEqual(self.m.rank_argument('T2000'), 2000)
        with self.assertRaises(Exception):
            self.m.rank_argument('999')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.m.main([])

    def test_cli_formats_current_and_predicted_scores_and_exits(self):
        config_path = self.root/'config.json'
        config_path.write_text(json.dumps(self.config))
        result = {'status':'ok', 'as_of':'2026-11-19T01:00:00+08:00',
            'predictions':[self.fake_predict(None, [dict(curve=StoreFixture.curves[0],
                 now=NOW, cutoff=NOW, score=50., progress=1/3)])[0]]}
        with patch.object(self.m, 'read_config', return_value=self.config), \
             patch.object(self.m, 'predict_once', return_value=result), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = self.m.main(['--rank', 'T1000', '--config', str(config_path)])
        self.assertEqual(code, 0)
        self.assertIn('T1000', output.getvalue())
        self.assertIn('105', output.getvalue())


if __name__ == '__main__':
    unittest.main()
