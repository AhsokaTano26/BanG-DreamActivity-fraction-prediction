"""Causal warm-residual calibrators: boundaries, metric, weighting and replay."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_START = 1706504400000
H = 3600000
EXPERTS = ('ewma_warm', 'ewma_warm_fast', 'kalman_level', 'kalman_trend', 'kernel_progress')


def fixture(events=7):
    rows = []
    for i in range(events):
        start = MIN_START + i * 72 * H
        for rank in (1000, 2000):
            for j in range(1 + i % 3):
                progress = .3 + .1 * j
                row = dict(case_id=f'h:{i}:{rank}:{j}', event=1000 - i * 13,
                    rank=rank, phase=1, progress=progress, start_ms=start,
                    end_ms=start + 48 * H, current=100., base_prediction=200.,
                    ewma_bias=-.12, event_type='normal', band_group='1',
                    duration_h=48., remaining_h=48 * (1-progress),
                    log_base_growth=np.log(2), rate_24h_relative=.8,
                    rate_6h_relative=1.2, log_error=.15 if rank == 1000 else .05)
                row.update({'expert__'+name: 200.*np.exp(-(.04 if name == 'ewma_warm' else .15))
                            for name in EXPERTS})
                rows.append(row)
    history = pd.DataFrame(rows)
    query = MIN_START + (events + 1) * 72 * H
    target = pd.DataFrame([dict(rows[0], case_id='t:1000', event=17,
        start_ms=query, end_ms=query + 48 * H), dict(rows[0], case_id='t:2000',
        event=17, rank=2000, start_ms=query, end_ms=query + 48 * H)])
    target = target.drop(columns=['log_error', *['expert__'+name for name in EXPERTS]])
    return history, target, query


class NextCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.v3_next_calibration'),
            'warm calibration module is missing')
        return __import__('analysis.v3_next_calibration', fromlist=['predict_all'])

    def predict(self, history, target, query, name='model'):
        return self.module().predict_all(history, target, query_start_ms=query,
            model_directory=self.root/name)

    def test_fixed_menu_small_pool_bounded_and_no_input_mutation(self):
        module = self.module()
        history, target, query = fixture()
        pristine = history.copy(deep=True)
        scores, diagnostics, experts = self.predict(history, target, query)
        self.assertEqual(set(scores), set(module.CONFIGS) |
            {f'factorized_mlp_seed{s}' for s in (42, 123, 2026)})
        self.assertNotIn('ewma_reference', scores)
        self.assertEqual(tuple(experts), EXPERTS)
        self.assertEqual(diagnostics['factorized_mlp']['status'], 'ewma_fallback')
        for values in [*scores.values(), *experts.values()]:
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.all(values >= target.current))
            self.assertTrue(np.all(values <= target.current*np.exp(4)))
            self.assertLessEqual(np.max(np.abs(np.log(target.base_prediction/values))), .3+1e-12)
        for seed in (42, 123, 2026):
            np.testing.assert_array_equal(scores[f'factorized_mlp_seed{seed}'], scores['ewma_warm'])
        pd.testing.assert_frame_equal(pristine, history)
        json.dumps(diagnostics, allow_nan=False)

    def test_strict_clock_dense_start_self_and_answers_before_fallback(self):
        history, target, query = fixture(1)
        variants = []
        for column, value in [('end_ms', query), ('end_ms', query+1),
                ('start_ms', MIN_START-1), ('event', 17)]:
            bad = history.copy(); bad[column] = value
            variants.append((bad, target, query))
        variants.append((pd.concat([history, history]), target, query))
        variants.append((history, target, query+1))
        for name in ('actual', 'target', 'final', 'log_error', 'actual_final', 'expert__ewma_warm'):
            variants.append((history, target.assign(**{name: 1}), query))
        for i, (past, current, clock) in enumerate(variants):
            with self.subTest(i=i), self.assertRaises(ValueError):
                self.predict(past, current, clock, str(i))
            self.assertFalse((self.root/str(i)).exists())

    def test_observation_cutoff_checks_and_missing_pair_fallback(self):
        history, target, query = fixture(2)
        target['observed_at'] = target.start_ms + 10*H
        target['cutoff'] = target.start_ms + 9*H
        with self.assertRaises(ValueError):
            self.predict(history, target, query)
        target = target.drop(columns=['observed_at', 'cutoff'])
        history = history[history['rank'] == 1000]
        scores, diagnostics, _ = self.predict(history, target, query)
        for name in ('coupled_gap', 'coupled_kalman'):
            np.testing.assert_array_equal(scores[name], scores['ewma_warm'])
            self.assertEqual(diagnostics[name]['target_states'][0]['paired_events'], 0)

    def test_mape_log_median_matches_actual_percentage_objective(self):
        module = self.module()
        errors = np.asarray([-.2, -.1, .1])
        weights = np.asarray([.4, .15, .45])
        optimum = module.mape_log_median(errors, weights)
        grid = np.linspace(-.3, .3, 6001)
        losses = np.sum(weights*np.abs(np.exp(errors[None, :]-grid[:, None])-1), axis=1)
        self.assertAlmostEqual(optimum, grid[np.argmin(losses)], places=10)
        self.assertEqual(optimum, .1)  # plain weighted median would be -.1

    def test_bounded_breakpoint_optimum_includes_current_floor(self):
        module = self.module()
        errors = np.asarray([.01, .2, .25])
        growth = np.asarray([.02, .9, .9])
        weights = np.asarray([.6, .2, .2])
        panel = pd.DataFrame({'log_error': errors, 'current': 100.,
            'base_prediction': 100.*np.exp(growth)})
        optimum, objective = module.bounded_mape_bias(panel, weights)
        grid = np.linspace(-.3, .3, 6001)
        loss = np.abs(np.expm1(np.clip(growth[None,:]-grid[:,None], 0., 4.)
            +errors[None,:]-growth[None,:])) @ weights
        self.assertAlmostEqual(optimum, grid[np.argmin(loss)], places=10)
        self.assertAlmostEqual(objective, loss.min(), places=12)
        self.assertAlmostEqual(module.mape_log_median(errors,weights), .01)
        self.assertAlmostEqual(optimum,.25)

    def test_expert_recipes_and_history_only_numeric_preprocessing_are_explicit(self):
        module = self.module()
        for recipe in module.EXPERT_CONFIGS.values():
            self.assertEqual(recipe['initial_residuals'],'completed sealed prequential forecast events')
        history, target, query = fixture(2)
        history['rate_24h_relative'] = np.nan
        weights = module._event_weights(history)
        prep = module.NumericPreprocessor.fit(history,weights)
        state = prep.state()
        poisoned = target.assign(event_type='unseen',band_group='unknown',
            peer_log_ratio=np.inf,rate_6h_relative=np.nan,remaining_h=1e12)
        result = prep.transform(poisoned)
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(np.max(np.abs(result)),10.)
        self.assertEqual(state,prep.state())
        self.assertNotIn('event_type',state['numeric_columns'])
        self.assertNotIn('band_group',state['numeric_columns'])
        np.testing.assert_allclose(pd.Series(weights).groupby(history.event).sum(),1.)

    def test_empty_history_has_no_feedback_and_warm_zero_correction(self):
        history, target, query = fixture(1)
        scores, diagnostics, experts = self.predict(history.iloc[:0],target,query)
        for values in scores.values():
            np.testing.assert_allclose(values,target.base_prediction,atol=1e-12)
        self.assertEqual(diagnostics['_boundary']['history_events'],0)
        self.assertEqual(diagnostics['factorized_mlp']['status'],'ewma_fallback')

    def test_snapshot_density_does_not_add_event_weight(self):
        history, target, query = fixture(4)
        first, diag1, _ = self.predict(history, target, query, 'first')
        extra = pd.concat([history[history.event == history.event.iloc[0]]]*20, ignore_index=True)
        extra['case_id'] = [f'extra:{i}' for i in range(len(extra))]
        second, diag2, _ = self.predict(pd.concat([history, extra]), target, query, 'second')
        for name in self.module().CONFIGS:
            np.testing.assert_allclose(first[name], second[name], atol=1e-12)
        for name in ('mape_median', 'mape_kernel'):
            self.assertEqual(diag1[name]['target_states'], diag2[name]['target_states'])
        self.assertEqual(diag2['hedge_rank10']['groups'][0]['n_events'], 4)

    def test_hedge_uses_only_sealed_expert_scores_and_event_losses(self):
        history, target, query = fixture(7)
        scores, diagnostics, _ = self.predict(history, target, query)
        detail = next(g for g in diagnostics['hedge_rank30']['groups'] if g['rank'] == 1000)
        self.assertGreater(detail['weights']['ewma_warm_fast'], .075)
        self.assertLess(detail['weights']['ewma_warm'], .7)
        self.assertAlmostEqual(sum(detail['weights'].values()), 1.)
        self.assertAlmostEqual(detail['losses']['ewma_warm_fast'], 0., places=12)
        poisoned = history.copy()
        poisoned['expert__ewma_warm_fast'] = 200.*np.exp(.15)
        different, _, _ = self.predict(poisoned, target, query, 'different')
        self.assertNotEqual(scores['hedge_rank30'][0], different['hedge_rank30'][0])
        missing = history.drop(columns=['expert__kalman_trend'])
        _, diag, _ = self.predict(missing, target, query, 'missing')
        self.assertTrue(all(g['n_events'] == 0 for g in diag['hedge_rank10']['groups']))

    def test_atomic_models_and_replay_refuse_overwrite(self):
        history, target, query = fixture(1)
        first, _, _ = self.predict(history, target, query)
        replay = self.module().replay_neural(self.root/'model'/'factorized_mlp', target,
            query_start_ms=query)
        for name in replay:
            np.testing.assert_array_equal(first[name], replay[name])
        before = (self.root/'model'/'diagnostics.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.predict(history, target, query)
        self.assertEqual(before, (self.root/'model'/'diagnostics.json').read_bytes())

    def test_neural_true_bounded_mape_eventweights_and_three_seed_score_mean(self):
        code = textwrap.dedent('''
            import json, sys
            from pathlib import Path
            import numpy as np
            import pandas as pd
            sys.path.insert(0, str(Path.cwd()/'tests'))
            from test_v3_next_calibration import fixture
            from analysis.v3_next_calibration import predict_all, replay_neural, bounded_mape_loss
            import torch
            root = Path(sys.argv[1])
            history, target, query = fixture(8)
            scores, diag, _ = predict_all(history, target, query_start_ms=query, model_directory=root/'first')
            replay = replay_neural(root/'first'/'factorized_mlp', target, query_start_ms=query)
            second, _, _ = predict_all(history, target, query_start_ms=query, model_directory=root/'second')
            seeds = [f'factorized_mlp_seed{s}' for s in (42,123,2026)]
            np.testing.assert_array_equal(scores['factorized_mlp'], np.mean([scores[k] for k in seeds], axis=0))
            for name in replay:
                np.testing.assert_array_equal(scores[name], replay[name])
                np.testing.assert_array_equal(scores[name], second[name])
            weights = pd.read_csv(root/'first'/'factorized_mlp'/'training_weights.csv')
            np.testing.assert_allclose(weights.groupby('event').weight.sum(), 1.)
            assert diag['factorized_mlp']['baseline_source'] == 'sealed expert__ewma_warm final score'
            assert torch.get_num_threads() == 2
            for seed in (42,123,2026):
                checkpoint = torch.load(root/'first'/'factorized_mlp'/f'seed{seed}.pt', map_location='cpu', weights_only=False)
                assert checkpoint['epochs'] == 80 and checkpoint['optimizer_state_dict']
                assert checkpoint['recipe']['loss'] == 'event-equal actual bounded MAPE'
            current = torch.tensor([100.,100.], dtype=torch.float64)
            base = torch.tensor([110.,200.], dtype=torch.float64)
            bias = torch.tensor([.3,-.3], dtype=torch.float64, requires_grad=True)
            actual = torch.tensor([120.,180.], dtype=torch.float64)
            per = bounded_mape_loss(current, base, bias, actual)
            expected = np.abs(np.clip(np.asarray([110.,200.])*np.exp(-np.asarray([.3,-.3])), 100., 100.*np.exp(4))/np.asarray([120.,180.])-1)
            np.testing.assert_allclose(per.detach().numpy(), expected, atol=1e-12)
            per.sum().backward()
            assert torch.isfinite(bias.grad).all()
            prep1 = json.loads((root/'first'/'factorized_mlp'/'preprocessor.json').read_text())
            assert 'band_group' not in str(prep1) and 'event_type' not in str(prep1)
            print('bounded MAPE three-seed replay verified')
        ''')
        env = os.environ.copy()
        if sys.platform == 'darwin':
            env.setdefault('DYLD_LIBRARY_PATH', '/opt/homebrew/opt/libomp/lib')
        process = subprocess.run([sys.executable, '-c', code, str(self.root)], cwd=ROOT,
            env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn('bounded MAPE three-seed replay verified', process.stdout)


if __name__ == '__main__':
    unittest.main()
