"""Detached-frame, event-balanced learned residual correction regressions."""
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

import joblib
import numpy as np
import pandas as pd

from analysis.data_policy import MIN_EVENT_START_MS

ROOT = Path(__file__).resolve().parents[1]
H = 3600000


def learned_fixture(events=8):
    """Nonmonotone IDs and unequal prefix counts, with no real answer access."""
    rows = []
    for i in range(events):
        event = 900 - i * 37
        start = MIN_EVENT_START_MS + i * 72 * H
        for j in range(1 + i % 3):
            progress = .2 + j * .25
            rows.append(dict(case_id=f'h:{i}:{j}', event=event, rank=1000,
                phase=0 if progress < .25 else 1, progress=progress,
                start_ms=start, end_ms=start + 48 * H, current=100. + i,
                base_prediction=200. + i, ewma_bias=.08, event_type='normal',
                band_group='1', duration_h=48., remaining_h=48 * (1-progress),
                log_base_growth=np.log(2), rate_24h_relative=.8,
                rate_6h_relative=1.2, log_error=.2 + .01 * (i % 2)))
    history = pd.DataFrame(rows)
    query = MIN_EVENT_START_MS + (events + 1) * 72 * H
    target = pd.DataFrame([dict(rows[0], case_id='t:1', event=17,
        start_ms=query, end_ms=query + 48 * H, ewma_bias=.07),
        dict(rows[0], case_id='t:2', event=17, rank=2000, phase=2,
        start_ms=query, end_ms=query + 48 * H, progress=.95, ewma_bias=-.02)])
    return history, target.drop(columns='log_error'), query


class RollingResidualModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.rolling_residual_models'),
                             'learned residual correction module is missing')
        return __import__('analysis.rolling_residual_models', fromlist=['fit_predict'])

    def fit(self, name, history, target, query, directory='model'):
        return self.module().fit_predict(name, history, target,
            query_start_ms=query, model_directory=self.root/directory)

    def test_small_history_falls_back_for_every_recipe_and_saves_audit(self):
        module = self.module()
        history, target, query = learned_fixture(7)
        target['ewma_bias'] = [.9, -.9]
        for name in module.CONFIGS:
            with self.subTest(name=name):
                result, audit = self.fit(name, history, target, query, name)
                expected = ({'residual_mlp_seed42', 'residual_mlp_seed123',
                    'residual_mlp_seed2026'} if name == 'residual_mlp' else {name})
                self.assertEqual(set(result), expected)
                for bias in result.values():
                    np.testing.assert_array_equal(bias, [.3, -.3])
                self.assertEqual(audit['status'], 'ewma_fallback')
                self.assertEqual(audit['history_event_count'], 7)
                self.assertTrue((self.root/name/'diagnostics.json').is_file())
                json.dumps(audit, allow_nan=False)

    def test_guard_checks_all_history_before_small_history_fallback(self):
        history, target, query = learned_fixture(1)
        variants = []
        for label, column, value in [('boundary', 'end_ms', query),
                ('future', 'end_ms', query + 1),
                ('old_history', 'start_ms', MIN_EVENT_START_MS - 1),
                ('self', 'event', target.event.iloc[0])]:
            bad = history.copy(); bad.loc[0, column] = value
            variants.append((label, bad, target, query))
        bad = target.copy(); bad['start_ms'] = MIN_EVENT_START_MS - 1
        variants.append(('old_target', history.iloc[:0], bad, MIN_EVENT_START_MS - 1))
        variants.append(('clock', history, target, query + 1))
        variants.append(('duplicate', pd.concat([history, history]), target, query))
        for label, prior, current, clock in variants:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.fit('ridge20', prior, current, clock, label)
                self.assertFalse((self.root/label).exists())

    def test_target_answer_columns_are_rejected_before_fallback(self):
        history, target, query = learned_fixture(1)
        for label in ('actual', 'target', 'final', 'log_error'):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.fit('ridge20', history, target.assign(**{label: 10}), query, label)

    def test_empty_released_history_has_explicit_unfitted_audit(self):
        history, target, query = learned_fixture(1)
        result, audit = self.fit('ridge100', history.iloc[:0], target, query)
        np.testing.assert_array_equal(result['ridge100'], target.ewma_bias)
        self.assertEqual(audit['history_event_count'], 0)
        self.assertFalse((self.root/'model'/'preprocessor.joblib').exists())

    def test_regression_learns_remaining_error_and_applies_global_shrink(self):
        history, target, query = learned_fixture()
        history['log_error'] = .2
        result, audit = self.fit('ridge20', history, target, query)
        expected = target.ewma_bias.to_numpy() + (.2-.08)*8/28
        np.testing.assert_allclose(result['ridge20'], expected, atol=1e-12)
        self.assertAlmostEqual(audit['learned_shrinkage'], 8/28)
        replay = self.module().replay_predict(self.root/'model', target,
                                           query_start_ms=query)
        np.testing.assert_array_equal(result['ridge20'], replay['ridge20'])

    def test_event_weights_and_predictions_ignore_repeated_identical_prefixes(self):
        history, target, query = learned_fixture()
        first, _ = self.fit('ridge100', history, target, query, 'first')
        extra = pd.concat([history[history.event == history.event.iloc[0]]]*15,
                          ignore_index=True)
        extra['case_id'] = [f'repeat:{i}' for i in range(len(extra))]
        second, _ = self.fit('ridge100', pd.concat([history, extra]), target,
                             query, 'second')
        np.testing.assert_allclose(first['ridge100'], second['ridge100'], atol=1e-12)
        weights = pd.read_csv(self.root/'second'/'training_weights.csv')
        np.testing.assert_allclose(weights.groupby('event').weight.sum(), 1., atol=1e-12)
        self.assertEqual(set(weights.case_id), set(history.case_id) | set(extra.case_id))

    def test_preprocessing_is_fit_only_on_history_and_handles_unseen_categories(self):
        history, target, query = learned_fixture()
        history['rate_24h_relative'] = np.nan
        history.loc[history.index[0], 'rate_6h_relative'] = np.inf
        first, _ = self.fit('ridge20', history, target, query, 'first')
        poisoned = target.copy()
        poisoned['band_group'] = 'unseen-band'; poisoned['event_type'] = 'unseen-type'
        poisoned['current'] = 1e9; poisoned['rank'] = 99999
        result, audit = self.fit('ridge20', history, poisoned, query, 'second')
        state1 = json.loads((self.root/'first'/'preprocessor.json').read_text())
        state2 = json.loads((self.root/'second'/'preprocessor.json').read_text())
        self.assertEqual(state1, state2)
        self.assertNotIn('unseen-band', state2['categories']['band_group'])
        self.assertNotIn('unseen-type', state2['categories']['event_type'])
        self.assertNotIn('99999', state2['categories']['rank'])
        self.assertTrue(np.isfinite(result['ridge20']).all())
        for forbidden in ('event', 'case_id', 'start_ms', 'end_ms', 'log_error'):
            self.assertNotIn(forbidden, state2['numeric_columns'])
            self.assertNotIn(forbidden, state2['categorical_columns'])
        self.assertTrue(np.isfinite(first['ridge20']).all())
        json.dumps(audit, allow_nan=False)

    def test_total_bias_is_clipped_and_artifacts_cannot_be_overwritten(self):
        history, target, query = learned_fixture()
        history['log_error'] = 100.
        result, _ = self.fit('ridge20', history, target, query)
        np.testing.assert_array_equal(result['ridge20'], [.3, .3])
        before = (self.root/'model'/'config.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.fit('ridge20', history, target, query)
        self.assertEqual(before, (self.root/'model'/'config.json').read_bytes())

    def test_tree_checkpoint_replays_and_uses_frozen_recipe(self):
        history, target, query = learned_fixture()
        result, audit = self.fit('residual_tree', history, target, query)
        replay = self.module().replay_predict(self.root/'model', target,
                                           query_start_ms=query)
        np.testing.assert_array_equal(result['residual_tree'], replay['residual_tree'])
        config = json.loads((self.root/'model'/'config.json').read_text())['recipe']
        for key, value in dict(iterations=200, depth=2, learning_rate=.03,
                l2_leaf_reg=25, random_seed=42, thread_count=2).items():
            self.assertEqual(config[key], value)
        self.assertEqual(audit['history_event_count'], 8)

    def test_three_neural_seeds_checkpoint_replay_and_repeat_fit_in_subprocess(self):
        # Keep PyTorch outside the parent test runner, which also fits tree models.
        code = textwrap.dedent('''
            import json, sys
            from pathlib import Path
            import numpy as np
            sys.path.insert(0, str(Path.cwd()/'tests'))
            from test_rolling_residual_models import learned_fixture
            from analysis.rolling_residual_models import fit_predict, replay_predict, CONFIGS
            import torch
            root = Path(sys.argv[1])
            history, target, query = learned_fixture()
            first, audit = fit_predict('residual_mlp', history, target,
                query_start_ms=query, model_directory=root/'first')
            second, _ = fit_predict('residual_mlp', history, target,
                query_start_ms=query, model_directory=root/'second')
            replay = replay_predict(root/'first', target, query_start_ms=query)
            assert set(first) == {f'residual_mlp_seed{s}' for s in (42,123,2026)}
            assert torch.get_num_threads() == 2
            assert CONFIGS['residual_mlp']['epochs'] == 80
            for seed in (42,123,2026):
                name = f'residual_mlp_seed{seed}'
                np.testing.assert_array_equal(first[name], replay[name])
                np.testing.assert_array_equal(first[name], second[name])
                checkpoint = torch.load(root/'first'/f'seed{seed}.pt',
                    map_location='cpu', weights_only=False)
                assert checkpoint['seed'] == seed
                assert checkpoint['epochs'] == 80
                assert checkpoint['optimizer_state_dict']
                assert np.isfinite(first[name]).all()
                assert np.max(np.abs(first[name])) <= .3
            json.dumps(audit, allow_nan=False)
            print('three-seed checkpoint replay verified')
        ''')
        env = os.environ.copy()
        if sys.platform == 'darwin':
            env.setdefault('DYLD_LIBRARY_PATH', '/opt/homebrew/opt/libomp/lib')
        result = subprocess.run([sys.executable, '-c', code, str(self.root)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('three-seed checkpoint replay verified', result.stdout)


if __name__ == '__main__':
    unittest.main()
