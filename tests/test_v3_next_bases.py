"""Metric targets, as-of peers, and sealed causal rolling-base checks."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis import rolling_benchmark, rolling_dataset
from analysis.event_features import EventFeatures
from test_rolling_v3 import fixture


class NextBaseTests(unittest.TestCase):
    def make(self, root):
        parent, curves = fixture(root)
        rolling_dataset.build_rolling(parent, root/'data', curves, source_identity={'fixture': True})
        return rolling_dataset.RollingDataset(root/'data')

    def test_mixed_split_pairing_preserves_original_ids_and_asof_peer(self):
        m = importlib.import_module('analysis.v3_next_bases')
        with tempfile.TemporaryDirectory() as tmp:
            data = self.make(Path(tmp)); meta = EventFeatures.load(data.metadata_directory)
            samples = list(data.inputs('train')) + list(data.inputs('validation'))
            arrays = m.prepare_arrays(samples, meta)
            self.assertEqual([c['case_id'] for c in arrays['cases']], [s['case_id'] for s in samples])
            self.assertEqual(arrays['pair_audit'].case_id.tolist(), [s['case_id'] for s in samples])
            audit = arrays['pair_audit'].query('peer_available == 1')
            self.assertTrue((audit.peer_observed_at <= audit.observed_at).all())
            # A peer's suffix after the own observed time may not affect own features.
            own = next(copy.deepcopy(s) for s in samples if len(s['times']) > 8)
            peer = next(copy.deepcopy(s) for s in samples
                if s['cutoff'] == own['cutoff'] and s['curve'].event == own['curve'].event
                and s['curve'].rank != own['curve'].rank)
            own['now'] = int(own['times'][-3]); own['times'] = own['times'][:-2]
            own['scores'] = own['scores'][:-2]; own['curve'].times = own['times'].copy()
            own['curve'].scores = own['scores'].copy(); own['score'] = own['scores'][-1]
            from analysis.backtest import snapshot
            own = snapshot(own['curve'], own['cutoff']) | {'case_id': own['case_id']}
            first = m.prepare_arrays([own, peer], meta)['paired'].iloc[0]
            peer['curve'].scores[-1] *= 1000
            second = m.prepare_arrays([own, peer], meta)['paired'].iloc[0]
            pd.testing.assert_series_equal(first, second)

    def test_four_targets_bounds_train_vocab_clock_and_event_equal_weights(self):
        m = importlib.import_module('analysis.v3_next_bases')
        with tempfile.TemporaryDirectory() as tmp:
            data = self.make(Path(tmp)); meta = EventFeatures.load(data.metadata_directory)
            arrays = m.prepare_arrays(list(data.inputs('train')), meta)
            start = min(r['start_ms'] for r in data.event_index.values() if r['split'] == 'validation')
            history = data.history_before(start)
            finals = {(c.event, c.rank): c.final for c in history}
            labels = {c['case_id']: finals[c['event'], c['rank']] for c in arrays['cases']}
            target = m.prepare_arrays(list(data.inputs('validation')), meta)
            target['event_type'][:] = 'unseen-type'
            target['band_shares'] = [{'future-band': 1.} for _ in target['cases']]
            for name in m.CONFIGS:
                bundle = m.fit_base(name, arrays, labels, fit_before_ms=start, iterations=2)
                prediction = m.predict_base(bundle, target)
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue((prediction >= target['current']).all())
                self.assertTrue((prediction <= target['current'] * np.exp(4) + 1e-6).all())
                self.assertNotIn('unseen-type', bundle['context_design'].types)
                self.assertNotIn('future-band', bundle['context_design'].bands)
                weights = pd.DataFrame(bundle['weight_rows'])
                for _, part in weights.groupby('model_rank'):
                    mass = part.groupby('event').weight.sum()
                    np.testing.assert_allclose(mass, mass.iloc[0])
                with self.assertRaisesRegex(ValueError, 'completed|clock'):
                    m.fit_base(name, arrays, labels, fit_before_ms=history[0].end, iterations=2)
                with self.assertRaisesRegex(ValueError, 'labels'):
                    m.fit_base(name, arrays, dict(labels, foreign=1.), fit_before_ms=start, iterations=2)

    def test_ewma_uses_one_mean_per_event_and_uncorrected_errors(self):
        m = importlib.import_module('analysis.v3_next_bases')
        state = m.OnlineEwma(.6)
        frame = pd.DataFrame({'case_id': ['a', 'b'], 'event': [1, 1],
            'rank': [1000, 1000], 'phase': [0, 0], 'predicted': [200., 200.]})
        state.observe(frame, {'a': 100., 'b': 100.}, event=1, end_ms=10)
        self.assertAlmostEqual(state.bias(1000, 0), np.log(2)/11)
        frame = pd.DataFrame({'case_id': ['c'], 'event': [2], 'rank': [1000], 'phase': [0], 'predicted': [100.]})
        state.observe(frame, {'c': 200.}, event=2, end_ms=20)
        self.assertAlmostEqual(state.bias(1000, 0), -.2*np.log(2)*2/12)

    def test_runner_guard_seal_selection_before_test_inputs_and_exact_controls(self):
        m = importlib.import_module('analysis.v3_next_base_runner')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = self.make(root); run = root/'new'
            rolling_benchmark.run_experiment(data, root/'base', iterations=2, allow_fixture=True)
            original_reveal, original_inputs = data.reveal_event, data.inputs
            def guarded(event, *, before_ms):
                row = data.event_index[event]
                self.assertLess(row['end_ms'], before_ms)
                if row['split'] != 'train':
                    self.assertTrue((run/'events'/str(event)/'event_manifest.json').exists())
                return original_reveal(event, before_ms=before_ms)
            def inputs(split):
                if split == 'test': self.assertTrue((run/'selection.json').exists())
                return original_inputs(split)
            with patch.object(data, 'reveal_event', side_effect=guarded), \
                 patch.object(data, 'inputs', side_effect=inputs), \
                 patch.object(data, 'labels', side_effect=AssertionError('No answer files')):
                selected = m.run_experiment(data, run, base_run=root/'base', iterations=2, allow_fixture=True)
            self.assertEqual(set(selected['validation_scores']), set(m.METHODS))
            seal = rolling_benchmark.verify_seal(run/'blind_manifest.json')
            self.assertEqual(len(seal['methods']), 14)
            old = pd.read_csv(root/'base/test_predictions.csv')
            new = pd.read_csv(run/'test_predictions.csv')
            for ours, theirs in [('uncorrected', 'rolling_context'), ('ewma_reference', 'rolling_context_ewma')]:
                np.testing.assert_allclose(new.query('method == @ours').set_index('case_id').predicted.sort_index(),
                    old.query('method == @theirs').set_index('case_id').predicted.sort_index(), rtol=1e-12)
            for event in [700, 4, 600, 5]:
                audit = json.loads((run/f'events/{event}/audit.json').read_text())
                self.assertTrue(all(r['end_ms'] < audit['query_start_ms'] for r in audit['released_history']))
                self.assertNotIn(event, audit['training_events'])
                self.assertTrue(all(r['event'] != event for r in audit['feedback_records']['rank_log400']))
            with patch.object(data, 'reveal_event', side_effect=AssertionError('Resume cannot open outcomes')):
                self.assertEqual(m.run_experiment(data, run, base_run=root/'base', iterations=2,
                    allow_fixture=True, resume=True), selected)

    def test_unreleased_current_and_future_outcomes_cannot_change_predictions(self):
        m = importlib.import_module('analysis.v3_next_base_runner')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = self.make(root)
            rolling_benchmark.run_experiment(data, root/'base', iterations=2, allow_fixture=True)
            m.run_experiment(data, root/'clean', base_run=root/'base', iterations=2, allow_fixture=True)
            original = data.reveal_event
            def poison(event, *, before_ms):
                curves = original(event, before_ms=before_ms)
                if event == 600:
                    for curve in curves:
                        curve.scores[-1] *= 100
                return curves
            with patch.object(data, 'reveal_event', side_effect=poison):
                # The poison becomes legal feedback only for event 5. Exact
                # cached-control replay then detects its changed prior outcome.
                with self.assertRaisesRegex(ValueError, 'EWMA replay'):
                    m.run_experiment(data, root/'poison', base_run=root/'base', iterations=2, allow_fixture=True)
            for event in [700, 4, 600]:
                self.assertEqual((root/f'clean/events/{event}/predictions.csv').read_bytes(),
                    (root/f'poison/events/{event}/predictions.csv').read_bytes())
