"""Detached-history production fits, causal inference and sealed replay."""
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from analysis.backtest import Curve, snapshot
from analysis.data_policy import MIN_EVENT_START_MS
from analysis.holdout_dataset import prefix_curve

H = 3_600_000


def model_fixture(root, *, ranks=(1000, 2000, 5000)):
    metadata = root / 'metadata'
    metadata.mkdir()
    curves = []
    for index, event in enumerate((900, 2, 800)):
        start = MIN_EVENT_START_MS + index * 72 * H
        end = start + 48 * H
        times = np.arange(start, end + 1, H // 4, dtype=np.int64)
        for rank in ranks:
            scores = (1 + np.arange(len(times), dtype=float) ** (1.08 + index / 100)) * 10000 / rank
            curves.append(Curve(event, rank, 'normal', start, end, times.copy(), scores))
    events = {str(c.event): {'characters': [{'characterId': 1}]} for c in curves}
    values = {'events': events, 'characters': {'1': {'bandId': 1}},
              'bands': {'1': {'bandName': ['A'] * 5}}, 'cards': {}}
    for name, value in values.items():
        (metadata / (name + '.json')).write_text(json.dumps(value))
    return curves, metadata


def live_sample(curves, metadata, rank=1000):
    original = next(c for c in reversed(curves) if c.rank == rank)
    delta = original.duration + 24 * H
    c = replace(original, event=1, start=original.start + delta,
                end=original.end + delta, times=original.times + delta)
    values = json.loads((metadata / 'events.json').read_text())
    values['1'] = {'characters': [{'characterId': 7}]}
    (metadata / 'events.json').write_text(json.dumps(values))
    characters = json.loads((metadata / 'characters.json').read_text())
    characters['7'] = {'bandId': 7}
    (metadata / 'characters.json').write_text(json.dumps(characters))
    cutoff = c.start + 12 * H
    prefix = prefix_curve(c, cutoff)
    sample = snapshot(prefix, cutoff)
    sample.update(case_id=f'live:{c.event}:{rank}:{cutoff}', stage='live')
    return sample


class ProductionModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assertIsNotNone(importlib.util.find_spec('production.model'), 'production.model missing')
        from production import model
        self.model = model
        self.curves, self.metadata = model_fixture(self.root)
        self.before = max(c.end for c in self.curves) + H

    def fit(self, **kwargs):
        return self.model.train_pack(self.curves, self.metadata, self.root / 'versions',
            before_ms=self.before, ranks=[1000, 2000, 5000, 30000], iterations=3, **kwargs)

    def test_quality_and_actual_date_clock_not_event_ids(self):
        good = self.curves[0]
        old = replace(good, event=99999, start=MIN_EVENT_START_MS - 1)
        future = replace(good, event=4, end=self.before)
        sparse = replace(good, event=5, times=good.times[:4], scores=good.scores[:4])
        stale = replace(good, event=6, times=good.times[:-1] - 2 * H, scores=good.scores[:-1])
        negative = replace(good, event=7, scores=good.scores * -1)
        nan = replace(good, event=8, scores=good.scores * np.nan)
        self.curves += [old, future, sparse, stale, negative, nan]
        manifest = self.model.verify_pack(self.fit())
        self.assertEqual(set(manifest['history_event_ids']), {900, 2, 800})
        self.assertFalse(manifest['production_ready'])
        self.assertEqual(manifest['rank_support']['30000']['events'], 0)
        self.assertLess(manifest['history_max_end_ms'], self.before)

    def test_delayed_first_observation_skips_early_origins_but_keeps_later_cases(self):
        curve = self.curves[0]
        keep = curve.times >= curve.start + 2 * H
        self.curves[0] = replace(curve, times=curve.times[keep], scores=curve.scores[keep])
        path = self.fit()
        rows = json.loads((path / 'training_weights.json').read_text())['context']
        cases = [r for r in rows if r['event'] == curve.event and r['rank'] == curve.rank]
        self.assertTrue(cases)
        self.assertNotIn(f'production:{curve.event}:{curve.rank}:{curve.start + H}',
                         {r['case_id'] for r in cases})
        self.assertIn(f'production:{curve.event}:{curve.rank}:{curve.start + 3 * H}',
                      {r['case_id'] for r in cases})
        self.assertEqual(self.model.verify_pack(path)['rank_support']['1000']['events'], 3)

    def test_corrected_closed_scores_omit_unsupported_prefixes_and_keep_actual_final(self):
        originals = {}
        for index, curve in enumerate(self.curves):
            if curve.event == 900 and curve.rank in (1000, 5000):
                scores = curve.scores.copy()
                scores[curve.times <= curve.start + 24 * H] = curve.final * 2
                self.curves[index] = replace(curve, scores=scores)
                originals[curve.rank] = scores.copy()
        path = self.fit()
        manifest = self.model.verify_pack(path)
        quality = json.loads((path / 'training_quality.json').read_text())
        self.assertEqual(manifest['training_quality'], quality)
        self.assertGreater(quality['unsupported_label_cases_by_event_rank']['900:1000'], 0)
        self.assertGreater(quality['unsupported_label_cases_by_event_rank']['900:5000'], 0)
        self.assertEqual(quality['unsupported_label_cases'],
                         sum(quality['unsupported_label_cases_by_event_rank'].values()))
        self.assertIn('verified final', quality['reason'])
        weights = json.loads((path / 'training_weights.json').read_text())
        for family, rows in weights.items():
            masses = {}
            for row in rows:
                masses[row['event']] = masses.get(row['event'], 0.) + row['weight']
            self.assertTrue(np.allclose(list(masses.values()), next(iter(masses.values()))))
            for rank in (1000, 5000):
                if family == 'context' and rank == 5000:
                    continue
                keys = {r['case_id'] for r in rows if r['event'] == 900 and r['rank'] == rank}
                self.assertNotIn(f'production:900:{rank}:{self.curves[0].start + 24 * H}', keys)
                self.assertIn(f'production:900:{rank}:{self.curves[0].start + 36 * H}', keys)
        for curve in self.curves:
            if curve.event == 900 and curve.rank in originals:
                np.testing.assert_array_equal(curve.scores, originals[curve.rank])
        context = joblib.load(path / 'context.joblib')
        actual = next(c.final for c in self.curves if c.event == 900 and c.rank == 1000)
        self.assertEqual(next(r['final'] for r in context['history_provenance']
                              if r['event'] == 900 and r['rank'] == 1000), actual)

    def test_unchanged_content_reuses_version_without_fit_or_clock_digest(self):
        path = self.fit()
        events = json.loads((self.metadata / 'events.json').read_text())
        events['999999'] = {'characters': [{'characterId': 1}]}
        (self.metadata / 'events.json').write_text(json.dumps(events))
        with patch('production.model.fit_context', side_effect=AssertionError('retraining')):
            again = self.model.train_pack(list(reversed(self.curves)), self.metadata,
                self.root / 'versions', before_ms=self.before + H,
                ranks=[1000, 2000, 5000, 30000], iterations=3)
        self.assertEqual(path, again)
        self.assertEqual(len(list((self.root / 'versions').iterdir())), 1)

    def test_equal_total_event_weights_unique_physical_cutoffs(self):
        # Sparse one rank of one event changes case counts, never activity mass.
        self.curves[0] = replace(self.curves[0], times=self.curves[0].times[::4], scores=self.curves[0].scores[::4])
        path = self.fit()
        rows = json.loads((path / 'training_weights.json').read_text())
        for family in ('context', 'shared'):
            part = rows[family]
            self.assertEqual(len({r['case_id'] for r in part}), len(part))
            if family == 'shared':
                self.assertEqual({r['rank'] for r in part}, {1000, 2000, 5000})
            masses = {}
            for row in part:
                masses[row['event']] = masses.get(row['event'], 0) + row['weight']
            self.assertTrue(np.allclose(list(masses.values()), next(iter(masses.values()))))

    def test_reload_exact_prediction_and_new_event_metadata_unknown_vocab(self):
        path = self.fit()
        sample = live_sample(self.curves, self.metadata)
        first = self.model.predict_pack(path, [sample], metadata_directory=self.metadata)
        second = self.model.predict_pack(path, [sample], metadata_directory=self.metadata)
        self.assertEqual(first, second)
        row = first[0]
        self.assertGreaterEqual(row['predicted_final_ep'], row['current_ep'])
        self.assertTrue(math.isfinite(row['raw_predicted_final_ep']))
        self.assertEqual(row['training_events'], 3)
        self.assertEqual(row['event'], 1)
        saved = joblib.load(path / 'metadata.joblib')
        self.assertEqual(saved.character_ids, [1])
        self.assertEqual(saved.band_ids, [1])
        self.assertNotIn('1', saved.events)
        # The shared recipe also replays with current metadata and its frozen schema.
        shared = live_sample(self.curves, self.metadata, rank=5000)
        self.assertTrue(self.model.predict_pack(path, [shared], metadata_directory=self.metadata)[0]['available'])

    def test_inference_rejects_full_curve_future_self_and_unsafe_old_pack(self):
        path = self.fit()
        sample = live_sample(self.curves, self.metadata)
        future = dict(sample)
        future['curve'] = replace(sample['curve'], times=np.append(sample['times'], sample['cutoff'] + 1),
                                  scores=np.append(sample['scores'], sample['score'] + 1))
        with self.assertRaisesRegex(ValueError, 'prefix|future'):
            self.model.predict_pack(path, [future], metadata_directory=self.metadata)
        own = dict(sample)
        own['curve'] = replace(sample['curve'], event=self.curves[0].event)
        with self.assertRaisesRegex(ValueError, 'history|training|own'):
            self.model.predict_pack(path, [own], metadata_directory=self.metadata)
        early = dict(sample)
        early['curve'] = replace(sample['curve'], start=self.curves[0].start)
        with self.assertRaises(ValueError):
            self.model.predict_pack(path, [early], metadata_directory=self.metadata)
        (path / 'metadata.joblib').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'artifact|changed|corrupt'):
            self.model.verify_pack(path)
        # Restore the exact metadata used for this pack to request its same identity.
        for source in (path / 'metadata').glob('*.json'):
            (self.metadata / source.name).write_bytes(source.read_bytes())
        with self.assertRaises(ValueError):
            self.fit()

    def test_formal_defaults_and_calibration_policy_and_event_clock(self):
        path = self.model.train_pack(self.curves, self.metadata, self.root / 'versions',
            before_ms=self.before, ranks=[1000, 2000, 5000])
        manifest = self.model.verify_pack(path)
        self.assertTrue(manifest['production_ready'])
        self.assertEqual(manifest['config']['context']['iterations'], 400)
        self.assertEqual(manifest['config']['shared']['iterations'], 600)
        sample = live_sample(self.curves, self.metadata)
        coverage_row = self.model.predict_pack(path, [sample], metadata_directory=self.metadata)[0]
        self.assertTrue(coverage_row['provisional'])
        self.assertEqual(coverage_row['reason'], 'insufficient historical activities for this rank')
        self.assertEqual(manifest['config']['minimal_support_for_confident'], 20)
        old_state = {'groups': {'1000:0': {'count': 9, 'mean': .2}}}
        with self.assertRaisesRegex(ValueError, 'policy|calibration|Calibration'):
            self.model.predict_pack(path, [sample], metadata_directory=self.metadata, calibration=old_state)
        record = dict(event=5, rank=1000, phase=0, start_ms=sample['curve'].start,
                      end_ms=sample['curve'].end, mean_log_error=.1)
        future_state = self.model.build_bias([record], before_ms=sample['curve'].end + 1)
        with self.assertRaisesRegex(ValueError, 'Calibration|calibration'):
            self.model.predict_pack(path, [sample], metadata_directory=self.metadata, calibration=future_state)

    def test_calibration_keeps_large_finite_inputs_bounded(self):
        path = self.fit()
        sample = live_sample(self.curves, self.metadata)
        multiplier = 1e308 / sample['score']
        sample['curve'] = replace(sample['curve'], scores=sample['curve'].scores * multiplier)
        state = self.model.build_bias([dict(event=7, rank=1000, phase=1,
            start_ms=MIN_EVENT_START_MS, end_ms=MIN_EVENT_START_MS + 1, mean_log_error=-100.)],
            before_ms=sample['curve'].start)
        predict = self.model._base_predictions
        def huge(directory, samples, metadata):
            if samples[0]['case_id'].startswith('live:'):
                return {s['case_id']: float(np.finfo(float).max) for s in samples}
            return predict(directory, samples, metadata)
        with patch('production.model._base_predictions', side_effect=huge):
            row = self.model.predict_pack(path, [sample], metadata_directory=self.metadata, calibration=state)[0]
        self.assertTrue(math.isfinite(row['predicted_final_ep']))
        self.assertGreaterEqual(row['predicted_final_ep'], row['current_ep'])

    def test_missing_rank_is_reported_without_a_fabricated_forecast(self):
        path = self.fit()
        sample = live_sample(self.curves, self.metadata)
        sample['curve'] = replace(sample['curve'], rank=30000)
        row = self.model.predict_pack(path, [sample], metadata_directory=self.metadata)[0]
        self.assertFalse(row['available'])
        self.assertIsNone(row['predicted_final_ep'])

    def test_bias_rebuild_completion_order_group_once_and_clock(self):
        m = self.model
        start = MIN_EVENT_START_MS
        rows = [dict(event=e, rank=1000, phase=1, start_ms=start + i * 100,
                     end_ms=start + i * 100 + 50, mean_log_error=error,
                     cases=2, origin='issued')
                for i, (e, error) in enumerate(((900, .2), (2, -.1), (800, .4)))]
        state = m.build_bias(rows[::-1] + [dict(rows[0])], before_ms=start + 1000)
        expected = ((.2 * .7 - .1 * .3) * .7 + .4 * .3) * 3 / 13
        self.assertAlmostEqual(m.bias_for(state, 1000, 1), expected)
        self.assertEqual(state['groups']['1000:1']['count'], 3)
        limited = m.build_bias(rows, before_ms=rows[1]['end_ms'])
        self.assertEqual(limited['groups']['1000:1']['count'], 1)
        excluded = m.build_bias(rows, before_ms=start + 1000, exclude_event_id=2)
        self.assertEqual(excluded['groups']['1000:1']['count'], 2)
        old = dict(rows[0], event=999, start_ms=start - 1)
        self.assertEqual(m.build_bias([old], before_ms=start + 1000)['groups'], {})
        with self.assertRaises(ValueError):
            m.build_bias(rows + [dict(rows[0], mean_log_error=.9)], before_ms=start + 1000)
        huge = m.build_bias([dict(r, mean_log_error=100.) for r in rows], before_ms=start + 1000)
        self.assertEqual(m.bias_for(huge, 1000, 1), .3)


if __name__ == '__main__':
    unittest.main()
