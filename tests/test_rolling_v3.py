"""Causal release, frozen predictions, and independent V3 scoring regressions."""
import gzip
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from analysis.backtest import Curve, snapshot
from analysis.data_policy import MIN_EVENT_START_MS
from analysis.holdout_dataset import curve_record, prefix_curve, sha256, write_json
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import TemporalDataset, STAGES, origin_groups

H = 3600000


def fixture(root):
    """Eight nonmonotone IDs, 4/2/2 chronological events and real blind inputs."""
    parent = root/'parent'; parent.mkdir(); (parent/'metadata').mkdir()
    groups = {'train': [900, 2, 800, 3], 'validation': [700, 4], 'test': [600, 5]}
    curves, partitions = [], []
    events = sum(groups.values(), [])
    for i, event in enumerate(events):
        start = MIN_EVENT_START_MS + i*72*H; end = start + 48*H
        times = np.arange(start, end+1, H//4, dtype=np.int64)
        for rank in (1000, 2000):
            scores = (1+np.arange(len(times), dtype=float)**1.15)*(1+i/20)*1000/rank
            c = Curve(event, rank, 'normal', start, end, times.copy(), scores)
            curves.append(c)
            split = next(s for s, ids in groups.items() if event in ids)
            partitions.append({'event': event, 'rank': rank, 'split': split,
                'start_ms': start, 'end_ms': end, 'observations': len(times)})
    for name, content in {'events': {str(e): {'characters': [{'characterId': 1}]} for e in events},
                          'characters': {'1': {'bandId': 1}}, 'bands': {'1': {'bandName': ['A']*5}}, 'cards': {}}.items():
        write_json(parent/'metadata'/f'{name}.json', content)
    pd.DataFrame(partitions).to_csv(parent/'partitions.csv', index=False)
    splits = {}
    for split, ids in groups.items():
        cases, memberships, labels, records = [], [], [], []
        for c in curves:
            if c.event not in ids: continue
            for group in origin_groups(c):
                cutoff = group['cutoff']
                if not c.start < cutoff < c.end: continue
                sample = snapshot(c, cutoff)
                if sample is None: continue
                case_id = f'{split}:{c.event}:{c.rank}:{cutoff}'
                records.append(curve_record(prefix_curve(c, cutoff)) | {'case_id': case_id,
                    'stage': group['stages'][0]['stage_id'], 'cutoff': cutoff})
                cases.append({'case_id': case_id, 'event': c.event, 'rank': c.rank,
                    'start_ms': c.start, 'end_ms': c.end, 'cutoff': cutoff,
                    'current': sample['score'], 'observation_age_minutes': (cutoff-sample['now'])/60000,
                    'event_type': 'normal', 'band_group': '1', 'calendar_quarter': '2024Q1'})
                memberships.extend({'case_id': case_id, **stage} for stage in group['stages'])
                labels.append({'case_id': case_id, 'actual': c.final})
        for suffix, rows in [('cases', cases), ('memberships', memberships), ('labels', labels)]:
            pd.DataFrame(rows).to_csv(parent/f'{split}_{suffix}.csv', index=False)
        with gzip.open(parent/f'{split}_inputs.jsonl.gz', 'wt') as f:
            for record in records: f.write(json.dumps(record)+'\n')
        splits[split] = {'event_ids': ids, 'cases': len(cases), 'stage_memberships': len(memberships)}
    with gzip.open(parent/'train_curves.jsonl.gz', 'wt') as f:
        for c in curves:
            if c.event in groups['train']: f.write(json.dumps(curve_record(c))+'\n')
    manifest = {'protocol': {'ranks': [1000, 2000], 'stages': STAGES,
        'exposure': 'Previously explored test cohort.', 'metadata_caveat': 'Retrospective metadata.'},
        'splits': splits, 'artifacts': {str(p.relative_to(parent)): sha256(p) for p in parent.rglob('*') if p.is_file()}}
    manifest['dataset_id'] = identity(manifest); write_json(parent/'manifest.json', manifest)
    return TemporalDataset(parent), curves


class RollingV3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def module(self, name):
        self.assertIsNotNone(importlib.util.find_spec('analysis.'+name), name+' missing')
        return __import__('analysis.'+name, fromlist=[name])

    def dataset(self):
        m = self.module('rolling_dataset'); parent, curves = fixture(self.root)
        m.build_rolling(parent, self.root/'rolling', curves, source_identity={'fixture': True})
        return m.RollingDataset(self.root/'rolling'), parent, curves

    def test_release_is_strictly_by_dates_and_returns_detached_curves(self):
        data, parent, curves = self.dataset(); first = curves[0]
        self.assertEqual(data.history_before(first.end), [])
        released = data.history_before(first.end+1)
        self.assertEqual({c.event for c in released}, {900})
        released[0].scores[-1] = 1e100
        self.assertEqual(data.reveal_event(900, before_ms=first.end+1)[0].final, first.final)
        for event, boundary in [(900, first.end), (2, first.start), (5, first.end+1)]:
            with self.assertRaisesRegex(ValueError, 'completed|boundary|release'):
                data.reveal_event(event, before_ms=boundary)
        for split in ('train', 'validation', 'test'):
            self.assertEqual(sha256(data.root/f'{split}_inputs.jsonl.gz'), sha256(parent.root/f'{split}_inputs.jsonl.gz'))
            pd.testing.assert_frame_equal(data.case_index(split), parent.case_index(split))
        with self.assertRaises(FileExistsError):
            self.module('rolling_dataset').build_rolling(parent, data.root, curves, source_identity={})

    def test_builder_rejects_lost_duplicate_old_and_mistimed_curves_atomically(self):
        m = self.module('rolling_dataset'); parent, curves = fixture(self.root)
        variants = [curves[:-1], curves+[curves[0]],
                    [replace(curves[0], start=MIN_EVENT_START_MS-1)]+curves[1:],
                    [curves[0], replace(curves[1], end=curves[1].end+1)]+curves[2:]]
        for i, variant in enumerate(variants):
            dest = self.root/f'bad{i}'
            with self.assertRaises(ValueError): m.build_rolling(parent, dest, variant, source_identity={})
            self.assertFalse(dest.exists())

    def test_builder_rejects_case_identity_disagreeing_with_its_blind_prefix(self):
        m = self.module('rolling_dataset'); parent, curves = fixture(self.root)
        path = parent.root/'train_cases.csv'; cases = pd.read_csv(path)
        cases.loc[0, 'event'] = 2  # Another valid train event cannot relabel this prefix.
        cases.to_csv(path, index=False)
        manifest = parent.manifest.copy(); manifest['artifacts'][path.name] = sha256(path)
        manifest['dataset_id'] = identity({k: v for k, v in manifest.items() if k != 'dataset_id'})
        write_json(parent.root/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'case|prefix'):
            m.build_rolling(TemporalDataset(parent.root), self.root/'invalid', curves, source_identity={})
        self.assertFalse((self.root/'invalid').exists())

    def test_online_bias_counts_events_not_snapshots_and_shrinks_log_pred_over_actual(self):
        m = self.module('rolling_benchmark'); state = m.OnlineBias()
        rows = pd.DataFrame({'case_id': ['a', 'b', 'c'], 'event': [1]*3,
            'rank': [1000]*3, 'phase': [0]*3, 'predicted': [200., 200., 200.]})
        state.observe(rows, {'a': 100., 'b': 100., 'c': 100.}, event=1, end_ms=100)
        self.assertAlmostEqual(state.bias(1000, 0), np.log(2)/11)
        rows2 = pd.DataFrame({'case_id': ['d'], 'event': [2], 'rank': [1000], 'phase': [0], 'predicted': [50.]})
        state.observe(rows2, {'d': 100.}, event=2, end_ms=200)
        self.assertAlmostEqual(state.bias(1000, 0), .4*np.log(2)*2/12)
        with self.assertRaises(ValueError): state.observe(rows, {'a': 100., 'b': 100., 'c': 100.}, event=1, end_ms=100)

    def test_real_runner_seals_before_release_and_preserves_static_history_and_event_weights(self):
        m = self.module('rolling_benchmark'); data, _, _ = self.dataset(); run = self.root/'run'
        original = type(data).reveal_event
        def guarded(instance, event, *, before_ms):
            row = instance.event_index[int(event)]
            self.assertLess(row['end_ms'], before_ms)
            if row['split'] != 'train':
                self.assertTrue((run/'events'/str(event)/'event_manifest.json').exists(), 'Outcome opened before event prediction seal')
            return original(instance, event, before_ms=before_ms)
        with patch.object(type(data), 'reveal_event', guarded), \
             patch.object(type(data), 'labels', side_effect=AssertionError('No answer file in runner')), \
             patch('sqlite3.connect', side_effect=AssertionError('No database in runner')):
            selected = m.run_experiment(data, run, iterations=3, allow_fixture=True)
        self.assertEqual(set(selected['validation_scores']), set(m.METHODS))
        for event in [700, 4, 600, 5]:
            audit = json.loads((run/'events'/str(event)/'audit.json').read_text())
            self.assertEqual(audit['static_history_events'], [900, 2, 800, 3])
            self.assertTrue(all(r['end_ms'] < audit['query_start_ms'] for r in audit['released_history']))
            self.assertNotIn(event, audit['rolling_training_events'])
            weights = pd.read_csv(run/'events'/str(event)/'training_weights.csv').groupby('event').weight.sum()
            np.testing.assert_allclose(weights, weights.iloc[0])
        test_audit = json.loads((run/'events/5/audit.json').read_text())
        self.assertIn(600, test_audit['rolling_training_events'])
        self.assertIn(4, test_audit['rolling_training_events'])
        with patch.object(type(data), 'reveal_event', side_effect=AssertionError('Resume must not read outcomes')):
            self.assertEqual(m.run_experiment(data, run, iterations=3, allow_fixture=True, resume=True), selected)
        evaluator = self.module('rolling_evaluate')
        occupied = self.root/'occupied'; occupied.mkdir(); (occupied/'stage_metrics.csv').write_text('preserve')
        with self.assertRaises(FileExistsError): evaluator.evaluate(run, occupied)
        self.assertEqual((occupied/'stage_metrics.csv').read_text(), 'preserve')
        result = evaluator.evaluate(run)
        for method, score in selected['validation_scores'].items():
            self.assertAlmostEqual(result['splits']['validation'][method]['overall']['event_macro_mape_pct'], score)
        self.assertEqual(result['splits']['test']['static_context']['overall']['events'], 2)
        self.assertIn('by_event_type', result['splits']['test']['rolling_context'])
        self.assertTrue(result['paired_bootstrap'])
        with patch.object(type(data), 'labels', side_effect=AssertionError('Cached score must not read answers')):
            self.assertEqual(evaluator.evaluate(run), result)
        with patch('analysis.rolling_evaluate.platform.python_version', return_value='different-runtime'):
            with self.assertRaisesRegex(FileExistsError, 'evaluator'):
                evaluator.evaluate(run)
            audit = evaluator.evaluate(run, self.root/'audit')
            self.assertNotEqual(audit['evaluation_id'], result['evaluation_id'])
        with self.assertRaises(ValueError): m.run_experiment(data, run, iterations=4, allow_fixture=True, resume=True)
        sealed = run/'events/700/predictions.csv'; sealed.write_text(sealed.read_text()+'\n')
        with self.assertRaises(ValueError): evaluator.evaluate(run)


    def test_poisoned_current_and_future_finals_cannot_change_unreleased_predictions(self):
        m = self.module('rolling_benchmark'); d = self.module('rolling_dataset')
        data, _, _ = self.dataset(); m.run_experiment(data, self.root/'clean', iterations=2, allow_fixture=True)
        for poisoned_event in [700, 5]:
            subroot = self.root/f'poison{poisoned_event}'; subroot.mkdir()
            parent, curves = fixture(subroot)
            changed = []
            for curve in curves:
                scores = curve.scores.copy()
                if curve.event == poisoned_event: scores[-1] *= 10
                changed.append(replace(curve, scores=scores))
            split = 'validation' if poisoned_event == 700 else 'test'
            path = parent.root/f'{split}_labels.csv'; labels = pd.read_csv(path)
            cases = parent.case_index(split); poisoned_ids = set(cases.loc[cases.event == poisoned_event, 'case_id'])
            labels.loc[labels.case_id.isin(poisoned_ids), 'actual'] *= 10; labels.to_csv(path, index=False)
            manifest = json.loads((parent.root/'manifest.json').read_text())
            manifest['artifacts'][path.name] = sha256(path)
            manifest['dataset_id'] = identity({k: v for k, v in manifest.items() if k != 'dataset_id'})
            write_json(parent.root/'manifest.json', manifest)
            parent = TemporalDataset(parent.root)
            d.build_rolling(parent, subroot/'rolling', changed, source_identity={'poisoned_event': poisoned_event})
            m.run_experiment(d.RollingDataset(subroot/'rolling'), subroot/'run', iterations=2, allow_fixture=True)
            events_to_check = [700] if poisoned_event == 700 else [700, 4, 600, 5]
            for event in events_to_check:
                pd.testing.assert_frame_equal(pd.read_csv(self.root/'clean/events'/str(event)/'predictions.csv'),
                    pd.read_csv(subroot/'run/events'/str(event)/'predictions.csv'))
            if poisoned_event == 700:
                clean = pd.read_csv(self.root/'clean/events/4/predictions.csv')
                modified = pd.read_csv(subroot/'run/events/4/predictions.csv')
                self.assertGreater(np.max(np.abs(clean.predicted-modified.predicted)), 1.,
                    'The ended event outcome should influence a later event, after its seal')

    def test_failure_keeps_event_seal_and_incomplete_resume_fails_closed(self):
        m = self.module('rolling_benchmark'); data, _, _ = self.dataset(); run = self.root/'partial'
        actual_fit = m.fit_context; calls = 0
        def fail_third(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3: raise RuntimeError('simulated interruption')
            return actual_fit(*args, **kwargs)
        with patch.object(m, 'fit_context', side_effect=fail_third):
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                m.run_experiment(data, run, iterations=2, allow_fixture=True)
        first = run/'events/700/event_manifest.json'; old = first.read_bytes()
        with self.assertRaisesRegex(FileExistsError, 'incomplete|new run'):
            m.run_experiment(data, run, iterations=2, allow_fixture=True, resume=True)
        self.assertEqual(first.read_bytes(), old)


if __name__ == '__main__': unittest.main()
