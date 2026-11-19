import importlib.util
import unittest
import numpy as np


class SearchSelectionTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.research_search'))
        from analysis import research_search
        return research_search

    def test_combinations_use_final_scores_and_fixed_validation_rank_routes(self):
        m = self.module()
        arrays = {'current': np.array([10., 10., 10., 10.]), 'rank': np.array([0, 1, 0, 1]), 'progress': np.array([.1, .1, .95, .95]),
                  'cases': [{'case_id': str(i), 'event': e, 'rank': r} for i, e, r in [(0, 1, 1000), (1, 1, 2000), (2, 2, 1000), (3, 2, 2000)]]}
        predictions = {'a': np.array([100., 120., 100., 120.]),
                       'b': np.array([120., 100., 120., 100.]),
                       'c': np.array([130., 130., 130., 130.]), 'reference_paired': np.array([140.]*4)}
        labels = {str(i): 100. for i in range(4)}
        specs = m.combination_specs(arrays, labels, predictions, {'remaining': 'a', 'fusion': 'b', 'shift': 'c'})
        np.testing.assert_allclose(m.combine(specs['family_mean'], predictions, arrays), [350/3, 350/3, 350/3, 350/3])
        np.testing.assert_allclose(m.combine(specs['rank_route'], predictions, arrays), 100.)
        np.testing.assert_allclose(m.combine(specs['blend_50'], predictions, arrays), [120., 130., 120., 130.])
        # Routes are frozen validation choices, even if another batch reverses errors.
        changed = {'a': np.array([500., 1., 500., 1.]), 'b': np.array([1., 500., 1., 500.])}
        np.testing.assert_allclose(m.combine(specs['rank_route'], changed, arrays), 500.)

    def test_phase_routes_are_selected_on_validation_and_reused_without_labels(self):
        m = self.module()
        arrays = {'rank': np.array([0, 0, 0]), 'progress': np.array([.1, .5, .95]),
                  'cases': [{'case_id': str(i), 'event': 1} for i in range(3)]}
        predictions = {'a': np.array([100., 120., 140.]), 'b': np.array([120., 100., 120.]),
                       'c': np.array([140., 120., 100.]), 'reference_paired': np.array([150.]*3)}
        specs = m.combination_specs(arrays, {str(i): 100. for i in range(3)}, predictions,
                                    {'remaining': 'a', 'fusion': 'b', 'shift': 'c'})
        np.testing.assert_allclose(m.combine(specs['phase_route'], predictions, arrays), 100.)
        changed = dict(arrays, progress=np.array([.95, .5, .1]))
        np.testing.assert_allclose(m.combine(specs['phase_route'], predictions, changed), [140., 100., 140.])

    def test_metric_counts_activities_equally_with_unequal_prefix_counts(self):
        m = self.module()
        arrays = {'cases': [{'case_id': 'a', 'event': 1}, {'case_id': 'b', 'event': 1}, {'case_id': 'c', 'event': 2}]}
        self.assertAlmostEqual(m.event_mape(arrays, {'a': 100., 'b': 100., 'c': 100.}, np.array([120., 120., 100.])), 10.)

    def test_search_is_blind_and_freezes_selection_before_test_inputs(self):
        from unittest.mock import patch
        import pandas as pd
        import json
        import joblib
        import importlib
        import test_temporal_training as fixtures
        from analysis.temporal_rank_subset import build_subset
        from analysis.temporal_dataset import TemporalDataset
        from analysis.holdout_dataset import sha256, write_json
        from analysis.temporal_evaluate import verify_predictions, evaluate
        from analysis.temporal_paired_features import prepare_paired
        m = self.module()
        helper = fixtures.TemporalTrainingTests(); self.addCleanup(helper.doCleanups)
        root, parent = helper.fixture(); build_subset(parent, root/'subset')
        data = TemporalDataset(root/'subset'); reference = root/'reference'; reference.mkdir()
        for split in ['train', 'validation', 'test']:
            a = prepare_paired(data.inputs(split)); frames = []
            for name in m.REFERENCE_METHODS.values():
                p = a['current'] if name == 'hold_current' else a['current']/a['progress']
                frames.append(m.rows(a, name, p))
            pd.concat(frames).to_csv(reference/f'{split}_predictions.csv', index=False)
        write_json(reference/'blind_manifest.json', {'dataset_id': data.manifest['dataset_id'],
            'methods': list(m.REFERENCE_METHODS.values()),
            'artifacts': {p.name: sha256(p) for p in reference.iterdir()}})
        label_reader, input_reader = TemporalDataset.labels, TemporalDataset.inputs
        def labels(instance, split, **kwargs):
            if split == 'test':
                raise AssertionError('Test labels forbidden in search')
            return label_reader(instance, split, **kwargs)
        def inputs(instance, split):
            if split == 'test':
                self.assertTrue((root/'search/selection.json').exists())
            return input_reader(instance, split)
        with patch.object(TemporalDataset, 'labels', labels), patch.object(TemporalDataset, 'inputs', inputs), \
             patch('sqlite3.connect', side_effect=AssertionError('SQLite forbidden')):
            selection = m.run_experiment(data, reference, root/'search', limit_per_family=1, iterations_override=3)
        self.assertEqual(selection['selected_recipe'], min(selection['validation'], key=selection['validation'].get))
        seal = verify_predictions(root/'search')
        self.assertEqual(seal['reference_method'], selection['selected_recipe'])
        self.assertEqual(len(seal['methods']), 15)
        config = json.loads((root/'search/run_manifest.json').read_text())
        test = prepare_paired(data.inputs('test'))
        saved = pd.read_csv(root/'search/test_predictions.csv').pivot(index='case_id', columns='method', values='predicted')
        for name, spec in config['recipes'].items():
            module = importlib.import_module(m.FAMILIES[spec['family']])
            bundle = joblib.load(root/f'search/models/{name}.joblib')
            prediction = module.predict_recipe(bundle, test)
            np.testing.assert_allclose(prediction, saved.loc[[c['case_id'] for c in test['cases']], name])
        score = evaluate(root/'search')
        self.assertEqual(len(score['splits']['test']), 15)
        with self.assertRaises(FileExistsError):
            m.run_experiment(data, reference, root/'search')
        prediction_file = root/'search/test_predictions.csv'
        prediction_file.write_text(prediction_file.read_text()+'\n')
        with self.assertRaises(ValueError):
            verify_predictions(root/'search')
