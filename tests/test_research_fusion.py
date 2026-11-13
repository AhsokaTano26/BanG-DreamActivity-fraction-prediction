"""Contracts preventing row loss, peer dependence, and train leakage in fusion."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd


def fixture():
    events = [1] * 6 + [2] * 4 + [3] * 2
    progress = np.tile([.12, .45, .94], 4)
    ranks = np.arange(12) % 2
    current = (1 + np.arange(12)) * 100.
    single = pd.DataFrame({'progress': progress, 'rate': current / 3,
                           'event_type': ['normal', 'challenge'] * 6,
                           'log_rank': np.log(np.where(ranks == 0, 1000, 2000)),
                           'empty_train_column': [None] * 12})
    paired = single.assign(peer_available=1, peer_age_hours=0.,
                           peer__score=current / 2,
                           pair_log_score_ratio=np.log(2), peer__present_00=1)
    cases = [{'case_id': f'train:{e}:{r}:{i}', 'event': e, 'rank': r}
             for i, (e, r) in enumerate(zip(events, ranks))]
    arrays = {'single': single, 'paired': paired, 'current': current,
              'progress': progress, 'rank': ranks, 'cases': cases,
              'audit': pd.DataFrame()}
    labels = {c['case_id']: x / p * (1 + .1 * np.sin(i))
              for i, (c, x, p) in enumerate(zip(cases, current, progress))}
    return arrays, labels


class ResearchFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('analysis.research_fusion') is None:
            raise AssertionError('Train-only robust fusion module is required')
        from analysis import research_fusion
        cls.module = research_fusion

    def config(self, name):
        config = next(c for c in self.module.recipes() if c['name'] == name)
        config.update(iterations=7, n_estimators=9, min_samples_leaf=1)
        return config

    def test_saved_bundle_replays_and_predictions_follow_input_order(self):
        arrays, labels = fixture()
        for name in ['fusion_cat_rmse_progress', 'fusion_cat_rank_progress',
                     'fusion_extra_progress']:
            with self.subTest(name=name):
                bundle = self.module.fit_recipe(self.config(name), arrays, labels)
                before = self.module.predict_recipe(bundle, arrays)
                self.assertEqual(before.shape, (12,))
                self.assertTrue(np.isfinite(before).all())
                self.assertTrue((before >= arrays['current']).all())
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'model.joblib'
                    joblib.dump(bundle, path)
                    loaded = joblib.load(path)
                    np.testing.assert_array_equal(before, self.module.predict_recipe(loaded, arrays))
                order = np.array([8, 0, 11, 3, 2])
                subset = {k: (v.iloc[order].reset_index(drop=True) if isinstance(v, pd.DataFrame) and len(v) == 12
                              else [v[i] for i in order] if k == 'cases'
                              else v[order] if isinstance(v, np.ndarray) else v)
                          for k, v in arrays.items()}
                np.testing.assert_array_equal(before[order], self.module.predict_recipe(bundle, subset))

    def test_absent_peer_uses_single_branch_even_with_poisoned_peer_columns(self):
        arrays, labels = fixture()
        bundle = self.module.fit_recipe(self.config('fusion_cat_rmse_progress'), arrays, labels)
        missing = deepcopy(arrays)
        missing['paired']['peer_available'] = 0
        missing['paired']['peer_age_hours'] = np.nan
        first = self.module.predict_recipe(bundle, missing)
        poisoned = deepcopy(missing)
        poisoned['paired']['peer__score'] = 1e30
        poisoned['paired']['pair_log_score_ratio'] = -1e30
        poisoned['paired']['peer__present_00'] = 1
        np.testing.assert_array_equal(first, self.module.predict_recipe(bundle, poisoned))
        single_only = deepcopy(bundle)
        single_only['config']['gate'] = 'constant'
        single_only['config']['paired_weight'] = 0.
        np.testing.assert_array_equal(first, self.module.predict_recipe(single_only, arrays))

    def test_global_event_weights_survive_rank_heads_and_dropout(self):
        arrays, labels = fixture()
        bundle = self.module.fit_recipe(self.config('fusion_cat_rank_progress'), arrays, labels)
        weights = np.asarray(bundle['training_weights'])
        np.testing.assert_allclose(weights[:6], np.repeat(2 / 3, 6))
        np.testing.assert_allclose(weights[6:10], np.repeat(1., 4))
        np.testing.assert_allclose(weights[10:], np.repeat(2., 2))
        for head in bundle['heads'].values():
            for branch in ['single', 'paired']:
                self.assertAlmostEqual(head[branch]['training_weight_sum'],
                                       weights[head['training_indices']].sum())
                self.assertAlmostEqual(head[branch]['effective_weight_sum'],
                                       weights[head['training_indices']].sum())

    def test_fit_rejects_nontraining_cases_and_nonfinite_labels(self):
        arrays, labels = fixture()
        arrays['cases'][0]['case_id'] = 'validation:1:1000:0'
        with self.assertRaises(ValueError):
            self.module.fit_recipe(self.config('fusion_cat_rmse_progress'), arrays, labels)
        arrays, labels = fixture()
        labels[arrays['cases'][0]['case_id']] = float('nan')
        with self.assertRaises(ValueError):
            self.module.fit_recipe(self.config('fusion_cat_rmse_progress'), arrays, labels)

    def test_wide_paired_features_fit_without_fragmentation_warnings(self):
        arrays, labels = fixture()
        extra = pd.DataFrame({f'peer__wide_{i}': np.arange(12, dtype=float) for i in range(130)})
        arrays['paired'] = pd.concat([arrays['paired'], extra], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('error', pd.errors.PerformanceWarning)
            bundle = self.module.fit_recipe(self.config('fusion_cat_rmse_progress'), arrays, labels)
            self.assertTrue(np.isfinite(self.module.predict_recipe(bundle, arrays)).all())


if __name__ == '__main__':
    unittest.main()
