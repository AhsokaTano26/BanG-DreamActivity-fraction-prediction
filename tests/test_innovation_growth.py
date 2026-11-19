"""Synthetic contracts for the terminal-only positive-growth decoder ablation."""
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import numpy as np
import pandas as pd


def growth_module(test):
    test.assertIsNotNone(importlib.util.find_spec("analysis.innovation_growth"),
                         "the growth family has not been implemented")
    return importlib.import_module("analysis.innovation_growth")


class GrowthAnalyticTests(unittest.TestCase):
    def test_integrals_are_in_hours_and_separate_absolute_end_windows(self):
        growth = growth_module(self)
        values = growth.basis_integrals(
            np.array([.5, .5, .75, 1.]), np.array([48., 24., 24., 0.]),
            np.array([96., 48., 96., 96.]), dual_clock=True)
        np.testing.assert_allclose(values, [
            [4., 16., 28., 12., 3.], [2., 8., 14., 12., 3.],
            [.5, 5., 18.5, 12., 3.], [0., 0., 0., 0., 0.]], atol=1e-12)

    def test_positive_coefficients_give_nondecreasing_future_path(self):
        growth = growth_module(self)
        horizons = np.array([0., 1., 6., 12., 24., 48.])
        integrals = growth.basis_integrals(
            np.full(6, .5), np.full(6, 48.), np.full(6, 96.),
            dual_clock=True, horizon_h=horizons)
        increments = integrals @ np.array([2., .5, 3., 4., 5.])
        self.assertEqual(increments[0], 0.)
        self.assertTrue(np.all(np.diff(increments) >= 0.))
        np.testing.assert_allclose(integrals[:, :3].sum(axis=1), horizons)
        np.testing.assert_array_equal(integrals[:4, 3:], np.zeros((4, 2)))

    def test_event_weights_do_not_treat_prefixes_as_independent_events(self):
        growth = growth_module(self)
        weights = growth.event_weights([11, 11, 11, 22, 33, 33])
        np.testing.assert_allclose([weights[:3].sum(), weights[3], weights[4:].sum()],
                                   [2., 2., 2.])

    def test_train_only_preprocessor_preserves_missing_columns_and_unknowns(self):
        growth = growth_module(self)
        training = {"base": pd.DataFrame({"x": [1., 3., np.nan],
                                          "empty": [None, None, None],
                                          "kind": ["known", "known", "other"]}),
                    "rank": np.array([0, 1, 0])}
        prep = growth.fit_preprocessor(training)
        before = prep.transform(training).copy()
        query = {"base": pd.DataFrame({"x": [1e6, None], "kind": ["unseen", None],
                                       "extra_untrained": [99., 99.]}),
                 "rank": np.array([1, 0])}
        transformed = prep.transform(query)
        self.assertEqual(transformed.shape[1], before.shape[1])
        self.assertTrue(np.isfinite(transformed).all())
        # Missing numeric columns use the saved all-missing training fallback.
        self.assertTrue(np.all(transformed[:, prep.feature_names.index("empty")] == 0.))
        categorical = [i for i, name in enumerate(prep.feature_names) if name.startswith("kind=")]
        np.testing.assert_array_equal(transformed[:, categorical], np.zeros((2, 2)))
        np.testing.assert_array_equal(prep.transform(training), before)

    def test_never_observed_numeric_columns_stay_zero_when_query_supplies_values(self):
        import io
        import joblib

        growth = growth_module(self)
        training = {"base": pd.DataFrame({"seen": [1., 3.],
                                          "empty": [None, None],
                                          "nonfinite": [np.inf, -np.inf]}),
                    "rank": np.array([0, 1])}
        prep = growth.fit_preprocessor(training)
        query = {"base": pd.DataFrame({"seen": [1., 3.],
                                       "empty": [100., -100.],
                                       "nonfinite": [10., -10.]}),
                 "rank": np.array([0, 1])}
        expected = np.array([[-1., 0., 0., -1.], [1., 0., 0., 1.]], dtype=np.float32)
        np.testing.assert_array_equal(prep.transform(query), expected)
        serialized = io.BytesIO()
        joblib.dump(prep, serialized)
        serialized.seek(0)
        restored = joblib.load(serialized)
        np.testing.assert_array_equal(restored.transform(query), expected)


class GrowthSubprocessTests(unittest.TestCase):
    def run_script(self, script):
        environment = os.environ.copy()
        environment["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib"
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                                cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_module_import_does_not_load_torch(self):
        growth_module(self)
        self.run_script("""
            import sys
            import analysis.innovation_growth
            assert 'torch' not in sys.modules
        """)

    def test_three_seed_fit_final_mean_zero_horizon_and_saved_replay(self):
        growth_module(self)
        self.run_script("""
            from pathlib import Path
            import tempfile
            import joblib
            import numpy as np
            import pandas as pd
            from analysis import innovation_growth as growth

            progress = np.array([.25, .5, .75, .25, .5, .75], dtype=float)
            current = np.array([100., 220., 360., 150., 330., 540.])
            arrays = {
                'base': pd.DataFrame({'log_score': np.log(current),
                    **{f'prefix_{i:02}': np.full(6, i / 32.) for i in range(33)},
                    'kind': ['a', 'a', 'a', 'b', 'b', 'b'], 'empty': [None] * 6}),
                'cases': [{'case_id': f'case-{i}', 'event': [1, 1, 1, 2, 3, 3][i]}
                          for i in range(6)],
                'current': current, 'progress': progress,
                'remaining_h': 96. * (1. - progress), 'duration_h': np.full(6, 96.),
                'rank': np.array([0, 0, 0, 1, 1, 1]),
                'history': object(), 'event_type': object(), 'band_shares': object(),
            }
            labels = {f'case-{i}': [520., 520., 520., 800., 800., 800.][i]
                      for i in range(6)}
            for config in growth.recipes():
                bundle = growth.fit_recipe(dict(config, epochs=2), arrays, labels)
                assert bundle['parameters'] <= 20_000
                assert bundle['training_event_count'] == 3
                assert set(bundle['members']) == {'seed_42', 'seed_123', 'seed_2026'}
                assert all(len(item['trace']) == 2 for item in bundle['members'].values())
                members = growth.predict_members(bundle, arrays)
                prediction = growth.predict_recipe(bundle, arrays)
                assert np.isfinite(prediction).all()
                assert (prediction >= current).all()
                np.testing.assert_array_equal(prediction, np.mean(list(members.values()), axis=0))
                assert not np.array_equal(members['seed_42'], members['seed_123'])
                frozen = joblib.hash(bundle)
                unseen = dict(arrays, base=arrays['base'].assign(kind='unseen'))
                assert np.isfinite(growth.predict_recipe(bundle, unseen)).all()
                assert joblib.hash(bundle) == frozen
                ended = dict(arrays, progress=np.ones(6), remaining_h=np.zeros(6))
                np.testing.assert_array_equal(growth.predict_recipe(bundle, ended), current)
                fractional_end = dict(ended, current=np.array([.7, .1, 1.7, .2, .4, .8]))
                np.testing.assert_array_equal(growth.predict_recipe(bundle, fractional_end),
                                              fractional_end['current'])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'bundle.joblib'
                    joblib.dump(bundle, path)
                    restored = joblib.load(path)
                    np.testing.assert_array_equal(growth.predict_recipe(restored, arrays), prediction)
                if config['head'] == 'dual_clock':
                    repeated = growth.fit_recipe(dict(config, epochs=2), arrays, labels)
                    np.testing.assert_array_equal(growth.predict_recipe(repeated, arrays), prediction)
        """)


if __name__ == "__main__":
    unittest.main()
