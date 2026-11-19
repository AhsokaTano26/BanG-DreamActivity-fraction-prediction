"""Deterministic mathematical checks; the Monte Carlo grid is never run here."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class SyntheticMechanismMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module('analysis.paper_upgrade_synthetic')

    def test_equal_filter_coordinate_change_preserves_all_external_bounds(self):
        # Different horizons, first-observation initialization and binding caps.
        original = np.array([[.8, -.4], [-.5, .9], [.12, .04], [.7, -.6]])
        for n in (1, 2, 4):
            panel = original[:n]
            independent = self.module.equal_filter_bias(panel, alpha=.3, shrinkage=10.)
            decomposed = self.module.equal_filter_anchor_bias(panel, alpha=.3, shrinkage=10.)
            np.testing.assert_allclose(independent, decomposed, rtol=0., atol=1e-12)
            for scale in (1., 100.):
                first = np.clip(independent * scale, -.3, .3)
                second = np.clip(decomposed * scale, -.3, .3)
                base, current = np.array([1., 40.]), np.array([1., 1.])
                np.testing.assert_allclose(
                    np.clip(base * np.exp(-first), current, current * np.exp(4)),
                    np.clip(base * np.exp(-second), current, current * np.exp(4)),
                    rtol=0., atol=1e-12)

    def test_intermediate_component_clipping_breaks_the_equivalence(self):
        panel = np.array([[.2, .6]])
        direct = np.clip(self.module.equal_filter_bias(panel, alpha=.3, shrinkage=0.), -.3, .3)
        anchor = np.clip(.6, -.3, .3)
        contrast = np.clip(.2 - .6, -.3, .3)
        premature = np.clip(anchor + contrast, -.3, .3)
        self.assertGreater(abs(float(direct[0]) - premature), .1)

    def test_same_filter_requires_the_same_temporal_weights(self):
        panel = np.array([[.7, .4], [-.2, .5], [.3, -.4]])
        own = self.module.equal_filter_bias(panel, alpha=.3, shrinkage=10.)
        heterogeneous = self.module.anchor_gap_bias(panel)
        self.assertGreater(np.max(np.abs(own - heterogeneous)), 1e-3)

    def test_anchor_contrast_has_negative_noise_covariance(self):
        transform = self.module.ANCHOR_TRANSFORM
        for variance in (.0004, .0064):
            actual = transform @ (variance * np.eye(2)) @ transform.T
            expected = variance * np.array([[1., -1.], [-1., 2.]])
            np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-15)
            self.assertGreater(np.linalg.eigvalsh(actual).min(), 0.)

    def test_full_covariance_kalman_is_invariant_to_coordinates(self):
        observations = np.array([[.1, -.2], [.3, .4], [-.1, .6], [.4, .3]])
        process = np.array([[.003, .001], [.001, .002]])
        noise = np.array([[.02, .003], [.003, .01]])
        prior = np.array([[.1, .02], [.02, .08]])
        transform = self.module.ANCHOR_TRANSFORM
        original = self.module.kalman_level_bias(observations, process, noise, prior)
        changed = self.module.kalman_level_bias(
            observations @ transform.T,
            transform @ process @ transform.T,
            transform @ noise @ transform.T,
            transform @ prior @ transform.T)
        np.testing.assert_allclose(changed @ np.linalg.inv(transform).T,
                                   original, rtol=0., atol=1e-12)

    def test_kalman_uses_prior_process_noise_before_each_observation(self):
        result = self.module.kalman_level_bias(
            np.array([[1., 2.]]), .001 * np.eye(2), .01 * np.eye(2), .1 * np.eye(2))
        expected_gain = .101 / .111
        np.testing.assert_allclose(result, expected_gain * np.array([1., 2.]), rtol=0., atol=1e-12)

    def test_pooling_risk_threshold_matches_direct_bias_variance(self):
        for delta in (0., .1, .2):
            v1, v2, covariance = .01, .02, .003
            for weight in (0., .25, .5, 1.):
                direct = ((weight * delta) ** 2 + (1 - weight) ** 2 * v1
                          + weight ** 2 * v2 + 2 * weight * (1 - weight) * covariance - v1)
                formula = self.module.sharing_risk_difference(delta, v1, v2, covariance, weight)
                self.assertAlmostEqual(formula, direct, places=14)
        self.assertLess(self.module.sharing_risk_difference(.1, .01, .01, 0., .5), 0.)
        self.assertGreater(self.module.sharing_risk_difference(.2, .01, .01, 0., .5), 0.)
        self.assertEqual(self.module.sharing_risk_difference(0., .01, .01, .01, .5), 0.)

    def test_scalar_steady_gain_identity(self):
        for alpha in (.2, .3, .6):
            ratio = alpha ** 2 / (1 - alpha)
            gain = (np.sqrt(ratio ** 2 + 4 * ratio) - ratio) / 2
            self.assertAlmostEqual(gain, alpha, places=14)

    def test_state_and_predictive_ape_are_distinct_scores(self):
        state = np.zeros((2, 2))
        transient = np.array([[.1, -.1], [.2, -.2]])
        scores = self.module.scores_for_bias(state, transient, np.zeros((2, 2)), clip=.3)
        np.testing.assert_array_equal(scores['state_mse'], np.zeros(2))
        self.assertTrue(np.all(scores['synthetic_unbounded_ape_pct'] > 0.))

    def test_plan_seal_rejects_changed_configuration(self):
        plan = {'status': 'sealed', 'grid': {'q_common': [0.]}, 'source_seal': None}
        digest = self.module.plan_digest(plan)
        plan['source_seal'] = {'plan_sha256': digest}
        self.assertEqual(self.module.plan_digest(plan), digest)
        plan['grid']['q_common'] = [1.]
        self.assertNotEqual(self.module.plan_digest(plan), digest)

    def test_known_zero_state_oracle_ignores_transitory_historical_errors(self):
        plan = json.loads((Path(__file__).resolve().parents[1] /
                           'analysis/paper_upgrade_records/20261118/reports/B_synthetic_plan.json').read_text())
        history = np.array([[[.1, -.2], [.3, -.4], [.2, -.1]],
                            [[-.3, .2], [.1, .5], [.2, -.2]]])
        cell = {'q_common': 0., 'q_private': 0., 'observation_variance': .0004,
                'mismatch_loading': .5}
        result = self.module._method_biases(history, cell, plan['fixed_parameters'])
        np.testing.assert_array_equal(result['dlm_oracle'], np.zeros((2, 2)))
        self.assertGreater(np.max(np.abs(result['independent_ewma'])), 0.)

    def test_unsealed_run_stops_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / 'plan.json'
            plan.write_text(json.dumps({'source_seal': None}))
            output = Path(directory) / 'must_not_exist'
            with self.assertRaisesRegex(ValueError, 'unsealed'):
                self.module.run(plan, output)
            self.assertFalse(output.exists())

    def test_semantic_plan_digest_ignores_json_key_order(self):
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / 'analysis/paper_upgrade_records/20261118/reports/B_synthetic_plan.json').read_text())
        plan['grid'] = dict(reversed(list(plan['grid'].items())))
        plan['source_seal'] = {'plan_sha256': self.module.plan_digest(plan), 'source_sha256': {}}
        self.module.verify_plan(plan, root)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'existing'
            output.mkdir()
            sentinel = output / 'sentinel'
            sentinel.write_text('preserve')
            with self.assertRaises(FileExistsError):
                self.module.prepare_output(output)
            self.assertEqual(sentinel.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
