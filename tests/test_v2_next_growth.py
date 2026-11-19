"""Synthetic safety and training contracts for V2 multi-horizon supervision."""
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

from analysis.backtest import Curve


HOUR = 3_600_000


def growth_module(test):
    test.assertIsNotNone(importlib.util.find_spec("analysis.v2_next_growth"),
                         "V2 multi-horizon growth has not been implemented")
    return importlib.import_module("analysis.v2_next_growth")


def fixture(events=(1, 1, 2), hours=(12., 36., 12.), *, duration=72.):
    """Hourly linear TRAIN curves, with deliberately later scheduled cutoffs."""
    history = [Curve(event, 1000, "normal", 0, int(duration * HOUR),
                     np.arange(int(duration) + 1) * HOUR,
                     100. + 10. * np.arange(int(duration) + 1))
               for event in sorted(set(events))]
    cases, samples = [], []
    for i, (event, hour) in enumerate(zip(events, hours)):
        curve = next(c for c in history if c.event == event)
        keep = curve.times <= hour * HOUR
        prefix = Curve(event, 1000, curve.kind, curve.start, curve.end,
                       curve.times[keep].copy(), curve.scores[keep].copy())
        cases.append({"case_id": f"case-{i}", "event": event, "rank": 1000,
                      "start_ms": curve.start, "end_ms": curve.end})
        samples.append({"case_id": f"case-{i}", "curve": prefix,
                        "now": int(hour * HOUR), "cutoff": int((hour + .5) * HOUR),
                        "score": float(prefix.scores[-1])})
    current = 100. + 10. * np.asarray(hours)
    arrays = {"cases": cases, "samples": samples, "history": history,
              "base": pd.DataFrame({"log_current": np.log(current),
                                     "progress": np.asarray(hours) / duration}),
              "current": current, "progress": np.asarray(hours) / duration,
              "remaining_h": duration - np.asarray(hours),
              "duration_h": np.full(len(events), duration),
              "rank": np.zeros(len(events), dtype=int)}
    return arrays


class AuxiliaryTargetTests(unittest.TestCase):
    def test_targets_start_at_observed_time_and_never_at_later_cutoff(self):
        growth = growth_module(self)
        aux = growth.build_auxiliary_targets(fixture(events=(1,), hours=(12.,)))
        np.testing.assert_array_equal(aux["available"], [[True, True, True]])
        np.testing.assert_allclose(aux["horizon_h"], [[6., 24., 30.]])
        np.testing.assert_allclose(aux["increment"], [[60., 240., 300.]])
        np.testing.assert_allclose(aux["target_score"], [[280., 460., 520.]])

    def test_terminal_boundary_and_duplicate_horizons_are_not_auxiliary_samples(self):
        growth = growth_module(self)
        aux = growth.build_auxiliary_targets(fixture(hours=(66., 60., 48.)))
        np.testing.assert_array_equal(aux["available"], [
            [False, False, True], [True, False, False], [True, False, True]])
        self.assertEqual(aux["reasons"][0, 0], "not_before_end")
        self.assertEqual(aux["reasons"][1, 2], "duplicate_horizon")
        np.testing.assert_allclose(aux["increment"][aux["available"]], [30., 60., 60., 120.])

    def test_interpolation_accepts_local_brackets_but_rejects_large_gaps(self):
        growth = growth_module(self)
        arrays = fixture(events=(1,), hours=(12.,))
        curve = arrays["history"][0]
        keep = curve.times != 18 * HOUR
        curve.times, curve.scores = curve.times[keep], curve.scores[keep]
        aux = growth.build_auxiliary_targets(arrays)
        self.assertTrue(aux["available"][0, 0])
        self.assertEqual(aux["reasons"][0, 0], "interpolated")
        self.assertEqual(aux["increment"][0, 0], 60.)
        keep = (curve.times < 17 * HOUR) | (curve.times > 19 * HOUR)
        curve.times, curve.scores = curve.times[keep], curve.scores[keep]
        aux = growth.build_auxiliary_targets(arrays)
        self.assertFalse(aux["available"][0, 0])
        self.assertEqual(aux["reasons"][0, 0], "interpolation_gap")
        self.assertTrue(np.isnan(aux["target_score"][0, 0]))

    def test_auxiliary_mass_is_equal_per_event_despite_unequal_prefix_coverage(self):
        growth = growth_module(self)
        arrays = fixture(events=(1, 1, 1, 2, 3), hours=(12., 36., 66., 66., 12.))
        arrays["history"] = [c for c in arrays["history"] if c.event != 3]
        aux = growth.build_auxiliary_targets(arrays)
        np.testing.assert_allclose([aux["weights"][:3].sum(), aux["weights"][3].sum(),
                                    aux["weights"][4].sum()], [5 / 3, 5 / 3, 0.])
        self.assertEqual(aux["available"].shape, (5, 3))
        self.assertTrue((aux["weights"][~aux["available"]] == 0.).all())
        # Missing auxiliary answers retain their original terminal training mass.
        terminal = growth.event_weights([1, 1, 1, 2, 3])
        np.testing.assert_allclose([terminal[:3].sum(), terminal[3], terminal[4]],
                                   [5 / 3, 5 / 3, 5 / 3])
        for strength in (.25, 1.):
            joint = growth.build_auxiliary_targets(arrays, auxiliary_lambda=strength)
            terminal_mass = np.array([joint["terminal_weights"][:3].sum(),
                                      joint["terminal_weights"][3], joint["terminal_weights"][4]])
            auxiliary_mass = np.array([joint["weights"][:3].sum(),
                                       joint["weights"][3].sum(), joint["weights"][4].sum()])
            np.testing.assert_allclose(terminal_mass + strength * auxiliary_mass, [5 / 3] * 3)
            self.assertAlmostEqual(terminal_mass[2], 5 / 3)
            self.assertTrue((joint["terminal_weights"] > 0.).all())
            np.testing.assert_allclose(terminal_mass[:2], [5 / 3 / (1 + strength)] * 2)

    def test_other_activity_rank_and_future_history_cannot_supply_targets(self):
        growth = growth_module(self)
        arrays = fixture(events=(1,), hours=(12.,))
        expected = growth.build_auxiliary_targets(arrays)
        for event, rank, start in [(88, 1000, 0), (1, 2000, 0), (99, 1000, 100 * HOUR)]:
            arrays["history"].append(Curve(event, rank, "normal", start, start + 72 * HOUR,
                                             start + np.arange(73) * HOUR,
                                             np.full(73, 1e12)))
        actual = growth.build_auxiliary_targets(arrays)
        np.testing.assert_array_equal(actual["target_score"], expected["target_score"])
        arrays["history"] = arrays["history"][1:]
        actual = growth.build_auxiliary_targets(arrays)
        self.assertFalse(actual["available"].any())
        self.assertTrue((actual["reasons"] == "missing_training_curve").all())

    def test_mismatched_anchor_and_future_prefix_are_rejected(self):
        growth = growth_module(self)
        arrays = fixture(events=(1,), hours=(12.,))
        arrays["history"][0].scores[12] += 1.
        with self.assertRaisesRegex(ValueError, "anchor"):
            growth.build_auxiliary_targets(arrays)
        arrays = fixture(events=(1,), hours=(12.,))
        arrays["samples"][0]["cutoff"] = 11 * HOUR
        with self.assertRaisesRegex(ValueError, "observed|cutoff|future"):
            growth.build_auxiliary_targets(arrays)

    def test_zero_growth_is_valid_and_negative_growth_is_unavailable(self):
        growth = growth_module(self)
        arrays = fixture(events=(1,), hours=(12.,))
        arrays["history"][0].scores[18] = 220.
        aux = growth.build_auxiliary_targets(arrays)
        self.assertTrue(aux["available"][0, 0])
        self.assertEqual(aux["increment"][0, 0], 0.)
        arrays["history"][0].scores[18] = 219.
        aux = growth.build_auxiliary_targets(arrays)
        self.assertFalse(aux["available"][0, 0])
        self.assertEqual(aux["reasons"][0, 0], "negative_growth")

    def test_positive_integrals_form_monotonic_paths_to_final_score(self):
        growth = growth_module(self)
        horizon = np.array([0., 6., 24., 30., 60.])
        for dual in (False, True):
            integral = growth.basis_integrals(np.full(5, 1 / 6), np.full(5, 60.),
                                               np.full(5, 72.), dual_clock=dual,
                                               horizon_h=horizon)
            prediction = 220. + integral @ np.arange(1, integral.shape[1] + 1)
            self.assertEqual(prediction[0], 220.)
            self.assertTrue((np.diff(prediction) > 0.).all())
            np.testing.assert_allclose(integral[:, :3].sum(axis=1), horizon)


class GrowthTrainingSubprocessTests(unittest.TestCase):
    def run_script(self, script):
        environment = os.environ.copy()
        environment["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib"
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                                cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_keeps_torch_out_of_test_discovery(self):
        growth_module(self)
        self.run_script("""
            import sys
            import analysis.v2_next_growth
            assert 'torch' not in sys.modules
        """)

    def test_fixed_seeds_controls_auxiliary_training_and_saved_replay(self):
        growth_module(self)
        self.run_script("""
            import io
            import joblib
            import numpy as np
            from analysis import innovation_growth as old
            from analysis import v2_next_growth as growth
            from tests.test_v2_next_growth import fixture

            arrays = fixture()
            # Event 2 has no auxiliary answers but must retain identical TOTAL
            # objective mass to covered event 1 for every registered lambda.
            arrays['history'] = [curve for curve in arrays['history'] if curve.event != 2]
            labels = {case['case_id']: 820. for case in arrays['cases']}
            query = dict(arrays)
            # Inference has neither full history nor the auxiliary target builder's samples.
            query.pop('history')
            query.pop('samples')
            controls = {config['head']: config for config in old.recipes()}
            assert {config['head'] for config in growth.recipes()
                    if config['auxiliary_lambda'] == 0.} == {'direct', 'progress', 'dual_clock'}
            results = {}
            for config in growth.recipes():
                bundle = growth.fit_recipe(config, arrays, labels)
                assert bundle['training_case_ids'] == ['case-0', 'case-1', 'case-2']
                assert bundle['training_event_count'] == 2
                assert set(bundle['members']) == {'seed_42', 'seed_123', 'seed_2026'}
                assert all(member['epochs_run'] == 80 for member in bundle['members'].values())
                prediction = growth.predict_recipe(bundle, query)
                members = growth.predict_members(bundle, query)
                np.testing.assert_array_equal(prediction, np.mean(list(members.values()), axis=0))
                assert np.isfinite(prediction).all() and (prediction >= query['current']).all()
                assert bundle['auxiliary_diagnostics']['available_targets'] == 6
                assert len(bundle['auxiliary_diagnostics']['by_event_rank_horizon']) == 6
                audit = bundle['diagnostics']['event_weight_audit']
                np.testing.assert_allclose(
                    [row['terminal_weight'] + config['auxiliary_lambda'] * row['auxiliary_weight']
                     for row in audit], [1.5, 1.5])
                frozen = joblib.hash(bundle)
                ended = dict(query, progress=np.ones(3), remaining_h=np.zeros(3))
                np.testing.assert_array_equal(growth.predict_recipe(bundle, ended), query['current'])
                assert joblib.hash(bundle) == frozen
                output = io.BytesIO()
                joblib.dump(bundle, output)
                output.seek(0)
                restored = joblib.load(output)
                np.testing.assert_array_equal(growth.predict_recipe(restored, query), prediction)
                assert 'history' not in bundle and 'target_score' not in bundle
                if config['auxiliary_lambda'] == 0.:
                    original = old.fit_recipe(controls[config['head']], arrays, labels)
                    np.testing.assert_array_equal(old.predict_recipe(original, query), prediction)
                results[config['name']] = prediction
            assert not np.array_equal(results['next_growth_dual_clock'],
                                      results['next_growth_dual_clock_aux025'])
            # The recipe API rejects changing epochs, seeds or target rules after registration.
            for change in ({'epochs': 2}, {'seeds': [42]}, {'max_endpoint_distance_h': 2.}):
                try:
                    growth.fit_recipe(dict(growth.recipes()[0], **change), arrays, labels)
                except ValueError:
                    pass
                else:
                    raise AssertionError('accepted an unregistered training recipe')
        """)


if __name__ == "__main__":
    unittest.main()
