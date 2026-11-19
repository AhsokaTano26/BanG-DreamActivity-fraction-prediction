"""Synthetic, label-isolated checks for the fixed hierarchy family."""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd


def fixture():
    cases, rows, current, progress, kinds, bands = [], [], [], [], [], []
    for event in range(1, 13):
        for j in range(2 if event % 2 else 3):
            p = (.25, .65, .9)[j]
            score = 100.0 + 8 * event + 15 * j
            cases.append({"case_id": f"{event}/{j}", "event": event,
                          "rank": 1000 if j % 2 == 0 else 2000,
                          "start_ms": event * 10, "end_ms": event * 10 + 6})
            rows.append({"own_scale": score, "phase": p,
                         "missing": np.nan,
                         "observed_regime": "early" if event <= 6 else "later"})
            current.append(score)
            progress.append(p)
            kinds.append("live" if event % 2 else "challenge")
            bands.append({"A": 1.0} if event % 2 else {"A": .25, "B": .75})
    progress = np.asarray(progress)
    arrays = {"cases": cases, "samples": [None] * len(cases), "base": pd.DataFrame(rows),
              "current": np.asarray(current), "progress": progress,
              "duration_h": np.full(len(cases), 120.0),
              "remaining_h": (1 - progress) * 120,
              "rank": np.asarray([0 if c["rank"] == 1000 else 1 for c in cases]),
              "event_type": np.asarray(kinds), "band_shares": bands,
              "history": []}
    labels = {c["case_id"]: float(s / p * (1 + .015 * (c["event"] % 3)))
              for c, s, p in zip(cases, current, progress)}
    return arrays, labels


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        # Removing the family implementation must fail this boundary, not silently skip.
        self.assertIsNotNone(importlib.util.find_spec("analysis.innovation_hierarchy"),
                             "The hierarchy family implementation is missing")
        self.h = importlib.import_module("analysis.innovation_hierarchy")

    def test_unknown_type_and_band_fall_back_to_shared_effects(self):
        arrays, _ = fixture()
        design = self.h.ContextDesign.fit(arrays, interactions=True)
        query = deepcopy(arrays)
        query["event_type"] = np.full(len(arrays["cases"]), "never-seen-type")
        query["band_shares"] = [{"never-seen-band": 1.0} for _ in arrays["cases"]]
        x = design.transform(query)
        np.testing.assert_allclose(x[:, 4:], 0)
        np.testing.assert_allclose(x[0, :4], [1, 0, -.5, np.exp(-90 / 12)])
        self.assertNotIn("never-seen-band", design.bands)
        self.assertNotIn("never-seen-type", design.types)

    def test_chronological_oof_excludes_heldout_unfinished_and_equal_end_events(self):
        arrays, _ = fixture()
        for case in arrays["cases"]:
            if case["event"] == 4:
                case["end_ms"] = 95
            if case["event"] == 6:
                case["end_ms"] = 70
        folds = self.h.chronological_oof_splits(arrays)
        self.assertEqual(len(folds), 3)
        first = folds[0]
        self.assertEqual(set(first["train_events"]), {1, 2, 3, 5})
        self.assertEqual(set(first["heldout_events"]), {7, 8})
        represented = []
        for fold in folds:
            heldout = [arrays["cases"][i] for i in fold["heldout_indices"]]
            start = min(c["start_ms"] for c in heldout)
            for i in fold["train_indices"]:
                self.assertLess(arrays["cases"][i]["end_ms"], start)
                self.assertNotIn(arrays["cases"][i]["event"], fold["heldout_events"])
            represented.extend(c["case_id"] for c in heldout)
        self.assertCountEqual(represented,
                              [c["case_id"] for c in arrays["cases"] if c["event"] >= 7])

    def test_inner_preprocessing_does_not_see_heldout_values_or_vocabulary(self):
        arrays, labels = fixture()
        config = self.h.recipes()[1] | {"iterations": 6}
        bundle = self.h.fit_recipe(config, arrays, labels)
        first = bundle["oof_folds"][0]
        prep = first["preprocessor"]
        self.assertEqual(prep.categories["observed_regime"], ("early",))
        self.assertEqual(prep.medians["missing"], 0)
        self.assertLess(prep.medians["own_scale"], 160)
        probe = pd.DataFrame({"own_scale": [np.nan], "phase": [np.nan],
                              "missing": [np.nan], "observed_regime": ["later"]})
        transformed = prep.transform(probe)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed[0, -1], 0)
        heldout_ids = set(first["heldout_case_ids"])
        train_ids = set(first["train_case_ids"])
        self.assertFalse(heldout_ids & train_ids)
        self.assertTrue(np.isnan(bundle["oof_predictions"][:15]).all())
        self.assertTrue(np.isfinite(bundle["oof_predictions"][15:]).all())

    def test_oof_predictions_ignore_their_own_answers_and_unfinished_answers(self):
        arrays, labels = fixture()
        for case in arrays["cases"]:
            if case["event"] == 4:
                case["end_ms"] = 95
        config = self.h.recipes()[1] | {"iterations": 6}
        first = self.h.fit_recipe(config, arrays, labels)
        poisoned = dict(labels)
        for case in arrays["cases"]:
            if case["event"] in {4, 7, 8, 12}:
                poisoned[case["case_id"]] *= 7
        second = self.h.fit_recipe(config, arrays, poisoned)
        indices = first["oof_folds"][0]["heldout_indices"]
        np.testing.assert_array_equal(first["oof_predictions"][indices],
                                      second["oof_predictions"][indices])

    def test_backbone_slice_weights_and_correction_event_totals_are_preserved(self):
        arrays, labels = fixture()
        bundle = self.h.fit_recipe(self.h.recipes()[2] | {"iterations": 6}, arrays, labels)
        weights = bundle["train_weights"]
        self.assertAlmostEqual(weights[0], 1.25)
        self.assertAlmostEqual(weights[2], 5 / 6)
        for fold in bundle["oof_folds"]:
            np.testing.assert_array_equal(fold["train_weights"], weights[fold["train_indices"]])
        correction = bundle["correction_weights"]
        for event in range(1, 13):
            indices = [i for i, c in enumerate(arrays["cases"]) if c["event"] == event]
            self.assertAlmostEqual(correction[indices].sum(), 0 if event <= 6 else 1)

    def test_all_recipes_predict_all_cases_and_survive_joblib_replay(self):
        arrays, labels = fixture()
        query = deepcopy(arrays)
        query["event_type"] = np.full(len(arrays["cases"]), "unknown")
        query["band_shares"] = [{"unknown": 1.0} for _ in arrays["cases"]]
        query["base"]["observed_regime"] = "unknown"
        query["base"]["missing"] = np.nan
        for config in self.h.recipes():
            with self.subTest(recipe=config["name"]), tempfile.TemporaryDirectory() as temporary:
                bundle = self.h.fit_recipe(config | {"iterations": 6}, arrays, labels)
                prediction = self.h.predict_recipe(bundle, query)
                self.assertEqual(len(prediction), len(query["cases"]))
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue((prediction >= query["current"]).all())
                path = Path(temporary) / "bundle.joblib"
                joblib.dump(bundle, path)
                np.testing.assert_array_equal(prediction,
                    self.h.predict_recipe(joblib.load(path), query))

    def test_no_oof_history_keeps_complete_backbone_predictions(self):
        arrays, labels = fixture()
        for case in arrays["cases"]:
            case["end_ms"] = 1000
        bundle = self.h.fit_recipe(self.h.recipes()[1] | {"iterations": 6}, arrays, labels)
        self.assertEqual(bundle["calibration_events"], [])
        self.assertTrue(np.isnan(bundle["oof_predictions"]).all())
        np.testing.assert_array_equal(bundle["correction_coefficients"], 0)
        prediction = self.h.predict_recipe(bundle, arrays)
        self.assertEqual(len(prediction), len(arrays["cases"]))
        self.assertTrue(np.isfinite(prediction).all())

    def test_correction_solves_the_weighted_oof_residual_objective(self):
        arrays, labels = fixture()
        bundle = self.h.fit_recipe(self.h.recipes()[2] | {"iterations": 6}, arrays, labels)
        support = np.isfinite(bundle["oof_predictions"])
        actual = np.asarray([labels[c["case_id"]] for c in arrays["cases"]])
        expected = np.log(actual[support]) - np.log(arrays["current"][support])
        expected += np.log(arrays["progress"][support]) - bundle["oof_predictions"][support]
        np.testing.assert_allclose(bundle["oof_residuals"][support], expected, atol=1e-14)
        x = bundle["context_design"].transform(arrays)[support]
        beta = bundle["correction_coefficients"]
        weights = bundle["correction_weights"][support]
        gradient = x.T @ (weights * (x @ beta - expected))
        gradient += bundle["correction_penalties"] * beta
        np.testing.assert_allclose(gradient, 0, atol=1e-12)

    def test_additive_group_effects_are_constant_but_phase_effects_change(self):
        arrays, _ = fixture()
        additive = self.h.ContextDesign.fit(arrays).transform(arrays)
        phase = self.h.ContextDesign.fit(arrays, interactions=True).transform(arrays)
        np.testing.assert_array_equal(additive[0, 4:], additive[1, 4:])
        # Columns 4:8 are challenge/live and fractional A/B membership.
        np.testing.assert_array_equal(additive[0, 4:], [0, 1, 1, 0])
        np.testing.assert_allclose(phase[0, 8:12], [0, -.5, -.5, 0])
        np.testing.assert_allclose(phase[1, 8:12], [0, .3, .3, 0])

    def test_split_uses_start_time_instead_of_numeric_event_id(self):
        arrays, _ = fixture()
        for case in arrays["cases"]:
            case["event"] = 13 - case["event"]
        first = self.h.chronological_oof_splits(arrays)[0]
        self.assertEqual(set(first["heldout_events"]), {5, 6})
        self.assertEqual(set(first["train_events"]), {7, 8, 9, 10, 11, 12})

    def test_single_event_constant_features_have_a_complete_fallback(self):
        arrays, _ = fixture()
        arrays = {k: v for k, v in arrays.items() if k not in {"samples", "duration_h", "history"}}
        arrays["cases"] = arrays["cases"][:2]
        arrays["base"] = pd.DataFrame({"constant": [1., 1.]})
        arrays["current"] = np.array([100., 100.])
        arrays["progress"] = np.array([.5, .5])
        arrays["remaining_h"] = np.array([60., 60.])
        arrays["rank"] = np.array([0, 1])
        arrays["event_type"] = np.array(["live", "live"])
        arrays["band_shares"] = [{"A": 1.}, {"A": 1.}]
        labels = {c["case_id"]: 200. for c in arrays["cases"]}
        for config in self.h.recipes():
            with self.subTest(recipe=config["name"]):
                bundle = self.h.fit_recipe(config | {"iterations": 6}, arrays, labels)
                np.testing.assert_allclose(self.h.predict_recipe(bundle, arrays), [200., 200.])


if __name__ == "__main__":
    unittest.main()
