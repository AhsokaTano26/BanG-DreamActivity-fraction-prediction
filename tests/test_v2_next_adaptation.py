"""Synthetic checks for frozen TRAIN-only historical adaptation."""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import joblib
import numpy as np

from analysis.backtest import Curve, snapshot
from analysis.innovation_common import prepare, subset_arrays


HOUR = 3_600_000


def fixture():
    history, samples, labels = [], [], {}
    for event in range(1, 13):
        start, end = event * 240 * HOUR, (event * 240 + 120) * HOUR
        for rank in (1000, 2000):
            times = start + np.arange(21) * 6 * HOUR
            final = (10000 + 800 * event) * (2 if rank == 1000 else 1)
            scores = final * np.linspace(0, 1, 21) ** (1.1 + event * .015)
            curve = Curve(event, rank, "live" if event % 2 else "challenge",
                          start, end, times, scores)
            history.append(curve)
            for hour in ((30, 60) if event % 2 else (30, 60, 108)):
                sample = snapshot(curve, start + hour * HOUR)
                # prepare must receive blind prefixes, exactly as in the runner.
                stop = np.searchsorted(times, sample["now"], side="right")
                sample["curve"] = Curve(event, rank, curve.kind, start, end,
                                        times[:stop].copy(), scores[:stop].copy())
                sample["case_id"] = f"train:{event}:{rank}:{hour}"
                samples.append(sample)
                labels[sample["case_id"]] = float(final)
    metadata = SimpleNamespace(parties={e: {"bands": {str(e % 2): 1.}}
                                        for e in range(1, 13)})
    arrays = prepare(samples, metadata)
    arrays["history"] = history
    return arrays, labels


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("analysis.v2_next_adaptation"),
                             "The adaptation family is missing")
        self.a = importlib.import_module("analysis.v2_next_adaptation")

    def test_recent_features_exclude_current_unfinished_and_future_history(self):
        arrays, _ = fixture()
        query = subset_arrays(arrays, [i for i, c in enumerate(arrays["cases"])
                                       if c["event"] == 7])
        query_start = query["cases"][0]["start_ms"]
        history = deepcopy(arrays["history"])
        for curve in history:
            if curve.event == 4:
                curve.end = query_start + HOUR
            if curve.event == 6:
                curve.end = query_start
        frame, audit = self.a.recent_history_features(query, history, self.a.recipes()[1])
        self.assertEqual(audit[0]["eligible_events"], [1, 2, 3, 5])
        self.assertAlmostEqual(frame.iloc[0]["recent_log_rank_ratio"], np.log(2))
        poisoned = deepcopy(history)
        for curve in poisoned:
            if curve.event not in {1, 2, 3, 5}:
                curve.scores *= 1000
        second, _ = self.a.recent_history_features(query, poisoned, self.a.recipes()[1])
        np.testing.assert_array_equal(frame.to_numpy(), second.to_numpy())

    def test_missing_history_and_stale_sprint_observations_are_explicit(self):
        arrays, _ = fixture()
        first = subset_arrays(arrays, [0])
        frame, _ = self.a.recent_history_features(first, arrays["history"], self.a.recipes()[1])
        self.assertEqual(frame.iloc[0]["recent_log_final_missing"], 1)
        self.assertEqual(frame.iloc[0]["recent_log_rank_ratio_missing"], 1)
        query = subset_arrays(arrays, [i for i, c in enumerate(arrays["cases"])
                                       if c["event"] == 2][:1])
        history = deepcopy(arrays["history"][:2])
        for curve in history:
            keep = curve.times != curve.end - 6 * HOUR
            curve.times, curve.scores = curve.times[keep], curve.scores[keep]
        frame, _ = self.a.recent_history_features(query, history, self.a.recipes()[1])
        self.assertEqual(frame.iloc[0]["recent_log_sprint_6h_missing"], 1)
        self.assertEqual(frame.iloc[0]["recent_log_sprint_24h_missing"], 0)

    def test_ewma_uses_one_prior_event_update_with_direction_and_shrinkage(self):
        arrays, _ = fixture()
        query = subset_arrays(arrays, [0])
        query["cases"][0] = dict(query["cases"][0], event=9, start_ms=1000, end_ms=1200)
        query["progress"] = np.array([.5])
        records = [{"event": 1, "rank": 1000, "phase": 1, "end_ms": 100,
                    "start_ms": 0, "log_error": .2, "case_ids": ["a", "b"]},
                   {"event": 2, "rank": 1000, "phase": 1, "end_ms": 200,
                    "start_ms": 110, "log_error": -.1, "case_ids": ["c"]},
                   {"event": 9, "rank": 1000, "phase": 1, "end_ms": 300,
                    "start_ms": 210, "log_error": 5., "case_ids": ["own"]},
                   {"event": 3, "rank": 1000, "phase": 1, "end_ms": 1000,
                    "start_ms": 310, "log_error": 5., "case_ids": ["equal"]},
                   {"event": 4, "rank": 2000, "phase": 1, "end_ms": 400,
                    "start_ms": 320, "log_error": 5., "case_ids": ["rank"]}]
        full = {"config": self.a.recipes()[3], "correction_records": records}
        shrunk = {"config": self.a.recipes()[2], "correction_records": records}
        # Start from the first event error; .7*.2 + .3*(-.1) = .11.
        value, audit = self.a.correction_values(full, query)
        self.assertAlmostEqual(value[0], .11)
        self.assertEqual(audit[0]["eligible_events"], [1, 2])
        self.assertAlmostEqual(self.a.correction_values(shrunk, query)[0][0], .11 * 2 / 12)
        negative = deepcopy(full)
        negative["correction_records"] = [dict(records[0], log_error=-.8)]
        self.assertAlmostEqual(self.a.correction_values(negative, query)[0][0], -.3)

    def test_oof_prediction_ignores_own_and_later_labels(self):
        arrays, labels = fixture()
        config = self.a.recipes()[2] | {"iterations": 5}
        original = self.a.fit_recipe(config, arrays, labels)
        poisoned = dict(labels)
        for case in arrays["cases"]:
            if case["event"] >= 7:
                poisoned[case["case_id"]] *= 4
        changed = self.a.fit_recipe(config, arrays, poisoned)
        indices = original["oof_folds"][0]["heldout_indices"]
        np.testing.assert_array_equal(original["oof_predictions"][indices],
                                      changed["oof_predictions"][indices])
        self.assertTrue(np.isnan(original["oof_predictions"][:30]).all())
        for fold in original["oof_folds"]:
            self.assertLess(fold["training_latest_end"], fold["heldout_earliest_start"])
            self.assertTrue(set(fold["history_events"]).issubset(fold["training_events"]))
        expected = np.log([labels[c["case_id"]] for c in arrays["cases"]])
        available = np.isfinite(original["oof_predictions"])
        expected[available] -= np.log(original["oof_predictions"][available])
        np.testing.assert_allclose(original["oof_residuals"][available], expected[available])

    def test_training_and_oof_event_mass_is_equal_and_audit_is_serializable(self):
        arrays, labels = fixture()
        bundle = self.a.fit_recipe(self.a.recipes()[2] | {"iterations": 5}, arrays, labels)
        events = np.array([c["event"] for c in arrays["cases"]])
        mass = [bundle["train_weights"][events == event].sum() for event in range(1, 13)]
        np.testing.assert_allclose(mass, 5.)
        for fold in bundle["oof_folds"]:
            np.testing.assert_array_equal(fold["train_weights"],
                                          bundle["train_weights"][fold["train_indices"]])
        for event in range(1, 13):
            self.assertAlmostEqual(bundle["calibration_case_weights"][events == event].sum(),
                                   0. if event <= 6 else 1.)
        _, audit = self.a.correction_values(bundle, arrays)
        for case, row in zip(arrays["cases"], audit):
            self.assertNotIn(case["event"], row["eligible_events"])
            for event in row["eligible_events"]:
                self.assertLess((event * 240 + 120) * HOUR, case["start_ms"])
        records = bundle["correction_records"]
        keys = [(r["event"], r["rank"], r["phase"]) for r in records]
        self.assertEqual(len(keys), len(set(keys)))
        json.dumps(bundle["diagnostics"], allow_nan=False)

    def test_sparse_case_history_keeps_history_only_events_and_block_freeze(self):
        arrays, labels = fixture()
        extra = deepcopy(arrays["history"][:2])
        for curve in extra:
            curve.event = 99  # Ordering must depend on dates, not ID magnitude.
            curve.start -= 480 * HOUR
            curve.end -= 480 * HOUR
            curve.times -= 480 * HOUR
        arrays["history"].extend(extra)
        bundle = self.a.fit_recipe(self.a.recipes()[4] | {"iterations": 5}, arrays, labels)
        first = bundle["oof_folds"][0]
        self.assertIn(99, first["history_events"])
        self.assertNotIn(99, first["training_events"])
        for row in first["heldout_history_audit"]:
            self.assertEqual(row["eligible_events"], [99, 1, 2, 3, 4, 5, 6])
        group = next(r for r in bundle["correction_records"]
                     if (r["event"], r["rank"], r["phase"]) == (8, 1000, 1))
        # 30h and 60h both belong to the middle phase, producing one update.
        self.assertEqual(len(group["case_ids"]), 2)
        values = [bundle["oof_residuals"][i] for i, c in enumerate(arrays["cases"])
                  if c["case_id"] in group["case_ids"]]
        self.assertAlmostEqual(group["log_error"], (values[0] + values[1]) / 2)

    def test_incomplete_training_event_is_excluded_from_oof_supervision_and_history(self):
        arrays, labels = fixture()
        late_end = 7 * 240 * HOUR
        for case in arrays["cases"]:
            if case["event"] == 4:
                case["end_ms"] = late_end
        for curve in arrays["history"]:
            if curve.event == 4:
                curve.end = late_end
        config = self.a.recipes()[4] | {"iterations": 5}
        original = self.a.fit_recipe(config, arrays, labels)
        changed_labels = dict(labels)
        for case in arrays["cases"]:
            if case["event"] in {4, 7, 8, 12}:
                changed_labels[case["case_id"]] *= 5
        changed = self.a.fit_recipe(config, arrays, changed_labels)
        first = original["oof_folds"][0]
        self.assertNotIn(4, first["training_events"])
        self.assertNotIn(4, first["history_events"])
        indices = first["heldout_indices"]
        np.testing.assert_array_equal(original["oof_predictions"][indices],
                                      changed["oof_predictions"][indices])

    def test_no_completed_oof_history_keeps_all_predictions(self):
        arrays, labels = fixture()
        for case in arrays["cases"]:
            case["end_ms"] = 10000 * HOUR
        for curve in arrays["history"]:
            curve.end = 10000 * HOUR
        bundle = self.a.fit_recipe(self.a.recipes()[2] | {"iterations": 5}, arrays, labels)
        self.assertEqual(bundle["correction_records"], [])
        self.assertTrue(np.isnan(bundle["oof_predictions"]).all())
        prediction = self.a.predict_recipe(bundle, arrays)
        self.assertEqual(len(prediction), len(labels))
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_array_equal(self.a.correction_values(bundle, arrays)[0], 0.)

    def test_all_recipes_ignore_external_query_history_and_roundtrip(self):
        arrays, labels = fixture()
        query = deepcopy(arrays)
        query["history"] = [object()]
        for config in self.a.recipes():
            with self.subTest(recipe=config["name"]), tempfile.TemporaryDirectory() as folder:
                bundle = self.a.fit_recipe(config | {"iterations": 5}, arrays, labels)
                expected = self.a.predict_recipe(bundle, arrays)
                np.testing.assert_array_equal(expected, self.a.predict_recipe(bundle, query))
                path = Path(folder) / "model.joblib"
                joblib.dump(bundle, path)
                np.testing.assert_array_equal(expected,
                    self.a.predict_recipe(joblib.load(path), query))
                self.assertTrue(np.isfinite(expected).all())
                self.assertTrue((expected >= arrays["current"]).all())
                # A caller mutating the original training library cannot change the fit.
                changed = deepcopy(arrays)
                changed["history"][0].scores *= 20
                np.testing.assert_array_equal(expected, self.a.predict_recipe(bundle, changed))


if __name__ == "__main__":
    unittest.main()
