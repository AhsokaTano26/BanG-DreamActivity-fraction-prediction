"""Synthetic leakage and persistence checks; never open a frozen dataset."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from analysis.backtest import Curve
from analysis import innovation_retrieval


HOUR = 3_600_000


def history_curve(event, *, rank=1000, start=None, multiplier=2.0, scale=1.0):
    start = event * 20 * HOUR if start is None else start
    scores = np.r_[np.arange(1, 6) * 20.0, np.linspace(100, 100 * multiplier, 6)[1:]] * scale
    return Curve(event, rank, "normal", start, start + 10 * HOUR,
                 start + np.arange(1, 11) * HOUR, scores)


def query_arrays(events=(100,), *, histories=(), start_base=2000 * HOUR):
    samples, cases, rows = [], [], []
    current, progress = [], []
    for i, event in enumerate(events):
        start = start_base + i * 20 * HOUR
        value = 100.0 + i * 7
        times = start + np.arange(1, 6) * HOUR
        scores = np.arange(1, 6) * value / 5
        curve = Curve(event, 1000 + (i % 2) * 1000, "normal", start, start + 10 * HOUR, times, scores)
        case_id = f"case-{event}-{i}"
        samples.append({"curve": curve, "case_id": case_id, "times": times, "scores": scores,
                        "now": int(times[-1]), "cutoff": int(times[-1] + HOUR // 2),
                        "score": value, "progress": 0.5})
        cases.append({"case_id": case_id, "event": event, "rank": curve.rank,
                      "start_ms": start, "end_ms": curve.end})
        rows.append({"log_current": np.log1p(value), "progress": 0.5,
                     "event_type": "normal" if i % 2 else "challenge", "all_missing": np.nan})
        current.append(value)
        progress.append(0.5)
    return {"samples": samples, "cases": cases, "base": pd.DataFrame(rows),
            "current": np.asarray(current), "progress": np.asarray(progress),
            "remaining_h": np.full(len(events), 5.0), "duration_h": np.full(len(events), 10.0),
            "rank": np.arange(len(events)) % 2, "event_type": np.array(["normal"] * len(events)),
            "band_shares": [{"1": 1.0} for _ in events], "history": list(histories)}


class RetrievalTests(unittest.TestCase):
    module = innovation_retrieval

    def setUp(self):
        self.history = [history_curve(i, multiplier=2 + 0.2 * (i - 1)) for i in range(1, 6)]

    def test_same_rank_completed_different_event_only(self):
        arrays = query_arrays(histories=self.history)
        clean = self.module.retrieval_features(arrays, self.history)
        query_start = arrays["cases"][0]["start_ms"]
        poisons = [history_curve(100, start=0, multiplier=1e9),
                   history_curve(101, start=query_start - 5 * HOUR, multiplier=1e9),
                   history_curve(102, start=query_start - 10 * HOUR, multiplier=1e9),
                   history_curve(103, start=0, rank=2000, multiplier=1e9)]
        polluted = self.module.retrieval_features(arrays, self.history + poisons)
        pd.testing.assert_frame_equal(clean, polluted)
        self.assertEqual(clean.loc[0, "retrieval_eligible_count"], 5)
        self.assertEqual(clean.loc[0, "retrieval_neighbor_count"], 5)
        self.assertAlmostEqual(clean.loc[0, "retrieval_prior"], 240.0)
        self.assertGreater(clean.loc[0, "retrieval_log_multiplier_iqr"], 0)

    def test_exactly_five_nearest_with_deterministic_distance(self):
        history = self.history + [history_curve(6, multiplier=1e6, scale=100)]
        result = self.module.retrieval_features(query_arrays(), history)
        self.assertEqual(result.loc[0, "retrieval_eligible_count"], 6)
        self.assertEqual(result.loc[0, "retrieval_neighbor_count"], 5)
        self.assertAlmostEqual(result.loc[0, "retrieval_prior"], 240.0)
        self.assertAlmostEqual(result.loc[0, "retrieval_distance_mean"], 0.0)

    def test_sparse_or_absent_history_keeps_linear_prior_and_coverage(self):
        arrays = query_arrays(events=(100, 101))
        for history in ([], self.history[:4]):
            with self.subTest(size=len(history)):
                result = self.module.retrieval_features(arrays, history)
                np.testing.assert_allclose(result["retrieval_prior"], [200, 214])
                np.testing.assert_array_equal(result["retrieval_fallback"], [1, 1])
                np.testing.assert_array_equal(result["retrieval_neighbor_count"], [0, 0])
                self.assertEqual(result.loc[0, "retrieval_eligible_count"], len(history))
                self.assertEqual(result.loc[1, "retrieval_eligible_count"], 0)
                self.assertEqual(result.loc[0, "retrieval_query_lag_minutes"], 30)

    def test_early_fallback_keeps_true_linear_prior_before_final_decode_bound(self):
        arrays = query_arrays()
        arrays["progress"][0] = 0.005
        result = self.module.retrieval_features(arrays, [])
        self.assertAlmostEqual(result.loc[0, "retrieval_prior"], 20000)

    def test_insufficient_query_shape_retains_case_with_explicit_fallback(self):
        arrays = query_arrays()
        arrays["samples"][0]["times"] = arrays["samples"][0]["times"][-1:]
        arrays["samples"][0]["scores"] = arrays["samples"][0]["scores"][-1:]
        result = self.module.retrieval_features(arrays, self.history)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "retrieval_prefix_quality_count"], 5)
        self.assertEqual(result.loc[0, "retrieval_eligible_count"], 0)
        self.assertEqual(result.loc[0, "retrieval_prior"], 200)
        self.assertEqual(result.loc[0, "retrieval_fallback"], 1)

    def test_stale_historical_prefix_is_excluded_and_lag_is_recorded(self):
        accepted = history_curve(1)
        accepted.times = np.delete(accepted.times, 4)
        accepted.scores = np.delete(accepted.scores, 4)
        stale = history_curve(2)
        stale.times = stale.start + np.array([0.5, 1, 2, 3.9, 6, 7, 8, 9, 10]) * HOUR
        stale.scores = np.array([10, 20, 40, 78, 120, 140, 160, 180, 200], dtype=float)
        result = self.module.retrieval_features(query_arrays(), [accepted, stale] + self.history[2:])
        self.assertEqual(result.loc[0, "retrieval_past_count"], 5)
        self.assertEqual(result.loc[0, "retrieval_prefix_quality_count"], 4)
        self.assertEqual(result.loc[0, "retrieval_eligible_count"], 4)
        all_accepted = [accepted] + self.history[1:]
        result = self.module.retrieval_features(query_arrays(), all_accepted)
        self.assertEqual(result.loc[0, "retrieval_lag_max_minutes"], 60)
        self.assertEqual(result.loc[0, "retrieval_lag_mean_minutes"], 12)

    def test_query_future_suffix_cannot_change_features(self):
        arrays = query_arrays()
        before = self.module.retrieval_features(arrays, self.history)
        poisoned = copy.deepcopy(arrays)
        query = poisoned["samples"][0]
        query["times"] = np.r_[query["times"], query["now"] + HOUR]
        query["scores"] = np.r_[query["scores"], 1e100]
        query["curve"].times = query["times"]
        query["curve"].scores = query["scores"]
        after = self.module.retrieval_features(poisoned, self.history)
        pd.testing.assert_frame_equal(before, after)

    def test_duplicate_event_rank_library_fails_instead_of_double_weighting(self):
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            self.module.retrieval_features(query_arrays(), self.history + [copy.deepcopy(self.history[0])])

    def test_each_event_receives_equal_total_training_weight(self):
        weights = self.module.event_weights([7, 7, 7, 8, 9, 9])
        self.assertAlmostEqual(weights[:3].sum(), 2)
        self.assertAlmostEqual(weights[3], 2)
        self.assertAlmostEqual(weights[4:].sum(), 2)
        self.assertAlmostEqual(weights.mean(), 1)

    def test_all_recipes_persist_schema_ignore_new_history_and_align_outputs(self):
        arrays = query_arrays(events=(100, 100, 101, 102, 103, 104), histories=copy.deepcopy(self.history))
        labels = {c["case_id"]: arrays["current"][i] * (1.5 + 0.1 * i) for i, c in enumerate(arrays["cases"])}
        query = query_arrays(events=(201, 202, 203), start_base=4000 * HOUR)
        query["base"]["event_type"] = "unseen-type"
        query["base"]["all_missing"] = [500, np.nan, -500]
        query["base"]["extra_column"] = 99
        for config in self.module.recipes():
            with self.subTest(recipe=config["name"]):
                bundle = self.module.fit_recipe(config | {"iterations": 8}, arrays, labels)
                before = self.module.predict_recipe(bundle, query)
                self.assertEqual(before.shape, (3,))
                self.assertTrue(np.isfinite(before).all())
                self.assertTrue((before >= query["current"]).all())
                poisoned = copy.deepcopy(query)
                poisoned["history"] = [history_curve(500, start=0, multiplier=1e9)]
                np.testing.assert_array_equal(before, self.module.predict_recipe(bundle, poisoned))
                for curve in arrays["history"]:
                    curve.scores[-1] = 1e80
                np.testing.assert_array_equal(before, self.module.predict_recipe(bundle, query))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "bundle.joblib"
                    joblib.dump(bundle, path)
                    loaded = joblib.load(path)
                    np.testing.assert_array_equal(before, self.module.predict_recipe(loaded, query))
                reversed_arrays = {k: (v.iloc[::-1].reset_index(drop=True) if isinstance(v, pd.DataFrame)
                                       else v[::-1] if k != "history" else v) for k, v in query.items()}
                np.testing.assert_array_equal(before[::-1], self.module.predict_recipe(bundle, reversed_arrays))
                self.assertEqual(len(bundle["history_provenance"]), len(self.history))
                self.assertEqual(len(bundle["feature_columns"]), len(set(bundle["feature_columns"])))
                self.assertNotIn("extra_column", bundle["feature_columns"])
                arrays["history"] = copy.deepcopy(self.history)

    def test_decode_clamps_negative_learned_growth_to_current(self):
        arrays = query_arrays(events=(100, 101, 102, 103))
        labels = {case["case_id"]: 0.1 * current for case, current in zip(arrays["cases"], arrays["current"])}
        config = next(r for r in self.module.recipes() if r["mode"] == "control") | {"iterations": 4}
        bundle = self.module.fit_recipe(config, arrays, labels)
        np.testing.assert_array_equal(self.module.predict_recipe(bundle, arrays), arrays["current"])

    def test_control_does_not_use_retrieval_and_prediction_requires_no_labels(self):
        arrays = query_arrays(events=(100, 101, 102, 103, 104, 105), histories=self.history)
        labels = {c["case_id"]: arrays["current"][i] * (1.4 + i / 10) for i, c in enumerate(arrays["cases"])}
        config = next(r for r in self.module.recipes() if r["mode"] == "control") | {"iterations": 8}
        clean = self.module.fit_recipe(config, arrays, labels)
        altered = copy.deepcopy(arrays)
        altered["history"] = []
        other = self.module.fit_recipe(config, altered, labels)
        np.testing.assert_array_equal(self.module.predict_recipe(clean, arrays),
                                      self.module.predict_recipe(other, arrays))

    def test_recipe_cannot_claim_unimplemented_neighbor_or_tree_settings(self):
        arrays = query_arrays(events=(100, 101, 102, 103))
        labels = {case["case_id"]: 250.0 for case in arrays["cases"]}
        config = self.module.recipes()[1] | {"iterations": 4}
        for alteration in ({"neighbors": 6}, {"depth": 7}, {"scale_distance_weight": 0.9}):
            with self.subTest(alteration=alteration), self.assertRaisesRegex(ValueError, "[Ff]ixed"):
                self.module.fit_recipe(config | alteration, arrays, labels)


if __name__ == "__main__":
    unittest.main()
