"""Preregistered V2 TRAIN-history features and frozen chronological OOF EWMA.

Only fit_recipe receives labels. All history enters through the trusted
training arrays; prediction uses the fitted copy, never query-supplied history.
Recent weighting applies to descriptive history and to one OOF update per
event/group. It never changes the supervised loss's equal event totals.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np
import pandas as pd

from analysis.innovation_common import chronological_folds, event_weights, subset_arrays
from analysis.innovation_hierarchy import (
    ContextDesign, _check_arrays, _event_times, _fit_backbone, _static_frame,
)
from analysis.innovation_retrieval import _snapshot_history


HOUR = 3_600_000


def recipes() -> list[dict[str, Any]]:
    """Complete menu, frozen by the coordinator before any formal scoring."""
    common = {
        "family": "adaptation", "iterations": 400, "depth": 4,
        "learning_rate": .04, "l2_leaf_reg": 12., "seed": 42, "thread_count": 2,
        "target": "log(final/current)+log(progress)",
        "supervision_weights": "equal total weight per event; no recency weighting",
        "history_half_life_days": 90., "sprint_hours": [24, 6],
        "history_max_lag_hours": 1.,
        "history_boundary": "different event; end_ms strictly below query start_ms",
        "ewma_alpha": .3, "ewma_shrinkage_events": 10., "log_bias_clip": .3,
        "phase_edges": [.25, .9], "phase_boundary": "left closed; right open",
        "oof_schedule": "innovation_common.chronological_folds on fitted-case events; "
                        "first half warmup, three blocks; earlier completed history fixed per block",
        "oof_error": "log actual minus log uncorrected bounded final prediction",
        "oof_update": "one mean base log error per event/rank/phase; ordered by completion",
        "ewma_initialization": "first eligible event mean log error",
        "correction_order": "EWMA then n/(n+10) for shrinkage, then clip",
        "final_log_growth_bounds": [0., 4.],
    }
    return [common | {"name": "adaptation_context_control", "recent_features": False,
                       "correction": "none"},
            common | {"name": "adaptation_recent_features", "recent_features": True,
                       "correction": "none"},
            common | {"name": "adaptation_context_ewma_shrink", "recent_features": False,
                       "correction": "shrinkage"},
            common | {"name": "adaptation_context_ewma_full", "recent_features": False,
                       "correction": "full"},
            common | {"name": "adaptation_recent_ewma_shrink", "recent_features": True,
                       "correction": "shrinkage"}]


def _config(config):
    choices = {r["name"]: r for r in recipes()}
    if config.get("name") not in choices:
        raise ValueError("Unknown fixed adaptation recipe")
    fixed = choices[config["name"]]
    for key, value in config.items():
        if key != "iterations" and (key not in fixed or value != fixed[key]):
            raise ValueError("Fixed adaptation recipe field cannot change: " + key)
    result = fixed | dict(config)
    # Small synthetic tests may reduce iterations; formal values are sealed by
    # the coordinator together with all other family configuration fields.
    if (int(result["iterations"]) != result["iterations"] or result["iterations"] < 1):
        raise ValueError("Iterations must be a positive integer")
    return result


def _summaries(history, config):
    events, keys = {}, set()
    for curve in history:
        event, rank = int(curve.event), int(curve.rank)
        if rank not in (1000, 2000) or (event, rank) in keys:
            raise ValueError("Historical curves require unique event/rank pairs at two ranks")
        keys.add((event, rank))
        row = events.setdefault(event, {"event": event, "start_ms": int(curve.start),
                                       "end_ms": int(curve.end), "ranks": {}})
        if (row["start_ms"], row["end_ms"]) != (curve.start, curve.end):
            raise ValueError("Historical ranks disagree on event dates")
        if curve.end <= curve.start or not np.isfinite(curve.final) or curve.final <= 0:
            raise ValueError("Invalid historical event dates or final score")
        values = {"log_final": float(np.log(curve.final))}
        times, scores = np.asarray(curve.times), np.asarray(curve.scores)
        for hours in config["sprint_hours"]:
            cutoff = curve.end - hours * HOUR
            last = int(np.searchsorted(times, cutoff, side="right")) - 1
            valid = (last >= 0 and cutoff > curve.start and
                     0 <= cutoff - times[last] <= config["history_max_lag_hours"] * HOUR
                     and np.isfinite(scores[last]) and scores[last] > 0)
            values[f"log_sprint_{hours}h"] = (float(np.log(curve.final / scores[last]))
                                             if valid else None)
        row["ranks"][rank] = values
    for row in events.values():
        ranks = row["ranks"]
        row["log_rank_ratio"] = (ranks[1000]["log_final"] - ranks[2000]["log_final"]
                                 if 1000 in ranks and 2000 in ranks else None)
    return sorted(events.values(), key=lambda row: (row["end_ms"], row["event"]))


def _recent_from_summaries(arrays, summaries, config):
    rows, audit = [], []
    feature_names = ["log_final", "log_rank_ratio"] + [
        f"log_sprint_{hours}h" for hours in config["sprint_hours"]]
    for case in arrays["cases"]:
        eligible = [r for r in summaries if r["event"] != int(case["event"])
                    and r["end_ms"] < int(case["start_ms"])]
        row, used = {}, {}
        for name in feature_names:
            values, ends, events = [], [], []
            for record in eligible:
                value = (record[name] if name == "log_rank_ratio" else
                         record["ranks"].get(int(case["rank"]), {}).get(name))
                if value is not None:
                    values.append(value)
                    ends.append(record["end_ms"])
                    events.append(record["event"])
            prefix = "recent_" + name
            if values:
                # Subtract newest completion before exponentiating to avoid
                # underflow for a query far after the training history freezes.
                ages = (max(ends) - np.asarray(ends, dtype=float)) / (24 * HOUR)
                weights = np.exp2(-ages / config["history_half_life_days"])
                row[prefix] = float(np.average(values, weights=weights))
                row[prefix + "_effective_events"] = float(weights.sum() ** 2 / (weights @ weights))
            else:
                row[prefix], row[prefix + "_effective_events"] = np.nan, 0.
            row[prefix + "_missing"] = float(not values)
            row[prefix + "_events"] = float(len(values))
            used[name] = events
        rows.append(row)
        audit.append({"case_id": str(case["case_id"]), "event": int(case["event"]),
                      "query_start_ms": int(case["start_ms"]),
                      "eligible_events": [r["event"] for r in eligible],
                      "latest_eligible_end_ms": max((r["end_ms"] for r in eligible), default=None),
                      "feature_events": used})
    columns = ["recent_" + name + suffix for name in feature_names
               for suffix in ("", "_effective_events", "_missing", "_events")]
    return pd.DataFrame(rows, columns=columns), audit


def recent_history_features(arrays, history, config):
    """Audit-friendly pure feature builder over explicitly permitted curves."""
    return _recent_from_summaries(arrays, _summaries(history, config), config)


def _frame(arrays, design, summaries, config):
    frame = _static_frame(arrays, design)
    audit = []
    if config["recent_features"]:
        recent, audit = _recent_from_summaries(arrays, summaries, config)
        if set(frame.columns) & set(recent.columns):
            raise ValueError("Historical features collide with blind-prefix columns")
        frame = pd.concat([frame, recent], axis=1)
    return frame, audit


def _final_from_residual(residual, arrays, config):
    current = np.asarray(arrays["current"], dtype=float)
    growth = np.clip(np.asarray(residual) - np.log(arrays["progress"]),
                     *config["final_log_growth_bounds"])
    prediction = np.maximum(current, np.exp(np.minimum(np.log(current) + growth,
                                                       np.log(np.finfo(float).max))))
    if prediction.shape != current.shape or not np.isfinite(prediction).all():
        raise ValueError("Adaptation produced invalid predictions")
    return prediction


def _schedule(cases):
    if len({c["event"] for c in cases}) < 6:
        return []
    try:
        return chronological_folds(cases)
    except ValueError as error:
        # The common helper takes max(empty) when an overlapping synthetic
        # schedule has no completed training activity. Preserve its exact block
        # memberships and mark those folds unavailable instead of leaking data.
        if not str(error).startswith("max()") or "empty" not in str(error):
            raise
        times = {int(c["event"]): c for c in cases}
        ordered = sorted(times, key=lambda event: (times[event]["start_ms"], event))
        folds = []
        for number, block in enumerate(np.array_split(ordered[len(ordered) // 2:], 3), 1):
            heldout = [int(event) for event in block]
            start = min(times[event]["start_ms"] for event in heldout)
            train = [event for event in ordered if times[event]["end_ms"] < start]
            folds.append({"fold": number, "training_events": train, "heldout_events": heldout,
                          "training_latest_end": max((times[e]["end_ms"] for e in train), default=None),
                          "heldout_earliest_start": start})
        return folds


def _phase(progress, config):
    return int(np.searchsorted(config["phase_edges"], progress, side="right"))


def _correction_records(arrays, errors, config):
    groups = defaultdict(list)
    event_counts = Counter(int(case["event"]) for case, error in zip(arrays["cases"], errors)
                           if np.isfinite(error))
    for index, (case, error) in enumerate(zip(arrays["cases"], errors)):
        if np.isfinite(error):
            groups[(int(case["event"]), int(case["rank"]),
                    _phase(arrays["progress"][index], config))].append(index)
    records = []
    for (event, rank, phase), indices in groups.items():
        first = arrays["cases"][indices[0]]
        records.append({"event": event, "rank": rank, "phase": phase,
                        "start_ms": int(first["start_ms"]), "end_ms": int(first["end_ms"]),
                        "log_error": float(np.mean(errors[indices])),
                        "event_normalized_case_mass": len(indices) / event_counts[event],
                        "case_ids": [str(arrays["cases"][i]["case_id"]) for i in indices]})
    return sorted(records, key=lambda row: (row["end_ms"], row["event"], row["rank"], row["phase"]))


def correction_values(bundle, arrays):
    """Return frozen correction and case-level provenance without updating state."""
    config = bundle["config"]
    records = bundle.get("correction_records", ())
    result, audit = [], []
    for index, case in enumerate(arrays["cases"]):
        phase = _phase(arrays["progress"][index], config)
        eligible = [r for r in records if r["event"] != int(case["event"])
                    and r["rank"] == int(case["rank"]) and r["phase"] == phase
                    and r["end_ms"] < int(case["start_ms"])]
        eligible.sort(key=lambda row: (row["end_ms"], row["event"]))
        state = 0.
        for number, record in enumerate(eligible):
            state = (record["log_error"] if number == 0 else
                     (1 - config["ewma_alpha"]) * state + config["ewma_alpha"] * record["log_error"])
        n = len(eligible)
        strength = (n / (n + config["ewma_shrinkage_events"])
                    if config["correction"] == "shrinkage" else 1.)
        correction = float(np.clip(state * strength, -config["log_bias_clip"], config["log_bias_clip"]))
        if config["correction"] == "none":
            correction = 0.
        result.append(correction)
        audit.append({"case_id": str(case["case_id"]), "event": int(case["event"]),
                      "query_start_ms": int(case["start_ms"]), "rank": int(case["rank"]),
                      "phase": phase, "eligible_events": [r["event"] for r in eligible],
                      "latest_eligible_end_ms": max((r["end_ms"] for r in eligible), default=None),
                      "log_bias": correction})
    return np.asarray(result, dtype=float), audit


def fit_recipe(config: Mapping[str, Any], train_arrays: Mapping[str, Any],
               train_labels: Mapping[str, float]) -> dict[str, Any]:
    """Fit using supplied training labels and TRAIN curves; no file access."""
    _check_arrays(train_arrays, training=True)
    config = _config(config)
    times = _event_times(train_arrays)
    history, provenance = _snapshot_history(train_arrays.get("history", ()))
    for curve in history:
        if curve.event in times and times[curve.event] != (curve.start, curve.end):
            raise ValueError("Training cases and historical curves disagree on event dates")
    summaries = _summaries(history, config)
    actual = np.asarray([train_labels[c["case_id"]] for c in train_arrays["cases"]], dtype=float)
    current = np.asarray(train_arrays["current"], dtype=float)
    if not np.isfinite(actual).all() or (actual < current).any():
        raise ValueError("Final training scores must be finite and at least current")
    target = np.log(actual / current) + np.log(train_arrays["progress"])
    weights = event_weights(train_arrays["cases"])
    oof, folds = np.full(len(actual), np.nan), []
    if config["correction"] != "none":
        for schedule in _schedule(train_arrays["cases"]):
            train_events, heldout_events = set(schedule["training_events"]), set(schedule["heldout_events"])
            ti = np.asarray([i for i, c in enumerate(train_arrays["cases"])
                             if c["event"] in train_events], dtype=int)
            hi = np.asarray([i for i, c in enumerate(train_arrays["cases"])
                             if c["event"] in heldout_events], dtype=int)
            # Include TRAIN events without a prefix (e.g. 57/68) if completed
            # before the block; do not add earlier members of the heldout block.
            allowed = [row for row in summaries if row["end_ms"] < schedule["heldout_earliest_start"]]
            fold = schedule | {"train_indices": ti, "heldout_indices": hi,
                               "train_weights": weights[ti].copy(),
                               "history_events": [row["event"] for row in allowed],
                               "train_case_ids": [str(train_arrays["cases"][i]["case_id"]) for i in ti],
                               "heldout_case_ids": [str(train_arrays["cases"][i]["case_id"]) for i in hi]}
            if not len(ti):
                fold["status"] = "no_completed_training_cases"
                folds.append(fold)
                continue
            fitting, heldout = subset_arrays(train_arrays, ti, history=[]), subset_arrays(train_arrays, hi, history=[])
            design = ContextDesign.fit(fitting)
            train_frame, train_audit = _frame(fitting, design, allowed, config)
            heldout_frame, heldout_audit = _frame(heldout, design, allowed, config)
            prep, model = _fit_backbone(train_frame, target[ti], weights[ti], config)
            oof[hi] = _final_from_residual(model.predict(prep.transform(heldout_frame)), heldout, config)
            fold.update({"status": "fitted", "preprocessor": prep, "context_design": design,
                         "train_history_audit": train_audit, "heldout_history_audit": heldout_audit})
            folds.append(fold)
    errors = np.full(len(actual), np.nan)
    support = np.isfinite(oof)
    errors[support] = np.log(actual[support]) - np.log(oof[support])
    calibration_counts = Counter(case["event"] for case, valid in zip(train_arrays["cases"], support)
                                 if valid)
    calibration_weights = np.asarray([1. / calibration_counts[case["event"]] if valid else 0.
                                      for case, valid in zip(train_arrays["cases"], support)])
    records = _correction_records(train_arrays, errors, config)
    design = ContextDesign.fit(train_arrays)
    frame, train_audit = _frame(train_arrays, design, summaries, config)
    prep, model = _fit_backbone(frame, target, weights, config)
    bundle = {"config": config, "model": model, "preprocessor": prep, "context_design": design,
              "history": history, "history_provenance": provenance, "history_summaries": summaries,
              "feature_columns": list(frame.columns), "train_weights": weights,
              "train_case_ids": [str(c["case_id"]) for c in train_arrays["cases"]],
              "oof_folds": folds, "oof_predictions": oof, "oof_residuals": errors,
              "calibration_case_weights": calibration_weights,
              "correction_records": records}
    _, correction_audit = correction_values(bundle, train_arrays)
    masses = defaultdict(float)
    for case, weight in zip(train_arrays["cases"], weights):
        masses[str(case["event"])] += float(weight)
    fold_diagnostics = []
    for fold in folds:
        diagnostic = {key: value for key, value in fold.items()
                      if key not in {"train_indices", "heldout_indices", "train_weights",
                                     "preprocessor", "context_design"}}
        if fold["status"] == "fitted":
            diagnostic["preprocessing_medians"] = dict(fold["preprocessor"].medians)
            diagnostic["preprocessing_categories"] = {key: list(value) for key, value
                                                        in fold["preprocessor"].categories.items()}
        diagnostic["train_weights"] = fold["train_weights"].tolist()
        fold_diagnostics.append(diagnostic)
    bundle["diagnostics"] = {
        "supervised_event_mass": dict(masses), "training_cases": len(actual),
        "training_events": len(masses), "history_events": len(summaries),
        "history_provenance": provenance, "oof_schedule": config["oof_schedule"],
        "oof_available_cases": int(support.sum()), "oof_unavailable_cases": int((~support).sum()),
        "calibration_case_weights": calibration_weights.tolist(),
        "oof_folds": fold_diagnostics, "train_history_audit": train_audit,
        "correction_records": records, "train_correction_audit": correction_audit,
        "calibration_scope": "supplied TRAIN only; one uncorrected OOF error mean per event/rank/phase; "
                             "each query excludes own and not-yet-completed events",
    }
    return bundle


def predict_recipe(bundle: Mapping[str, Any], arrays: Mapping[str, Any]) -> np.ndarray:
    """Predict every blind case using only immutable fitted historical state."""
    _check_arrays(arrays)
    _event_times(arrays)
    if not arrays["cases"]:
        return np.empty(0, dtype=float)
    frame, _ = _frame(arrays, bundle["context_design"], bundle["history_summaries"], bundle["config"])
    raw = np.asarray(bundle["model"].predict(bundle["preprocessor"].transform(frame)))
    # OOF calibration errors refer to the same bounded uncorrected prediction.
    # Apply multiplicative correction once, then enforce the final score bounds.
    base = _final_from_residual(raw, arrays, bundle["config"])
    correction, _ = correction_values(bundle, arrays)
    residual = np.log(base / np.asarray(arrays["current"])) + np.log(arrays["progress"]) + correction
    return _final_from_residual(residual, arrays, bundle["config"])
