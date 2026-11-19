"""Fixed historical retrieval recipes over supplied blind-prefix arrays only.

The runner owns dataset access and the allowed training-history boundary. This
module never loads datasets, metadata, labels, or curves from a filesystem. A
fitted bundle snapshots its supplied history; prediction ignores query history.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


HOUR = 3_600_000
SHAPE_FRACTIONS = np.asarray([0.25, 0.5, 0.75, 0.9])
NEIGHBORS = 5


def recipes():
    """The complete, fixed menu; the runner seals this before validation."""
    common = {
        "family": "retrieval", "iterations": 600, "depth": 5,
        "learning_rate": 0.035, "l2_leaf_reg": 12, "seed": 42, "thread_count": 2,
        "neighbors": NEIGHBORS, "shape_fractions": SHAPE_FRACTIONS.tolist(),
        "scale_distance_weight": 0.15, "type_distance_weight": 0.2,
        "history_alignment": "actual query progress; same rank; different event; end < query start",
        "history_prefix": "at least 4 points; at most 60 minutes lag; at least 2 joint shape anchors",
        "fallback": "linear prior when fewer than 5 eligible historical events",
        "target": "log final / prior", "final_log_growth_bounds": [0.0, 4.0],
        "weights": "each event has equal total weight",
    }
    return [common | {"name": f"retrieval_{mode}", "mode": mode}
            for mode in ("control", "features", "residual")]


def event_weights(events):
    """Mean-one case weights, with identical summed weight for each event."""
    counts = Counter(events)
    weights = np.asarray([1.0 / counts[event] for event in events], dtype=float)
    return weights / weights.mean() if len(weights) else weights


@dataclass(frozen=True)
class HistoricalCurve:
    """Self-contained, joblib-safe copy of one permitted training curve."""
    event: int
    rank: int
    kind: str
    start: int
    end: int
    times: np.ndarray
    scores: np.ndarray

    @property
    def duration(self):
        return self.end - self.start

    @property
    def final(self):
        return float(self.scores[-1])


def _check_history(history):
    keys = [(int(curve.event), int(curve.rank)) for curve in history]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate event/rank in permitted history library")


def _snapshot_history(history):
    history = list(history)
    _check_history(history)
    library, provenance = [], []
    for curve in history:
        times = np.array(curve.times, dtype=np.int64, copy=True)
        scores = np.array(curve.scores, dtype=float, copy=True)
        if (times.ndim != 1 or scores.ndim != 1 or len(times) != len(scores)
                or not len(times) or np.any(np.diff(times) <= 0)
                or np.any(~np.isfinite(scores)) or np.any(scores < 0)
                or curve.end <= curve.start or scores[-1] <= 0):
            raise ValueError(f"Invalid training history curve: {curve.event}/{curve.rank}")
        times.setflags(write=False)
        scores.setflags(write=False)
        frozen = HistoricalCurve(int(curve.event), int(curve.rank), str(curve.kind),
                                 int(curve.start), int(curve.end), times, scores)
        record = {"event": frozen.event, "rank": frozen.rank, "kind": frozen.kind,
                  "start_ms": frozen.start, "end_ms": frozen.end,
                  "observations": len(times), "final": frozen.final}
        digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode())
        digest.update(times.astype("<i8", copy=False).tobytes())
        digest.update(scores.astype("<f8", copy=False).tobytes())
        provenance.append(record | {"sha256": digest.hexdigest()})
        library.append(frozen)
    return tuple(library), provenance


def _shape(times, scores, start, matched_at):
    result = np.full(len(SHAPE_FRACTIONS), np.nan)
    if not len(times) or scores[-1] <= 0:
        return result
    checkpoints = start + SHAPE_FRACTIONS * (matched_at - start)
    observed = (checkpoints >= times[0]) & (checkpoints <= times[-1])
    result[observed] = np.interp(checkpoints[observed], times, scores) / scores[-1]
    return result


def _linear_prior(arrays):
    current = np.asarray(arrays["current"], dtype=float)
    progress = np.asarray(arrays["progress"], dtype=float)
    if (len(current) != len(arrays["cases"]) or current.shape != progress.shape
            or np.any(~np.isfinite(current)) or np.any(current <= 0)
            or np.any(~np.isfinite(progress)) or np.any(progress <= 0)):
        raise ValueError("Cases require aligned finite positive current and progress")
    # The training prior remains true linear extrapolation even at very early
    # progress. Only the final prediction decoder applies the growth bound.
    return current / np.minimum(progress, 1.0)


def retrieval_features(arrays, history):
    """Return aligned retrieval statistics, including explicit sparse fallback.

    Each candidate is the same rank and a different event that ended strictly
    before this query started. Shape interpolation sees only observations up to
    the matched progress. Historical final scores enter multipliers only after
    this filter. Five neighbors are selected by shape + .15 scale + .2 type
    distance, ties by event ID. Sparse queries retain a linear prior and zero
    selected neighbors; eligible count still reports their available coverage.
    """
    history = list(history)
    _check_history(history)
    by_rank = {}
    for curve in history:
        by_rank.setdefault(int(curve.rank), []).append(curve)
    linear = _linear_prior(arrays)
    if len(arrays["samples"]) != len(linear):
        raise ValueError("Samples must align with cases")
    rows = []
    for index, query in enumerate(arrays["samples"]):
        case = arrays["cases"][index]
        query_curve = query["curve"]
        start = int(case["start_ms"])
        now = min(int(query["now"]), int(query.get("cutoff", query["now"])))
        # Defensive truncation makes an accidentally attached suffix inert.
        times = np.asarray(query["times"])
        scores = np.asarray(query["scores"], dtype=float)
        prefix = times <= now
        times, scores = times[prefix], scores[prefix]
        shape = _shape(times, scores, start, now)
        current = float(arrays["current"][index])
        progress = float(arrays["progress"][index])
        candidates, past, quality = [], 0, 0
        for curve in by_rank.get(int(case["rank"]), ()):
            if int(curve.event) == int(case["event"]) or curve.end >= start:
                continue
            past += 1
            matched_at = curve.start + progress * curve.duration
            stop = int(np.searchsorted(curve.times, matched_at, side="right"))
            if stop < 4:
                continue
            ht, hs = curve.times[:stop], curve.scores[:stop]
            lag = (matched_at - ht[-1]) / 60_000.0
            if hs[-1] <= 0 or not np.isfinite(hs[-1]) or lag > 60:
                continue
            quality += 1
            historical_shape = _shape(ht, hs, curve.start, matched_at)
            joint = np.isfinite(shape) & np.isfinite(historical_shape)
            if joint.sum() < 2:
                continue
            shape_distance = float(np.abs(shape[joint] - historical_shape[joint]).mean())
            scale_distance = float(abs(np.log(current / float(hs[-1]))))
            type_mismatch = float(str(query_curve.kind) != str(curve.kind))
            distance = shape_distance + 0.15 * scale_distance + 0.2 * type_mismatch
            candidates.append({
                "event": int(curve.event), "distance": distance,
                "shape_distance": shape_distance, "scale_distance": scale_distance,
                "type_match": 1.0 - type_mismatch,
                "multiplier": max(1.0, float(curve.final) / float(hs[-1])),
                "lag_minutes": float(lag),
                "observed_progress": float((ht[-1] - curve.start) / curve.duration),
                "age_days": float((start - curve.end) / (24 * HOUR)),
            })
        fallback = len(candidates) < NEIGHBORS
        selected = [] if fallback else sorted(candidates, key=lambda x: (x["distance"], x["event"]))[:NEIGHBORS]
        multipliers = np.asarray([item["multiplier"] for item in selected], dtype=float)
        prior = linear[index] if fallback else current * float(np.median(multipliers))
        log_multipliers = np.log(multipliers) if selected else np.array([np.log(prior / current)])
        p10, p25, p50, p75, p90 = np.quantile(log_multipliers, [0.1, 0.25, 0.5, 0.75, 0.9])
        row = {
            "retrieval_prior": float(prior), "retrieval_fallback": int(fallback),
            "retrieval_past_count": past, "retrieval_prefix_quality_count": quality,
            "retrieval_eligible_count": len(candidates), "retrieval_neighbor_count": len(selected),
            "retrieval_eligible_fraction": len(candidates) / past if past else 0.0,
            "retrieval_query_lag_minutes": max(0.0, (int(query.get("cutoff", now)) - now) / 60_000.0),
            "retrieval_log_multiplier_p10": float(p10), "retrieval_log_multiplier_p25": float(p25),
            "retrieval_log_multiplier_median": float(p50), "retrieval_log_multiplier_p75": float(p75),
            "retrieval_log_multiplier_p90": float(p90), "retrieval_log_multiplier_iqr": float(p75 - p25),
            "retrieval_log_multiplier_std": float(np.std(log_multipliers)),
        }
        for source, prefix_name in (("distance", "distance"), ("shape_distance", "shape_distance"),
                                    ("scale_distance", "scale_distance"), ("lag_minutes", "lag"),
                                    ("observed_progress", "observed_progress"), ("age_days", "age_days")):
            values = [item[source] for item in selected]
            suffix = "_minutes" if prefix_name == "lag" else ""
            for aggregation, operation in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                row[f"retrieval_{prefix_name}_{aggregation}{suffix}"] = float(operation(values)) if values else np.nan
        row["retrieval_type_match_fraction"] = float(np.mean([item["type_match"] for item in selected])) if selected else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_frame(arrays, history, mode):
    base = arrays["base"].reset_index(drop=True).copy()
    if len(base) != len(arrays["cases"]):
        raise ValueError("Base features must align with cases")
    linear = _linear_prior(arrays)
    if mode == "control":
        return base, linear
    retrieval = retrieval_features(arrays, history)
    if set(base.columns) & set(retrieval.columns):
        raise ValueError("Base features overlap retrieval column names")
    prior = retrieval["retrieval_prior"].to_numpy() if mode == "residual" else linear
    return pd.concat([base, retrieval], axis=1), prior


def _coerce_frame(frame, columns, categorical):
    result = frame.reindex(columns=columns).copy()
    for name in columns:
        if name in categorical:
            result[name] = result[name].where(result[name].notna(), "__missing__").astype(str)
        else:
            result[name] = pd.to_numeric(result[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return result


def fit_recipe(config, train_arrays, train_labels):
    """Fit one sealed recipe; only this function receives answer values."""
    config = dict(config)
    mode = config.get("mode")
    if mode not in {"control", "features", "residual"}:
        raise ValueError(f"Unknown retrieval recipe mode: {mode}")
    fixed = next(recipe for recipe in recipes() if recipe["mode"] == mode)
    for key, value in fixed.items():
        if key != "iterations" and key in config and config[key] != value:
            raise ValueError(f"Fixed retrieval recipe field cannot change: {key}")
    config = fixed | config
    if int(config["iterations"]) != config["iterations"] or config["iterations"] < 1:
        raise ValueError("Iterations must be a positive integer")
    if not len(train_arrays["cases"]):
        raise ValueError("Cannot fit an empty training fold")
    library, provenance = _snapshot_history(train_arrays.get("history", ()))
    frame, prior = _feature_frame(train_arrays, library, mode)
    columns = list(frame.columns)
    categorical = [name for name in columns
                   if not pd.api.types.is_numeric_dtype(frame[name]) and frame[name].notna().any()]
    numeric = [name for name in columns if name not in categorical]
    frame = _coerce_frame(frame, columns, categorical)
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    x = preprocessor.fit_transform(frame)
    actual = np.asarray([train_labels[case["case_id"]] for case in train_arrays["cases"]], dtype=float)
    if np.any(~np.isfinite(actual)) or np.any(actual <= 0):
        raise ValueError("Training answers must be finite and positive")
    target = np.log(actual / prior)
    weights = event_weights([case["event"] for case in train_arrays["cases"]])
    model = CatBoostRegressor(
        iterations=int(config.get("iterations", 600)), depth=5, learning_rate=0.035,
        l2_leaf_reg=12, random_seed=42, thread_count=2, loss_function="RMSE",
        allow_const_label=True, verbose=False, allow_writing_files=False,
    )
    model.fit(x, target, sample_weight=weights)
    return {"config": config, "model": model, "preprocessor": preprocessor,
            "feature_columns": columns, "categorical_columns": categorical,
            "history": library, "history_provenance": provenance,
            "training_case_ids": [case["case_id"] for case in train_arrays["cases"]],
            "training_weights": weights}


def predict_recipe(bundle, arrays):
    """Return finite final scores in input order, never below current score."""
    if not len(arrays["cases"]):
        return np.empty(0, dtype=float)
    frame, prior = _feature_frame(arrays, bundle["history"], bundle["config"]["mode"])
    frame = _coerce_frame(frame, bundle["feature_columns"], bundle["categorical_columns"])
    correction = np.asarray(bundle["model"].predict(bundle["preprocessor"].transform(frame)))
    current = np.asarray(arrays["current"], dtype=float)
    prediction = current * np.exp(np.clip(np.log(prior / current) + correction, 0.0, 4.0))
    if prediction.shape != current.shape or np.any(~np.isfinite(prediction)):
        raise ValueError("Invalid retrieval predictions")
    return np.maximum(prediction, current)
