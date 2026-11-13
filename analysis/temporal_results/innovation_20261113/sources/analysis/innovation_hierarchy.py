"""Fixed type/band shrinkage recipes consuming injected blind-prefix arrays only.

Targets and OOF predictions are log residuals around current / progress.  No
dataset, filesystem, metadata store, or held-out labels are accessed here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd


def recipes() -> list[dict[str, Any]]:
    """The three choices were fixed before formal validation or test scoring."""
    common = {"iterations": 600, "depth": 5, "learning_rate": .035,
              "l2_leaf_reg": 12, "seed": 42, "thread_count": 2,
              "inner_folds": 3, "warmup_fraction": .5,
              "intercept_penalty": 5., "global_penalty": 10.,
              "context_penalty": 30., "interaction_penalty": 60.}
    return [common | {"name": "hierarchy_static_context", "mode": "static"},
            common | {"name": "hierarchy_additive_shrinkage", "mode": "additive"},
            common | {"name": "hierarchy_phase_shrinkage", "mode": "phase"}]


@dataclass
class FramePreprocessor:
    """Training-only schema, finite median imputation, and unknown-safe one-hot."""

    columns: tuple[str, ...]
    medians: dict[str, float]
    categories: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> FramePreprocessor:
        if frame.empty:
            raise ValueError("Cannot fit preprocessing on an empty training frame")
        medians, categories = {}, {}
        for name in frame.columns:
            series = frame[name]
            numeric = pd.to_numeric(series, errors="coerce")
            if series.notna().sum() == numeric.notna().sum():
                values = np.asarray(numeric, dtype=float)
                finite = values[np.isfinite(values)]
                medians[name] = float(np.median(finite)) if len(finite) else 0.
            else:
                categories[name] = tuple(sorted({str(v) for v in series if pd.notna(v)}))
        return cls(tuple(frame.columns), medians, categories)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = []
        frame = frame.reindex(columns=self.columns)
        for name, median in self.medians.items():
            values = np.asarray(pd.to_numeric(frame[name], errors="coerce"), dtype=float)
            blocks.append(np.where(np.isfinite(values), values, median)[:, None])
        for name, vocabulary in self.categories.items():
            values = [str(v) if pd.notna(v) else None for v in frame[name]]
            blocks.append(np.asarray([[float(v == category) for category in vocabulary]
                                      for v in values], dtype=float).reshape(len(frame), -1))
        return np.concatenate(blocks, axis=1) if blocks else np.ones((len(frame), 1))


@dataclass
class ContextDesign:
    """Shared global effects plus shrinkable type and fractional band effects.

    The optional interaction is low dimensional in time: each main effect is
    multiplied by two fixed phase functions, not by a sparse type-band tensor.
    Unknown categories receive zero group effect and retain global effects.
    """

    types: tuple[str, ...]
    bands: tuple[str, ...]
    interactions: bool = False

    @classmethod
    def fit(cls, arrays: Mapping[str, Any], interactions: bool = False) -> ContextDesign:
        types = tuple(sorted({str(v) for v in arrays["event_type"] if pd.notna(v)}))
        bands = tuple(sorted({str(b) for shares in arrays["band_shares"]
                              for b, value in shares.items()
                              if np.isfinite(value) and value > 0}))
        return cls(types, bands, interactions)

    @property
    def column_names(self) -> tuple[str, ...]:
        groups = tuple("type:" + v for v in self.types) + tuple("band:" + v for v in self.bands)
        result = ("global:intercept", "global:rank", "global:progress", "global:rush") + groups
        if self.interactions:
            result += tuple(v + "*progress" for v in groups) + tuple(v + "*rush" for v in groups)
        return result

    def transform(self, arrays: Mapping[str, Any]) -> np.ndarray:
        n = len(arrays["cases"])
        progress = np.clip(np.asarray(arrays["progress"], dtype=float), 0., 1.)
        phase = 2 * progress - 1
        remaining = np.maximum(np.asarray(arrays["remaining_h"], dtype=float), 0.)
        rush = np.exp(-remaining / 12.)
        rank = np.asarray(arrays["rank"], dtype=float)
        global_effects = np.column_stack([np.ones(n), rank, phase, rush])
        type_lookup = {v: i for i, v in enumerate(self.types)}
        band_lookup = {v: i + len(self.types) for i, v in enumerate(self.bands)}
        groups = np.zeros((n, len(self.types) + len(self.bands)))
        for i, value in enumerate(arrays["event_type"]):
            j = type_lookup.get(str(value))
            if j is not None:
                groups[i, j] = 1.
        for i, shares in enumerate(arrays["band_shares"]):
            for band, value in shares.items():
                j = band_lookup.get(str(band))
                if j is not None and np.isfinite(value) and value > 0:
                    groups[i, j] += value
        blocks = [global_effects, groups]
        if self.interactions:
            blocks.extend([groups * phase[:, None], groups * rush[:, None]])
        return np.concatenate(blocks, axis=1)


def _event_weights(cases: list[dict[str, Any]], *, normalize_mean: bool) -> np.ndarray:
    counts = Counter(c["event"] for c in cases)
    weights = np.asarray([1. / counts[c["event"]] for c in cases])
    return weights / weights.mean() if normalize_mean and len(weights) else weights


def _event_times(arrays: Mapping[str, Any]) -> dict[Any, tuple[float, float]]:
    result = {}
    for case in arrays["cases"]:
        time = (float(case["start_ms"]), float(case["end_ms"]))
        if not np.isfinite(time).all() or time[1] <= time[0]:
            raise ValueError("Every event requires finite start_ms < end_ms")
        previous = result.setdefault(case["event"], time)
        if previous != time:
            raise ValueError("Cases from one event disagree on start/end timestamps")
    return result


def chronological_oof_splits(arrays: Mapping[str, Any], *, blocks: int = 3,
                             warmup_fraction: float = .5) -> list[dict[str, Any]]:
    """Expanding OOF blocks, restricted to completed earlier training events.

    This is inner calibration on the *supplied* training events. Official
    memberships and the runner's outer diagnostic folds are never modified.
    All cases of each held-out event stay together, including both ranks.
    """
    if blocks < 1 or not 0 < warmup_fraction < 1:
        raise ValueError("Invalid chronological calibration schedule")
    times = _event_times(arrays)
    events = sorted(times, key=lambda event: (times[event][0], str(event)))
    warmup = max(1, int(len(events) * warmup_fraction))
    folds = []
    for number, heldout_array in enumerate(np.array_split(np.asarray(events[warmup:], dtype=object), blocks)):
        heldout_events = heldout_array.tolist()
        if not heldout_events:
            continue
        start = min(times[event][0] for event in heldout_events)
        heldout_set = set(heldout_events)
        train_events = [event for event in events
                        if event not in heldout_set and times[event][1] < start]
        train_set = set(train_events)
        train_indices = np.asarray([i for i, c in enumerate(arrays["cases"])
                                    if c["event"] in train_set], dtype=int)
        heldout_indices = np.asarray([i for i, c in enumerate(arrays["cases"])
                                      if c["event"] in heldout_set], dtype=int)
        folds.append({"fold": number, "heldout_start_ms": start,
                      "train_events": train_events, "heldout_events": heldout_events,
                      "train_indices": train_indices, "heldout_indices": heldout_indices,
                      "train_case_ids": [arrays["cases"][i]["case_id"] for i in train_indices],
                      "heldout_case_ids": [arrays["cases"][i]["case_id"] for i in heldout_indices]})
    return folds


def _slice(arrays: Mapping[str, Any], indices: np.ndarray) -> dict[str, Any]:
    return {"cases": [arrays["cases"][i] for i in indices],
            "base": arrays["base"].iloc[indices].reset_index(drop=True),
            "band_shares": [arrays["band_shares"][i] for i in indices],
            **{name: np.asarray(arrays[name])[indices] for name in
               ("current", "progress", "remaining_h", "rank", "event_type")}}


def _check_arrays(arrays: Mapping[str, Any], *, training: bool = False) -> None:
    n = len(arrays["cases"])
    if training and not n:
        raise ValueError("Training arrays are empty")
    for name in ("base", "current", "progress", "remaining_h", "rank", "event_type", "band_shares"):
        if len(arrays[name]) != n:
            raise ValueError("Misaligned arrays: " + name)
    case_ids = [case["case_id"] for case in arrays["cases"]]
    if len(set(case_ids)) != n:
        raise ValueError("Duplicate case identifiers")
    current = np.asarray(arrays["current"], dtype=float)
    progress = np.asarray(arrays["progress"], dtype=float)
    if not np.isfinite(current).all() or (current <= 0).any():
        raise ValueError("Current scores must be finite and positive")
    if not np.isfinite(progress).all() or (progress <= 0).any() or (progress > 1).any():
        raise ValueError("Progress must lie in (0, 1]")
    if not np.isfinite(np.asarray(arrays["remaining_h"], dtype=float)).all():
        raise ValueError("Remaining hours must be finite")
    if not np.isin(np.asarray(arrays["rank"]), [0, 1]).all():
        raise ValueError("Shared hierarchy rank must be the 0/1 rank indicator")
    if training:
        _event_times(arrays)


@dataclass
class _ConstantRegressor:
    value: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value)


def _fit_backbone(frame: pd.DataFrame, target: np.ndarray, weights: np.ndarray,
                  config: Mapping[str, Any]) -> tuple[FramePreprocessor, Any]:
    preprocessor = FramePreprocessor.fit(frame)
    x = preprocessor.transform(frame)
    if len(x) < 2 or not np.any(np.ptp(x, axis=0) > 0):
        model = _ConstantRegressor(float(np.average(target, weights=weights)))
    else:
        model = CatBoostRegressor(iterations=int(config["iterations"]), depth=int(config["depth"]),
                                  learning_rate=float(config["learning_rate"]),
                                  l2_leaf_reg=float(config["l2_leaf_reg"]),
                                  random_seed=int(config["seed"]), thread_count=int(config["thread_count"]),
                                  verbose=False, allow_writing_files=False, allow_const_label=True)
        model.fit(x, target, sample_weight=weights)
    return preprocessor, model


def _static_frame(arrays: Mapping[str, Any], design: ContextDesign) -> pd.DataFrame:
    frame = arrays["base"].reset_index(drop=True).copy()
    groups = design.transform(arrays)[:, 4:]
    context = pd.DataFrame(groups, columns=["hierarchy_context_" + str(i) for i in range(groups.shape[1])])
    if set(frame.columns) & set(context.columns):
        raise ValueError("Own-line features collide with hierarchy context names")
    return pd.concat([frame, context], axis=1)


def fit_recipe(config: Mapping[str, Any], train_arrays: Mapping[str, Any],
               train_labels: Mapping[str, float]) -> dict[str, Any]:
    """Fit a fixed recipe using only labels belonging to injected training cases."""
    _check_arrays(train_arrays, training=True)
    config = recipes()[0] | dict(config)
    mode = config["mode"]
    if mode not in {"static", "additive", "phase"}:
        raise ValueError("Unknown hierarchy recipe mode")
    if int(config["iterations"]) < 1 or int(config["inner_folds"]) < 1:
        raise ValueError("Iterations and inner_folds must be positive")
    for key in ("intercept_penalty", "global_penalty", "context_penalty", "interaction_penalty"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Shrinkage penalties must be finite and positive")
    actual = np.asarray([train_labels[c["case_id"]] for c in train_arrays["cases"]], dtype=float)
    current = np.asarray(train_arrays["current"], dtype=float)
    if not np.isfinite(actual).all() or (actual < current).any():
        raise ValueError("Training final scores must be finite and at least current")
    target = np.log(actual / current) + np.log(train_arrays["progress"])
    # This one vector is global to the supplied fit; inner folds slice unchanged.
    weights = _event_weights(train_arrays["cases"], normalize_mean=True)
    bundle = {"config": config, "train_weights": weights,
              "train_case_ids": [c["case_id"] for c in train_arrays["cases"]],
              "weighting": "backbone: event totals equal; inner slices preserve global weights; "
                           "correction: represented event totals exactly one",
              "oof_folds": []}
    if mode == "static":
        design = ContextDesign.fit(train_arrays)
        frame = _static_frame(train_arrays, design)
        preprocessor, model = _fit_backbone(frame, target, weights, config)
        bundle.update({"context_design": design, "preprocessor": preprocessor, "model": model})
        return bundle

    oof = np.full(len(target), np.nan)
    folds = chronological_oof_splits(train_arrays, blocks=int(config["inner_folds"]),
                                     warmup_fraction=float(config["warmup_fraction"]))
    for fold in folds:
        train_indices, heldout_indices = fold["train_indices"], fold["heldout_indices"]
        fold["train_weights"] = weights[train_indices].copy()
        if not len(train_indices):
            fold.update({"status": "no_completed_history", "preprocessor": None, "model": None})
            continue
        preprocessor, model = _fit_backbone(train_arrays["base"].iloc[train_indices],
                                            target[train_indices], weights[train_indices], config)
        oof[heldout_indices] = model.predict(preprocessor.transform(train_arrays["base"].iloc[heldout_indices]))
        fold.update({"status": "fitted", "preprocessor": preprocessor, "model": model})

    indices = np.flatnonzero(np.isfinite(oof))
    calibration = _slice(train_arrays, indices)
    # Zero-support categories contribute zero; even the correction vocabulary is
    # learned from calibration training cases rather than prediction queries.
    design = ContextDesign.fit(calibration, interactions=mode == "phase")
    penalties = np.full(len(design.column_names), float(config["context_penalty"]))
    penalties[0], penalties[1:4] = config["intercept_penalty"], config["global_penalty"]
    group_count = len(design.types) + len(design.bands)
    if mode == "phase":
        penalties[4 + group_count:] = config["interaction_penalty"]
    coefficients = np.zeros(len(penalties))
    correction_weights = np.zeros(len(target))
    residuals = np.full(len(target), np.nan)
    if len(indices):
        x = design.transform(calibration)
        local_weights = _event_weights(calibration["cases"], normalize_mean=False)
        correction_weights[indices] = local_weights
        residuals[indices] = target[indices] - oof[indices]
        coefficients = np.linalg.solve(x.T @ (local_weights[:, None] * x) + np.diag(penalties),
                                       x.T @ (local_weights * residuals[indices]))
    # The final backbone learns all supplied training events, without held-out
    # labels or a refit of calibration on its own in-sample residuals.
    preprocessor, model = _fit_backbone(train_arrays["base"], target, weights, config)
    bundle.update({"preprocessor": preprocessor, "model": model, "context_design": design,
                   "oof_folds": folds, "oof_predictions": oof, "oof_residuals": residuals,
                   "correction_weights": correction_weights, "correction_penalties": penalties,
                   "correction_coefficients": coefficients,
                   "calibration_events": sorted({c["event"] for c in calibration["cases"]}, key=str),
                   "calibration_case_ids": [c["case_id"] for c in calibration["cases"]]})
    return bundle


def predict_recipe(bundle: Mapping[str, Any], arrays: Mapping[str, Any]) -> np.ndarray:
    """Return one finite positive final prediction for every supplied blind case."""
    _check_arrays(arrays)
    if not len(arrays["cases"]):
        return np.zeros(0, dtype=float)
    if bundle["config"]["mode"] == "static":
        frame = _static_frame(arrays, bundle["context_design"])
    else:
        frame = arrays["base"]
    residual = np.asarray(bundle["model"].predict(bundle["preprocessor"].transform(frame)))
    if bundle["config"]["mode"] != "static":
        residual += bundle["context_design"].transform(arrays) @ bundle["correction_coefficients"]
    log_growth = np.clip(-np.log(arrays["progress"]) + residual, 0., 4.)
    current = np.asarray(arrays["current"], dtype=float)
    prediction = np.maximum(current, np.exp(np.minimum(np.log(current) + log_growth,
                                                       np.log(np.finfo(float).max))))
    if not np.isfinite(prediction).all():
        raise ValueError("Hierarchy model produced a nonfinite prediction")
    return prediction
