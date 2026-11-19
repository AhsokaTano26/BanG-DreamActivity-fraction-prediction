"""Pinned local foundation-model controls for the frozen two-rank V2 protocol.

This module accepts blind arrays; it never opens project datasets or answers.
Bundles contain training contexts and relative checkpoint references, not copied
pretrained tensors. A single ephemeral runtime model is retained at a time.
"""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path("analysis/foundation_assets/20261114")
SEEDS = (42, 123, 2026)
HOUR = 3_600_000
MODEL_SPECS = {
    "tabpfn_v2": {
        "repository": "Prior-Labs/TabPFN-v2-reg",
        "revision": "02a44705ff7790b50cded404bc59adb94da456a1",
        "package": "tabpfn", "version": "7.1.1",
        "files": {
            "tabpfn-v2-regressor.ckpt": "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736",
            "config.json": "365aefad99bdcc8842403b850cdd6ef5a485e11af6742243b305f0e63cd30a83",
        },
    },
    "chronos2": {
        "repository": "amazon/chronos-2",
        "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "package": "chronos-forecasting", "version": "2.3.2",
        "files": {
            "model.safetensors": "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
            "config.json": "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031",
        },
    },
    "timesfm25": {
        "repository": "google/timesfm-2.5-200m-pytorch",
        "revision": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
        "package": "timesfm", "version": "2.0.2",
        "files": {
            "model.safetensors": "2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8",
            "config.json": "cd3315b760d5cc7e278d7afdf41b897031ced888fc6c115bd9b3ac0ea2c47408",
        },
    },
}
_RUNTIME = {"key": None, "model": None}


def _local_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def file_identity(path):
    path = _local_path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    try:
        stored = str(path.relative_to(ROOT))
    except ValueError:
        stored = str(path)
    return {"path": stored, "sha256": digest, "bytes": path.stat().st_size}


def verify_file(identity):
    if file_identity(identity["path"]) != identity:
        raise ValueError("Local foundation checkpoint fingerprint changed: " + identity["path"])


def _software():
    names = ("tabpfn", "tabpfn-common-utils", "chronos-forecasting", "timesfm", "torch",
             "numpy", "pandas", "scipy", "scikit-learn", "transformers", "huggingface-hub",
             "safetensors", "accelerate", "einops", "pydantic", "pydantic-settings", "joblib")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def availability():
    """No imports of torch and no network calls; report unavailable candidates."""
    software = _software()
    result = {}
    for backend, spec in MODEL_SPECS.items():
        reasons = []
        if software[spec["package"]] != spec["version"]:
            reasons.append(f"Requires {spec['package']}=={spec['version']}")
        for filename, digest in spec["files"].items():
            path = ROOT / ASSETS / backend / filename
            if not path.is_file():
                reasons.append("Missing local file: " + str(path.relative_to(ROOT)))
            elif file_identity(path)["sha256"] != digest:
                reasons.append("Checkpoint SHA256 mismatch: " + str(path.relative_to(ROOT)))
        result[backend] = {"available": not reasons, "reasons": reasons}
    return result


def provenance():
    software = _software()
    statuses = availability()
    return {
        "software": software, "availability": statuses,
        "python": platform.python_version(), "platform": platform.platform(),
        "models": {
            backend: {**spec, "files": [file_identity(ASSETS / backend / filename)
                        for filename in spec["files"] if (ROOT / ASSETS / backend / filename).is_file()]}
            for backend, spec in MODEL_SPECS.items()
        },
        "inference": "local only; telemetry disabled; HF and transformers offline",
        "asset_archive": str(ASSETS),
        "devices": {"tabpfn_v2": "mps", "chronos2": "cpu", "timesfm25": "cpu"},
    }


def recipes():
    common = {"family": "foundation", "seeds": list(SEEDS), "torch_threads": 2,
              "checkpoint_policy": "pinned local weights; verify SHA256; offline only"}
    tabular = common | {
        "name": "foundation_tabpfn_v2", "backend": "tabpfn_v2", "zero_shot": False,
        "feature_mode": "base",
        "device": "mps", "n_estimators": 1, "fit_mode": "fit_with_cache", "query_batch_size": 256,
        "target": "log(final/current)", "final_log_growth_bounds": [0.0, 4.0],
        "context_balance": "max cases per event; retain all originals; balanced cyclic repeats plus seeded remainder",
        "seed_role": "three independent event-balanced in-context ensembles; no task weight updates",
        "training_predictions": "resubstitution queries include own labels in context; never OOF calibration",
    }
    temporal = common | {
        "zero_shot": True, "device": "cpu", "step_hours": 2.0, "max_context": 256,
        "max_horizon": 256, "batch_size": 8, "torch_compile": False,
        "input": "cumulative score/current; prefix-only linear interpolation; grid anchored at observed now",
        "missingness": "audit age and gaps >6h; keep all cases; interpolate only between observed prefix points",
        "long_event_policy": "increase step by integer multiples of 2h to preserve full span and end within caps",
        "output": "median cumulative forecast; monotonic projection; linear interpolation to exact event end",
        "seed_role": "deterministic repeated inference seeds; not three trained models",
        "cross_learning": False,
    }
    return [tabular, tabular | {"name": "foundation_tabpfn_v2_context", "feature_mode": "static_context"},
            temporal | {"name": "foundation_chronos2", "backend": "chronos2"},
            temporal | {"name": "foundation_timesfm25", "backend": "timesfm25"}]


def balanced_context_indices(cases, seed):
    """Equal context mass per event without throwing away any original case."""
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Expected nonempty unique physical training cases")
    events = np.asarray([int(c["event"]) for c in cases])
    counts = Counter(events.tolist())
    maximum = max(counts.values())
    rng = np.random.default_rng(seed)
    chunks = []
    for event in sorted(counts):
        original = np.flatnonzero(events == event)
        repeat, remainder = divmod(maximum, len(original))
        chunks.append(np.concatenate([np.tile(original, repeat), rng.permutation(original)[:remainder]]))
    indices = rng.permutation(np.concatenate(chunks)).astype(np.int64)
    return indices, {"seed": int(seed), "original_cases": len(cases), "context_rows": len(indices),
                     "duplicated_rows": len(indices) - len(cases), "independent_events": len(counts),
                     "context_rows_per_event": maximum, "original_cases_per_event": {str(k): v for k, v in counts.items()},
                     "all_original_cases_retained": len(set(indices.tolist())) == len(cases)}


def fit_preprocessor(frame):
    """Compact training-only imputation/encoding, with explicit unknown category."""
    columns = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) or not series.notna().any():
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            valid = values[np.isfinite(values)]
            columns.append({"name": name, "kind": "numeric", "fill": float(np.median(valid)) if len(valid) else 0.0})
        else:
            columns.append({"name": name, "kind": "category",
                            "values": sorted(set(series.dropna().astype(str)))})
    return {"columns": columns}


def build_feature_frame(arrays, mode, design=None):
    """Freeze type/band vocabularies on train; unknown/missing groups stay zero."""
    if mode == "base":
        return arrays["base"].reset_index(drop=True).copy(), None
    if mode != "static_context":
        raise ValueError("Unknown foundation feature mode")
    from analysis.innovation_hierarchy import ContextDesign, _static_frame
    if design is None:
        design = ContextDesign.fit(arrays, interactions=False)
    return _static_frame(arrays, design), design


def transform_features(transform, frame):
    values = []
    for column in transform["columns"]:
        series = frame[column["name"]]
        if column["kind"] == "numeric":
            data = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            values.append(np.where(np.isfinite(data), data, column["fill"]))
        else:
            lookup = {value: i for i, value in enumerate(column["values"])}
            values.append(np.asarray([lookup.get(str(value), -1) if pd.notna(value) else -1 for value in series]))
    return np.stack(values, axis=1).astype(np.float64)


def resample_prefix(sample, *, step_hours, max_context, max_horizon):
    """Regularize only supplied observations; nominal cutoff adds no observations."""
    curve = sample["curve"]
    times = np.asarray(curve.times, dtype=np.float64)
    scores = np.asarray(curve.scores, dtype=np.float64)
    now, cutoff, end = float(sample["now"]), float(sample["cutoff"]), float(curve.end)
    current = float(sample["score"])
    if (times.ndim != 1 or not len(times) or times.shape != scores.shape
            or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0)
            or np.any(~np.isfinite(scores)) or np.any(scores < 0)
            or times[-1] != now or now > cutoff or end < now or current <= 0
            or not np.isfinite(current) or scores[-1] != current
            or step_hours <= 0 or max_context < 2 or max_horizon < 1):
        raise ValueError("Expected finite chronological blind prefix ending at observed now")
    earliest = max(float(curve.start), times[0])
    base_step = float(step_hours) * HOUR
    multiplier = max(1, math.ceil((now - earliest) / (max_context - 1) / base_step),
                     math.ceil((end - now) / max_horizon / base_step))
    step = base_step * multiplier
    grid = now - step * np.arange(int(math.floor((now - earliest) / step)), -1, -1)
    context = np.interp(grid, times, scores) / current
    end_steps = (end - now) / step
    gaps = np.diff(times)
    large_gap = np.zeros(len(grid), dtype=bool)
    for left, right in zip(times[:-1], times[1:]):
        if right - left > 6 * HOUR:
            large_gap |= (grid > left) & (grid < right)
    return {"context": context.astype(np.float32), "horizon": int(math.ceil(end_steps)),
            "end_steps": end_steps, "current": current,
            "diagnostics": {"case_id": sample.get("case_id"), "observed_at": int(now),
                            "observation_age_hours": (cutoff - now) / HOUR,
                            "max_observation_gap_hours": float(np.max(gaps) / HOUR) if len(gaps) else 0.0,
                            "grid_points_in_large_gap": int(large_gap.sum()),
                            "context_points": len(context), "effective_step_hours": step / HOUR,
                            "forecast_steps": int(math.ceil(end_steps)), "end_steps": end_steps}}


def decode_endpoint(series, normalized_forecast):
    horizon = series["horizon"]
    if horizon == 0:
        return series["current"]
    values = np.asarray(normalized_forecast, dtype=float).reshape(-1)
    if len(values) < horizon or not np.isfinite(values[:horizon]).all():
        raise ValueError("Foundation forecast must cover every requested step with finite values")
    cumulative = np.maximum.accumulate(np.r_[1.0, values[:horizon]])
    result = np.interp(series["end_steps"], np.arange(horizon + 1), cumulative) * series["current"]
    if not np.isfinite(result):
        raise ValueError("Nonfinite foundation endpoint")
    return max(float(result), series["current"])


def _offline_torch():
    for name in ("TABPFN_DISABLE_TELEMETRY", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    import torch
    torch.set_num_threads(2)
    return torch


def _runtime(bundle, seed=None):
    config = bundle["config"]
    for identity in bundle["checkpoint"]["files"]:
        verify_file(identity)
    key = (bundle["context_id"] if not config["zero_shot"] else config["backend"], seed)
    if _RUNTIME["key"] == key:
        return _RUNTIME["model"]
    torch = _offline_torch()
    _RUNTIME.update(key=None, model=None)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    root = ROOT / ASSETS / config["backend"]
    if config["backend"] == "tabpfn_v2":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Frozen TabPFNv2 recipe requires the registered MPS device")
        from tabpfn import TabPFNRegressor
        torch.manual_seed(seed)
        model = TabPFNRegressor(model_path=root / "tabpfn-v2-regressor.ckpt", device="mps",
                                n_estimators=1, fit_mode="fit_with_cache", random_state=seed,
                                n_preprocessing_jobs=1,
                                categorical_features_indices=bundle["categorical_indices"])
        idx = bundle["context_indices"][f"seed_{seed}"]
        start = time.perf_counter()
        model.fit(bundle["train_x"][idx], bundle["train_target"][idx])
        bundle["runtime_diagnostics"].append({"seed": seed, "operation": "fit_training_context",
                                              "seconds": time.perf_counter() - start, "rows": len(idx)})
    elif config["backend"] == "chronos2":
        from chronos import Chronos2Pipeline
        model = Chronos2Pipeline.from_pretrained(str(root), device_map="cpu", dtype=torch.float32,
                                                 local_files_only=True)
    elif config["backend"] == "timesfm25":
        import timesfm
        # load_checkpoint is explicitly local, avoiding even hub metadata lookup.
        model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        model.load_checkpoint(str(root))
        model.compile(timesfm.ForecastConfig(max_context=config["max_context"], max_horizon=config["max_horizon"],
                       normalize_inputs=True, per_core_batch_size=config["batch_size"],
                       force_flip_invariance=False, infer_is_positive=True))
    else:
        raise ValueError("Unknown foundation backend")
    _RUNTIME.update(key=key, model=model)
    return model


def fit_recipe(config, train_arrays, train_labels):
    fixed = next((r for r in recipes() if r["name"] == config.get("name")), None)
    if fixed is None or any(config.get(key, value) != value for key, value in fixed.items()):
        raise ValueError("Foundation recipe differs from the registered fixed configuration")
    config = fixed.copy()
    status = availability()[config["backend"]]
    if not status["available"]:
        raise RuntimeError("Unavailable foundation backend: " + "; ".join(status["reasons"]))
    if not train_arrays["cases"]:
        raise ValueError("Expected nonempty training arrays")
    current = np.asarray(train_arrays["current"], dtype=float)
    if current.shape != (len(train_arrays["cases"]),) or not np.isfinite(current).all() or np.any(current <= 0):
        raise ValueError("Expected aligned finite positive training current scores")
    p = provenance()
    bundle = {"config": config, "checkpoint": p["models"][config["backend"]], "software": p["software"],
              "training_case_ids": [c["case_id"] for c in train_arrays["cases"]], "runtime_diagnostics": [],
              "seed_role": config["seed_role"]}
    if config["zero_shot"]:
        # train_labels intentionally unused: no fitting, no history, no calibration.
        rows = [resample_prefix(s, step_hours=config["step_hours"], max_context=config["max_context"],
                                max_horizon=config["max_horizon"])["diagnostics"] for s in train_arrays["samples"]]
        bundle["training_grid_diagnostics"] = rows
        bundle["context_id"] = "zero_shot_" + config["backend"]
        return bundle
    frame, design = build_feature_frame(train_arrays, config["feature_mode"])
    if len(frame) != len(current):
        raise ValueError("Training features must align with cases")
    transform = fit_preprocessor(frame)
    x = transform_features(transform, frame)
    actual = np.asarray([train_labels[c["case_id"]] for c in train_arrays["cases"]], dtype=float)
    if not np.isfinite(actual).all() or np.any(actual <= 0):
        raise ValueError("Training final answers must be finite and positive")
    y = np.log(actual / current)
    indices, audits = {}, {}
    for seed in SEEDS:
        indices[f"seed_{seed}"], audits[f"seed_{seed}"] = balanced_context_indices(train_arrays["cases"], seed)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(json.dumps(transform, sort_keys=True).encode())
    digest.update(x.tobytes()); digest.update(y.tobytes())
    for idx in indices.values():
        digest.update(idx.tobytes())
    bundle.update(preprocessor=transform, context_design=design, train_x=x, train_target=y, context_indices=indices,
                  context_audits=audits, context_id=digest.hexdigest(),
                  categorical_indices=[i for i, c in enumerate(transform["columns"]) if c["kind"] == "category"])
    for seed in SEEDS:
        _runtime(bundle, seed)
    return bundle


def _temporal_predict(bundle, arrays):
    config = bundle["config"]
    series = [resample_prefix(s, step_hours=config["step_hours"], max_context=config["max_context"],
                              max_horizon=config["max_horizon"]) for s in arrays["samples"]]
    model = _runtime(bundle)
    result = np.empty(len(series), dtype=float)
    for start in range(0, len(series), config["batch_size"]):
        batch = series[start:start + config["batch_size"]]
        horizon = max(1, max(s["horizon"] for s in batch))
        context = [s["context"] for s in batch]
        if config["backend"] == "chronos2":
            quantiles, _ = model.predict_quantiles(context, prediction_length=horizon, quantile_levels=[0.5],
                            batch_size=config["batch_size"], context_length=config["max_context"], cross_learning=False)
            forecast = [q.detach().cpu().numpy().reshape(-1) for q in quantiles]
        else:
            forecast, _ = model.forecast(horizon=horizon, inputs=context)
        if len(forecast) != len(batch):
            raise ValueError("Foundation model changed the number of query cases")
        result[start:start + len(batch)] = [decode_endpoint(s, f) for s, f in zip(batch, forecast)]
    return result


def predict_members(bundle, arrays):
    """Three final-score members; zero-shot members are deterministic repeats."""
    current = np.asarray(arrays["current"], dtype=float)
    if not len(current):
        return {f"seed_{seed}": np.empty(0, dtype=float) for seed in SEEDS}
    if len(current) != len(arrays["cases"]) or not np.isfinite(current).all() or np.any(current <= 0):
        raise ValueError("Prediction cases must have aligned positive current scores")
    config = bundle["config"]
    if config["zero_shot"]:
        prediction = _temporal_predict(bundle, arrays)
        return {f"seed_{seed}": prediction.copy() for seed in SEEDS}
    frame, _ = build_feature_frame(arrays, config["feature_mode"], bundle["context_design"])
    x = transform_features(bundle["preprocessor"], frame)
    result = {}
    for seed in SEEDS:
        model = _runtime(bundle, seed)
        predicted = np.concatenate([np.asarray(model.predict(x[i:i + config["query_batch_size"]]))
                                    for i in range(0, len(x), config["query_batch_size"])])
        if predicted.shape != current.shape or not np.isfinite(predicted).all():
            raise ValueError("TabPFNv2 failed to predict every input case")
        result[f"seed_{seed}"] = current * np.exp(np.clip(predicted, *config["final_log_growth_bounds"]))
    return result


def predict_recipe(bundle, arrays):
    return np.maximum(np.mean(list(predict_members(bundle, arrays).values()), axis=0), arrays["current"])


def diagnostics(bundle, arrays=None):
    """JSON-compatible audit export; optional blind query grid coverage."""
    result = {key: bundle[key] for key in ("context_audits", "runtime_diagnostics", "training_grid_diagnostics", "seed_role")
              if key in bundle}
    if "preprocessor" in bundle:
        result["feature_columns"] = [c["name"] for c in bundle["preprocessor"]["columns"]]
    design = bundle.get("context_design")
    if design is not None:
        result["training_context_vocabulary"] = {"event_types": list(design.types), "bands": list(design.bands)}
    config = bundle["config"]
    if arrays is not None and config["zero_shot"]:
        result["query_grid_diagnostics"] = [resample_prefix(s, step_hours=config["step_hours"],
                    max_context=config["max_context"], max_horizon=config["max_horizon"])["diagnostics"]
                    for s in arrays["samples"]]
    return result
