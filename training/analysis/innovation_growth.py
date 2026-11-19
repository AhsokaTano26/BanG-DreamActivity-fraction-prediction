"""Small terminal-supervised own-prefix networks with positive integral decoders.

Only ``fit_recipe`` accepts labels. All data are supplied by the common blind
adapter; this module never opens datasets, metadata files, or snapshot stores.
Torch is imported only inside training/inference functions so test discovery can
keep its CPU runtime separate from the repository's tree-model tests.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import copy
from typing import Mapping

import numpy as np
import pandas as pd


SEEDS = (42, 123, 2026)
_RANK_COLUMN = "__growth_rank_index"
_MISSING_CATEGORY = "__growth_missing__"


def recipes():
    """Return the three predeclared decoders, with identical encoder/training."""
    common = {
        "family": "growth", "features": "base_plus_rank", "hidden_widths": [32, 16],
        "dropout": .1, "epochs": 80, "learning_rate": .001,
        "weight_decay": .005, "batch_size": 128, "gradient_clip": 1.,
        "seeds": list(SEEDS), "loss": "event_equal_squared_log_final",
        "rate_floor_quantile": .01, "minimum_elapsed_h": 1. / 60.,
        "basis": "bernstein_degree_2", "end_windows_h": [24., 6.],
        "supervision": "terminal_only", "early_stopping": False,
    }
    return [dict(copy.deepcopy(common), name=f"growth_{head}", head=head)
            for head in ("direct", "progress", "dual_clock")]


def event_weights(events):
    """Mean-one row weights with exactly equal total weight per activity."""
    events = list(events)
    if not events:
        raise ValueError("Cannot weight empty training cases")
    counts = Counter(events)
    weights = np.asarray([1. / counts[event] for event in events], dtype=np.float64)
    return weights / weights.mean()


def _feature_frame(arrays):
    frame = arrays["base"].copy()
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("base must be a DataFrame with unique column names")
    if _RANK_COLUMN in frame:
        raise ValueError("Reserved rank feature is already present")
    rank = np.asarray(arrays["rank"])
    if rank.shape != (len(frame),) or not np.isin(rank, [0, 1]).all():
        raise ValueError("rank must be an aligned array of 0/1 rank indices")
    frame[_RANK_COLUMN] = rank.astype(float)
    return frame


def _numeric_column(frame, name):
    if name not in frame:
        return np.full(len(frame), np.nan)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return np.where(np.isfinite(values), values, np.nan)


def _categorical_column(frame, name):
    if name not in frame:
        return np.full(len(frame), _MISSING_CATEGORY, dtype=object)
    return np.asarray([_MISSING_CATEGORY if pd.isna(value) else str(value)
                       for value in frame[name]], dtype=object)


@dataclass
class PrefixPreprocessor:
    """A frozen train-only schema, median imputation, scaling and one-hot map."""

    numeric_columns: tuple
    categorical_columns: tuple
    medians: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    observed_in_training: np.ndarray
    categories: dict
    feature_names: tuple

    def transform(self, arrays):
        frame = _feature_frame(arrays)
        numeric = np.column_stack([_numeric_column(frame, name)
                                   for name in self.numeric_columns])
        numeric = np.where(np.isfinite(numeric), numeric, self.medians)
        # Fixed clipping prevents one unseen extreme from overflowing the MLP;
        # it does not learn any query-dependent normalization.
        scaled = np.clip((numeric - self.centers) / self.scales, -20., 20.)
        # A never-observed input has no trained response: query values must not
        # activate its otherwise untrained random first-layer weights.
        columns = [np.where(self.observed_in_training, scaled, 0.)]
        for name in self.categorical_columns:
            values = _categorical_column(frame, name)
            columns.append(np.column_stack([values == category
                                             for category in self.categories[name]]))
        return np.concatenate(columns, axis=1).astype(np.float32)


def fit_preprocessor(arrays):
    frame = _feature_frame(arrays)
    if len(frame) == 0:
        raise ValueError("Cannot fit preprocessing on empty training cases")
    categorical = tuple(name for name in frame
                        if not pd.api.types.is_numeric_dtype(frame[name])
                        and frame[name].notna().any())
    numeric = tuple(name for name in frame if name not in categorical)
    values = np.column_stack([_numeric_column(frame, name) for name in numeric])
    observed_in_training = np.isfinite(values).any(axis=0)
    # Explicit zero keeps all-missing training columns in the frozen schema.
    medians = np.asarray([np.median(column[np.isfinite(column)])
                          if np.isfinite(column).any() else 0.
                          for column in values.T])
    values = np.where(np.isfinite(values), values, medians)
    centers = values.mean(axis=0)
    scales = values.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.)
    categories = {name: tuple(sorted(set(_categorical_column(frame, name))))
                  for name in categorical}
    names = numeric + tuple(f"{name}={category}" for name in categorical
                            for category in categories[name])
    return PrefixPreprocessor(numeric, categorical, medians, centers, scales,
                              observed_in_training, categories, names)


def _clocks(progress, remaining_h, duration_h):
    p, remaining, duration = (np.asarray(value, dtype=np.float64)
                              for value in (progress, remaining_h, duration_h))
    if p.ndim != 1 or remaining.shape != p.shape or duration.shape != p.shape:
        raise ValueError("Clock arrays must be aligned one-dimensional arrays")
    if not all(np.isfinite(value).all() for value in (p, remaining, duration)):
        raise ValueError("Clock arrays must be finite")
    if ((p < 0.) | (p > 1.) | (remaining < 0.) | (duration <= 0.)).any():
        raise ValueError("Invalid progress, remaining hours or duration")
    if not np.allclose(remaining, duration * (1. - p), rtol=1e-7, atol=1e-7):
        raise ValueError("Both clocks must describe the same actual observed time")
    return p, remaining, duration


def basis_integrals(progress, remaining_h, duration_h, *, dual_clock=False, horizon_h=None):
    """Integrate fixed nonnegative rate bases analytically, returning hours.

    The progress bases are ``(1-p)^2, 2*p*(1-p), p^2``. The two optional
    remaining-time bases are ``max(1-h/24, 0), max(1-h/6, 0)``. ``horizon_h``
    integrates only a future portion of the interval, for checking cumulative
    paths; the model's terminal forecast uses the complete remaining interval.
    All clocks start at the actual latest observation supplied by the adapter.
    """
    p, remaining, duration = _clocks(progress, remaining_h, duration_h)
    horizon = remaining if horizon_h is None else np.asarray(horizon_h, dtype=np.float64)
    if horizon.shape != p.shape or not np.isfinite(horizon).all():
        raise ValueError("horizon_h must be an aligned finite array")
    if ((horizon < 0.) | (horizon > remaining + 1e-9)).any():
        raise ValueError("Integration horizon must be within the remaining interval")
    horizon = np.minimum(horizon, remaining)
    delta = horizon / duration
    # These are exact polynomial averages over [p, p + horizon / duration].
    b0 = (1. - p) ** 2 - (1. - p) * delta + delta ** 2 / 3.
    b1 = 2. * (p * (1. - p) + (1. - 2. * p) * delta / 2. - delta ** 2 / 3.)
    b2 = p ** 2 + p * delta + delta ** 2 / 3.
    bases = [horizon * np.maximum(value, 0.) for value in (b0, b1, b2)]
    if dual_clock:
        for window in (24., 6.):
            start = np.maximum(1. - remaining / window, 0.)
            end = np.maximum(1. - (remaining - horizon) / window, 0.)
            bases.append(window / 2. * np.maximum(end ** 2 - start ** 2, 0.))
    return np.column_stack(bases)


def _values(arrays):
    p, remaining, duration = _clocks(arrays["progress"], arrays["remaining_h"], arrays["duration_h"])
    current = np.asarray(arrays["current"], dtype=np.float64)
    if current.shape != p.shape or not np.isfinite(current).all() or (current < 0.).any():
        raise ValueError("Current scores must be aligned, finite and nonnegative")
    if len(arrays["base"]) != len(p) or len(arrays["cases"]) != len(p):
        raise ValueError("Features and cases must align with the clock arrays")
    return current, p, remaining, duration


def _weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, quantile * cumulative[-1], side="left")
    return float(values[order[min(index, len(order) - 1)]])


def _decoder_arrays(config, arrays, rate_floor):
    current, p, remaining, duration = _values(arrays)
    rate = np.maximum(current / np.maximum(duration - remaining, config["minimum_elapsed_h"]),
                      rate_floor)
    # A zero observed score still has a small train-derived positive scale.
    direct_scale = np.where(current > 0., current, rate_floor)
    integrals = basis_integrals(p, remaining, duration,
                               dual_clock=config["head"] == "dual_clock")
    return current, remaining, rate, direct_scale, integrals


def _network(torch, inputs, config):
    nn = torch.nn
    outputs = {"direct": 1, "progress": 3, "dual_clock": 5}[config["head"]]
    model = nn.Sequential(nn.Linear(inputs, 32), nn.GELU(), nn.Dropout(config["dropout"]),
                          nn.Linear(32, 16), nn.GELU(), nn.Dropout(config["dropout"]),
                          nn.Linear(16, outputs))
    # Unit progress amplitudes start at the own-prefix average rate; the dual
    # clock also starts with unit end-window amplitudes. The direct head starts
    # with log(F/current)=log(2).
    nn.init.zeros_(model[-1].weight)
    nn.init.constant_(model[-1].bias, 0. if config["head"] == "direct" else float(np.log(np.expm1(1.))))
    return model


def _decode(torch, raw, values, head):
    current, remaining, rate, direct_scale, integrals = values
    raw = raw.to(dtype=torch.float64)
    if head == "direct":
        # exp(softplus(z)) - 1 == exp(z): a direct nonnegative log-growth head.
        increment = direct_scale * torch.exp(raw[:, 0].clamp(min=-20., max=20.))
    else:
        coefficients = torch.nn.functional.softplus(raw)
        increment = rate * (coefficients * integrals).sum(dim=1)
    return current + torch.where(remaining > 0., increment, torch.zeros_like(increment))


def fit_recipe(config, train_arrays, train_labels: Mapping):
    """Fit all three fixed seeds on supplied training cases, without validation."""
    config = copy.deepcopy(config)
    if config["head"] not in ("direct", "progress", "dual_clock"):
        raise ValueError("Unknown growth decoder")
    if tuple(config["seeds"]) != SEEDS or config["hidden_widths"] != [32, 16]:
        raise ValueError("This experiment fixes the three seeds and encoder widths")
    if config["early_stopping"] or int(config["epochs"]) != config["epochs"] or config["epochs"] < 1:
        raise ValueError("A positive fixed epoch count without early stopping is required")
    current, _, remaining, duration = _values(train_arrays)
    if len(current) == 0:
        raise ValueError("Cannot train on zero cases")
    cases = train_arrays["cases"]
    case_ids = [case["case_id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Training case IDs must be unique")
    target = np.asarray([train_labels[case_id] for case_id in case_ids], dtype=np.float64)
    if not np.isfinite(target).all() or (target <= 0.).any() or (target < current).any():
        raise ValueError("Training final scores must be finite, positive and at least current")
    events = [case["event"] for case in cases]
    weights = event_weights(events)
    rates = current / np.maximum(duration - remaining, config["minimum_elapsed_h"])
    rate_floor = max(_weighted_quantile(rates, weights, config["rate_floor_quantile"]), 1e-6)
    preprocessor = fit_preprocessor(train_arrays)
    x = preprocessor.transform(train_arrays)
    decoder = _decoder_arrays(config, train_arrays, rate_floor)

    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    features = torch.from_numpy(x)
    values = tuple(torch.as_tensor(value, dtype=torch.float64) for value in decoder)
    truth = torch.as_tensor(np.log(target), dtype=torch.float64)
    sample_weights = torch.as_tensor(weights, dtype=torch.float64)
    bundle = {"version": 1, "config": config, "preprocessor": preprocessor,
              "input_features": x.shape[1], "rate_floor": rate_floor,
              "training_case_count": len(cases), "training_event_count": len(set(events)),
              "training_case_ids": case_ids, "members": {},
              "limitation": "Terminal-only supervision does not identify basis amplitudes as behavior."}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = _network(torch, x.shape[1], config)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if parameters > 20_000:
            raise ValueError(f"Growth encoder exceeds the 20,000-parameter cap: {parameters}")
        bundle["parameters"] = parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        generator = torch.Generator().manual_seed(seed)
        batch_size = int(config["batch_size"])
        trace = []
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            order = torch.randperm(len(target), generator=generator)
            loss_sum = 0.
            for begin in range(0, len(order), batch_size):
                indices = order[begin:begin + batch_size]
                optimizer.zero_grad(set_to_none=True)
                prediction = _decode(torch, model(features[indices]),
                                     tuple(value[indices] for value in values), config["head"])
                errors = (torch.log(prediction.clamp_min(1e-6)) - truth[indices]).square()
                weighted = (errors * sample_weights[indices]).sum()
                # A fixed denominator also avoids upweighting the short final
                # minibatch: each event has equal total objective weight.
                loss = weighted / batch_size
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite growth training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
                loss_sum += float(weighted.detach())
            trace.append({"epoch": epoch, "train_weighted_log_mse": loss_sum / len(target)})
        bundle["members"][f"seed_{seed}"] = {
            "seed": seed, "epochs_run": int(config["epochs"]), "parameters": parameters,
            "state": {name: value.detach().cpu().numpy().copy()
                      for name, value in model.state_dict().items()}, "trace": trace,
        }
    return bundle


def predict_members(bundle, arrays):
    """Return seed final scores for reporting, never for selecting one seed."""
    current, _, _, _ = _values(arrays)
    if len(current) == 0:
        return {f"seed_{seed}": np.empty(0, dtype=np.float64) for seed in SEEDS}
    x = bundle["preprocessor"].transform(arrays)
    decoder = _decoder_arrays(bundle["config"], arrays, bundle["rate_floor"])

    import torch

    torch.set_num_threads(2)
    features = torch.from_numpy(x)
    values = tuple(torch.as_tensor(value, dtype=torch.float64) for value in decoder)
    predictions = {}
    with torch.no_grad():
        for seed in SEEDS:
            name = f"seed_{seed}"
            model = _network(torch, bundle["input_features"], bundle["config"])
            model.load_state_dict({key: torch.from_numpy(value.copy())
                                   for key, value in bundle["members"][name]["state"].items()})
            model.eval()
            prediction = _decode(torch, model(features), values, bundle["config"]["head"]).numpy()
            if not np.isfinite(prediction).all() or (prediction < current).any():
                raise ValueError("Invalid growth final-score prediction")
            predictions[name] = prediction
    return predictions


def predict_recipe(bundle, arrays):
    """The selectable recipe is the arithmetic mean of three final scores."""
    members = predict_members(bundle, arrays)
    mean = np.mean(list(members.values()), axis=0)
    current = np.asarray(arrays["current"], dtype=np.float64)
    # Averaging three identical floats can round below/above that same float.
    # Preserve the decoder's mathematical lower bound and exact end boundary.
    return np.where(np.asarray(arrays["remaining_h"]) == 0., current, np.maximum(mean, current))
