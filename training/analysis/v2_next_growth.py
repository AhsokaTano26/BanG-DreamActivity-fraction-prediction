"""Frozen V2 small-network controls and training-only multi-horizon targets.

The terminal decoder, encoder, optimizer, seeds and terminal loss form are the
same as ``innovation_growth``. Joint terminal/auxiliary objective mass remains
equal per event, including events without auxiliary coverage. Targets add no inference features and
are drawn exclusively from the supplied training curve with matching event and
rank. This module opens no datasets or labels. Torch remains a lazy import.
"""
from __future__ import annotations

from collections import Counter
import copy
from typing import Mapping

import numpy as np

from analysis import innovation_growth as terminal


SEEDS = terminal.SEEDS
HORIZONS = ("6h", "24h", "remaining_half")
HOUR_MS = 3_600_000
MAX_ENDPOINT_DISTANCE_H = 1.
MAX_BRACKETING_GAP_H = 2.
event_weights = terminal.event_weights
basis_integrals = terminal.basis_integrals


def recipes():
    """Six preregistered recipes, including a terminal control for each decoder."""
    baseline = {config["head"]: config for config in terminal.recipes()}
    settings = (("direct", 0., "next_growth_direct"),
                ("dual_clock", 0., "next_growth_dual_clock"),
                ("dual_clock", .25, "next_growth_dual_clock_aux025"),
                ("dual_clock", 1., "next_growth_dual_clock_aux100"),
                ("progress", .25, "next_growth_progress_aux025"),
                ("progress", 0., "next_growth_progress"))
    result = []
    for head, strength, name in settings:
        config = copy.deepcopy(baseline[head])
        config.update(name=name, auxiliary_lambda=strength,
                      supervision="terminal_only" if strength == 0. else "multi_horizon",
                      auxiliary_horizons=list(HORIZONS),
                      max_endpoint_distance_h=MAX_ENDPOINT_DISTANCE_H,
                      max_bracketing_gap_h=MAX_BRACKETING_GAP_H,
                      auxiliary_loss="event_equal_squared_log_intermediate_score",
                      auxiliary_normalization="joint_event_mass_then_prefix_then_unique_horizon",
                      auxiliary_empty_event="full_terminal_mass_zero_auxiliary_mass")
        result.append(config)
    return result


def _registered_config(config):
    registered = {spec["name"]: spec for spec in recipes()}
    if config.get("name") not in registered or config != registered[config["name"]]:
        raise ValueError("Only the six unchanged preregistered V2 growth recipes are allowed")
    return copy.deepcopy(config)


def _matching_history(arrays):
    """Ignore unrelated pool members; never borrow another event or rank."""
    keys = {(case["event"], case["rank"]) for case in arrays["cases"]}
    index = {}
    for curve in arrays["history"]:
        key = (curve.event, curve.rank)
        if key not in keys:
            continue
        if key in index:
            raise ValueError("Duplicate event/rank in supplied training curves")
        times = np.asarray(curve.times)
        scores = np.asarray(curve.scores)
        if (times.ndim != 1 or scores.shape != times.shape or not len(times)
                or not np.isfinite(times).all() or not np.isfinite(scores).all()
                or (np.diff(times) <= 0).any() or (scores < 0.).any()):
            raise ValueError("Invalid supplied training curve observations")
        index[key] = curve
    return index


def _observed_time(arrays, row):
    sample, case = arrays["samples"][row], arrays["cases"][row]
    prefix = sample["curve"]
    now = float(sample["now"])
    if (sample["case_id"] != case["case_id"]
            or (prefix.event, prefix.rank) != (case["event"], case["rank"])
            or (prefix.start, prefix.end) != (case["start_ms"], case["end_ms"])):
        raise ValueError("Training case and prefix identities must agree")
    if (not len(prefix.times) or not np.isfinite(now) or prefix.times[-1] != now
            or now > sample["cutoff"] or now < prefix.start or now > prefix.end):
        raise ValueError("Auxiliary origin must be the actual observed time, before cutoff")
    expected = prefix.end - float(arrays["remaining_h"][row]) * HOUR_MS
    if (not np.isclose(expected, now, rtol=0., atol=1e-3)
            or not np.isclose(prefix.duration / HOUR_MS, arrays["duration_h"][row],
                              rtol=0., atol=1e-9)
            or int(arrays["rank"][row]) != {1000: 0, 2000: 1}.get(prefix.rank)
            or not np.isclose(prefix.scores[-1], arrays["current"][row], rtol=0., atol=1e-6)):
        raise ValueError("Auxiliary observed anchor does not match supplied clocks or score")
    return now


def _target_score(curve, timestamp):
    """Exact observations or <=1h each-side interpolation; never extrapolate."""
    position = int(np.searchsorted(curve.times, timestamp, side="left"))
    if position < len(curve.times) and curve.times[position] == timestamp:
        return float(curve.scores[position]), "observed"
    if position == 0 or position == len(curve.times):
        return np.nan, "unbracketed"
    left, right = curve.times[position - 1:position + 1]
    # Even a near-end target cannot use a post-end snapshot as an endpoint.
    if left < curve.start or right > curve.end:
        return np.nan, "outside_event_bracket"
    if (timestamp - left > MAX_ENDPOINT_DISTANCE_H * HOUR_MS
            or right - timestamp > MAX_ENDPOINT_DISTANCE_H * HOUR_MS
            or right - left > MAX_BRACKETING_GAP_H * HOUR_MS):
        return np.nan, "interpolation_gap"
    fraction = (timestamp - left) / (right - left)
    value = curve.scores[position - 1] + fraction * (
        curve.scores[position] - curve.scores[position - 1])
    return float(value), "interpolated"


def _auxiliary_weights(cases, available):
    """Unscaled auxiliary weights, subsequently sharing the joint event budget.

    Every event with auxiliary coverage receives N/E total weight, where N and
    E count ALL original terminal cases and events. Within such an event, its
    eligible prefixes are equal, then each prefix's unique available horizons
    are equal. Joint-objective normalization divides both terminal and auxiliary
    weights of covered events by (1 + lambda). An uncovered event contributes
    zero auxiliary weight and keeps its entire terminal mass.
    """
    events = [case["event"] for case in cases]
    counts = available.sum(axis=1)
    eligible = Counter(event for event, count in zip(events, counts) if count)
    weights = np.zeros(available.shape, dtype=np.float64)
    mass = len(events) / len(set(events)) if events else 0.
    for row, (event, count) in enumerate(zip(events, counts)):
        if count:
            weights[row, available[row]] = mass / eligible[event] / count
    return weights


def _coverage(cases, available, reasons, weights, terminal_weights, auxiliary_lambda):
    groups, horizons, events = {}, {}, {}
    for row, case in enumerate(cases):
        event, rank = int(case["event"]), int(case["rank"])
        event_audit = events.setdefault(event, {
            "event": event, "terminal_cases": 0, "auxiliary_cases": 0,
            "terminal_weight": 0., "auxiliary_weight": 0.})
        event_audit["terminal_cases"] += 1
        event_audit["auxiliary_cases"] += int(available[row].any())
        event_audit["terminal_weight"] += float(terminal_weights[row])
        event_audit["auxiliary_weight"] += float(weights[row].sum())
        for column, name in enumerate(HORIZONS):
            key = (event, rank, name)
            group = groups.setdefault(key, {"event": event, "rank": rank, "horizon": name,
                                            "cases": 0, "available": 0, "reasons": {}})
            horizon = horizons.setdefault(name, {"horizon": name, "cases": 0,
                                                 "available": 0, "reasons": {}})
            for summary in (group, horizon):
                summary["cases"] += 1
                summary["available"] += int(available[row, column])
                reason = str(reasons[row, column])
                summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1
    for event_audit in events.values():
        event_audit["effective_auxiliary_weight"] = auxiliary_lambda * event_audit["auxiliary_weight"]
        event_audit["combined_objective_weight"] = (
            event_audit["terminal_weight"] + event_audit["effective_auxiliary_weight"])
    return {"training_cases": len(cases), "training_events": len(events),
            "auxiliary_lambda": float(auxiliary_lambda),
            "available_targets": int(available.sum()),
            "cases_with_auxiliary": int(available.any(axis=1).sum()),
            "events_with_auxiliary": sum(item["auxiliary_cases"] > 0 for item in events.values()),
            "by_event_rank_horizon": [groups[key] for key in sorted(groups)],
            "by_horizon": [horizons[name] for name in HORIZONS if name in horizons],
            "event_weight_audit": [events[event] for event in sorted(events)],
            "origin": "actual latest observed timestamp, not scheduled cutoff",
            "interpolation": {"maximum_endpoint_distance_h": MAX_ENDPOINT_DISTANCE_H,
                              "maximum_bracketing_gap_h": MAX_BRACKETING_GAP_H,
                              "extrapolation": False, "post_end_endpoint": False},
            "normalization": "N/E combined objective mass per event: covered terminal N/E/(1+lambda) "
                             "plus effective auxiliary N/E*lambda/(1+lambda); uncovered terminal N/E. "
                             "Terminal weight splits equally over original prefixes; auxiliary weight "
                             "splits equally over eligible prefixes, then unique horizons."}


def build_auxiliary_targets(train_arrays, *, auxiliary_lambda=0.):
    """Construct masked TRAIN-only intermediate scores and interval increments.

    Unavailable entries remain NaN, have zero weight, and never enter a loss.
    A repeated physical endpoint (e.g. six hours and half of twelve remaining
    hours) is supervised once, in fixed horizon order. Diagnostic coverage has
    no target scores and is safe to retain in the serialized model bundle.
    Returned terminal_weights and weights share an equal total event budget:
    sum(terminal_weights) + lambda * sum(weights) is N/E for every event.
    """
    current, _, remaining, _ = terminal._values(train_arrays)
    if not np.isfinite(auxiliary_lambda) or auxiliary_lambda < 0.:
        raise ValueError("Auxiliary strength must be finite and nonnegative")
    cases = train_arrays["cases"]
    if len(train_arrays["samples"]) != len(cases):
        raise ValueError("Training prefix samples must align with cases")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Training case IDs must be unique")
    history = _matching_history(train_arrays)
    horizons = np.column_stack([np.full(len(cases), 6.), np.full(len(cases), 24.), remaining / 2.])
    target = np.full(horizons.shape, np.nan)
    reasons = np.full(horizons.shape, "not_before_end", dtype=object)
    available = np.zeros(horizons.shape, dtype=bool)
    for row, case in enumerate(cases):
        now = _observed_time(train_arrays, row)
        curve = history.get((case["event"], case["rank"]))
        if curve is not None:
            if (curve.start, curve.end) != (case["start_ms"], case["end_ms"]):
                raise ValueError("Training curve dates do not match observed anchor")
            anchor = int(np.searchsorted(curve.times, now))
            if (anchor == len(curve.times) or curve.times[anchor] != now
                    or not np.isclose(curve.scores[anchor], current[row], rtol=0., atol=1e-6)):
                raise ValueError("Full training curve must contain the exact observed anchor")
        seen = set()
        for column, horizon in enumerate(horizons[row]):
            timestamp = now + horizon * HOUR_MS
            if horizon <= 0. or horizon >= remaining[row] or timestamp >= case["end_ms"]:
                continue
            if timestamp in seen:
                reasons[row, column] = "duplicate_horizon"
                continue
            seen.add(timestamp)
            if curve is None:
                reasons[row, column] = "missing_training_curve"
                continue
            score, reason = _target_score(curve, timestamp)
            if np.isfinite(score) and score < current[row]:
                score, reason = np.nan, "negative_growth"
            if np.isfinite(score):
                target[row, column] = score
                available[row, column] = True
            reasons[row, column] = reason
    weights = _auxiliary_weights(cases, available)
    terminal_weights = event_weights([case["event"] for case in cases]) if cases else np.empty(0)
    covered_events = {case["event"] for case, mask in zip(cases, available) if mask.any()}
    scale = np.array([1. / (1. + auxiliary_lambda) if case["event"] in covered_events else 1.
                      for case in cases])
    terminal_weights = terminal_weights * scale
    weights = weights * scale[:, None]
    return {"horizon_h": horizons, "target_score": target,
            "increment": target - current[:, None], "available": available,
            "weights": weights, "terminal_weights": terminal_weights, "reasons": reasons,
            "diagnostics": _coverage(cases, available, reasons, weights,
                                     terminal_weights, auxiliary_lambda)}


def _attach_diagnostics(bundle, auxiliary):
    bundle["version"] = 2
    bundle["auxiliary_diagnostics"] = auxiliary["diagnostics"]
    bundle["diagnostics"] = auxiliary["diagnostics"]
    bundle["limitation"] = (
        "Sparse training-only intermediate supervision constrains growth paths; "
        "basis amplitudes are not identified causal behavior. Missing auxiliary "
        "observations do not remove terminal training cases.")
    return bundle


def fit_recipe(config, train_arrays, train_labels: Mapping):
    """Fit exactly the fixed three seeds and 80 epochs, with no validation input."""
    config = _registered_config(config)
    auxiliary = build_auxiliary_targets(train_arrays, auxiliary_lambda=config["auxiliary_lambda"])
    if config["auxiliary_lambda"] == 0.:
        # Calling the original fitter also preserves its floating point order,
        # minibatch ordering and dropout draws for the terminal-only controls.
        return _attach_diagnostics(terminal.fit_recipe(config, train_arrays, train_labels), auxiliary)

    current, progress, remaining, duration = terminal._values(train_arrays)
    if len(current) == 0:
        raise ValueError("Cannot train on zero cases")
    cases = train_arrays["cases"]
    case_ids = [case["case_id"] for case in cases]
    target = np.asarray([train_labels[case_id] for case_id in case_ids], dtype=np.float64)
    if not np.isfinite(target).all() or (target <= 0.).any() or (target < current).any():
        raise ValueError("Training final scores must be finite, positive and at least current")
    events = [case["event"] for case in cases]
    weights = event_weights(events)
    rates = current / np.maximum(duration - remaining, config["minimum_elapsed_h"])
    rate_floor = max(terminal._weighted_quantile(rates, weights, config["rate_floor_quantile"]), 1e-6)
    preprocessor = terminal.fit_preprocessor(train_arrays)
    x = preprocessor.transform(train_arrays)
    decoder = terminal._decoder_arrays(config, train_arrays, rate_floor)
    # Invalid horizons are replaced only in integration tensors, never as
    # synthetic target answers. Their auxiliary weight remains exactly zero.
    safe_horizons = np.where(auxiliary["available"], auxiliary["horizon_h"], 0.)
    integrals = np.stack([
        basis_integrals(progress, remaining, duration, dual_clock=config["head"] == "dual_clock",
                        horizon_h=safe_horizons[:, column]) for column in range(len(HORIZONS))], axis=1)

    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    features = torch.from_numpy(x)
    values = tuple(torch.as_tensor(value, dtype=torch.float64) for value in decoder)
    truth = torch.as_tensor(np.log(target), dtype=torch.float64)
    sample_weights = torch.as_tensor(auxiliary["terminal_weights"], dtype=torch.float64)
    aux_integrals = torch.as_tensor(integrals, dtype=torch.float64)
    aux_weights = torch.as_tensor(auxiliary["weights"], dtype=torch.float64)
    # A neutral finite placeholder avoids NaN * 0; it is never supervised.
    aux_truth = torch.as_tensor(np.log(np.where(auxiliary["available"],
                                np.maximum(auxiliary["target_score"], 1e-6), 1.)), dtype=torch.float64)
    bundle = {"version": 2, "config": config, "preprocessor": preprocessor,
              "input_features": x.shape[1], "rate_floor": rate_floor,
              "training_case_count": len(cases), "training_event_count": len(set(events)),
              "training_case_ids": case_ids, "members": {}}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = terminal._network(torch, x.shape[1], config)
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
            terminal_sum, auxiliary_sum = 0., 0.
            for begin in range(0, len(order), batch_size):
                indices = order[begin:begin + batch_size]
                optimizer.zero_grad(set_to_none=True)
                raw = model(features[indices])
                prediction = terminal._decode(torch, raw, tuple(value[indices] for value in values),
                                              config["head"])
                errors = (torch.log(prediction.clamp_min(1e-6)) - truth[indices]).square()
                weighted = (errors * sample_weights[indices]).sum()
                coefficients = torch.nn.functional.softplus(raw.to(dtype=torch.float64))
                aux_increment = values[2][indices, None] * (
                    coefficients[:, None, :] * aux_integrals[indices]).sum(dim=2)
                aux_prediction = values[0][indices, None] + aux_increment
                aux_errors = (torch.log(aux_prediction.clamp_min(1e-6)) - aux_truth[indices]).square()
                aux_weighted = (aux_errors * aux_weights[indices]).sum()
                # Both losses retain a fixed denominator for the short final
                # minibatch. Their joint mass is equal per event even when an
                # entire event has no reliable auxiliary supervision.
                loss = (weighted + config["auxiliary_lambda"] * aux_weighted) / batch_size
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite multi-horizon training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
                terminal_sum += float(weighted.detach())
                auxiliary_sum += float(aux_weighted.detach())
            trace.append({"epoch": epoch, "train_weighted_log_mse": terminal_sum / len(target),
                          "auxiliary_weighted_log_mse": auxiliary_sum / len(target),
                          "total_weighted_log_mse": (terminal_sum + config["auxiliary_lambda"]
                                                     * auxiliary_sum) / len(target)})
        bundle["members"][f"seed_{seed}"] = {
            "seed": seed, "epochs_run": int(config["epochs"]), "parameters": parameters,
            "state": {name: value.detach().cpu().numpy().copy()
                      for name, value in model.state_dict().items()}, "trace": trace}
    return _attach_diagnostics(bundle, auxiliary)


def predict_members(bundle, arrays):
    """Pure prefix inference: neither training history nor auxiliary answers."""
    return terminal.predict_members(bundle, arrays)


def predict_recipe(bundle, arrays):
    """Arithmetic mean of the three fixed-seed final scores."""
    return terminal.predict_recipe(bundle, arrays)
