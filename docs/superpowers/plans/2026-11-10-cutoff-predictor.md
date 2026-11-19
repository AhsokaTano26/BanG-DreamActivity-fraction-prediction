# Cutoff Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Turn the selected XGBoost remaining-growth method into a usable CN cutoff prediction command.

**Architecture:** Reuse the feature extraction, model parameters, and historical templates already evaluated in `analysis/backtest.py`. A forecasting module selects eligible historical curves and estimates an empirical error range; a small CLI loads live or local observations and formats the result.

**Tech Stack:** Existing project `.venv`, Python 3.14, NumPy, pandas, scikit-learn, XGBoost, httpx, SQLite, unittest.

**Spec:** The prior method recommendation accepted by the user's “继续”, clarified in the interface contract below.

## Interface Contract

- `predict.py --event ID --rank {1000,2000}` fetches public Bestdori metadata and cutoffs and predicts the final score.
- `--offline` uses the existing database; `--as-of ISO8601` replays a historical observation prefix. Naive dates use UTC+08:00.
- `--json` emits one JSON object to stdout; failures exit nonzero and go to stderr.
- Output includes observation time, current score, remaining time, XGBoost prediction, historical analog reference, and an empirical range when enough past calibration events exist.
- The range targets 80% coverage using the finite-sample order statistic of absolute log errors, grouped by event, ranking tier, and nearest evaluated stage. It is historical-error guidance, not guaranteed future coverage.
- Calibration rows must belong to eligible historical events that ended before the target began. At least 20 distinct events are required. Data-hash mismatches or missing calibration artifacts suppress the range with an explanation.
- Forecasts before the 25% evaluation stage carry an extrapolation note and no empirical range. Ended activities require an earlier `--as-of`; stale observations (>60 minutes) are rejected.

## Global Constraints

- Work in the existing project folder and use its `.venv`.
- Do not write the database or replace the old prediction scripts.
- Exclude the target event at both ranking tiers from training and calibration.
- Select history by actual start/end timestamps, not numeric activity ID.
- Preserve existing backtest behavior and metrics.
- Do not commit unrelated uncommitted database, image, or IDE files.

## Review Focus

- Historical replay must not expose suffix scores or another tier of the target event.
- Invalid/empty tracker responses and stale observations must fail clearly.
- Missing or stale calibration must not produce a fabricated range.
- Repeated calibration snapshots must not inflate the number of independent events.
- Commands run outside the repository must resolve the database relative to the script.

## Task 1: Forecasting and historical-error range

Files: `forecasting.py`, `analysis/backtest.py`, `tests/test_forecasting.py`.

- [x] Write tests for time-based history selection, suffix invariance, replay boundaries, and finite-sample calibration.
- [x] Run `.venv/bin/python -m unittest discover -s tests -v`; verify missing forecasting functionality fails.
- [x] Implement `select_history`, `predict_cutoff`, `calibrate_range`, and readonly database loading. Parameterize the existing backtest loader without changing defaults.
- [x] Run tests and confirm history eligibility, range bounds, and no database writes.

## Task 2: Live and offline command

Files: `predict.py`, `tests/test_predict_cli.py`, `README.md`.

- [x] Test time-zone parsing, nonzero errors, offline JSON output, path resolution, and failed HTTP payloads.
- [x] Observe failures, then implement bounded HTTP requests and the command interface.
- [x] Document environment activation, live use, historical replay, calibration behavior, and the full test command.
- [x] Verify the command against one live activity and one historical offline replay; compare database hashes.

## Task 3: Review and regression verification

- [x] Run the full unittest suite and the existing backtest. Compare all prior metric values before changing result artifacts.
- [x] Review the implementation independently for leakage, calibration, and user-visible failures; repair supported findings with regression tests.
- [x] Record outcomes and leave the current work uncommitted for user review.
