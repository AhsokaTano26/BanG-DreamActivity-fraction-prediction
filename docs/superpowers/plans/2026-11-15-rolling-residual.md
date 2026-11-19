# V3 residual exploration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test fixed statistical/contextual/neural residual correction methods against the existing dense V3 EWMA under identical causal base forecasts.
**Architecture:** A trusted coordinator replays guarded releases over a sealed base prediction cache; pure correction modules receive released frames and current blind frames only. Independently score after validation-only selection and all event seals.
**Tech Stack:** Existing Python .venv, NumPy/pandas/sklearn/CatBoost/PyTorch; no new dependencies.
**Spec:** docs/superpowers/specs/2026-11-15-rolling-residual-design.md

## Global constraints

- Dense V3 fixed IDs/cases and actual date cutoff1706504400000; models cannot access vault or answer files.
- Only strictly completed earlier forecast errors; activity-equal fitting; NN42/123/2026 and final-score ensemble.
- Exact existing EWMA replay; no changes to old experiment sources or seals.
- Prior test exposure disclosed; no post-test retuning; no commit/push/deploy.

## Review focus

- Future/current outcome poisoning must not change earlier predictions.
- Repeated phase cases cannot overweight a dense activity.
- Causal EWMA baseline reproduces old bounded scores exactly.
- Preprocessing handles unseen activity metadata without fitting on targets.
- NN seed aggregation averages bounded final scores, not log corrections.

### Task 1: Statistical corrections (parallel agent)
- [x] New analysis/rolling_residual_stats.py + boundary/weight/clock tests.
- [x] Expose CONFIGS and predict_bias(name,history,target,query_start_ms=...) -> (bias,diagnostics).

### Task 2: Learned corrections (parallel agent)
- [x] New analysis/rolling_residual_models.py + small-sample/clock/weights/seed tests (torch subprocess isolation).
- [x] Expose CONFIGS and fit_predict(name,history,target,query_start_ms=...,model_directory=...) -> (dict method->total_bias, diagnostics). MLP returns three seed members; coordinator creates final-score mean.

### Task 3: Coordinator and experiment (root)
- [x] Write clock/source/cache/baseline guards and meaningful integration/poison tests.
- [x] New analysis/rolling_residual_benchmark.py; write spec/config/source identities before first release; seal validation selection before test.
- [x] Run full fixed menu, store individual event models/states/feedback audits.

### Task 4: Scoring/report and independent audit
- [x] New analysis/rolling_residual_evaluate.py, independent score with cached receipts and event/block intervals.
- [x] Full tests, check all source/case/feedback/old-delivery hashes, source-backed report with all candidates and exposure caveats.
- [x] Final delivery index and concise conclusion; selected result remains validation-selected regardless of test rank.
