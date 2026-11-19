# Dense full retraining implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Retrain the implemented fixed-protocol model families on the current dense-era populations and deliver independently scored results.

**Architecture:** Keep old sources and sealed artifacts intact; use new policy-checked adapters for recipes with hard-coded old cohorts. V1 and V2 select using validation only, and all blind predictions precede independent scoring. The already trained current V3 four-rule experiment is verified and reported as its separate information regime.

**Tech Stack:** Project Python environment, existing tree/torch/foundation libraries and local pretrained assets.

**Spec:** AGENTS.md current dense data policy; docs/superpowers/specs/2026-11-15-dense-cohorts-and-rolling-v3.md; user instruction “全部重训，我要看看效果”.

## Global constraints

- Actual CN start >= 1706504400000; fixed 67/14/16 event memberships; no case deletion.
- V1/V2 use training histories only; V3 permits strictly completed prior events.
- NN seeds 42, 123, 2026; equal total training weight per event.
- No new hyperparameter search based on test results; carry over existing fixed recipes.
- Store models, preprocessing, source/dependency identities and predictions in new directories.
- Old results remain archival. Cohort-level old/new totals are not a causal model-improvement comparison.
- Known limitations: previously explored events/validation and retrospective metadata snapshot.
- Execution is explicitly authorized. No intermediate approval or commit/push is needed.

## Review focus

- Hard-coded old references must never import old-cohort predictions or answers into new fits.
- Snapshot-rich events must not dominate training or be treated as independent statistical units.
- Per-seed models and combinations must share exactly the frozen test case IDs.
- Foundation model context must consist of training rows only; zero-shot models must be labeled accurately.
- Saved-model replay and seals must distinguish real retraining from reuse of historical forecasts.

### Task 1: Core V1 and 19-rank V2

- [x] Run unchanged V1 core recipes with explicit policy-checked dense V1 and three seeds.
- [x] Run unchanged 19-rank temporal training recipes with explicit dense V2.
- [x] Independently score after sealed predictions; add V1 training replay diagnostics separately.

### Task 2: V1 extensions

Owner: dense_v1_all. New `analysis/dense_v1_*` adapter and corresponding tests.

- [x] Inventory relationship, structured neural, deadline, irregular-quality and innovation families.
- [x] Test policy/case/reference guards; retrain fixed recipes with new-cohort references.
- [x] Seal predictions and independently score every registered method.

### Task 3: V2 two-rank full families

Owner: dense_v2_all. New `analysis/dense_v2_*` adapter and corresponding tests.

- [x] Inventory temporal paired, research/fusion, innovation and next-generation recipes.
- [x] Test policy/case/reference guards and frozen source identities.
- [x] Retrain fixed recipes, including 3-seed NN and TabPFN contexts; rerun zero-shot baselines.
- [x] Seal predictions and independently score every registered method.

### Task 4: Verification and results

- [x] Verify existing dense V3 run and compare only within its regime.
- [x] Audit candidate coverage, population dates, event weights, source and prediction seals.
- [x] Run `.venv/bin/python -m unittest discover -s tests -v` and inspect failures.
- [x] Produce coverage and train/validation/test metric tables, rank/time breakdowns, and limitations.
- [x] Preserve old datasets/results; seal delivery evidence and report actual findings.

## Reporting selection rule (registered before independent new test scoring)

Within each identical dataset, the overview reports each runner's validation-selected method and the minimum validation event-macro MAPE among those methods. Individual NN seeds remain diagnostics, not extra selection candidates. No additional test-driven ensembles or parameter changes are introduced. V1, V2 two-rank, V2 nineteen-rank and V3 are separate rows; no cross-protocol ranking by their aggregate scores. Complete per-method scores and original within-run candidate lists remain available for audit.

## Execution rulings

- V3 is also rerun in full at `analysis/rolling_results/v3_dense_full_retrain_20261115` using the identical four registered rules. The previous dense baseline remains archived and verifies reproducibility; the overview uses this new run. No V3 hyperparameters or architecture menu change.
- Bootstrap uses distinct activities sorted by actual start, not repeated rank partition rows. Report both 5,000-draw paired IID intervals and circular time blocks of 3 and 5 activities.
- Reports bind evaluation and training-replay receipts to the exact run seals. The suite inventory is the exact seven V1, six V2 (including 19-rank), and one V3 run paths.
- Resource adjustment before test scoring: structured V1 convolutional training on CPU proved much slower than the original MPS recipe. Preserve the incomplete `dense_structured_20261115` attempt and restart all 9×3 fits with unchanged epochs/seeds on now-free MPS in `dense_structured_mps_20261115`. Use a separate adapter so existing source seals stay valid. The MPS run is the registered structured result; do not choose between device attempts using scores. Downstream references follow this new run.
