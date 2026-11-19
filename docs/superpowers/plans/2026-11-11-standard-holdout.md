# Standard Holdout Benchmark Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task. Work in the user's existing project folder and leave unrelated files untouched.

**Goal:** Establish one immutable, full-history, event-grouped chronological holdout dataset and use it for all subsequent primary comparisons.

**Architecture:** Freeze the raw SQLite database and metadata, emit fixed prefix inputs and separate labels, and fingerprint every artifact. Train and tune on training/validation only; write blind test predictions after model selection, then evaluate them once with a separate scorer.

**Tech Stack:** Existing Python 3.14 environment; SQLite, NumPy/pandas, existing tree/sequence backbones, unittest.

## Fixed Contract

- Group by activity ID across both ranks and every snapshot.
- Sort groups by real CN start time. Split eligible events 70% / 15% / remainder; seed 42 is recorded for stochastic training, never for shuffling chronology.
- Use all historical eras. At least five ordered nonnegative observations and a positive terminal score are required. Terminal observation must be within 60 seconds before event end or 300 seconds after it. Ongoing/invalid curves are listed as exclusions.
- Use sparse observed-prefix training samples, capped by progress bin, to retain older records. Do not require 20 points per curve.
- Fixed validation/test stages: 25%, 50%, 75%, remaining 24 hours and remaining 6 hours. At least four prefix points and <=60-minute staleness; ineligible cases are logged once.
- Train-only fitting, preprocessing and historical outcome statistics. Same-event peer input is a prefix only. Validation labels may select hyperparameters/checkpoints; test labels may only be read by the final evaluator.
- Primary metric: event-macro MAPE. Also report case MAPE, per-rank/stage/type metrics, MAE, RMSE, median/P90 APE.
- Fixed dataset directory `data/holdout_v1`, refusing overwrite; future DB updates do not move existing IDs or cases. New data belongs to future external validation or a separately named dataset version.
- Test events overlap previous exploratory experiments; record that exposure rather than claiming a pristine unseen cohort.

## Tasks

- [x] Test immutable construction, event grouping, chronological splits, sparse histories, exclusion logs, and test-label access guard.
- [x] Implement and freeze the dataset with hashes and fixed case lists; save partition/exclusion tables.
- [x] Add train-only historical pools and explicit peer-prefix inputs without changing old experiment defaults.
- [x] Implement one holdout training/prediction/evaluation entry, checkpoint and prediction artifacts, validation-only selection, and run fingerprints.
- [x] Rerun baselines, tree/context methods and nine neural families with three fixed seeds; score every method on identical test cases.
- [x] Independently review leakage boundaries and verify metrics, snapshots and fingerprints; document the permanent protocol in AGENTS.md and README.

## Execution evidence

- Dataset/scorer/feature tests were first run RED for missing implementations, then GREEN. Dataset frozen as `d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb`: 198/42/43 events, 4377/414/412 prefixes. All 198 training groups are retained, including sparse histories.
- A separate holdout feature builder is used instead of changing the legacy context builder, preserving previous experiment defaults. Its only complete-curve pool is frozen training data; peers are explicit prefix inputs.
- Full suite initially passed 44 tests. The end-to-end synthetic pipeline trains/checkpoints/reloads trees and networks while a test-label guard is active, then runs the separate scorer and verifies sealed-model tampering is rejected.
- Independent review confirmed grouping, fixed stage coverage, prefix limits, train-only history and macro weights without opening actual test labels. It identified scorer-version/overwrite integrity and bootstrap event-source consistency. Both are treated as required fixes for future reuse: regression tests first failed, then scorer fingerprints/immutable receipts and frozen-case bootstrap grouping were implemented. They do not change valid predictions, primary metrics or model training; actual scoring source is archived separately from the training run.
- Primary full run was preregistered before test scoring: 5 baselines, 14 tree/context recipes, 9 neural families plus peer TCN, each neural configuration with 3 fixed seeds. Train-only event-balanced weights; validation-selected checkpoints; 80 epoch cap, 25 minimum / 18 patience.
- Completed 44 fitted checkpoints and 60 blind prediction methods, each exactly 412 cases. Validation selected the arithmetic final-score combination of LightGBM peer, PatchTST ensemble3, CatBoost peer. Test labels were read only after prediction/selection sealing. Validation 4.3206%, test 8.9202% event-macro MAPE; 1000/2000 tiers 13.4953%/4.3451%.
- Full suite after review fixes: 45/45 passing; `uv pip check --python .venv/bin/python`: 48 packages compatible. Regression tests cover scorer identity changes, cached immutable scoring and frozen event IDs for paired bootstrap.
- Independent standard-library metric recomputation matched all 60 methods' event/case MAPE, MAE and RMSE (largest rounding difference 2.33e-10); verified all 24720 rows' frozen identities, all 30 neural validation checkpoint choices, and unchanged selection seal. Dataset and scoring hashes verified. Source DB SHA-256 remains `1d15dc07c06e4cc3a1ab6c442bb02cbcfa204b1057e48507db68d6d14e28d5aa`.
- Final artifacts and limitations are in `analysis/holdout_results/v1/README.md`; primary workflow and future fixed-dataset rules are in root README.md/AGENTS.md. No model retuning or recipe changes were made after seeing the final test results.
