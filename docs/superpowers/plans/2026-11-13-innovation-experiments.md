# Innovation Dataset Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. User has already authorized execution and parallel independent research; do not ask for another plan approval.

**Goal:** Implement and evaluate historical retrieval residuals, hierarchical context, and dual-clock nonnegative growth on frozen V1 and V2, with train-only chronological diagnostics before validation selection and independent test scoring.

**Architecture:** New modules only; no edits to sealed old model/scorer sources. A common blind-prefix adapter supplies a uniform array interface, each family owns independent recipes, and a common runner persists folds, models, sources, predictions and seals. Existing independent V1/V2 scorers remain authoritative.

**Tech Stack:** Project Python environment, numpy/pandas/scikit-learn/CatBoost/PyTorch; no new dependencies.

**Spec:** `docs/research/2026-11-13-dataset-innovation-agenda.md` and `AGENTS.md`.

## Global Constraints

- V1 dataset ID `d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb`; V2 two-rank ID `1f76f39af8151bba185496411bbcee064b16a4c5aca5c515c1d957a5056fe854`.
- Use dataset classes for complete integrity checks. Never parse test labels outside independent scorers; never read SQLite for model work.
- Preserve all frozen cases and activity memberships. Every training activity has equal total weight. Missing information causes an explicit fallback, never case deletion.
- Model features use only blind own prefixes plus static metadata and permitted completed TRAIN history. New family main inputs are own-line features; old paired models are separately identified references.
- Each chronological fold fits on earlier activities whose end is before the earliest held-out activity start. Its history library is fixed to those training activities, not updated with fold answers.
- First half of the original 198 training activities (by actual start) is warmup, remaining 99 form 3 contiguous groups. Groups are diagnostic only; do not move official split memberships. Use all available cases in each fold.
- All new recipes, fixed NN epochs, seeds 42/123/2026 and combination menu are saved before any new validation score. Three-seed arithmetic final-score means are selectable; individual seeds are reported but not selectable.
- Each protocol selects by its validation event-macro MAPE. Internal folds are stability diagnostics, no test-driven or post-validation expansion. No new feature changes after formal runs start.
- Report previous test exposure and retrospective metadata in every report. No claims of improvement from comparing different protocols/grids.

## Shared family interface

`recipes()` returns up to three named dicts per family.
`fit_recipe(config, train_arrays, train_labels)` returns a joblib-safe bundle.
`predict_recipe(bundle, arrays)` returns aligned finite final scores >= current.
Optional `predict_members(bundle, arrays)` returns seed-name to prediction arrays.

Common arrays from `analysis.innovation_common.prepare(samples, metadata)`:

- `samples`: blind sample list, `cases`: list of case_id/event/rank/start_ms/end_ms.
- `base`: DataFrame of existing own-line `multirank_model.feature_row` features (no contextual columns).
- `current`, `progress`, `remaining_h`, `duration_h`: float arrays; `rank`: 0/1 integer array.
- `event_type`: string array; `band_shares`: list of {string band ID: fractional share}; vocab fitted by model on training only.
- `history`: assigned by runner: only fold-training or official-training Curve objects, never validation/test curves. Retrieval bundle persists its allowed history and filters end < query.start and event != query.event.

Modules must not use a dataset class or filesystem to fetch additional data in fit/predict. Labels map case_id -> final; only fit accepts labels. Main preparation does not normalize using future/final values.

## Review Focus

- No/sparse same-rank historical neighbor: preserve query with explicit linear fallback and count/lag features.
- Same event, unfinished history and fold-held-out curve poison: must not change predictions/features.
- Unknown type/band and all-missing numeric column: train-only schema/scaler and safe global fallback.
- At end and near-end: nonnegative integrated growth; no decreasing future cumulative path; predictions need not be monotone between updates.
- Runner must persist selection before opening test inputs, and repeated scoring must verify cache without reparsing labels.

## Tasks and ledger

- [x] 1. Root: common adapter, strict chronological folds, frozen-case metric/weights, runner and guarded integration tests.
- [x] 2. Retrieval worker: `analysis/innovation_retrieval.py`, `tests/test_innovation_retrieval.py`. Up to three configs: own-line CatBoost control, retrieval-feature augmented CatBoost, retrieval-prior residual CatBoost; fixed 600 trees/depth5/lr .035/L2 12/seed42/threads2. Historical k=5 fixed from research hypothesis, no tuning on test. Include distribution/coverage/lag, positive final decode, full history provenance.
- [x] 3. Context worker: `analysis/innovation_hierarchy.py`, `tests/test_innovation_hierarchy.py`. Up to three fixed context/shrinkage recipes with clear main-effect and time-interaction ablation. Coefficients fitted train-only; use chronological OOF if correcting fitted backbone residuals. Strong shrinkage and unknown-category fallback. Own module uses only shared arrays interface.
- [x] 4. Growth worker: `analysis/innovation_growth.py`, `tests/test_innovation_growth.py`. Three fixed same-encoder controls: direct log-growth, progress-only positive basis integral, progress+remaining-hours positive basis integral; three seeds, fixed epochs (no validation dependency), train-fitted preprocessing, event-equal loss. Small own-prefix MLP encoder is deliberate first decoder ablation; no auxiliary new cases in this run. Integration tests run in isolated subprocesses.
- [x] 5. Root plus peer reviews: verify family boundaries, run full unittest suite, seal recipe manifest and run independent V1/V2 experiments in new directories. Compare each family, unweighted three-family final-score mean and 50/50 new-best+matched CatBoost baseline; fixed menu before scoring.
- [x] 6. Independent existing V1/V2 scorers, bootstrap against matched references, replay persisted models, write consolidated report with all candidates, folds, stages, ranks, coverage and limitations.

## Execution rulings

- Work in the existing user workspace using only new uniquely named files/directories; do not copy or discard the extensive pre-existing untracked research, and do not commit unrelated files. A dedicated local branch may be used without moving user changes.
- Earlier user explicitly requested several agents and comprehensive independent directions; independent family modules may be implemented concurrently with disjoint file ownership. Main integration and any shared edits remain sequential.
- This is one declared experiment round, not an unbounded search. Negative results are delivered alongside positive ones.

## Progress

Plan prepared before implementation. Prior research artifacts remain sealed. No new validation/test results have been parsed for this experiment.

Preflight review ruling: the old own-line feature helper also supplies event_type. The new adapter explicitly removes it from base and keeps it in the separate event_type array, preserving the planned curve-only/context ablation. Existing historical references retain their original information and remain visible in the final comparison; improvements against the new control alone will not establish improvement over old models.

Family implementations fixed: retrieval control/features/prior residual; hierarchy static context/additive OOF shrinkage/phase OOF shrinkage; growth direct/progress-integral/dual-clock-integral, all 80 epochs and seeds 42/123/2026. Root integration and scorer review ongoing; no performance-dependent recipe changes.

Implementation/review checkpoints complete: common metrics/fold tests, guarded runner, 13 retrieval tests, 11 hierarchy tests, 7 isolated growth tests, cached scorer boundary. Growth train-never-observed numeric columns now forced to zero at inference after RED/GREEN regression. Independent reviews found no remaining family blockers. Full suite: 172 tests passed in 120.447s, log `/tmp/bangdream_innovation_full_20261113.log`.

Formal V1 and V2 jobs started with all model source/config hashes frozen, output directories `analysis/holdout_results/innovation_20261113` and `analysis/temporal_results/innovation_20261113`; logs `/tmp/bangdream_innovation_v1_20261113.log` and `/tmp/bangdream_innovation_v2_20261113.log`. No model changes while jobs run. Both jobs finish blind predictions before independent final scoring.

Completed results: both validation selections are family_mean. V1 validation/test 6.0472/8.4728%, versus historical selected test8.0676%; V2 validation/test6.8952/9.2331%, versus historical selected9.1180%. No overall replacement. All nine models per protocol replayed every test prefix, all three outerfold case/library boundaries verified, internal calibration chronology verified, equal event weights verified. Cached official/comparison scorers succeed with label readers blocked. Original preservation manifest assets (including DB) unchanged. Delivery report: analysis/innovation_results/20261113/report/README.md. Final report consistency review complete: 267 displayed values verified. A timestamp-file reference corrected in new report_verified, with the first rendered report retained. Model/code freeze unchanged.

Final report: analysis/innovation_results/20261113/report_verified/README.md. Delivery verification: analysis/innovation_results/20261113/verification.json. All planned first-round work complete; no changes to deployed prediction entrypoints.
