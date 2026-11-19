# Deadline dynamics research plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user requested further data relationship research, better methods and reference materials in this existing project.

**Goal:** Establish whether deadline dynamics and honest out-of-fold residual learning offer a better next direction than additional generic network depth.

**Architecture:** Training-only curve diagnostics followed by small registered prototypes on training chronological folds and the fixed validation cohort. No new test scoring in this research stage. Preserve all sealed experiments.

**Tech Stack:** Existing project Python, NumPy/pandas/SciPy, CatBoost, scikit-learn and frozen HoldoutDataset.

**Spec:** Current user request; root AGENTS.md controls all data access and weighting.

## Approaches and decision

1. Deadline-aligned historical curve priors: adapt functional partial-curve prediction to learn remaining-rate, comparing end-relative and progress-relative histories. Low parameter count, explicit current-event/future exclusions.
2. Temporal out-of-fold residual correction: estimate systematic errors using only past-fold predictions. Avoid fitting a correction to optimistic in-sample errors.
3. Irregular-time neural models and pretrained tabular models: assess primary literature and practical constraints; do not install or claim results for untested models.

The working choice is to diagnose first and prototype 1 and a small version of 2; this is an assistant implementation choice within the user's authorized research scope. It is not a claim that either will improve.

## Fixed protocol

- Use holdout_v1 fingerprint d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb, with unchanged 4377 train/414 validation/412 test cases.
- Read full curves only from train_curves.jsonl; no test inputs/labels or new test scoring for this stage. Existing test results are context only, not a tuning target.
- Internal folds: first 98 training activities, then four consecutive 25-activity blocks by manifest chronology. Fit models and preprocessing within each fold. Frozen history pool contains only fit activities; do not append outcomes within a held-out block.
- All full historical statistics exclude the target activity and activities unfinished at its start. Equal global activity training weight across both ranks.
- Register three tree prototypes: existing rank-specific remaining-rate baseline; same model with causal history-prior features; same features but predict a residual around the deadline-shape prior. CatBoost 500/depth4/lr.03/L2=15/seed42; no hyperparameter search.
- Residual correction baseline: fit Ridge to past OOF log(final/prediction), with fixed rank × progress × type features and alpha100; clip corrections to ±0.25. First OOF block has no preceding calibration and remains unchanged. Three shrinkages 0.25/0.5/1; validation selects only after the whole configuration set is fixed.
- Historical prior settings: 720-day recency scale, type mismatch weight0.15, log-duration distance scale0.5, normalized log-rate similarity bandwidth0.5. Predict log remaining-average-rate multiplier. Record candidate counts and interpolation support; never treat unknown history as observed zero.
- Keep original mixture as reference. Report both failed and successful directions; validation gains are not fixed-test improvements.

## Review focus

1. Past full curves must truly precede each event, including feature construction inside training folds.
2. Endpoint-window statistics must disclose interpolation span and sensitivity to sparse observations.
3. Repeated prefixes/ranks are dependent: event-level resampling and global event weights.
4. Out-of-fold calibration may only learn earlier-fold errors; no in-sample residual training and no test answers.
5. References distinguish original papers from adapted prototypes and unchecked suggestions.

## Tasks

- [x] Collect primary references, including limitations for this dataset.
- [x] Compute training-only endgame dynamics, cross-rank gap changes and interpolation sensitivity.
- [x] Implement causal history features and small prototype comparisons; verify boundary invariants.
- [x] Run frozen training folds and validation once for all registered configurations.
- [x] Independently verify metrics and complete an evidence-linked Chinese research report with explicit next-step priorities.

Completed in `analysis/deadline_research/`. Final development is `prototypes_v2`;
v1 is preserved as the pre-review missing-shape-support implementation. This rerun
fixed demonstrated input semantics, not a hyperparameter search. Shape candidates
require at least one common valid 6h/24h rate; otherwise only pooled deadline/progress
priors remain. `hist_count` is the number of supported shape candidates. `hist_span_h`
is the shape-weighted mean of each candidate's maximum interpolation bracket across
the aligned point and the rate endpoints actually compared.

All new candidates failed to beat the frozen current mixture's validation 3.801843%.
The corrected history-feature prototype is 4.516080%; prior-residual is 5.127482%;
quarter-strength OOF correction is 4.293060%. No new test prediction or scoring.
Independent review completed after the missing-support fix; 63 tests pass.
