# Paper Research Upgrade Implementation Plan

> Agentic workers: use scoped subagent implementation and independent review. User has explicitly authorized execution without intermediate approval prompts.

Goal: strengthen the prediction paper through valid mechanisms, real fixed-rule experiments, statistical evidence and reproducible figures.

Architecture: immutable archived inputs feed a trusted chronological coordinator; detached predictors receive released history only; an independent scorer creates activity-level evidence; figures and LaTeX bind to verified outputs.

Tech Stack: project .venv, NumPy/pandas/SciPy, existing CatBoost/PyTorch, Matplotlib and editable SVG, XeLaTeX/BibTeX.

Spec: docs/superpowers/specs/2026-11-18-paper-research-upgrade.md

## Global Constraints

- Actual CN start>=1706504400000; frozen97 events,67/14/16; no rewritten datasets or historical scores.
- End strictly before query start; predict/seal before release; no model label-file or history_vault access.
- New candidates fixed before scoring; selection never uses explored test; repeated validation and retrospective metadata disclosed.
- One numeric job, one thread; neural seeds42/123/2026; raw predictions, manifests, failed methods retained.
- Work on owned new paths; preserve dirty working tree and production; no commit/push/publication requested.

## Review Focus

- Missing ranks and T1500 zero initial support: explicit fallback and equal case coverage.
- Overlapping events and delayed labels: release guards, end-sorted feedback, no premature use.
- Multiple time-axis memberships: unique case scoring, memberships for subpanels only.
- Changing base route: exact common prediction source and baseline parity.
- Repeated exploration: no minimum-test-score selector, no future score before real outcomes.

## Task 1: Archive and audit (PI,A,C,G)

- [x] Archive current source/PDF and record workspace state.
- [ ] A reads manuscript and verifies related work; C maps data/prediction/release interfaces.
- [ ] G independently checks fingerprints, case identities and baseline metrics.
- Inputs: current paper, AGENTS, sealed run/evaluation manifests.
- Outputs: baseline archive, source index, A/C/G reports.
- Accept: exact 7.08171658/6.67074608/6.54028074 reproduction on matched5220cases/16events or a documented discrepancy.

## Task 2: Mechanism and experiment freeze (B,C,PI)

- [ ] B derives sharing risk under explicit assumptions, synthetic grid, and falsifiable hypotheses.
- [ ] C specifies detached predictor interface and fixed real experiment menu.
- [ ] PI records frozen config/hash before any new validation/test score.
- Outputs: B design, preregistration.json, tests for time guards, scoring unit and missing support.
- Accept: no method uses current outcome or chooses by test results.

## Task 3: Real and synthetic experiments (C,B,PI)

- [ ] Test temporal release and replay parity first; watch meaningful failure then fix.
- [ ] Replay EWMA/window/Kalman, anchor/difference deletion, swap/pair/support/decay controls on identical predictions.
- [ ] Evaluate simple growth baselines using permissible training history.
- [ ] Run training-history learning curves and shared/independent sample-size comparisons; distinguish base-fit learning from residual-history learning.
- [ ] Run fixed synthetic grid with repeated independent seeds and Monte Carlo uncertainty.
- [ ] Save forecasts/config/source/runtime fingerprints before independent scoring.
- Accept: complete candidate ledger, no dropped difficult cases or unsupported synthetic-to-real claims.

## Task 4: Independent statistics (D,G)

- [ ] Recompute activity-equal metrics and paired differences from raw rows.
- [ ] Paired and moving-block bootstrap (several block lengths), leave-one-event-out, wins/concentration and tail metrics.
- [ ] Group and interval reliability analysis with support counts and exploratory multiplicity scope.
- Outputs: results CSV/JSON, D statistics and G evidence reports.
- Accept: verified primary numbers, honest small-sample and temporal dependence limits.

## Task 5: Scientific figures (E,PI)

- [ ] Map12–15 figures to research questions and verified inputs.
- [ ] Produce2–3 editable2.5D mechanism diagrams; all numerical panels use real outputs or explicitly labeled synthetic results.
- [ ] Export PDF/SVG/400dpiPNG and source/data mapping; inspect at manuscript size and grayscale.
- Accept: readable labels, explicit units/uncertainty/support, accurate arrows, no fabricated observations.

## Task 6: Writing and reproducible delivery (F,PI)

- [ ] Reconstruct Chinese paper and supplement around evidence, add mathematical proposition/pseudocode/related work.
- [ ] Bind values, references, captions, limitations and prospective waiting status.
- [ ] One command runs bounded experiments, statistics, figures and compilation; archived fast-build mode available.
- [ ] Build PDF, inspect every page and font/bbox/citation status; validate standalone archive.
- Accept: editable source and complete assets/data, no placeholders represented as finished research.

## Task 7: Final adversarial review (A,G,PI)

- [ ] A reviews research claims; G independently reproduces critical evidence and boundary checks.
- [ ] Resolve material findings, run appropriate tests plus required full unittest suite serially.
- [ ] Deliver exact completed experiments, supported/rejected hypotheses, independent-evidence gaps and submission fit.
