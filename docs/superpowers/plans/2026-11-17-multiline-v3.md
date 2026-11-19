# Multi-line V3 implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. User requires serial computation, no parallel implementation agents.

**Goal:** Train and compare calibration across all19requested ranking lines, with strict causal feedback.
**Architecture:** New generalized dataset adapter, shared-rank/context-routed causal base cache, pure multi-rank correction, independent scorer/report. Legacy sources/results remain unchanged.
**Tech Stack:** ExistingNumPy/pandas/CatBoost/PyTorch/joblib,CPU1,nice19,background.
**Spec:** docs/superpowers/specs/2026-11-17-multiline-v3-design.md

## Global constraints
Same97event groups/start cutoff and19rank case lists.19aggregate and2rank panels reported separately. T1500coldstart documented. All model/preprocessor fitting history beforequery start; preserve production. Newsource paths only. No scoring-driven menu expansion.

## Review focus
- Trusted19rank builder restores legacy cohort guard even on failure; legacy two-rank data never modified.
- Raw models use own blind inputs, legal labels and event-equal weights; cachedContext2rank history/cases exact.
- MissingT1500past must use explicit transfer/fallback, not test-label imputation or omitted predictions.
- Global/group/rank state uses each event once; neural input states and expert scores are sealed out of sample.
- Main model selected before test; rank coverage and2rank comparability never hidden by19aggregate.

### 1 Dataset and raw cache
Create analysis/multiline_dataset.py, analysis/multiline_base.py and tests/test_multiline_data_base.py.
- [x] Red/green tests for arbitraryrank cohort, strictclock release, exactcase copying, restorelegacy guard, treehistory/current-outcome poisoning and equal event weights.
- [x] Build new frozen19V3 dataset; save quality/cold-start coverage and parent comparison evidence.
- [x] Precompute blind rank features once; issue65causal shared600/context400routed base forecasts and save model checkpoints/seals.

### 2 Correction and coordinator
Create analysis/multiline_models.py, analysis/multiline_runner.py; tests/test_multiline_models.py and tests/test_multiline_runner.py.
- [x] Red/green tests for hierarchy/graph/factor masks, independentcontrol, eventmass, metadata boundaries, historicexpertloss, neuralseedreplay in subprocess.
- [x] Fix12candidates+3seeddiagnostics before formal prediction, sourcefreeze; generate65calibrationevent seals with guarded feedback and validation-only choice.

### 3 Verification and readout
Create analysis/multiline_evaluate.py, analysis/multiline_report.py and records analysis/multiline_records/20261117.
- [x] Full unittest serial; exactcase/clock/source/weights/basecontrol/NNreplay audit before scorer.
- [x] Independent all19score and original2rankpanel score, rank/time/type/band/support and paired/block uncertainty; allmethodsreported.
- [x] Final independent read-only review only after all computations, source/delivery/preservation manifests, no production promotion.

Execution choices are routine within user's explicit extension request; no additional approval or automatic commit/push. Work remains in current research folder, preserving prior uncommitted work.
