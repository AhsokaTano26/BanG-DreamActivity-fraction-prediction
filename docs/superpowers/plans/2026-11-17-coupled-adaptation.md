# Coupled adaptation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, one task at a time. User requires serial execution.

**Goal:** Test whether observable activity context and causal state improve two-rank residual calibration.
**Architecture:** Pure14method calibrator and guarded cached-forecast runner, unchanged independent scorer, source-bound report.
**Tech Stack:** Existing NumPy/pandas/SciPy/PyTorch/joblib; no dependencies added.
**Spec:** docs/superpowers/specs/2026-11-17-coupled-adaptation-design.md

## Global constraints

Same denseV3 dataset/cases, >=1706504400000, strictend<start; event-equal model weights, NN42/123/2026 final-scoremean. CPU1thread,nice19,background,one computation. Preserve production and old outputs. No menu expansion after new test scores.

## Review focus

- Cached source must be original sealed causal warm run; reject wrong history/hash/self outcome and mixed case panels.
- Current features cannot contain actuals or expert feedback; historical lag/expert inputs must retain their own preforecast values.
- Rank2000 phase missing in training gap labels falls back without deleting frozen training cases; snapshots do not change event mass.
- Expert loss updates use historical sealed forecasts, not fitted-current reconstruction; three seeds averaged as scores.
- Validation choice precedes test; no production promotion from exposed-test minima.

### 1 Pure calibrator

Create `analysis/v3_coupled_models.py`, `tests/test_v3_coupled_models.py`.
Interface `predict_all(history,target,references,*,query_start_ms,model_directory) -> scores,features,diagnostics`; `replay_neural(directory,features)`.
- [x] Red tests: clock/target outcomes; bounds/fallback; event duplication invariance; unknown metadata; conditional-gap signal; changepoint reset; fixed-share sealed losses; neural three-seed reload in isolated subprocess.
- [x] Implement fixed recipes, history-only preprocessing/robust ridge, Student-t run mixture, fixed-share and context gate; save equal event weights/model state.
- [x] Run targeted tests green before runner.

### 2 Causal runner

Create `analysis/v3_coupled_runner.py`, `tests/test_v3_coupled_runner.py`.
Interface `run_experiment(data,run,*,source_run,allow_fixture=False,resume=False)`; outputs standard rolling_residual_v3 seal and selection.
- [x] Red integration tests using old small fixture: no label-file reads, new seal before reveal, exact controls/cases, self-outcome poisoning invariant, cache identity/peer guard, completed-only resume.
- [x] Implement35+14+16cache queries with saved historical states, source identities, immutable artifacts and selection before test.
- [x] Targeted tests green, source/config freeze, single serial formal run in new directory.

### 3 Score and report

Create `analysis/v3_coupled_report.py`; records `analysis/v3_coupled_records/20261117/`.
- [x] Full `.venv/bin/python -m unittest discover -s tests -v` serial low priority; no simultaneous experiment.
- [x] Complete case/clock/equal-weight/control/source audit and saved neural replay before independent scorer.
- [x] Score once using existing rolling_residual_evaluate; all17results and paired/block CIs, training diagnostics, no test-driven reselection.
- [x] Final independent read-only review after computation; save source/preservation/delivery receipts and candidate artifacts.

Execution ruling: user's repeated instruction to continue research authorizes routine modeling choices; no extra approval handoff or automatic commit/push. Work remains in their current research workspace, preserving the prior uncommitted work. Plans and model code from earlier sealed rounds remain unchanged.


## Completion evidence (2026-11-17)

All65forecast events and17methods sealed; validation choice kernel_reference unchanged.352full-suite tests passed in803.885s,65NNreplays and1333old files verified; final_review.json passes without important findings.

New context_gate validation6.447163%,test7.041884%; previous factorizedNN6.566160%/7.861479%. Fixedprior6.538186%/7.164640%; gate incremental-.122756point has IID/block3/block5 intervals including zero. Main kernel_reference8.094901% did not beat coldEWMA7.994546%. Old coupled_gap6.539744% remains a post-hoc test lead; no production promotion.

All research computations sequential, one numerical thread,nice19,background. Before formal freeze, reviewed two formula descriptions: regression blends from realized bounded gap bias, gate inputs log(base/expert). Exact floor clipping prevents floating point mixture underflow; explicit rank flag separates ranking lines. Frozen models/spec did not change after formal run. Report and trained candidate/state inventory: analysis/v3_coupled_records/20261117/report/. Original reviewed checklist preserved in reviewed_plan.md.
