# B Adaptive Sharing Implementation Plan

> **For agentic workers:** implementation is delegated to B; B executes inline with TDD, H supplies independent audit through PI.

**Goal:** implement the frozen 12-candidate past-only sharing comparison without altering main source or historical outputs.

**Architecture:** one pure numerical module and a coordinator CLI, plus local tests. The coordinator owns input SHA checks, release, issuance and score/selection seals; the pure module sees only detached historical labels and an outcome-free current frame.

**Tech Stack:** project .venv, Python standard library, NumPy, pandas.

**Spec:** B_design.md and preregistration.json in this directory.

## Global constraints

Only write workspaces/B; no validation/test labels; start>=1706504400000; end<query start; JSON candidate order is immutable; CPU1 formal numerical run waits for PI.

## Review focus

- Equal end/query boundary and future current outcome fields must fail closed.
- Constant experts and small histories must return finite lambda0.
- Reconstructed historical experts must never replace originally issued expert values in risk fit.
- Prefix multiplicity cannot change an event's risk weight.
- Selection seals before final10 and no output overwrite.

## Task 1 — pure calibrator

- [ ] Write local failing tests for clock/current-outcome rejection, ridge oracle clipping/shrinkage, event weights, degenerate supports, shared/independent expert arithmetic, deterministic pools and only-one-rank application.
- [ ] Run them RED and save log; implement adaptive_sharing.py; run GREEN.
- [ ] Add a future-outcome perturbation fixture and prove earlier output invariance with real pure calls.

Interface: `predict(history: DataFrame, blind_current: DataFrame, configuration: dict) -> DataFrame` returns case_id, expert_ind, expert_shared, lambda, support, n_eff, A, C, sensitivity bounds, bias and prediction. Configuration includes recipe and frozen common settings.

## Task 2 — coordinator

- [ ] Write failing synthetic-35-event tests for issuance-before-release, strict end<start, whole selection block seal, no overwrite, input SHA tampering and score recomputation.
- [ ] Implement replay.py, preserving all candidate issuance files and a selection seal before event26.
- [ ] Run local GREEN. PI runs repository suite; B will not queue it during other numerical work.

## Task 3 — preregistered replay after PI permission

- [ ] Send design/source/test SHA and bounded command to PI.
- [ ] After explicit permission run once to new directory under CPU1/nice19.
- [ ] Preserve all candidate scores, per-event outputs, limitations, source/dependency SHA and elapsed time; request H independent check.

No commits, branches, or changes outside B are part of this delegated task.
