# Shared/rank/peer V3 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. User requires sequential computation; no parallel implementation agents. The architecture was accepted in the user's2026-11-18request; implement without additional permission checkpoints.

**Goal:** Train and independently evaluate four nested19rank neural residual designs, with saved reusable weights and strict causal inputs.
**Architecture:** Trusted blind peer cache, pure shared/rank/peer networks, guarded V3 coordinator, unchanged independent scorer, new audit/report. Preserve old sources and production.
**Tech Stack:** Existing.venv,NumPy/pandas/PyTorch/joblib/scipy;CPU1,nice19,background.
**Spec:** docs/superpowers/specs/2026-11-18-task-learning-design.md

## Global Constraints

Same dataset db4e9b9c...,97activities67/14/16,19ranks,all cases/start cutoff.32basewarm+35prequential training. Four modes×3seeds,21outputs,5eligible choices.60epochs/min12/hidden16/lr.002/shrink20/logbound.2 exactly as spec. All peer observations<=targetOWNobserved_at. No target outcome in model; graphs saved before issue. Old results never rewritten. New namespace only; no commit/push/deployment implied.

## Review Focus

- A later blind prefix contains earlier public observations but also future points: only trusted preparation may inspect its source container; detached model batches must expose no later values.
- Rank missing/stale/current clock disagreement cannot create fabricated zero observations or fit normalizers on current inputs.
- Same-event errors are unavailable at prediction time; saved historical graph/support features must never be reconstructed using later outcomes.
- Both-zero attention Q/K and multiplicatively zero heads/gates can silently kill learning; all modes need gradient/replay checks and matching initialization.
- ColdT1500 has no training labels or dedicated-head evidence; retain all cases and use only declared continuous-rank fallback, not heldout correlations.

### Task1: causal peer feature preparation

**Files:** create analysis/task_peer_data.py, tests/test_task_peer_data.py.
**Interface:** build(data,destination)->manifest; PeerCache(root).batch(case_ids)->detached PeerBatch(nodes,node_events,links,ages,frame,ranks); batches have exact own-node pointers and clocks.
- [x] Write and run failing tests for future peer poisoning, own-clock alignment, missing/stale mask, exact cases/source guards and training-only data boundary.
- [x] Implement trusted cache with deduplicated nodes, own-prefix verification and immutable manifest; run focused tests green.
- [x] Build formal blind cache once and save coverage/clock/provenance evidence.

### Task2: graph features and shared/rank/peer networks

**Files:** create analysis/task_learning_models.py, tests/test_task_learning_models.py.
**Interface:** relation_state(history,current_start,ranks)->JSON-safe graph/support; fit_predict(past_batch,current_batch,past_frame,current_frame,issued_graphs,model_directory,recipe)->scores/details; replay(directory,current_batch,current_frame,graph)->seed/mean predictions.
- [x] Test strict history/outcome guards, event-based graph evidence and cold rank fallback; then implement and pass.
- [x] In isolated subprocess test zero residual initialization, three-seed learning/replay, identical own initialization, missing-peer zero effect, zero cold adapter, history-only preprocessing and equal event weights; implement and pass.

### Task3: sealed sequential coordinator

**Files:** create analysis/task_learning_runner.py, tests/test_task_learning_runner.py.
**Interface:** run(data,run,reference_run,peer_cache,allow_fixture=False,resume=False)->selection; output compatible with unchanged multiline_evaluate.
- [x] Write/run failing integration test for exact old controls/all21methods/cases, immutable historical issue graphs, before-test selection, current-outcome poisoning and completed-only resume.
- [x] Implement own sealed releases, saved relation states/batches/NNweights, validation eligibility and output seals; focused tests green.
- [x] Freeze spec/source/recipes/dependencies and execute65query forecasts once serially.

### Task4: verification and independent readout

**Files:** create analysis/task_learning_audit.py, analysis/task_learning_report.py; records analysis/task_learning_records/20261118.
- [x] Run full .venv/bin/python -m unittest discover -s tests -v serially, then replay all saved seed predictions and audit input clocks/weights/scalers/issue graphs/history/selection and old-file preservation before scoring.
- [x] Run unchanged analysis.multiline_evaluate independently; report every candidate/seed, matched ablations and original2/cold panels with limitations and trained artifact inventory.
- [x] Conduct one sequential read-only final review, resolve material findings, save completion/source/delivery/preservation manifests. Do not promote test winner.

Execution rulings: work in current research branch/folder (user preference and necessary uncommitted local data/source); reuse existing environment; full suite368green from prior completed task is baseline evidence, run current suite once after changes; no automatic per-task commits in this shared uncommitted research workspace.
