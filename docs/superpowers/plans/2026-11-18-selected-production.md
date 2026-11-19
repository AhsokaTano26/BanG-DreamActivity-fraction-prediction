# Selected Production Recipe Implementation Plan

> Execute inline with superpowers:executing-plans. User selected the existing tested recipe; no further design/permission checkpoint. One sequential read-only reviewer after tests; no parallel implementation or training.

**Goal:** Make legacy_pair the default for new production activities and keep self-updates running.
**Architecture:** Pure selected calibration, trusted retrospective feedback migration, versioned worker/CLI/schedule over unchanged base packs and ledger.
**Tech:** Existing.venv,NumPy/pandas,SQLite,launchd; serial low priority.
**Spec:** docs/superpowers/specs/2026-11-18-selected-production-design.md

## Constraints and review focus

ActualCNstart>=1706504400000; strictend<activitystart; no own/future outcomes; all19ranks. Preserve old pins, predictions, source and weights. No test-best result presented as untouched confirmation. Missing pairs use the exact fallback. Seed RAW errors only; live feedback overrides retrospective seeds; archive replaced seeds. A running old cycle must not be killed during schedule migration. New default/effective old-pin method must be distinguishable. No commit/push or dependency changes.

### Task1: pure recipe and parity
- [ ] RED tests for hand-calculated coupled correction, missing pairs, cutoff/duplicates/bad clocks, bounded outputs and old-pin compatibility.
- [ ] Add production/selected_model.py; pass tests and all65frozen-query prediction parity, without opening label files.

### Task2: worker, seed and schedule
- [ ] RED tests for archived/idempotent seed replacement/live precedence, stable old/new pins, failed base update fallback, selected schedule and refusal of foreign installation.
- [ ] Add production/selected_feedback.py,selected_runtime.py,selected_cli.py,selected_schedule.py; pass focused tests.
- [ ] Document current entrypoint and current-activity pin behavior in docs/production/selected-model.md.

### Task3: verify and activate
- [ ] Full.venv unittest suite serial; source/prediction/old-weight preservation and exact historical parity.
- [ ] One sequential fresh read-only review; resolve material findings.
- [ ] Back up runtime/config/SQLite/schedule; atomically select default, run local cycle and install selected worker without changing current pin.
- [ ] Verify current output/default/effective method/schedule/automatic updates; save deployment receipt and artifact seal.
