# Dense Cohorts and Rolling V3 Implementation Plan

> **For agentic workers:** Use dispatching-parallel-agents for the independent policy and V3 components; root implements dense builders. Existing user authorization covers execution without another approval gate.

**Goal:** Exclude pre-dense reporting from new V1/V2/V3 modeling and execute an auditable first rolling comparison.

**Architecture:** A shared data policy owns the minimum actual event start. A new trusted builder derives dense fixed datasets from frozen sources, leaving originals intact. A separate rolling dataset/history gate and runner implement causal outcome feedback with independent scoring.

**Tech Stack:** Project `.venv`, Python, NumPy/pandas, SQLite read-only sources, existing CatBoost and sealed dataset/evaluation conventions.

**Spec:** `docs/superpowers/specs/2026-11-15-dense-cohorts-and-rolling-v3.md`.

## Global Constraints

- Minimum actual CN start: 2024-01-29T13:00:00+08:00; never numeric EventID filters.
- Frozen original eligible queue only, new67/14/16 chronological groups shared by all versions/ranks.
- Preserve old data/results and previous uncommitted work. No dependency changes, commits or pushes.
- V1/V2 train-only; V3 strictly completed-before-current-start feedback, activity-equal supervision.
- Model selection on validation only; all recipes registered before scoring; original exposure/metadata caveats remain.

## Review Focus

- Old-event histories surviving after only filtering case rows.
- Non-monotonic event IDs, equality at cutoff/end boundary, mismatched rank dates and overlapping groups.
- Trusted builders accidentally selecting cases by outcomes or changing frozen availability.
- Rolling encoders/OOF correction observing current/future labels or resuming changed predictions.
- Old production presets/calibrations silently bypassing the new population policy.

## Tasks

- [x] 1. Implement `analysis/data_policy.py` and scoped production filtering with meaningful regression tests; expose minimum-start constant and `eligible_history` filtering. Preserve legacy reproducibility paths.
- [x] 2. Add `analysis/dense_dataset.py` and fixture tests. Build V1 from a cutoff-filtered snapshot of the frozen eligible cohort; reassign V2 case prefixes, labels, memberships and histories without changing their origin grid; emit immutable manifests and lineage/migration audit.
- [x] 3. Add `analysis/rolling_dataset.py`, `analysis/rolling_benchmark.py`, `analysis/rolling_evaluate.py` and focused tests. Implement gated history release, fixed static/rolling/EWMA controls, validation freeze, event seals, provenance and independent cached scoring.
- [x] 4. Run complete tests and independent review before formal data preparation/first experiment. Build new datasets, verify cutoff/cohorts/prefixes/peer timing and old artifact preservation.
- [x] 5. Execute first V3 two-rank comparison, seal predictions and independently score. Update AGENTS.md and README with primary new paths, exact IDs/commands and legacy-only status of old defaults; save dataset counts, limitations and a delivery audit.
