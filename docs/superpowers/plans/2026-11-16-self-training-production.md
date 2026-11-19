# Self-training production implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver and start a local trained forecasting model that fetches new data, predicts repeatedly and learns verified completed outcomes automatically.
**Architecture:** Independent data store/sync and immutable model/calibration components; a trusted scheduled coordinator binds their clocks, pinning models per event and exposing outputs/status/rollback.
**Tech Stack:** Existing .venv, sqlite3/httpx/CatBoost/sklearn; macOSlaunchd. No new third-party dependencies.
**Spec:** docs/superpowers/specs/2026-11-16-self-training-production-design.md

## Global constraints
- Dense actual-start cutoff1706504400000; no current-event/future outcome fitting.
- Rank scope configurable,19 default pending optional user response; T1000/T2000 dedicated V3base+EWMA.
- Old benchmark data/results remain immutable. Production metrics clearly distinguished.
- Runtime model versions/issued forecasts immutable; atomic promotion and idempotent feedback.

## Review focus
- Feed retries/delayed finals cannot train on unfinished or invalid targets.
- Activity pins avoid changing a current event's model/calibration unexpectedly.
- Same observations/history do not cause duplicate feedback or repeated training.
- Failed updates preserve useful last model/output and expose health.
- Machine restart/awake/login behavior is explicit and tested where possible.

### Data component (agent)
- [x] production/data.py + tests/test_production_data.py: bootstrap local history, schema, quality guards, incremental bounded public sync and metadata snapshots.
- [x] InterfaceStore(root), load_curves(before_ms,ranks=None,complete_only=True), sync_data(store,now_ms,ranks,transport=None); stable documented outputs.

### Model component (agent)
- [x] production/model.py + tests/test_production_model.py: fit immutable modelpacks, predict base+EWMA, validate/reload versions, persist feedback states; models only receive detachedcurves/metadata.
- [x] Freeze interfaces with root coordinator before implementation.

### Coordinator and service (root)
- [x] production/runtime.py, cli.py, launchd.py + lifecycle tests: locks, pins, feedback ledger, registry, statuses, atomic forecasts, schedule install/remove.
- [x] Bootstrap/sync real dense history, train full recipe, run real prediction and inspect artifacts.

### Verification/delivery
- [x] Fulltests and independent review, repair material findings.
- [x] Install/start local schedule, inspect launch status and one real completed cycle.
- [x] docs/production/README.md + runtime model card/commands, preserved old artifacts check and evidence receipt.
