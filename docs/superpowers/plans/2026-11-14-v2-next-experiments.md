# V2 后续实验 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for implementation; independent method modules may be delegated following dispatching-parallel-agents. The user has explicitly requested execution of the preceding recommendations.

**Goal:** Complete reproducible V2 experiments on the frozen two-rank dataset, select only on validation, and report all results without replacing prior experiments.

**Architecture:** New family modules implement the established blind-array interface. A new coordinator registers configurations and provenance, saves independent development checkpoints, freezes selection, emits blind predictions, and invokes the existing independent evaluator. Existing sealed sources and runs remain unchanged.

**Tech Stack:** Project Python `.venv`, NumPy/pandas, CatBoost, PyTorch; separately pinned optional foundation-model dependencies.

**Spec:** `docs/superpowers/specs/2026-11-14-v2-next-experiments-design.md`.

## Global Constraints

Apply root AGENTS.md in full. Frozen V2 fingerprint `1f76f39af8151bba185496411bbcee064b16a4c5aca5c515c1d957a5056fe854`; 2229/1548/1514 cases;196 fitted activities. All supervision uses equal total activity weights. Three NN seeds42/123/2026. Only training histories and chronological OOF errors can construct features/calibration. Never fit on validation, read test answers during development, or overwrite sealed results.

## Review Focus

- Auxiliary target construction across sparse gaps and beyond activity boundaries.
- OOF calibration accidentally including the query activity or incomplete historical activities.
- Unequal activity mass in augmented or foundation-model contexts.
- Foundation model failure causing silently dropped cases or an undisclosed replacement.
- Resume/provenance checks permitting changed source, weights, or predictions.

## Tasks

- [x] 1. Implement `analysis/v2_next_growth.py` with terminal controls and multi-horizon ablations; add `tests/test_v2_next_growth.py` boundary, leakage, equal-weight, monotonic-integral and subprocess round-trip tests.
- [x] 2. Implement `analysis/v2_next_adaptation.py` with legal recent-history features and OOF EWMA controls; add `tests/test_v2_next_adaptation.py` time-boundary, history-injection, OOF and activity-weight tests.
- [x] 3. Inspect foundation-model feasibility using official sources, then implement `analysis/v2_next_foundation.py` and focused tests; freeze package/weight identities and document unavailable candidates without substituting models. Coordinate dependency installation through the primary agent.
- [x] 4. Implement `analysis/v2_next_runner.py` using existing blind-array helpers and evaluator contract; include old selected predictions as a preregistered baseline, validation-only combinations, cached development artifacts, immutable blind seal, source/dependency hashes and round-trip checks. Add fixture tests.
- [x] 5. Review implementations and execute the full project test suite. Freeze source/configuration before formal fitting. Run all available training prefixes, three chronological internal folds for trainable new families, and three NN seeds. Save unavailable/cost-limited foundation evidence before final candidate registration.
- [x] 6. Seal all test predictions; run `.venv/bin/python -m analysis.temporal_evaluate --run analysis/temporal_results/v2_next_20261114`. Independently compare existing predictions, calculate paired and block-bootstrap sensitivity, and publish a local report with full stage/rank coverage and limitations.
- [x] 7. Verify all saved models/predictions/manifests, prior data and runs; rerun required tests after final source changes. Report improvements or failures candidly. Do not commit/push unless requested for this new round.
