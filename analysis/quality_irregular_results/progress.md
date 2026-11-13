# Execution record — quality hierarchy and irregular-time experiment

Specification: root AGENTS.md. Plan: docs/superpowers/plans/2026-11-11-quality-irregular.md.
Explicit supplement: docs/superpowers/plans/2026-11-11-irregular-convergence.md.

- Complete: meaningful boundary tests failed before implementations existed,
  then passed. Future clipping, observed-node identity, units, padding, missing
  peers, Heun analytic convergence, gradient/reload, global activity weights,
  type sample size and train/validation-only integration are covered.
- Complete: six statistical recipes, four fixed chronological training folds,
  and two observed-time neural structures with all three required seeds.
- Complete: v1 sealed with max80 epochs. v2 separately registered and sealed
  with max240 after observing boundary-limited optimization; all other settings
  unchanged. Six extended runs stopped at114–195 epochs under original patience.
- Complete: all first80 epoch traces match exactly; all statistical control
  predictions match exactly. Every final model is independently replayed on
  both training and validation, and all metrics and seed means verified.
- Complete: independent review found no Critical/Important defects. Two minor
  audit items addressed without modifying sealed model code; see review.md.
- Complete: required unittest suite69/69; dependency check48 compatible.
- Complete: Chinese report, all negative findings, primary references and limits.

Ruling: Keep the initial candidate plan and supplement immutable; this separate
ledger records completion without changing their archived source fingerprints.

Ruling: Retain original audit_v1 and its source snapshot after enhancing independent
audit to check Python version/reference identity and distinguish unused live DB
from frozen provenance. Use audit_v1_reviewed and audit_v2 in the final report.

No new test data was requested or scored. Existing production defaults retained.
Existing test exposure and retrospective metadata limitations remain applicable.
