# Temporal V2 execution

Spec: ../specs/2026-11-12-temporal-holdout-v2.md. Native implementation, authorized
by user request; preserve all old artifacts and unrelated working-tree changes.

- [x] RED/GREEN dataset tests: grouped activity memberships, alias deduplication,
      future cutoff clipping, stale/missing cases, immutable labels/artifacts.
- [x] Build and seal temporal V2 from verified existing source snapshots.
- [x] RED/GREEN benchmark/scorer tests: frozen models, exact case coverage,
      same-case baseline comparisons, common panels, event-unit weighting,
      scorer collision protection and idempotence before reading test labels.
- [x] Run frozen-model predictions, independent scoring and time-grid report.
- [x] Independent code review, full unittest suite, source/model/dataset checks.

Review focus: identical cutoffs must not become independent observations;
near-end requested time differs from stale observed time; group membership cannot
follow event-ID order; no complete validation/test curve reaches inference; per-
stage case sets and missingness must stay identical across methods.

Completed: temporal dataset 5f1d11f35e0da7b0fa02a0a056659ff75390efab62bf06545debb2c82e9f4a27; fixed-model predictions and independent evaluation sealed. 15 original method/stage comparisons reproduce exactly. 96 tests passed. User-authorized V2 documented in AGENTS.md; old source/model/data/DB identities unchanged.
