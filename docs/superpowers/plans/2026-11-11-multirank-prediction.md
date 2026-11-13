# 19-rank continuation: design and execution ledger

User scope: resume after downloading 19 CN ranks; produce simultaneous forecasts.
Implementation choice: Python in the existing .venv, native execution; no changes
to old sealed model sources or holdout_v1. User already authorized continuation.

## Design

Finish independent ingestion audit and required 80-test suite first. New data is
not itself a validated 19-rank model. Construct `data/multirank_v1` as a separately
named supplementary dataset, inheriting EXACT event memberships from validated
holdout_v1. Existing T1000/T2000 inputs/labels/train curves are copied verbatim;
additional ranks use the same quality rules and stages. Events absent from v1 are
excluded from this supplementary benchmark. No replacement/redivision of v1.
Full histories only for training. Prefixes are gzip JSONL to preserve original
observations without an enormous uncompressed footprint. No test-label access
from model training or recipe selection. Fixed case lists and manifests.

Compare linear extrapolation, a shared rank-aware CatBoost, and its metadata
variant (type, band, CN-released cards). Use continuous log-rank to share patterns
with scarce/unseen ranks such as T1500, train-only encoders and event-balanced
weights. Select by validation event-macro MAPE; seal all blind test predictions,
then use a separate final scorer. This is supplementary performance, not a claim
of improvement on the unchanged two-rank benchmark. No repeated test tuning.

Run selected trained model on the latest local activity at a common recorded
as-of time. Every forecast uses its observed prefix only. Save current points,
model, data/code/software fingerprints and predictions for all ranks. Missing or
stale inputs must be explicit failures, not invented cutoffs. Current forecasts
are provisional if a rank lacks held-out support; no unsupported confidence
intervals. Further neural research is outside this first 19-rank baseline.

Alternatives considered: 19 independent models would waste sparse data; applying
the old two-rank recipe unchanged would conceal unsupported rank handling. A
shared rank model is the bounded first implementation. More complex joint neural
models require this same fixed supplementary dataset before comparison.

## Tasks / checks
- [x] Complete real ingestion audit, post-import tests and saved inventory.
- [x] Test then implement immutable supplementary dataset + blind prefix reader.
- [x] Test then implement rank-aware features, event weights, train/validation fit,
      sealed prediction artifacts, separate scorer, and 19-rank local forecast.
- [x] Run real frozen supplementary experiment once; independent final scoring.
- [x] Required full suite, fresh code review, report current 19-rank forecasts.

## Review focus
Future full-curve suffixes cannot affect features; event boundaries follow v1;
zero final scores are never MAPE targets; T1500 may have no train/validation
examples; malformed/duplicate predictions cannot silently drop cases. Frozen
artifacts and completed runs are never overwritten.

## Constraints / limitations
Train weights sum equally per event. No validation/test finals in historical
features. Test events previously exposed in exploration, retrospective metadata
without point-in-time archives. Keep both caveats in every evaluation report.
No git commit/push; preserve unrelated working-tree content. Implementation and
new artifact paths are additive; root README and old model sources stay unchanged.

Progress: ingestion audit and 80 post-import tests passed. Initial supplementary v1 sealed without T10 labels; test labels never scored. Found T10 continuing polls 33–38 hours after end. Ruling: additional-rank histories truncate after end+300s before applying unchanged [-60s,+300s] final window; v2 is separate and v1 remains preserved. Test for late tail rejecting a valid near-end label failed then was fixed. Main holdout_v1 unchanged.
Review: fixed unlisted archive file admission and scorer collision overwrite, RED→GREEN; added per-rank support/provisional flags and actual live dependency provenance. Full pre-window-change suite 86/86 passed; final suite pending.

Completed: v2 dataset e94e249b47b6d542faa997ff82c4cccd29187bb3409aba74069b050e313d5e7a; selected catboost_rank by validation 7.0006559%; independent test 8.4072939%, 43 events / 3757 cases. Current event ended during work, so explicitly archived 22:00 replay with only preceding observations. All 19 scores reproduced. Final suite 88/88 passed. No new neural architecture trained; first shared-rank baseline delivered.
