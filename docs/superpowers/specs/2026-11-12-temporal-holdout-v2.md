# V2: chronological activity holdout with explicit prediction-time evaluation

User request: preserve the previous protocol as V1 and add temporal testing on
top of whole-activity holdout. V2 tests the same activity from different observed
prefixes, not randomly shuffled snapshots. This is the authorized scope.

## Design choices

Keep exact 198/42/43 event memberships and the eligible event/rank cohort from
`data/multirank_v2`. The name of that older directory is a data-processing
revision, NOT this new temporal protocol. New version: `data/temporal_holdout_v2`.
Original holdout_v1, multirank datasets, model sources and results stay immutable.

Use the SAME 21 requested time stages for train/validation/test replay:
- elapsed hours: 1, 3, 6, 12, 24, 48, 72;
- progress: 10%, 25%, 50%, 75%, 90%, 95%;
- remaining hours: 48, 24, 12, 6, 3, 1, 0.5, 0.25.

Identical requested cutoffs within an event/rank are one physical case with
multiple stage memberships. Overall metrics count it once, per-stage reports may
reference it in multiple meaningful axes. Every input is a blind prefix. Keep
original eligibility (>=4 points, positive current, <=60min age); record requested
and observed horizons, age and prefix count. Report <=15min freshness separately.
Missing prefixes are frozen before predictions. Report missing eligible curves
separately from missing/stale prefixes and report distinct event counts.

First experiment deliberately fixes the existing sealed shared-rank and context
CatBoost models. Compare them with linear extrapolation and hold-current. No
retraining/recipe selection/test tuning; this isolates evaluation-time effects.
Training-event replay on this new grid is not identical to original training-fit
error. New training prefixes remain available for future V2 model training.

Primary view: event-macro MAPE by stage and rank, with MAE/RMSE/P90 and counts.
Also provide overall deduplicated event-macro MAPE, stage x rank, stage x calendar
quarter, freshness subsets, and common event/rank panels within each time axis.
The common panels distinguish time effects from changing sample composition.
For selected remaining-time stages, paired bootstrap resamples whole frozen event
groups for frozen shared model versus persistence/linear, never snapshots.

No rolling retraining on previous validation/test activity finals in this main
protocol. Such a production-history experiment would need a separately named
track; mixing it here would change the available-history policy.

## Implementation

Additive modules only: temporal_dataset.py, temporal_benchmark.py,
temporal_evaluate.py, temporal_report.py and tests/test_temporal_protocol.py.
Trusted dataset builder may access frozen two-rank SQLite and verified additional
raw curves; model entry may access only the new blind prefixes and sealed models.
Labels remain separate, read only by an independent scorer after all predictions
are sealed. Save code, source/model/data identities, actual dependency versions,
per-case forecasts, scores and reports. Never overwrite a sealed directory.

## Limits / references

V1 test cohort previously appeared in exploration; retrospective metadata lacks
point-in-time archives. V2 does not make those events newly unseen. Old vs new
aggregate MAPE is not an improvement comparison because the time grid changes.
References: https://otexts.com/fpp3/tscv.html and
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
Irregular snapshots are not independent equally-spaced series folds; split by
whole activities first and replay prescribed origins inside each activity.
