# Type hierarchy and irregular-time experiment

User-approved continuation of the deadline research. Execute inline, with one
independent review at the end. Root AGENTS.md is the binding specification.

## Frozen design

- New directory `analysis/quality_irregular_results/v1`; preserve every old run.
- Only train and fixed validation inputs/labels; no test prediction or scoring in
  this development stage. All preprocessing is fitted on training activities.
- Fixed four chronological training folds: first 98, then 25/25/25/25 activities.
  Identical cases, global activity weights, causal training history, all ranks.
- Statistical comparisons: old rank-specific remaining-rate CatBoost; additive
  spline baseline; type/rank hierarchical spline; quality-aware hierarchical
  spline; quality-aware shared CatBoost; partially pooled type-specific CatBoost.
- Hierarchical spline: degree-2 B-splines, four uniform knots; global Ridge alpha
  20, type deviations penalty x4, type/rank deviations x16. Smooth progress and
  log remaining hours; retain the 39 ordinary features. The quality version adds
  causal historical priors with reliability n_eff/(n_eff+10)/(1+span/2), plus
  reliability itself. Reliability is a heuristic feature, not a calibrated
  probability or an activity weight. Unknown shape information contributes zero
  shape deviation; pooled historical deadline/progress priors remain available.
- Tree parameters stay 500/depth4/lr.03/L2=15; type expert share is n/(n+20)
  based on distinct training activities, not prefixes. Unknown types use shared
  prediction. Quality shared and expert models have exactly the same inputs.
- Neural comparison: actual observed timestamps, maximum 32 observed nodes per
  rank, preserve first/last, no regular time resampling. Canonical rank slots;
  invalid/stale peers masked. Controls: progress, score/current, observation
  intensity log1p(original index)/log(1000), elapsed hours/168. Time-aware GRU
  reads controls and increments. Neural CDE uses piecewise-linear controls and
  Heun integration (one step per retained interval); bounded vector field.
  Both share the same 39-feature tabular input and prediction head, width 32,
  dropout .2, AdamW .001/wd .005, batch256, max80epochs, min30/patience20.
  Fixed seeds42/123/2026. Mixed loss .8 relative final error + .2 Huber(logk).
- Report every seed and arithmetic final-score ensemble, not best seed.
  Compare each statistical model and neural ensemble separately; predeclared
  25% candidate / 75% frozen reference mixtures on validation only. No adaptive
  feature additions or hyperparameter search after seeing results.
- Save full predictions, source snapshots, dependencies, models, preprocessing,
  trace and fingerprints. Metrics include event/case MAPE, MAE/RMSE/P90 and
  rank/stage/type/band. Activity bootstrap uses frozen case membership.
- Check CDE numerical sensitivity after training using two and four Heun steps
  per original interval; diagnostics only, never select a solver by validation
  score. Report any material sensitivity.
- Interpretation: validation is reused for development; earlier test exposure
  and retrospective metadata limitations persist. No causal claims.

## Execution ledger

Ruling: Work in the existing requested folder, with additive modules/new result
directories, because the user's prior instruction explicitly requests this folder
and the experiments/data are untracked here. No migration, commit or old-source
modification is needed.

- [ ] Boundary tests and statistical/irregular-time implementations.
- [ ] Run all fixed training folds and neural seeds; archive replayable results.
- [ ] Independent review, full required tests, metric/artifact verification.
- [ ] Chinese report and concise outcome with evidence and limitations.
