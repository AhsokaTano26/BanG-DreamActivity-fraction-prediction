# Relationship study

User authorizes statistical exploration and improved methods within the fixed benchmark. Analysis uses frozen training data; validation selects recipes; test is scored once after all candidates are frozen. No changes to holdout_v1 or old runs.

1. Train-only event-level statistics: paired ranking correlation, era/type/duration/band relationships. Prefix-level weighted regression uses event-cluster robust uncertainty, rank/progress interactions, and Benjamini–Hochberg adjustment. Coefficients are conditional associations, not causal effects. Record density/year confounding and limited independent event count.
2. Engineer compact prefix-only features: clock-to-end, growth velocity/momentum, canonical 1000/2000 gap and gap changes, sampling quality. Ablate bands, type, peer, quality and absolute score scale. Try separate-rank experts and spline additive regression. Every activity retains equal training weight; no recency weighting or label updates.
3. Fixed tree candidates: CatBoost RMSE/MAE and LightGBM with modest fixed capacity; variants full/no_band/no_peer/no_type/no_quality/invariant/invariant_rank. Spline ridge penalties 1/10/100. Train-only chronological cross-validation evaluates group ablations; official validation alone chooses final recipe and equal top-three combination. Old v1 selected ensemble is a preregistered reference; 25/50/75% mixtures may be selected on validation.
4. Freeze checkpoints, recipes, dependencies, sources and blind predictions in a new directory. Final scorer is unchanged. Report every candidate and paired event comparison to old v1, including failures. Test results do not trigger tuning. Keep prior test exposure and retrospective metadata limitations.

## Completed evidence

- 198 train groups; paired finals correlation .9793 across 132 complete pairs. Train-only event-cluster WLS/BH and year-fixed-effect sensitivity distinguish heat/absolute score from remaining growth; band and sampling-density significance is sensitive to era controls and is reported as such.
- Added a remaining-average-rate multiplier target alongside log-growth residual candidates; target inversion is exact on all training cases. Candidate set fixed before testing: 23 fitted recipes, reference, top3 and three preregistered blends, 28 reported methods total.
- Independent review found rank-expert reweighting, scale-dependent smoothing, missing-interaction construction and absent environment-version validation. Each received a failing regression test before repair. All fixes occurred before test scoring. Review-only v1/v2 development directories retained without test evaluation; final run is relationships_v3.
- Final suite 53/53 passes. Fixed test: validation-selected blend 8.1403% vs old 8.9202%, 32/43 events improve; paired event bootstrap difference CI [-1.1123,-0.4623] percentage points. 1000/2000 MAPE 12.5438%/3.7368%. Validation 3.8852% vs 4.3206%.
- The single remaining-rate CatBoost expert has a lower observed test score (8.0558%), but was not selected retroactively. No changes followed final scoring.
- Verified all 28×412 prediction identities, full dataset/development/prediction/scoring hashes and independent macro calculations; unchanged source DB hash. Full findings and statistical limitations are in analysis/holdout_results/relationships_v3/README.md.
