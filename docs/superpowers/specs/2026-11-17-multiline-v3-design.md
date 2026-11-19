# Extend paired residual calibration to19ranks

User authorizes extension beyond two lines to previously requested19ranks. Preserve all old datasets/results and running production. One research computation at a time, CPU1thread,nice19,macOS background. Existing Python dependencies only. No deployment/commit/push implied.

## Population and information

Parent `data/temporal_holdout_v2_dense_20261115` ID8ffd781fc9f4ebcad9ea691839777458e5e4c7cbeade58978dca15bce2440ad8, same97activities67/14/16 and actualCNstart>=1706504400000. Copy exact21984/4361/5220cases, time memberships and curve eligibility into a NEW generalized rolling dataset. Existing two-rank rolling dataset and its `_cohort` contract unchanged. T1500has0train,0validation,14testcurves; T10has13test/13validation. Report cold-start and missingness, do not add or move curves. Two-rank case subsets equal previous2452/488/560 and remain separate comparison panels. All19aggregate is NOT comparable to oldtwo-rank6.54%aggregate.

Trusted preparation reuses old build_rolling's complete-curve/prefix/answer reconciliation with an explicitly scoped19rank cohort adapter, restores original function afterwards, then reseals staged output as rolling_multiline_v3 with correct ranks and adapter source. Only this preparation may read frozen answers/archives. New dataset exposes original guarded release methods, never generalizes the legacy two-rank class in place.

## Base cache

Initial32train activities; issue35train warm forecasts plus14validation and16test queries, chronological by actual starts. Each query learns only fully ended earlier activities; prediction sealed before its outcome release. For17ranks fit a single shared rank-awareCatBoost600/depth5/lr.035/l2=12/seed42/thread1 on all legally completed19rank prefixes (event-equal weights). Features identical existing multirank.feature_row(metadata=None): own blind snapshots and logrank, no peer suffix or full-curve features; imputation/vocabulary fitted only legal history. Precompute blind feature rows once, then mask fitting pools. Save each model/preprocessor/weights/history/issued scores.

ForT1000/T2000 reuse exact uncorrected400treeContext predictions from immutable `v3_next_warm_20261116`, including causal train warm predictions. Verify source seals/history/clocks and own frozen prefix fields against prior two-rank dataset. New shared model still trains on every available rank but its two-rank outputs are diagnostic only; issued baseline routes to Context for those2lines. This preserves the original two-rank reference while matching production's routed model families. New information comes from other ranks' strictly completed errors. No current-event cross-rank observation features, so no later peer snapshots can enter.

## Calibration fixed menu (11candidates+3NNmembers)

All predicted error is log(raw/actual), grouped to ONE observation per completed activity/rank/phase (edges.25/.9). Same per-event start-frozen history for all its prefixes; scorebounds current..current*exp4 and logbias±.3.

1. uncorrected.
2. ewma_reference: independent warm rank/phaseEWMAalpha.3,n/(n+10).
3. ewma_cold: same, validation/testfeedbackonly.
4. legacy_pair: old exact coupled_gap for1000/2000; ewma_reference for others.
5. hierarchy: global slowanchor from per-event median errors among1000/2000/3000/4000/5000, alpha.2,n/(n+10); four rank bands <=50,<=500,<=5000,rest. Group deviation from event global anchor EWMAalpha.4,n/(n+10); rank deviation from event group median alpha.6,n/(n+5). Missing current rank history interpolates rank deviations in logrank within its band, else0. Missing global event anchor falls back to event median. Rank/group/global evidence each activity equal, not prefix counts.
6–7. graph025/graph1: start with per-rank historical mean mu and own rawEWMA alpha.3; smooth deviations around mu on adjacent-logrank chain with weights1/(1+logrankgap),lambda.25/1, observed-node diagonal1,unseen-node diagonal.1. RHS for seen ranks=(rawEWMA-mu)*n/(n+10); unseen0. Interpolate unseen mu in logrank. Return mu*n/(n+10)+smooth for seen, interpolatedmu*globalN/(globalN+10)+smooth for unseen. No claim smoothing always helps abrupt rank-specific jumps.
8–9. factor2/factor4: per-phase activity×rank mean-error matrix; history-only means/scales(std floor.02), missing entries standardized0, PCA2/4dimensions. Interpolate unseen rank means/scales/loadings in logrank. EWMAfactor scoresalpha.5 plus idiosyncratic per-rank EWMAalpha.6*n/(n+10); total seen prediction times n/(n+10), unseen uses globalNshrink. Activity rows have equal PCA mass, missing rank masks preserved. All factor estimation only current legal past; no current outcomes.
10. share_gate: four experts warmEWMA,legacy_pair,hierarchy,factor2 with prior[.4,.2,.2,.2]. Per-rank/phase fixed-share weight update only on that past event's SEALED expert scores: eventMAPEclip[0,1],eta5,share.05. No per-snapshot repeated activity updates.
11. neural_gate: same4experts/prior; history-only numeric/categorical preprocessing, logrank/base-family-flag/progress/growth/rates/logremaining/logduration plus causal ownfast/ownslow/ownN/group/global states and current log(raw/expert). Phase/type/band categories. SharedMLP8SiLU/dropout.1/zerooutput4, prior-logit softmax, n/(n+20) weight shrink toward prior, min12past forecast activities;80epochsAdamWlr.001wd.05, event-equal actual finalMAPE,3seeds42/123/2026,CPU1. Arithmetic3finalscoremean. Historical state/expert columns must be those saved at that event's forecast, not refitted using later history.
12. prior_mix: fixed final-score mixture of the same4experts with weights[.4,.2,.2,.2], for a matched neural control.

Thus FINAL12candidates+3NNmembers=15reported methods. Cold controls isolate extra calibration history; legacy_pair isolates multi-line sharing on original two-rank panel; prior_mix isolates learning versus expert composition. No extra candidate/parameter menu after new scoring.

Validation-only minimum event-macro (all19fixed validation cases) selects; seed members ineligible. Test inputs are blind; fixed online updates allowed only after prior seals/end. Separately seal calibration outputs before independent scoring with a NEW scorer using generalized dataset and old frozen metric/bootstrap functions. Score all19main and original two-rank panels; 5000pairedactivitybootstrap and blocks3/5; allranks/times/types/bands,MAE/RMSE/P90 and cold-start support. Do not substitute new totals for oldV3two-line main results. Exposed test/reused validation/retrospective metadata caveats mandatory. Save trained candidate states and weights; production unchanged.
