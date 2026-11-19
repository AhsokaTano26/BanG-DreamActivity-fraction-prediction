# Small shared/rank/peer residual networks over frozen19rank V3

User accepted the preceding architecture and requested implementation/experiments on2026-11-18. Continue in this research folder and existing .venv, one computation at a time, CPU1,nice19,macOS background. No production change, automatic commit, push or dependency installation. All old data/source/results remain immutable. This document records implementation choices within that accepted scope; no further approval checkpoint is required.

## Question and alternatives

Compare a shared residual encoder with one rank-conditioned output, small individual-rank adapters, contemporaneous related-line messages, and phase-dependent relation gating. A separate full network per rank has much less independent activity support and larger maintenance cost; unrestricted sharing has already caused negative transfer. Choose the small nested architecture, retaining the present independent warm EWMA predictor as its immutable reference. This is an incremental multi-task/attention design inspired by shared/private expert methods, not a claim to reproduce fullPLE or MMoE.

## Frozen population and reference

Dataset `data/rolling_holdout_v3_dense_19rank_20261117`, ID `db4e9b9c84bd64a6de5ee30a090cfac83efbe85c2f6dd899304516e7d031cdf1`; same97activities67/14/16, actualCNstart>=1706504400000, all frozen21984/4361/5220prefixes. Preserve19ranks and exactcase/time membership; no new curve filters. Source reference `analysis/rolling_results/v3_multiline_calibration_20261117`, runID `14888ceeff945c1fd387a0e03265ddde1283901a654a678b9b81ac4a71699f5d`. Its `ewma_reference` is the anchor. Also retain `uncorrected`, `legacy_pair`, previous `neural_gate`, previous `graph1` as descriptive controls. Only anchor and four new seed ensembles are eligible for this experiment's validation selection; old alternatives and seed members are ineligible. The old test minimum is not promoted or reselected.

Use the same32initial base activities,35training-period causal forecasts,14validation and16test queries. Every query sees only start-frozen, strictly completed and own-sealed earlier forecast outcomes. Training targets are the remaining errors of previously issued anchor predictions. Never use fitted/in-sample tree errors. Validation chooses once before the first test; no early stopping, test tuning or expanded menu after scores.

## Trusted blind peer cache

New `analysis/task_peer_data.py`, output `data/peer_prefix_v3_19rank_20261118`. This is a blind feature cache, not a new evaluation split. Trusted preparation uses only existing `*_inputs.jsonl.gz`, case indexes, manifest and metadata already present in blind indexes; never reads any labels, SQLite, complete curves or history vault. The longest existing blind prefix per activity/rank is a source container. For each exact frozen target, reconstruct companion prefixes strictly through its OWN last-observed time, not through the requested cutoff. Snapshot quality is4points, positive score and<=60min peer age; missing peers remain masked. Own prefix is checked against the exact target record.

Deduplicate encoder nodes by(activity,rank,last-observed timestamp). Save numeric own-prefix features using existing snapshot/33relative-prefix coordinates, logrank, log-transformed elapsed/remaining/duration and observed calendar time. Keep all original cases, node clocks, source point counts, peer node IDs/masks and relative ages. The model receives only detached batches selected by the coordinator. It never receives a whole cache, dataset paths or future-prefix source containers. Future peer values must not affect earlier extracted features. Node scaling is fit on historical OWN nodes with equal activity mass; peer-only observations and current nodes never fit preprocessing.

Formal cache seal is pinned to `015fe62951413c8237bcbda98814c07d131135acb1e3bb56458cff3b3591fdf2`; runner additionally checks exact rank scope, all7blind source hashes and frozen builder source before creating a run. A valid self-seal/dataset ID alone is insufficient. Explicit fixtures bypass only this formal pin/source requirement, retaining dataset/rank/case/clock guards.

## Causal relation features

Predictive relations are estimated from previously sealed anchor log-errors log(anchor/actual), one mean per activity/rank (static) or activity/rank/phase (phase model). Phase edges are[.25,.9], right-boundary convention. For each pair with>=8shared activities, use Spearman rho, shrink n/(n+10), positive-rho squared plus remaining0.1*exp(-abs(logrankdiff))prior. Otherwise use the continuous logrank-distance prior. Exclude self, retain top3deterministically (rank-order ties), normalize. Keep pair counts and missing correlations explicitly; T1500 has zero own support initially and a declared rank-distance fallback, never heldout-derived edges.

Each query saves its graph BEFORE predictions/outcome release. Historical NN training rows use the relation graph and own-support count saved at their OWN issuance, not a graph reconstructed with later outcomes. Current query uses the graph of its legally released history. Missing current peers remove edges and renormalize; no available peers gives an exact zero peer contribution.

## Fixed architecture menu

Four modes, each with seeds42/123/2026 and arithmetic mean of final-score predictions:

1. `shared_single`: shared prefix encoder16SiLU16, continuous logrank/context, one shared residual head8SiLU1.
2. `rank_heads`: same plus one small linear adapter per rank, zero initialized; multiply by issuance-time ownN/(ownN+10). Cold rank adapter is exactly inactive.
3. `related_static`: same plus a shared peer residual head over the static top3message, with a learned constant sigmoid amplitude initialized atlogit-2.
4. `related_phase`: same but uses the appropriate saved phase graph, query/key attention over available top3peers with age penalty, and an amplitude gate conditioned on own encoding/context. Q/K use ordinary seeded initialization (not both zero); residual output heads are zero initialized. These phase graph and gate changes form one composite final ablation.

Instantiate identical module shapes and RNG order for all modes, disabling unused paths so own-path initialization matches seeds. Context includes known activity type/band vocab fit only historical cases, phase one-hot, continuous logrank/progress, own current/reference growth, duration/remaining and base-family flag. Missing numeric fields get historical weighted means plus missing indicators; std floor1e-4 and standardized clipping[-8,8]. No historical outcome summaries except declared causal support/graph features.

Training: minimum12released forecast activities, otherwise return anchor.60epochs full-batch AdamW lr.002,weight_decay.05, dropout.1 in shared encoder/head, CPU1, no early stop. Overall residual shrink N/(N+20). Bound its log change to +/-0.2 using0.2*tanh(output/0.2), apply anchor*exp(delta), then the existing current..current*exp(4) bounds. Optimize activity-equal actual final MAPE plus0.05*weighted squared delta. Each activity total training weight1. Node/context preprocessing uses historical rows only. Save preprocessing, issuance graphs, training weights, all seeds/optimizers/losses and source/dependency fingerprints.

The anchor already includes original calibration. Do NOT deduct its old bias a second time or apply an extra old-error correction. New networks learn remaining anchor error; any future post-network online calibrator would need its own sealed final-predictor errors and is outside this initial ablation.

## Evaluation and delivery

21reported methods:5source controls,4new ensembles,12seed diagnostics. Validation chooses among anchor+4ensembles on all4361cases. Independent unchanged `analysis.multiline_evaluate --run <newrun>` scores all5220cases, original2panel560, T1500 and shared18 panels, rank/stage/type/band and paired/block uncertainty. New run `analysis/rolling_results/v3_task_learning_20261118`. New records `analysis/task_learning_records/20261118`.

Tests must prove prefix-clock/future poisoning/missingness, shared normalization history, strict release and immutable historical graphs, true zero cold support, equal activity weights, mode isolation, three-seed checkpoint replay, exact old controls, complete frozen panels and selection-before-test. Full unittest suite serial. Before independent scoring, replay every saved seed and audit graph/normalization/sample/selection/preservation. Then report all methods, primary selection, original2 comparison to historical6.5397 and coldT1500limits. Preserve exposed-test/reused-validation/retrospective-metadata caveats. Single sequential read-only final reviewer; no parallel implementation agents.
