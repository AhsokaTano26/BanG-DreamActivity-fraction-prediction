# Independent review and verification

Reviewer: /root/review_predictor. Full-batch review requested under the inline
execution skill, covering the model code, test suite, runner, audit and budget
extension. No test inputs/labels/scores used.

Result: no Critical or Important findings. Reviewer independently passed all six
new tests and additionally probed unknown-type fallback and finite predictions.

Verified: frozen event partitions; causal historical pools; global activity
weights across ranks/types; unknown-history masking; CDE Heun mathematics and
padding; solver sensitivity excluded from selection; arithmetic final-score
seed ensembles; no best-seed selection; fixed25% mixtures; artifact sealing.

Minor observations and disposition:

1. The sealed training runner records Python version but its verify_run helper
   does not compare it. The independent audit now explicitly compares the actual
   interpreter version with the manifest, leaving sealed inference source intact.
2. Live DB hash could be mistaken for data provenance. Reviewed audit now names
   it unused_live_database_sha256 and separately records the immutable snapshot
   and train_curves fingerprints. Models use HoldoutDataset only.

Audit also rechecks frozen reference identity and values; saves its actual source
in each reviewed audit directory. Original audit_v1 and source remain preserved.

The reviewer agreed that a single disclosed240-epoch ceiling sensitivity check
is reasonable after all six original fits hit80, provided the original results
are retained and no continued budget search follows. Exact first80 trajectories
and all statistical predictions were subsequently verified in
budget_verification.json. All extended fits stopped before240.

The report distinguishes fixed penalties from Bayesian posterior inference,
heuristic reliability from calibrated probabilities, observed timestamps from
the CDE's linear interpolating path, and development results from test accuracy.
Earlier test exposure and retrospective metadata limitations remain applicable.

Final report review: passed with no remaining factual corrections. Reviewer
independently recomputed the spline mean decomposition (total0.4171557,
observation-count contribution0.321822, rank-gap contribution0.096052), checked
tables/intervals/budgets/solver diagnostics against saved artifacts, and verified
the primary references. Two wording improvements were made: the last fold
partially offsets earlier gains; fixed-checkpoint solver sensitivity is not an
upper bound on numerical error or a test of retraining with another solver.
