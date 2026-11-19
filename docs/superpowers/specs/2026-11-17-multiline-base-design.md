# Immutable19rank causal base-cache specification

Part of the user-authorized extension in2026-11-17-multiline-v3-design.md. This raw cache is fixed independently of correction menus and reads no independent labels or scores. No base hyperparameter selection on heldout events.

New generalizedV3dataset copies exact dense19V2 population97events67/14/16,21984/4361/5220prefixes and actualstart>=1706504400000. Initial32train events establish base history;35remainingtrain +14val +16test are forecast chronologically, release only end strictly<query start after prior prediction seal. T1500has0training/validation curves,14test; keep it as a cold-start interpolation target.

Use original own-prefix multirank.feature_row(metadata=None), fixed schema, numeric logrank and33normalized prefix coordinates, no other-line/current suffix features. Cache blind features once; preprocessing and target fitting only on legal completed histories, all available prefixes with equal event totals.

SharedCatBoost600depth5lr.035l2=12seed42threads1 fits all available19rank past prefixes. Target log(actual/current)+log(observedprogress); decode current*exp(clip(rawresidual-logprogress,0,4)). Issue shared predictions for17ranks. For1000/2000 issue exact uncorrectedContext400 forecasts from immutable v3_next_warm_20261116 run1a2aea3070785aaa377817f105239961b6b8582749ffe6ba3369fe6ffe81a9a7. Verify source identity, old two-rank dataset clocks/history, exact frozen case/own prefix values. Shared2rank predictions are diagnostics, not issued baselines. Save model/preprocessor/weights, currentblind scores, full legalhistory identities and sourceeventseals for every query; seal before any own outcome release.

One computation at a time,CPU1,nice19,macOS background. Current production and all old data/results/source remain unchanged. On interruption preserve incomplete run, use newdirectory; resume only exactly identical completed cache. Any later correction experiment must preregister its own rules and run source identity before its forecasts. Neither calibration outputs nor test scores can modify this cache.
