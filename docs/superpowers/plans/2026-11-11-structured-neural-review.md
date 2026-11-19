# Structured neural review

Independent read-only review by review_predictor before real test prediction/scoring.
Reviewed paired_prefix.py, structured_neural_models.py, structured_neural_benchmark.py,
the fixed evaluation contract, plan and structured neural tests.

No Critical or Important issues; no necessary Minor fixes. Five synthetic tests
passed independently. Additional probes covered asynchronous and stale peer data,
extreme future suffixes, both time windows and thinning invariants. No real test
labels, prediction values or scores were accessed by the reviewer.

Confirmed canonical rank slots, clipping to target observation time, availability
masks, shared global event weights, train-only preprocessing, validation-only
checkpoint/ensemble selection, averaging in final-score space and sealed artifacts.
Confirmed source/dependency/reference drift and preprocessor changes reject reuse.

Interpretation limits: interpolation is approximate, clipped remaining-rate cannot
exactly represent zero growth, reused validation can be optimistic, three seeds do
not add independent events. Retain prior test exposure, retrospective metadata and
unmodeled serial event correlation in reporting.
