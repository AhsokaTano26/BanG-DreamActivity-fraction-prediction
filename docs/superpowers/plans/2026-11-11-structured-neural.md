# Structured Neural Forecasting Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user explicitly authorized continuing improvements and neural exploration in the existing workspace.

**Goal:** Test whether paired, rank-aware neural models of remaining-rate improve the fixed benchmark beyond relationships_v3.

**Architecture:** Prefix-only canonical 1000/2000 sequence slots with availability, interpolation span and observation age; compact dimensionless tabular features. Shared encoders with rank heads output log remaining-rate, decoded by current score + mean velocity × remaining hours × multiplier. Separate develop/predict/final-scoring phases with unchanged standard evaluator.

**Tech Stack:** Project Python environment, PyTorch MPS/CPU, existing holdout loader, scikit-learn preprocessing and immutable manifests.

**Spec:** User instruction to continue method improvement and neural exploration; root AGENTS.md is the evaluation contract.

## Global constraints

- Fixed holdout_v1 fingerprint and 4377/414/412 cases; event grouping, equal global event training weight, train-only preprocessing/history. Never read test labels during modeling.
- Fixed seeds 42/123/2026 for every neural configuration; report individual seeds and arithmetic final-score ensemble. Model/recipe/ensemble selection uses validation event-macro MAPE only.
- Nine registered configurations: shared MLP, rank-head MLP, paired GRU, paired TCN, full+recent TCN, cross-rank patch attention, gated TCN experts, TCN with relative-final-score loss, TCN with prefix thinning augmentation.
- Width48, dropout.2, AdamW lr.001/wd.005, batch256, max100 epochs, minimum30/patience20, gradient clip1. Standard Huber log-rate beta.2; metric variant .8 smooth relative-final error + .2 Huber. No search after test scoring.
- Reference is the validation-selected relationships_v3 combo. Candidate ensembles are each neural 3-seed mean, equal top3 neural means, and 25/50/75 percent neural-top3 versus reference. Seeds individually reported but never selected individually.
- Keep prior test exposure, retrospective metadata and correlated-event uncertainty limitations explicit. No promises that neural models will improve.

## Review focus

1. Peer values after target observation must never enter either timescale; missing/sparse peer cannot silently become a real zero-score curve.
2. Input representation and output units must remain coherent under common score rescaling; rank selection must use canonical slots.
3. All branches and experts share the same global event weights; augmented inputs preserve current time/score/target.
4. Checkpoint reload reproduces validation predictions; changed data/code/dependencies/reference reject resuming or predicting.
5. Validation is the only early-stop/selection source. Test prediction finishes and seals before the evaluator sees answers.

## Tasks

- [x] Add failing scientific-invariant tests for paired prefixes, scale/missingness/thinning, model gradients, endpoints and checkpoint replay; keep torch tests in subprocesses.
- [x] Implement paired representation and rank-aware backbones in dedicated modules.
- [x] Implement preregistered development/checkpointing/blind prediction, including exact reference identity and all pipeline fingerprints.
- [x] Run 27 fits, validation selection, independent review and necessary fixes before final scoring.
- [x] Score unchanged test cases once; independently verify metrics, report every method and compare event-paired differences. Update project docs and preserve old experiments.

Completed run: `analysis/holdout_results/structured_neural_v1/`.
Validation selected `blend_neural_0.25` (3.801843%); fixed test 8.067550%
versus reference 8.140302%. Paired event difference -0.072752 percentage points,
95% interval [-0.201040, +0.053843]; insufficient evidence for stable overall improvement.
All 27 fits and 41 reported predictions preserved. 58 tests passed; independent
metric discrepancy <= 1.8e-15. No tuning followed test scoring.
