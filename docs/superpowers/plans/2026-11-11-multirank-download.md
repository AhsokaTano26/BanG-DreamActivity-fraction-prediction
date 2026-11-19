# Nineteen-rank historical download plan

**Goal:** Download all available CN activity histories for the user's 19 ranks,
then safely add them to the existing local SQLite database. Prediction/training
will follow separately; the existing two-rank model cannot be called a validated
nineteen-rank model.

**Spec:** Latest user request; root AGENTS.md. Use the current project folder and
`.venv`. Existing holdout_v1 and all sealed experiments remain immutable.

**Design:** Bestdori events/all.5.json determines activities actually started in
CN, by start time rather than ID ranges. Eighteen ranks use tracker/data.
T10 uses eventtop/data?server=3&event=...&mid=0&interval=0; derive the tenth
highest score among distinct users at each complete timestamp. Never follow the
eventual tenth player's history backwards, never forward-fill incomplete top10.

Archive original responses compressed, normalized curves, fresh event metadata,
status/quality records, hashes and code. Rate-limited bounded workers, retries
and exact-code resume. Empty API data is an explicit missing result, not zero
and never a reason to erase an existing curve. Preserve score decreases as
quality flags, not silently corrected scores. Conflicting same-time cutoffs are
invalid rather than arbitrarily resolved.

Only after download and verification, take a SQLite backup, merge valid points
into event rows by CN/event/rank, with fresh data authoritative on overlapping
timestamps and old-only timestamps preserved. Use one transaction and record
all row/point changes. Reject duplicate logical keys and frozen-dataset paths.
Keep a separate manifest for the download and the later database application.

Files: `download_ranks.py` owns acquisition/normalization/import; new
`tests/test_download_ranks.py` owns meaningful boundary and offline integration
checks; `data/downloads/<new-run>/` contains outputs and coverage report.

Tasks:
- [x] Test and implement normalization (ordinary/T10), metadata selection and safe merge.
- [x] Test and implement resumable download and transactional backup/import.
- [x] Download all requested event/rank pairs, retry actual transient failures.
- [x] Verify coverage, data quality, SQLite integrity and unchanged holdout fingerprint.
- [x] Required full test suite and independent code review; report actual coverage.

Ruling: Existing API supports no unparameterized all-tier response. Bounded
per-rank requests are necessary; T10 requires its separate documented endpoint.
Public sources: Bestdori API and the installed bestdori-api package implementation.
