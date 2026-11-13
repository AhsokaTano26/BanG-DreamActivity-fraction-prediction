"""Independent post-seal, answer-free audit of the training-only pilot."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.temporal_dataset import TemporalDataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = HERE / "audit_result.json"
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((HERE / "prediction_manifest.json").read_text())
    scoring = json.loads((HERE / "scoring_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha(HERE / name) == expected, name
    for name, expected in scoring["files"].items():
        assert sha(HERE / name) == expected, name
    for name, expected in manifest["source_hashes"].items():
        assert sha(ROOT / name) == expected, name
    dataset = TemporalDataset(ROOT / "data/temporal_holdout_v2_t1000_t2000")
    assert dataset.manifest["dataset_id"] == manifest["dataset_id"]
    curves = {(c.event, c.rank): c for c in dataset.train_curves()}
    starts = {c.event: c.start for c in curves.values()}
    ordered = sorted(starts, key=lambda event: (starts[event], event))
    expected_blocks = [list(map(int, block)) for block in np.array_split(ordered[len(ordered)//2:], 3)]
    assert expected_blocks == manifest["evaluation_blocks"]
    events = {event for block in expected_blocks for event in block}
    case_index = dataset.case_index("train")
    expected_index = case_index[case_index.event.isin(events)].set_index("case_id")
    expected_cases = set(expected_index.index)
    queries = {q["case_id"]: q for q in dataset.inputs("train") if q["curve"].event in events}
    assert set(queries) == expected_cases
    predictions = pd.read_csv(HERE / "predictions.csv")
    methods = manifest["config"]["variants"]
    assert set(predictions.method) == set(methods)
    selected_history_count = 0
    for method, group in predictions.groupby("method"):
        assert not group.case_id.duplicated().any()
        assert set(group.case_id) == expected_cases
        for row in group.itertuples(index=False):
            frozen = expected_index.loc[row.case_id]
            assert row.event == frozen.event and row.rank == frozen["rank"]
            assert row.start_ms == frozen.start_ms and row.observed_at == frozen.observed_at
            assert row.cutoff == frozen.cutoff and np.isclose(row.current, frozen.current)
            query = queries[row.case_id]
            assert np.isclose(row.progress, query["progress"])
            neighbor_ids = json.loads(row.neighbors)
            assert len(set(neighbor_ids)) == len(neighbor_ids)
            linear = row.current / row.progress
            if method == "linear":
                assert not neighbor_ids and not row.fallback
                assert np.isclose(row.prediction, linear)
                continue
            spec = methods[method]
            assert bool(row.fallback) == (row.eligible_neighbors < spec["k"])
            if row.fallback:
                assert not neighbor_ids and np.isclose(row.prediction, linear)
                continue
            assert len(neighbor_ids) == spec["k"]
            multipliers = []
            for event in neighbor_ids:
                history = curves[(event, row.rank)]
                assert event != row.event and history.end < row.start_ms
                matched_at = history.start + row.progress * history.duration
                stop = int(np.searchsorted(history.times, matched_at, side="right"))
                assert stop >= 4 and history.scores[stop-1] > 0
                assert 0 <= matched_at - history.times[stop-1] <= 3600000
                multipliers.append(history.final / history.scores[stop-1])
                selected_history_count += 1
            assert np.isclose(row.prediction, row.current * np.median(multipliers), rtol=1e-12)
    coverage = pd.read_csv(HERE / "candidate_coverage.csv")
    assert set(coverage.case_id) == expected_cases and not coverage.case_id.duplicated().any()
    result = {
        "status": "passed", "dataset_id": manifest["dataset_id"],
        "evaluation_events": len(events), "unique_cases_per_method": len(expected_cases),
        "methods": len(methods), "prediction_rows_checked": len(predictions),
        "chosen_history_entries_checked": selected_history_count,
        "checks": ["frozen case sets exactly equal in every method", "frozen event/rank/timestamps/current match",
                   "chronological 99 warmup plus 3 x 33 split", "strict-past same-rank unique chosen history events",
                   "history prefix >=4 points and <=60min lag", "fallback consistency", "prediction arithmetic",
                   "saved prediction/scoring artifact hashes", "sealed source hashes"],
        "labels_parsed": [], "audit_source_sha256": sha(__file__),
        "prediction_manifest_sha256": sha(HERE / "prediction_manifest.json"),
        "scoring_manifest_sha256": sha(HERE / "scoring_manifest.json"),
    }
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
