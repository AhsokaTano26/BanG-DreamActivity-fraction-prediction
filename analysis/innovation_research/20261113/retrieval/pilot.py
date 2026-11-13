"""Fixed, training-only chronological retrieval feasibility pilot.

This is not V2 validation/test scoring. Prediction and training-label scoring
are separate commands. No validation/test answer is parsed. TemporalDataset
still hashes all frozen artifacts as required by the repository protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.temporal_dataset import TemporalDataset


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DATASET = ROOT / "data/temporal_holdout_v2_t1000_t2000"
DATASET_ID = "1f76f39af8151bba185496411bbcee064b16a4c5aca5c515c1d957a5056fe854"
CONFIG = {
    "scope": "train-only rolling feasibility pilot; not fixed holdout score",
    "query_split": "train",
    "warmup": "first floor(N/2) training events by actual start",
    "evaluation_blocks": "remaining training events, three chronological np.array_split blocks",
    "history": "same rank, different event, history.end < query.start",
    "history_prefix": "cut at query actual progress; >=4 points, positive current, <=60min lag",
    "shape_fractions": [0.25, 0.5, 0.75, 0.9],
    "shape_distance": "mean absolute normalized-score difference over >=2 jointly observed fractions",
    "variants": {
        "linear": {"formula": "current / actual_progress"},
        "shape_k5": {"k": 5, "scale_weight": 0.0, "type_weight": 0.0},
        "shape_scale_type_k5": {"k": 5, "scale_weight": 0.15, "type_weight": 0.2},
        "shape_scale_type_k10": {"k": 10, "scale_weight": 0.15, "type_weight": 0.2},
    },
    "scale_distance": "absolute log(query_current / historical_matched_current)",
    "type_distance": "0 if equal else 1",
    "aggregation": "query_current * median(history_final / historical_matched_current), equal history-event weight",
    "fallback": "linear if fewer than k eligible neighbors; no query deletion",
    "fit_or_tuning": "none; all three variants fixed before scoring",
    "metrics": "case averages within event, then equal average over events; physical cases unique overall",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def sources():
    names = [
        str(Path(__file__).relative_to(ROOT)),
        "analysis/temporal_dataset.py", "analysis/multirank_dataset.py",
        "analysis/holdout_dataset.py", "analysis/backtest.py",
    ]
    return {name: digest(ROOT / name) for name in names}


def load_dataset():
    dataset = TemporalDataset(DATASET)
    if dataset.manifest["dataset_id"] != DATASET_ID:
        raise ValueError("Unexpected frozen dataset")
    return dataset


def shape(times, scores, start, matched_at):
    """Interpolation only between observations inside the available prefix."""
    fractions = np.asarray(CONFIG["shape_fractions"])
    checkpoints = start + fractions * (matched_at - start)
    result = np.full(len(fractions), np.nan)
    valid = (checkpoints >= times[0]) & (checkpoints <= times[-1])
    result[valid] = np.interp(checkpoints[valid], times, scores) / scores[-1]
    return result


def candidates(query, histories):
    qc = query["curve"]
    qshape = shape(query["times"], query["scores"], qc.start, query["now"])
    eligible = []
    past = 0
    quality = 0
    for history in histories:
        if history.event == qc.event or history.end >= qc.start:
            continue
        past += 1
        matched_at = history.start + query["progress"] * history.duration
        stop = int(np.searchsorted(history.times, matched_at, side="right"))
        if stop < 4:
            continue
        ht, hs = history.times[:stop], history.scores[:stop]
        if hs[-1] <= 0 or matched_at - ht[-1] > 3600000:
            continue
        quality += 1
        hshape = shape(ht, hs, history.start, matched_at)
        joint = np.isfinite(qshape) & np.isfinite(hshape)
        if int(joint.sum()) < 2:
            continue
        eligible.append({
            "event": history.event,
            "shape": float(np.abs(qshape[joint] - hshape[joint]).mean()),
            "scale": float(abs(np.log(query["score"] / float(hs[-1])))),
            "type": int(qc.kind != history.kind),
            "multiplier": float(history.final / hs[-1]),
            "historical_observed_progress": float((ht[-1] - history.start) / history.duration),
            "historical_lag_minutes": float((matched_at - ht[-1]) / 60000),
        })
    assert len({row["event"] for row in eligible}) == len(eligible)
    return eligible, past, quality


def predict():
    if (HERE / "prediction_manifest.json").exists() or (HERE / "predictions.csv").exists():
        raise FileExistsError("Pilot predictions already saved; use a new directory for another experiment")
    dataset = load_dataset()
    save_json(HERE / "fixed_config.json", CONFIG)
    curves = list(dataset.train_curves())
    histories = {rank: [c for c in curves if c.rank == rank] for rank in (1000, 2000)}
    starts = {c.event: c.start for c in curves}
    assert len(starts) == len(dataset.manifest["splits"]["train"]["event_ids"]) == 198
    ordered = sorted(starts, key=lambda e: (starts[e], e))
    warmup_count = len(ordered) // 2
    warmup = ordered[:warmup_count]
    blocks = [list(map(int, block)) for block in np.array_split(ordered[warmup_count:], 3)]
    event_block = {e: i + 1 for i, block in enumerate(blocks) for e in block}
    split_rows = [{"event": e, "start_ms": starts[e], "role": "warmup" if e in warmup else "evaluate",
                   "block": event_block.get(e, 0)} for e in ordered]
    pd.DataFrame(split_rows).to_csv(HERE / "chronological_event_groups.csv", index=False)
    predictions = []
    coverage = []
    seen = set()
    inputs = list(dataset.inputs("train"))
    for query in inputs:
        qc = query["curve"]
        if qc.event not in event_block:
            continue
        assert query["case_id"] not in seen
        seen.add(query["case_id"])
        assert qc.times[-1] == query["now"] and qc.times[-1] <= query["cutoff"]
        neighbors, past, quality = candidates(query, histories[qc.rank])
        linear = query["score"] / query["progress"]
        meta = {"case_id": query["case_id"], "event": qc.event, "rank": qc.rank,
                "block": event_block[qc.event], "event_type": qc.kind, "progress": query["progress"],
                "current": query["score"], "start_ms": qc.start, "observed_at": query["now"],
                "cutoff": query["cutoff"]}
        coverage.append(meta | {"strictly_past_same_rank": past, "prefix_quality_candidates": quality,
                                "eligible_neighbors": len(neighbors)})
        for name, spec in CONFIG["variants"].items():
            used = []
            fallback = False
            prediction = linear
            if name != "linear":
                fallback = len(neighbors) < spec["k"]
                if not fallback:
                    ranked = sorted(neighbors, key=lambda n: (
                        n["shape"] + spec["scale_weight"] * n["scale"] + spec["type_weight"] * n["type"], n["event"]))
                    used = ranked[:spec["k"]]
                    prediction = query["score"] * np.median([n["multiplier"] for n in used])
            predictions.append(meta | {"method": name, "prediction": float(prediction), "fallback": fallback,
                "neighbors": json.dumps([n["event"] for n in used]),
                "eligible_neighbors": len(neighbors),
                "neighbor_median_lag_minutes": float(np.median([n["historical_lag_minutes"] for n in used])) if used else None})
    frame = pd.DataFrame(predictions)
    assert frame.groupby("method").case_id.nunique().nunique() == 1
    assert np.isfinite(frame.prediction).all()
    assert (frame.prediction >= frame.current).all()
    frame.to_csv(HERE / "predictions.csv", index=False)
    pd.DataFrame(coverage).to_csv(HERE / "candidate_coverage.csv", index=False)
    input_events = {q["curve"].event for q in inputs}
    manifest = {
        "dataset_id": DATASET_ID, "scope": CONFIG["scope"], "config": CONFIG,
        "source_hashes": sources(), "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "warmup_events": warmup, "evaluation_blocks": blocks,
        "all_train_cases": len(inputs), "evaluation_cases": len(seen),
        "evaluation_events": int(frame.event.nunique()),
        "train_events_without_available_prefix": sorted(set(ordered) - input_events),
        "data_reads": ["TemporalDataset integrity hashing of all frozen artifacts", "train_curves.jsonl.gz",
                       "train_inputs.jsonl.gz", "manifest.json"],
        "label_policy": "No labels parsed during prediction; only training labels will be parsed by score command",
        "files": {name: digest(HERE / name) for name in ("fixed_config.json", "chronological_event_groups.csv", "predictions.csv", "candidate_coverage.csv")},
    }
    save_json(HERE / "prediction_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("dataset_id", "evaluation_cases", "evaluation_events", "train_events_without_available_prefix")}, ensure_ascii=False))


def metrics(frame):
    ape = np.abs(frame.prediction - frame.actual) / frame.actual * 100
    err = frame.prediction - frame.actual
    return {"events": int(frame.event.nunique()), "cases": len(frame),
            "event_macro_mape": float(ape.groupby(frame.event).mean().mean()),
            "case_mape": float(ape.mean()), "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.square(err).mean())), "p90_ape": float(ape.quantile(.9)),
            "retrieval_use_case_pct": float((~frame.fallback).mean() * 100) if frame.method.iloc[0] != "linear" else 0,
            "retrieval_use_event_macro_pct": float((~frame.fallback).groupby(frame.event).mean().mean() * 100) if frame.method.iloc[0] != "linear" else 0}


def grouped_metrics(frame, columns):
    rows = []
    for key, group in frame.groupby(columns, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append(dict(zip(columns, key)) | metrics(group))
    return pd.DataFrame(rows)


def score():
    if (HERE / "scoring_manifest.json").exists():
        raise FileExistsError("Pilot already scored; do not overwrite sealed results")
    manifest = json.loads((HERE / "prediction_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert digest(HERE / name) == expected, name
    assert sources() == manifest["source_hashes"], "Source changed after prediction"
    dataset = load_dataset()
    predictions = pd.read_csv(HERE / "predictions.csv")
    labels = dataset.labels("train")
    predictions["actual"] = predictions.case_id.map(labels)
    assert predictions.actual.notna().all()
    predictions["ape"] = np.abs(predictions.prediction - predictions.actual) / predictions.actual * 100
    case_index = dataset.case_index("train")[["case_id", "band_group", "observation_age_minutes"]]
    predictions = predictions.merge(case_index, on="case_id", validate="many_to_one")
    outputs = {
        "summary.csv": grouped_metrics(predictions, ["method"]),
        "by_block.csv": grouped_metrics(predictions, ["method", "block"]),
        "by_rank.csv": grouped_metrics(predictions, ["method", "rank"]),
        "by_event_type.csv": grouped_metrics(predictions, ["method", "event_type"]),
        "by_band.csv": grouped_metrics(predictions, ["method", "band_group"]),
        "by_event.csv": grouped_metrics(predictions, ["method", "event"]),
    }
    membership = dataset.memberships("train")
    stage_frame = predictions.merge(membership, on="case_id", validate="many_to_many")
    outputs["by_stage.csv"] = grouped_metrics(stage_frame, ["method", "stage_id"])
    outputs["by_rank_stage.csv"] = grouped_metrics(stage_frame, ["method", "rank", "stage_id"])
    coverage = pd.read_csv(HERE / "candidate_coverage.csv")
    cov_stages = coverage.merge(membership, on="case_id", validate="one_to_many")
    cov_rows = []
    for key, group in cov_stages.groupby(["rank", "stage_id"]):
        cov_rows.append({"rank": int(key[0]), "stage_id": key[1], "cases": len(group),
                        "events": int(group.event.nunique()), "median_eligible": float(group.eligible_neighbors.median()),
                        "ge5_pct": float((group.eligible_neighbors >= 5).mean() * 100),
                        "ge10_pct": float((group.eligible_neighbors >= 10).mean() * 100)})
    outputs["coverage_by_rank_stage.csv"] = pd.DataFrame(cov_rows)
    for name, frame in outputs.items():
        frame.to_csv(HERE / name, index=False)
    prediction_events = outputs["by_event.csv"].pivot(index="event", columns="method", values="event_macro_mape")
    win_counts = {name: int((prediction_events[name] < prediction_events["linear"]).sum())
                  for name in CONFIG["variants"] if name != "linear"}
    summary = outputs["summary.csv"]
    block = outputs["by_block.csv"]
    report = [
        "# 训练组内部前推历史曲线检索试验", "",
        "本报告是训练组内部的可行性试验，**不是固定 V2 验证或测试成绩**，不能据此宣称正式模型改进。",
        f"冻结两档数据集：`{DATASET_ID}`。", "",
        "198 个训练活动按实际开始时间排序；前 99 个仅作 warmup，后 99 个分为三个连续的 33 活动块。",
        f"本次评估 {manifest['evaluation_cases']} 个唯一物理案例，{manifest['evaluation_events']} 个活动；活动 57、68 在冻结训练网格无可用前缀，仍保持原分组。",
        "总体指标不因时间轴重复成员关系重复计算，stage 表只在各自切片内统计。每个活动等权。", "",
        "## 预先固定的方法", "",
        "线性基线 current/progress；shape k=5；shape+0.15×log规模差+0.2×类型差 k=5、k=10。没有调参循环。",
        "形状是本期已观测进度的 25%、50%、75%、90% 位置的累计分数/当前分数；用至少两项共同可见的比例计算平均绝对差。",
        "候选只来自同排名、不同活动且在本期开始之前已经结束的训练活动；按查询实际进度截断候选，只取截断前观察。",
        "历史前缀要求至少 4 点、当前分数为正、距匹配时刻不超过 60 分钟，每个历史活动至多一个候选。",
        "预测为当前分数乘以邻居 final/current 比例的中位数；不足 k 个候选时整条案例回退线性。", "",
        "## 全体结果", "",
        "| 方法 | event-macro MAPE % | case-MAPE % | P90 APE % | 使用检索案例 % |", "|---|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        report.append(f"| {row['method']} | {row['event_macro_mape']:.3f} | {row['case_mape']:.3f} | {row['p90_ape']:.3f} | {row['retrieval_use_case_pct']:.2f} |")
    report += ["", "## 三个连续块", "", "| 方法 | 块 | 活动数 | 案例数 | event-macro MAPE % |", "|---|---:|---:|---:|---:|"]
    for row in block.to_dict("records"):
        report.append(f"| {row['method']} | {row['block']} | {row['events']} | {row['cases']} | {row['event_macro_mape']:.3f} |")
    report += ["", "## 覆盖与审计", "",
        f"具备 ≥5 合格历史邻居的案例占 {(coverage.eligible_neighbors >= 5).mean()*100:.2f}%；≥10 占 {(coverage.eligible_neighbors >= 10).mean()*100:.2f}%。",
        f"合格历史邻居数中位数 {coverage.eligible_neighbors.median():.0f}，最少 {coverage.eligible_neighbors.min()}，最多 {coverage.eligible_neighbors.max()}。",
        "逐活动相对线性获胜活动数（共 99）：" + json.dumps(win_counts, ensure_ascii=False) + "。",
        "分排名、阶段、排名×阶段、活动类型、乐队、活动及覆盖明细均保存为 CSV。预测文件保存每个查询的邻居活动 ID、回退标记与可用邻居数。",
        "预测与训练标签评分分为独立命令，评分前校验预测、配置与源码哈希。没有读取 SQLite、既有模型评分、验证或测试答案。",
        "TemporalDataset 的标准完整性校验会按字节计算所有冻结文件哈希，包括答案文件；这仅用于验证数据完整性，没有解析验证/测试答案。", "",
        "## 限制与后续方向", "",
        "这是与线性基线的训练内部可行性比较，不是与现有正式模型的同案例竞赛。已完成的训练历史会随查询活动的开始时间自然增加；没有把验证/测试最终分数追加到历史。",
        "固定队列来自不同时期，早年观察覆盖较疏；严格历史质量门槛可能导致早期阶段覆盖低，不能据保留样本的表现外推到缺失阶段。",
        "更值得正式开发的是将历史乘子及邻居离散度作为现有模型的训练期交叉拟合特征或弱残差分量，并按候选数、距离和查询阶段进行预先固定的回退/收缩；正式采用前仍需在冻结验证组选择，测试仅独立评分一次。",
        "该测试队列曾出现在早期探索实验中，不能称为完全未见过的盲测。元数据是当前回溯快照，缺少历史时点存档。", "",
        "复现命令（本目录产物已保存，禁止直接覆盖）：先 `PYTHONPATH=. .venv/bin/python analysis/innovation_research/20261113/retrieval/pilot.py predict`，再同路径 `score`。",
    ]
    (HERE / "report.md").write_text("\n".join(report) + "\n")
    scoring = {"dataset_id": DATASET_ID, "scope": CONFIG["scope"], "prediction_manifest_sha256": digest(HERE / "prediction_manifest.json"),
               "source_hashes": sources(), "labels_parsed": ["train_labels.csv"],
               "files": {name: digest(HERE / name) for name in [*outputs, "report.md"]}}
    save_json(HERE / "scoring_manifest.json", scoring)
    print(summary.to_string(index=False))
    print("\nBlock metrics:\n" + block[["method", "block", "event_macro_mape"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "score"))
    args = parser.parse_args()
    predict() if args.phase == "predict" else score()
