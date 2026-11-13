"""Forecasting using the exact features and models evaluated by the backtest."""
from __future__ import annotations

import ast
from contextlib import closing
from datetime import datetime
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from analysis.backtest import (
    ROOT, TZ, Curve, fit_models, load_curves, make_samples, snapshot, templates,
)

DEFAULT_DATABASE = ROOT / 'data' / 'db.sqlite3'


def select_history(target: Curve, curves: list[Curve]) -> list[Curve]:
    """Final scores must have been available before the target event began."""
    return [c for c in curves if c.event != target.event and c.end + 15 * 60000 < target.start]


def curve_from_points(event: int, rank: int, kind: str, start: int, end: int, points: list) -> Curve:
    if not isinstance(points, list) or not points:
        raise ValueError('没有可用的分数线观测数据。')
    if not isinstance(kind, str) or not kind or end <= start:
        raise ValueError('活动类型或起止时间无效。')
    by_time = {}
    for point in points:
        if not isinstance(point, dict) or any(type(point.get(k)) is not int for k in ('time', 'ep')):
            raise ValueError('分数线观测格式无效，time 和 ep 必须是整数。')
        t, ep = point['time'], point['ep']
        if t <= 0 or ep < 0:
            raise ValueError('分数线观测包含无效时间或负分数。')
        if t in by_time and by_time[t] != ep:
            raise ValueError('同一时间的分数线观测存在冲突。')
        by_time[t] = ep
    times = np.array(sorted(by_time), dtype=np.int64)
    return Curve(event, rank, kind, int(start), int(end), times,
                 np.array([by_time[t] for t in times], dtype=float))


def load_local(database: Path, event: int, rank: int) -> tuple[Curve, list[Curve]]:
    try:
        uri = Path(database).resolve().as_uri() + '?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM event WHERE EventID=? AND Rank=? AND Country='中国大陆'",
                (event, rank),
            ).fetchone()
        if row is None:
            raise ValueError(f'数据库没有国服活动 {event} 的 {rank} 线，请去掉 --offline 实时抓取。')
        target = curve_from_points(event, rank, row['EventType'], int(row['StartAt']),
                                   int(row['EndAt']), ast.literal_eval(row['PointRank']))
        history, _ = load_curves(database)
        return target, history
    except (sqlite3.Error, SyntaxError, TypeError, KeyError, OverflowError) as exc:
        raise ValueError(f'读取数据库失败：{exc}') from exc


def nearest_stage(progress: float, duration_hours: float) -> str:
    stages = { '25%': .25, '50%': .5, '75%': .75,
              '24h remaining': 1 - 24 / duration_hours,
              '6h remaining': 1 - 6 / duration_hours}
    return min(stages, key=lambda stage: abs(stages[stage] - progress))


def calibrate_range(predicted: float, current: float, rank: int, stage: str,
                    calibration: pd.DataFrame, history: list[Curve], *, coverage=.8) -> dict | None:
    """Historical absolute log-error range; the distinct event is the unit."""
    if not 0 < coverage < 1:
        raise ValueError('范围覆盖目标必须介于 0 和 1 之间。')
    required = {'event', 'rank', 'stage', 'method', 'predicted', 'actual'}
    if calibration is None or not required.issubset(calibration.columns):
        return None
    eligible = {c.event for c in history if c.rank == rank}
    rows = calibration[(calibration['event'].isin(eligible)) & (calibration['rank'] == rank)
                       & (calibration['stage'] == stage) & (calibration['method'] == 'xgb_growth')].copy()
    if rows.empty:
        return None
    predicted_values = pd.to_numeric(rows['predicted'], errors='coerce')
    actual_values = pd.to_numeric(rows['actual'], errors='coerce')
    valid = (predicted_values > 0) & (actual_values > 0) & np.isfinite(predicted_values) & np.isfinite(actual_values)
    rows = rows.loc[valid].copy()
    rows['error'] = abs(np.log(actual_values[valid] / predicted_values[valid]))
    errors = np.sort(rows.groupby('event')['error'].max().to_numpy())
    count = len(errors)
    if count < 20:
        return None
    order = math.ceil((count + 1) * coverage)
    if order > count:
        return None
    error = float(errors[order - 1])
    return {'lower_ep': max(current, predicted * math.exp(-error)),
            'upper_ep': predicted * math.exp(error), 'coverage_target': coverage,
            'calibration_events': count, 'stage': stage}


def predict_cutoff(target: Curve, curves: list[Curve], cutoff: int, *,
                   calibration: pd.DataFrame | None = None, calibration_note: str | None = None) -> dict:
    if cutoff < target.start:
        raise ValueError('活动尚未开始，当前模型需要活动开始后的观测。')
    if cutoff >= target.end:
        raise ValueError('活动已结束；请用 --as-of 指定结束前的时刻回放预测。')
    sample = snapshot(target, cutoff)
    if sample is None:
        raise ValueError('观测不足，或最新观测距指定时刻超过 60 分钟；请更新数据或调整 --as-of。')
    history = select_history(target, curves)
    training_events = len({c.event for c in history})
    if training_events < 20:
        raise ValueError(f'可用历史不足：需要至少 20 个已结束活动，当前只有 {training_events} 个。')
    samples, _ = make_samples(history)
    if len(samples) < 20:
        raise ValueError('历史活动的有效观测样本不足 20 个。')
    _, model = fit_models(samples)
    growth = float(model.predict(pd.DataFrame([sample['features']]))[0])
    predicted = float(sample['score'] * np.exp(np.clip(growth, 0, 4)))
    try:
        _, _, analog = templates(sample, history)
    except AssertionError as exc:
        raise ValueError('没有同排名、相近进度的历史曲线可用于参考；请补充历史数据。') from exc
    if not np.isfinite(predicted) or not np.isfinite(analog):
        raise ValueError('模型未能生成有限的预测值。')
    stage = nearest_stage(sample['progress'], target.duration / 3600000)
    notes = []
    if sample['progress'] < .25:
        interval = None
        notes.append('当前进度早于主要回测时点（25%），属于外推，不提供校准范围。')
    else:
        interval = calibrate_range(predicted, sample['score'], target.rank, stage, calibration, history)
        if interval is None:
            notes.append(calibration_note or '可用的同排名、相近阶段历史误差不足 20 个活动，暂不提供范围。')
    if interval is not None:
        notes.append('范围来自历史误差，目标覆盖 80%；未来活动的实际覆盖率可能变化。')
    return {
        'event_id': target.event, 'server': '中国大陆', 'rank': target.rank, 'event_type': target.kind,
        'as_of': datetime.fromtimestamp(cutoff / 1000, TZ).isoformat(),
        'observed_at': datetime.fromtimestamp(sample['now'] / 1000, TZ).isoformat(),
        'event_end': datetime.fromtimestamp(target.end / 1000, TZ).isoformat(),
        'progress': float(sample['progress']), 'current_ep': int(sample['score']),
        'remaining_hours': (target.end - cutoff) / 3600000,
        'predicted_final_ep': int(round(predicted)), 'analog_final_ep': int(round(analog)),
        'training_events': training_events, 'training_curves': len(history),
        'historical_error_range': interval, 'notes': notes,
    }
