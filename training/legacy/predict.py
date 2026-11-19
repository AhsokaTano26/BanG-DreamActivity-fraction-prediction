"""CN final-cutoff prediction command. Run with --help for examples."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import httpx
import pandas as pd

from analysis.backtest import ROOT, TZ, model_signature
from analysis.data_policy import POLICY_ID
from forecasting import DEFAULT_DATABASE, curve_from_points, load_curves, load_local, predict_cutoff


def parse_as_of(value: str) -> int:
    try:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if date.tzinfo is None:
            date = date.replace(tzinfo=TZ)
        return int(date.timestamp() * 1000)
    except (ValueError, OverflowError, OSError) as exc:
        raise argparse.ArgumentTypeError('请输入 ISO8601 时间，例如 2026-11-02T16:59:59+08:00。') from exc


def fetch_current(event: int, rank: int, *, client=None):
    owned = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(20., connect=10.), follow_redirects=True)
    try:
        metadata = client.get(f'https://bestdori.com/api/events/{event}.json')
        metadata.raise_for_status()
        info = metadata.json()
        start, end = int(info['startAt'][3]), int(info['endAt'][3])
        kind = info['eventType']
        response = client.get('https://bestdori.com/api/tracker/data',
                              params={'server': 3, 'event': event, 'tier': rank})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get('result') is not True:
            raise ValueError('分数线接口没有返回有效观测。')
        return curve_from_points(event, rank, kind, start, end, payload.get('cutoffs'))
    except httpx.HTTPError as exc:
        raise ValueError(f'Bestdori 请求失败：{exc}；可用 --offline --as-of 回放本地数据。') from exc
    except (KeyError, IndexError, TypeError, OverflowError) as exc:
        raise ValueError('Bestdori 活动信息无效，或该活动尚未在国服开放。') from exc
    finally:
        if owned:
            client.close()


def load_calibration(database: Path, *, directory=ROOT / 'analysis/results'):
    try:
        report = json.loads((directory / 'backtest.json').read_text())
        if report.get('database_sha256') != hashlib.sha256(Path(database).read_bytes()).hexdigest():
            return None, '数据库与回测时不同，需要按当前数据政策重新校准误差范围。'
        if report.get('model_signature') != model_signature():
            return None, '模型或特征已变化，需要按当前数据政策重新校准误差范围。'
        if report.get('data_policy_id') != POLICY_ID:
            return None, '旧回测未证明排除 2024-01-29 13:00 前开始的活动，暂不提供校准范围。'
        frame = pd.read_csv(directory / 'predictions.csv')
        required = {'event', 'rank', 'stage', 'method', 'predicted', 'actual'}
        if not required.issubset(frame.columns):
            return None, '历史误差文件格式无效，需要按当前数据政策重新校准。'
        frame.attrs['data_policy_id'] = POLICY_ID
        return frame, None
    except (OSError, ValueError, TypeError, AttributeError):
        return None, '缺少符合当前数据政策的历史误差记录，暂不提供校准范围。'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='预测国服活动最终分数线，复用时间顺序回测验证的模型。')
    parser.add_argument('--event', required=True, type=int, help='活动 ID')
    parser.add_argument('--rank', type=int, choices=[1000, 2000], default=2000, help='排名档，默认 2000')
    parser.add_argument('--as-of', type=parse_as_of, help='历史预测时刻；自动使用本地数据，无时区时按北京时间')
    parser.add_argument('--offline', action='store_true', help='仅使用本地数据库')
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE, help='历史数据库路径')
    parser.add_argument('--json', action='store_true', help='将结果作为 JSON 输出')
    args = parser.parse_args(argv)
    if args.event <= 0:
        parser.error('--event 必须为正整数。')
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    cutoff = args.as_of if args.as_of is not None else now
    if cutoff > now:
        parser.error('--as-of 不能晚于当前时间。')
    try:
        if args.offline or args.as_of is not None:
            target, history = load_local(args.database, args.event, args.rank)
        else:
            target = fetch_current(args.event, args.rank)
            history, _ = load_curves(args.database)
        # This CLI's model was developed for these two ranks. A wider local
        # archive must not silently change its training population.
        history = [curve for curve in history if curve.rank in (1000, 2000)]
        calibration, note = load_calibration(args.database)
        result = predict_cutoff(target, history, cutoff, calibration=calibration, calibration_note=note)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print(f"国服活动 {result['event_id']} · {result['rank']} 线")
            print(f"观测时间：{result['observed_at']}；活动进度：{result['progress']:.1%}")
            print(f"当前分数：{result['current_ep']:,}；距离结束：{result['remaining_hours']:.1f} 小时")
            print(f"预测最终分数：{result['predicted_final_ep']:,}")
            print(f"历史相似曲线参考：{result['analog_final_ep']:,}")
            interval = result['historical_error_range']
            if interval is not None:
                print(f"历史误差参考范围：{interval['lower_ep']:,.0f}～{interval['upper_ep']:,.0f}"
                      f"（{interval['calibration_events']} 个历史活动，目标覆盖 {interval['coverage_target']:.0%}）")
            print(f"训练使用 {result['training_events']} 个已结束活动。")
            for note in result['notes']:
                print(note)
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f'预测失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
