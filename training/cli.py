"""Explicit training commands; benchmark prediction and scoring remain separate."""
from __future__ import annotations
import argparse
import ast
from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import runpy
import sqlite3
import sys
import time

from training.paths import ROOT


def clock_argument(value):
    try:
        if str(value).isdigit():
            return int(value)
        from analysis.backtest import TZ
        parsed = datetime.fromisoformat(value)
        return int((parsed if parsed.tzinfo else parsed.replace(tzinfo=TZ)).timestamp() * 1000)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError('时间应为 epoch 毫秒或带时区的 ISO 日期') from error


def load_histories(database, before_ms, ranks):
    """Read original SQLite history without changing it or filtering by event ID."""
    import numpy as np
    from analysis.backtest import Curve
    from analysis.data_policy import MIN_EVENT_START_MS
    from download_ranks import normalize_points
    from production.data import _clock_points
    database = Path(database).expanduser().resolve()
    curves = []
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute('SELECT * FROM event WHERE Country=? AND StartAt>=? AND EndAt<? '
                          'ORDER BY StartAt,EventID,Rank', ('中国大陆', MIN_EVENT_START_MS, before_ms))
        for row in rows:
            if int(row['Rank']) not in ranks:
                continue
            points, _ = normalize_points(ast.literal_eval(row['PointRank']))
            points = _clock_points(points, int(row['StartAt']), int(row['EndAt']), before_ms)
            curves.append(Curve(int(row['EventID']), int(row['Rank']), str(row['EventType']),
                int(row['StartAt']), int(row['EndAt']),
                np.array([point['time'] for point in points], dtype=np.int64),
                np.array([point['ep'] for point in points], dtype=float)))
    return curves


def _module(name, arguments):
    original = sys.argv
    try:
        sys.argv = [name, *arguments]
        runpy.run_module(name, run_name='__main__')
        return 0
    finally:
        sys.argv = original


def main(argv=None):
    parser = argparse.ArgumentParser(description='显式训练、固定协议实验与独立评分')
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='按既定配方训练生产权重；输出新模型包，不切换网页模型')
    sources = fit.add_mutually_exclusive_group(required=True)
    sources.add_argument('--database', type=Path, help='原始历史 SQLite，只读')
    sources.add_argument('--state', type=Path, help='网页已有数据目录，只读取已结束活动')
    fit.add_argument('--metadata', type=Path, help='原始库配套 metadata 目录')
    fit.add_argument('--before', required=True, type=clock_argument, help='严格反馈截止时间（建议当前活动开始时刻）')
    fit.add_argument('--output', type=Path, default=ROOT/'training/models/versions')
    fit.add_argument('--rank', type=int, action='append', dest='ranks')
    fit.add_argument('--fixture-iterations', type=int, help=argparse.SUPPRESS)
    package = commands.add_parser('package', help='将已训练模型打包，供独立安装或模型发布使用')
    package.add_argument('--model', type=Path, required=True)
    package.add_argument('--output', type=Path, required=True, help='输出新的 model.tar.gz 文件路径')
    package.add_argument('--state', type=Path, help='可选：附带已有预测账本的合法终分反馈')
    for command in ['benchmark', 'evaluate']:
        item = commands.add_parser(command, help='固定密集队列实验' if command == 'benchmark' else '独立评分封存预测')
        item.add_argument('--protocol', choices=['v1', 'v2', 'v3'], required=True)
        item.add_argument('--run', type=Path, required=True)
    module = commands.add_parser('module', help='访问保留的 analysis.* 研究入口')
    module.add_argument('name')
    module.add_argument('options', nargs=argparse.REMAINDER)
    args, forwarded = parser.parse_known_args(argv)
    if forwarded and args.command not in {'benchmark', 'evaluate'}:
        parser.error('无法识别的参数：' + ' '.join(forwarded))
    if args.command == 'package':
        from production.model_release import build_release
        now = int(time.time() * 1000)
        feedback = []
        if args.state is not None:
            from production.state import Ledger
            feedback = Ledger(args.state).feedback_rows(before_ms=now)
        result = build_release(args.model, args.output, feedback_rows=feedback, created_ms=now)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'fit':
        from download_ranks import RANKS
        ranks = RANKS if args.ranks is None else sorted(set(args.ranks))
        if not ranks or not set(ranks) <= set(RANKS):
            parser.error('--rank 必须属于19条受支持分数线')
        if args.before > int(time.time() * 1000):
            parser.error('--before 不能晚于当前实际时间')
        if args.state is not None:
            from production.data import Store
            class ReadOnlyStore(Store):
                def __init__(self, root):
                    self.root = Path(root).expanduser().resolve()
                    self.database = self.root / 'data.sqlite3'
                    self.metadata_directory = self.root / 'metadata'
                    if not self.database.is_file():
                        raise ValueError('--state 必须已有网页历史数据库')

                def _connect(self):
                    connection = sqlite3.connect(self.database.as_uri() + '?mode=ro', uri=True)
                    connection.row_factory = sqlite3.Row
                    return closing(connection)
            store = ReadOnlyStore(args.state)
            curves, metadata = store.load_curves(args.before, ranks=ranks), store.metadata_directory
            pointer = store.root / 'models/current.json'
            if pointer.is_file():
                previous = Path(json.loads(pointer.read_text())['path'])
                support = json.loads((previous / 'curve_support.json').read_text())
                required = {(row['event'], row['rank']) for row in support if row['rank'] in ranks}
                if not required <= {(curve.event, curve.rank) for curve in curves}:
                    raise ValueError('网页历史不足以覆盖已安装模型，请使用完整原始历史库训练')
        else:
            if args.metadata is None:
                parser.error('--database 需要配套 --metadata')
            curves, metadata = load_histories(args.database, args.before, ranks), args.metadata
        from training.model_fit import train_pack
        from production.model import verify_pack
        output = train_pack(curves, metadata, args.output, before_ms=args.before,
                            ranks=ranks, iterations=args.fixture_iterations)
        manifest = verify_pack(output)
        print(json.dumps({'model_path': str(output.resolve()), 'model_id': manifest['model_id'],
            'production_ready': manifest['production_ready'], 'training_events': manifest['training_events'],
            'ranks': manifest['ranks'], 'activated': False,
            'scope': 'Production fit; historical test metrics are not a new evaluation of these weights.'},
            ensure_ascii=False, indent=2))
        return 0
    if args.command == 'benchmark':
        return _module('analysis.current_benchmark', ['--protocol', args.protocol, '--run', str(args.run), *forwarded])
    if args.command == 'evaluate':
        name = {'v1': 'analysis.holdout_evaluate', 'v2': 'analysis.temporal_evaluate', 'v3': 'analysis.rolling_evaluate'}[args.protocol]
        return _module(name, ['--run', str(args.run), *forwarded])
    if not args.name.startswith('analysis.') or '..' in args.name:
        parser.error('研究模块须为 analysis.<模块名>；生产入口使用 fit')
    return _module(args.name, args.options)


if __name__ == '__main__':
    raise SystemExit(main())
