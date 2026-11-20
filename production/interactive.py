"""Explicit one-shot predictions and opt-in foreground web service.

This module never fits a model in the single-rank path. The continuous service
delegates fitting and feedback to the existing, time-guarded coordinator.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from analysis.backtest import TZ, snapshot
from analysis.holdout_dataset import prefix_curve
from download_ranks import RANKS
from production.data import Store, sync_data
from production.model import verify_pack,inference_session
from production.runtime import DEFAULT_ROOT, _previous, cycle_lock, read_config, validate_root
from production.selected_feedback import ensure_archive, seed_feedback
from production.selected_model import build_bias, predict_pack
from production.state import Ledger, atomic_json


def rank_argument(value):
    try:
        rank = int(str(value).strip().upper().removeprefix('T'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('分数线应为1000或T1000等排名') from exc
    if rank not in RANKS:
        raise argparse.ArgumentTypeError('支持的分数线：' + '、'.join('T'+str(r) for r in RANKS))
    return rank


def predict_once(config, rank, *, now_ms=None, sync=True, transport=None):
    with inference_session():
        return _predict_once(config,rank,now_ms=now_ms,sync=sync,transport=transport)


def _predict_once(config, rank, *, now_ms=None, sync=True, transport=None):
    """Issue the requested rank using existing causal weights, then return."""
    rank = rank_argument(rank)
    if rank not in config['ranks']:
        raise ValueError('当前模型配置未包含此分数线')
    if config.get('forecast_method') != 'legacy_pair':
        raise ValueError('当前入口需要已配置的 selected 生产模型')
    root = validate_root(config['root'])
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    with cycle_lock(root) as acquired:
        if not acquired:
            return {'status': 'busy', 'predictions': [], 'as_of_ms': now,
                    'message': '已有预测或训练任务运行，请稍后重试'}
        store, ledger = Store(root), Ledger(root)
        if not ledger.fact('bootstrap_complete'):
            raise ValueError('尚无准备好的生产历史，请先运行 forecast.py serve 完成初始化')
        errors = []
        sync_result = {'status': 'offline', 'errors': []}
        if sync:
            try:
                sync_result = asyncio.run(sync_data(store, now_ms=now, ranks=[rank], transport=transport))
                errors.extend(sync_result.get('errors', []))
            except Exception as exc:
                errors.append({'stage': 'sync', 'error': f'{type(exc).__name__}: {exc}'})
                sync_result = {'status': 'degraded', 'errors': list(errors)}
        active = [e for e in store.events(now) if e['start_ms'] <= now < e['end_ms']]
        single_model=not config.get('automatic_training',True)
        query={'events':[e['event'] for e in active]} if single_model else {}
        curves = {(c.event, c.rank): c for c in store.load_curves(now, ranks=[rank], complete_only=False,**query)}
        rows = []
        for event in active:
            curve = curves.get((event['event'], rank))
            prefix = None if curve is None else prefix_curve(curve, now)
            sample = None if prefix is None else snapshot(prefix, now)
            if sample is None:
                rows.append({'event': event['event'], 'rank': rank, 'available': False,
                    'reason': '缺少足够观测，或最新观测已超过60分钟'})
                continue
            pin = ledger.pin(event['event'])
            if pin is None:
                # A new pin uses the same verified seed and event-start clock as
                # the continuous coordinator; no single-rank fitting occurs.
                ensure_archive(ledger)
                ledger.learn_completed(store.load_curves(now, ranks=config['ranks']),
                    now_ms=now, grace_ms=config.get('feedback_grace_ms', 600000))
                seed_feedback(store, ledger, config, before_ms=event['start_ms'], now_ms=now)
                path, manifest = _previous(root, before_ms=event['start_ms'],current_only=single_model)
                if not manifest.get('production_ready') or manifest['history_max_end_ms'] >= event['start_ms']:
                    raise ValueError('模型不满足期初历史约束')
                state = build_bias(ledger.feedback_rows(before_ms=event['start_ms']),
                                   before_ms=event['start_ms'], exclude_event_id=event['event'])
                pin = ledger.create_pin(event, path, manifest['model_id'], state, now_ms=now)
            if single_model:
                path, manifest = _previous(root,before_ms=event['start_ms'],current_only=True)
                if pin['model_id']!=manifest['model_id'] or Path(pin['model_path']).resolve()!=Path(path).resolve():
                    pin=ledger.replace_unusable_pin(event['event'],path,manifest['model_id'],
                        reason='Installed model replaced previous version',now_ms=now)
            manifest = verify_pack(pin['model_path'])
            if not manifest.get('production_ready') or manifest['history_max_end_ms'] >= event['start_ms']:
                raise ValueError('模型不满足期初历史约束')
            sample.update(case_id=f"live:{event['event']}:{rank}:{now}", stage='live')
            predicted = predict_pack(pin['model_path'], [sample],
                metadata_directory=store.metadata_directory, calibration=pin['calibration'])
            ledger.record(event, predicted, cutoff_ms=now, issued_ms=now, model_id=pin['model_id'])
            rows.extend(predicted)
        status = ('waiting_for_event' if not active else 'unavailable'
                  if not any(r.get('available') for r in rows) else 'degraded' if errors else 'ok')
        result = {'status': status, 'as_of_ms': now, 'as_of': datetime.fromtimestamp(now/1000, TZ).isoformat(),
                  'rank': rank, 'predictions': rows, 'active_events': [e['event'] for e in active],
                  'errors': errors, 'sync': sync_result, 'model_updates': False}
        atomic_json(root / f'requests/latest_t{rank}.json', result)
        return result


def print_prediction(result):
    messages = {'waiting_for_event': '当前没有进行中的国服活动',
                'busy': '已有预测或训练任务运行，请稍后重试',
                'unavailable': '当前没有可用预测，请检查观测数据'}
    if result['status'] in messages:
        print(messages[result['status']])
    if result.get('as_of'):
        print('预测时间：' + result['as_of'])
    for row in result.get('predictions', []):
        print(f"\n活动 {row['event']} · T{row['rank']}")
        if not row.get('available'):
            print('暂不可用：' + str(row.get('reason', '观测不足')))
            continue
        print(f"当前分数：{row['current_ep']:,.0f}")
        print(f"预测终分：{row['predicted_final_ep']:,.0f}")
        print(f"活动进度：{row['progress']:.1%}")
        if row.get('observed_at'):
            print('观测时间：' + datetime.fromtimestamp(row['observed_at']/1000, TZ).strftime('%Y-%m-%d %H:%M:%S'))
        if row.get('provisional'):
            print('提示：该排名历史活动支持不足，结果为暂定预测')
    if result.get('errors'):
        print('提示：数据更新失败，以上使用仍有效的缓存观测', file=sys.stderr)
    if result.get('sync', {}).get('status') == 'offline':
        print('数据来源：本地缓存（离线模式）')


def main(argv=None):
    parser = argparse.ArgumentParser(description='指定分数线预测，或手动启动持续预测网页')
    parser.add_argument('mode', nargs='?', choices=['predict', 'serve'], default='predict')
    parser.add_argument('--rank', type=rank_argument, help='分数线，例如1000或T1000')
    parser.add_argument('--config', type=Path, default=DEFAULT_ROOT/'config.json')
    parser.add_argument('--offline', action='store_true', help='使用本地观测，不联网同步')
    parser.add_argument('--json', action='store_true', help='单次预测输出JSON')
    parser.add_argument('--host', default='127.0.0.1', help='网页监听地址（默认仅本机）')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--interval', type=int, help='持续模式更新间隔，至少60秒，默认900秒')
    args = parser.parse_args(argv)
    if args.mode == 'predict' and args.rank is None:
        parser.error('单次预测需要 --rank，例如：forecast.py --rank 1000')
    if args.mode == 'serve' and args.json:
        parser.error('--json 仅适用于单次预测')
    if not 1 <= args.port <= 65535:
        parser.error('端口应在1至65535之间')
    if args.interval is not None and args.interval < 60:
        parser.error('更新间隔至少60秒')
    try:
        config = read_config(args.config)
        if args.mode == 'serve':
            from production.web import serve
            from production.deployment import DeploymentWorker
            config = config | {'automatic_training': False}
            atomic_json(args.config, config)
            return serve(config, args.config, host=args.host, port=args.port,
                         interval=args.interval, offline=args.offline, initial_rank=args.rank or 1000,
                         worker_factory=DeploymentWorker)
        result = predict_once(config, args.rank, sync=not args.offline)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print_prediction(result)
        return 0 if result['status'] in ('ok', 'degraded') else 2
    except KeyboardInterrupt:
        print('\n已停止。', file=sys.stderr)
        return 130
    except Exception as exc:
        if args.json:
            print(json.dumps({'status':'error', 'error':str(exc)}, ensure_ascii=False))
        else:
            print('运行失败：' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
