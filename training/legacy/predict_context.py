"""Metadata-aware CN prediction with the development-selected CatBoost recipe."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

from analysis.backtest import TZ,snapshot,templates
from analysis.data_policy import POLICY_ID,require_eligible_start
from forecasting import DEFAULT_DATABASE,load_curves,load_local,select_history
from predict import fetch_current,parse_as_of
from analysis.context_benchmark import ContextBuilder,arrays_for,fit_context,training_samples


def forecast(target,curves,cutoff,peer):
    require_eligible_start(target.start)
    if peer is not None:require_eligible_start(peer.start)
    if cutoff<target.start:raise ValueError('活动尚未开始，需要活动开始后的观测。')
    if cutoff>=target.end:raise ValueError('活动已结束，请用 --as-of 指定结束前的时刻回放。')
    sample=snapshot(target,cutoff)
    if sample is None:raise ValueError('观测不足或已滞后超过 60 分钟，请更新数据或调整 --as-of。')
    history=select_history(target,curves)
    events=len({c.event for c in history})
    if events<20:raise ValueError('至少需要 20 个已结束历史活动。')
    pool=history+[target]
    if peer is not None:pool.append(peer)
    builder=ContextBuilder(pool)
    # Historical features use this same pool, including for training samples.
    # Retain base card/character/band dictionaries but no unrelated event rows.
    event_ids={c.event for c in pool}
    builder.metadata.events={key:value for key,value in builder.metadata.events.items()
                             if int(key) in event_ids}
    builder.metadata.parties={key:value for key,value in builder.metadata.parties.items()
                              if key in event_ids}
    train_ids={c.event for c in history}
    builder.metadata.character_ids=sorted({character for event in train_ids
        for character in builder.metadata.parties[event]['characters']})
    builder.metadata.band_ids=sorted({band for event in train_ids
        for band in builder.metadata.parties[event]['bands']})
    if target.event not in builder.metadata.parties:
        raise ValueError('缺少该活动的乐队/角色元数据，请先更新 data/metadata。')
    if not builder.metadata.peer(sample,pool)['peer_available']:
        raise ValueError('缺少同一时刻另一排名档的观测；可用原 predict.py 进行单档预测。')
    train=training_samples(history)
    arrays=arrays_for(builder,train,[sample],history,'peer')
    correction,_=fit_context('catboost',arrays,train,[sample])
    predicted=float(sample['score']*np.exp(np.clip(arrays['test_base'][0]+correction[0],0,4)))
    try:analog=templates(sample,history)[2]
    except AssertionError:analog=None
    if not np.isfinite(predicted):raise ValueError('模型没有生成有效预测值。')
    context=builder.metadata.describe(target.event)
    return {'model':'catboost_context','event_id':target.event,'rank':target.rank,
        'event_type':target.kind,'event_bands':list(context['band_names'].values()),
        'mixed_band':context['band_group']=='mixed',
        'as_of':datetime.fromtimestamp(cutoff/1000,TZ).isoformat(),
        'observed_at':datetime.fromtimestamp(sample['now']/1000,TZ).isoformat(),
        'current_ep':int(sample['score']),'progress':float(sample['progress']),
        'predicted_final_ep':int(round(predicted)),
        'analog_final_ep':int(round(analog)) if analog is not None else None,
        'training_events':events,'historical_error_range':None,'data_policy_id':POLICY_ID,
        'notes':['乐队按活动加成角色构成识别；混团保留多个乐队。',
                 '新模型的误差范围尚未校准，请结合历史回测误差使用点预测。']}


def main(argv=None):
    parser=argparse.ArgumentParser(description='结合活动类型、乐队、角色与历史增长统计预测国服分数线。')
    parser.add_argument('--event',required=True,type=int)
    parser.add_argument('--rank',choices=[1000,2000],type=int,default=2000)
    parser.add_argument('--as-of',type=parse_as_of)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--database',type=Path,default=DEFAULT_DATABASE)
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args(argv)
    now=int(datetime.now(timezone.utc).timestamp()*1000)
    cutoff=now if args.as_of is None else args.as_of
    if args.event<=0:parser.error('--event 必须为正整数。')
    if cutoff>now:parser.error('--as-of 不能晚于当前时间。')
    try:
        if args.offline or args.as_of is not None:
            target,curves=load_local(args.database,args.event,args.rank)
            peer_rank=1000 if args.rank==2000 else 2000
            peer=next((c for c in curves if c.event==target.event and c.rank==peer_rank),None)
        else:
            target=fetch_current(args.event,args.rank)
            peer=fetch_current(args.event,1000 if args.rank==2000 else 2000)
            curves,_=load_curves(args.database)
        curves=[curve for curve in curves if curve.rank in (1000,2000)]
        result=forecast(target,curves,cutoff,peer)
        if args.json:print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        else:
            print(f"国服活动 {result['event_id']} · {result['rank']} 线 · {result['event_type']}")
            print('乐队：'+' / '.join(result['event_bands']))
            print(f"观测：{result['observed_at']}；当前分数：{result['current_ep']:,}")
            print(f"预测最终分数：{result['predicted_final_ep']:,}")
            if result['analog_final_ep'] is not None:print(f"历史曲线参考：{result['analog_final_ep']:,}")
            for note in result['notes']:print(note)
        return 0
    except (ValueError,OSError,KeyError,sqlite3.Error) as exc:
        print(f'预测失败：{exc}',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
