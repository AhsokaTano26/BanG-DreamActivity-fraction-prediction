"""Compare sealed predictions with actual-window labels; never fill missing finals."""
import csv,gzip,hashlib,json,math,sys
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from analysis.holdout_dataset import sha256,write_json,HoldoutDataset
RUN=Path(__file__).resolve().parent
PRED=ROOT/'analysis/multirank_predictions/20261111_2200_replay'
TZ=timezone(timedelta(hours=8))
for directory,filename in [(PRED,'prediction_manifest.json'),(RUN,'fetch_manifest.json')]:
    for name,digest in json.loads((directory/filename).read_text())['artifacts'].items():
        if sha256(directory/name)!=digest:raise ValueError('Artifact hash mismatch: '+name)
if (RUN/'comparison_manifest.json').exists():raise FileExistsError('Comparison already sealed')
source=json.loads((RUN/'actual_observations.json').read_text());prediction=json.loads((PRED/'prediction.json').read_text())
assert sha256(PRED/'prediction.json')==source['prediction_sha256']
lookup={r['rank']:r for r in source['rows']};rows=[]
for pred in prediction['predictions']:
    rank=pred['rank'];obs=lookup[rank];end=obs['end_ms']
    raw=json.loads(gzip.decompress((RUN/f'raw/{rank}.json.gz').read_bytes()))
    # Independent recomputation from raw responses, not downloader normalization.
    if rank==10:
        grouped=defaultdict(dict)
        for p in raw['points']:
            t,u,v=int(p['time']),int(p['uid']),int(p['value'])
            if u in grouped[t] and grouped[t][u]!=v:raise ValueError('Conflicting raw T10 user record')
            grouped[t][u]=v
        points={t:sorted(g.values())[-10] for t,g in grouped.items() if len(g)>=10}
    else:
        assert raw['result'] is True;points={}
        for p in raw['cutoffs']:
            t,v=int(p['time']),int(p['ep'])
            if t in points and points[t]!=v:raise ValueError('Conflicting raw cutoff')
            points[t]=v
    last=max(points);assert {'time':last,'ep':points[last]}==obs['latest_observation']
    near=[t for t in points if end-60000<=t<=end+300000]
    final={'time':max(near),'ep':points[max(near)]} if near else None
    assert final==obs['near_end_final']
    expected=pred['predicted_final_ep'];latest=points[last]
    difference=expected-latest
    latest_gap=float(Decimal(difference)/Decimal(latest)*100)
    assert math.isclose(latest_gap,100*(expected/latest-1),abs_tol=1e-10)
    row={'rank':rank,'predicted_ep':expected,'observed_at_prediction_ep':pred['current_ep'],
        'latest_observed_ep':latest,'latest_observed_at':datetime.fromtimestamp(last/1000,TZ).isoformat(),
        'final_label_status':'available_near_end' if final else 'missing_near_end',
        'actual_final_ep':final['ep'] if final else None,
        'actual_final_observed_at':datetime.fromtimestamp(final['time']/1000,TZ).isoformat() if final else None,
        'signed_final_error_pct':float(Decimal(expected-final['ep'])/Decimal(final['ep'])*100) if final else None,
        'signed_gap_to_latest_pct_not_final_error':latest_gap,
        'gap_to_latest_ep':difference,'last_observation_seconds_from_end':(last-end)/1000,
        'source_url':obs['url']}
    rows.append(row)
assert len(rows)==19 and len({r['rank'] for r in rows})==19
valid=[r for r in rows if r['actual_final_ep'] is not None]
summary={'event':328,'forecast_mode':'historical_replay','as_of':prediction['as_of'],'requested_ranks':19,
    'available_final_labels':len(valid),'missing_final_labels':19-len(valid),'full_event_mape_pct':None,
    'reason_full_metric_unavailable':'18 ranks lack observations within the predeclared end window; latest points are not final labels',
    'verified_final_rank_errors':{str(r['rank']):r['signed_final_error_pct'] for r in valid},
    'baseline_final_comparison':{str(r['rank']):{'model_ape_pct':abs(r['signed_final_error_pct']),
        'hold_current_ape_pct':100*abs(r['observed_at_prediction_ep']/r['actual_final_ep']-1)} for r in valid},
    'independent_activities':1,'forecasts_unchanged':True,
    'caveats':['22:00 replay made after activity end using truncated prefixes, not a prospectively published prediction.',
        'Final labels absent for 18 ranks. Do not interpret latest-observation gaps as final forecast errors.',
        'The earlier 8.41% metric belongs to a 43-event supplementary holdout, not this activity.',
        'Earlier test cohort exposed in exploration; metadata is retrospective.']}
with (RUN/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_json(RUN/'summary.json',summary)
lines=['| 排名 | 封存预测 | 最新观测 | 预测相对最新观测的差异 | 终分状态 |','|---|---:|---:|---:|---|']
for r in rows:
    label='结束附近记录可用' if r['actual_final_ep'] is not None else '终分缺失'
    lines.append(f"| T{r['rank']} | {r['predicted_ep']:,} | {r['latest_observed_ep']:,} | {r['signed_gap_to_latest_pct_not_final_error']:+.2f}% | {label} |")
text='''# 活动328：22:00封存回放预测与结果核对

截至2026-11-11 23:43（北京时间）的逐档请求，T10有结束后的记录；其余18档仅到22:45:02，早于22:59:59结束约14分57秒。**无法给出19档完整终分MAPE**。没有用末次观测冒充最终标签，也没有重新训练或修改预测。

T10预测41,090,399，结束后23:04:23.747记录41,090,399，后续到23:42:35.262也保持相同，分数误差0。该档在22:00输入时已经是41,090,399，因此“沿用当时分数”的基线同样误差0；不能把这一个命中解释为模型胜出。

以下是最新可用对照。除T10外，“差异”是与22:45:02观测比较，不是终分误差。正号表示预测较高，负号表示预测较低。

'''+ '\n'.join(lines)+'''

## 口径和证据

误差方向：(预测−实际)/实际×100%；终分仅取结束前60秒至结束后300秒内的最后观测，后续修订单独保留。T10在同刻至少10个不同玩家时取第10高分，独立从原始响应重新核对。每个排名固定一行，无缺失剔除后冒称完整结果。单一活动不提供活动级bootstrap置信区间。

原始响应在`raw/`，抓取时刻、HTTP日期和源链接在`actual_observations.json`，逐档对照见`comparison.csv`。封存预测的全部指纹通过核验，独立计算与浮点误差公式一致。源接口为[分数线追踪接口](https://github.com/WindowsSov8forUs/bestdori-api/blob/main/docs/api/eventtracker.md)，记录自带观测时间，抓取时间不等于数据时刻。

本次是活动结束后按22:00前缀回放的核对，不是赛前发布的前瞻预测；且距结束仅1小时，超出固定评测最晚6小时前时点。上次8.41%来自43个活动的19档补充测试，不能当作本期误差。原测试队列曾参与早期探索；元数据为回溯快照、无历史时点存档。这些限制保持不变。
'''
(RUN/'README.md').write_text(text)
assert HoldoutDataset().manifest['dataset_id']=='d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb'
write_json(RUN/'comparison_manifest.json',{'prediction_manifest_sha256':sha256(PRED/'prediction_manifest.json'),
    'fetch_manifest_sha256':sha256(RUN/'fetch_manifest.json'),
    'artifacts':{p.name:sha256(p) for p in [RUN/'compare.py',RUN/'comparison.csv',RUN/'summary.json',RUN/'README.md']}})
print(json.dumps(summary,ensure_ascii=False,indent=2));print('\n'.join(lines))
