"""Human-readable time-origin report from independently sealed evaluation files."""
import argparse,json,shutil
from pathlib import Path
import pandas as pd
from analysis.holdout_dataset import sha256,write_json
from analysis.temporal_dataset import TemporalDataset,STAGES
from analysis.temporal_evaluate import verify_predictions


def pct(value):return '—' if value is None else f'{value:.2f}%'

def stage_name(stage):
    axis,value=stage['axis'],stage['value']
    if axis=='elapsed':return f'开始后 {value:g}h'
    if axis=='progress':return f'进度 {value:.0%}'
    return f'结束前 {value:g}h' if value>=1 else f'结束前 {value*60:g}min'


def report(run,output):
    run=Path(run);output=Path(output);seal=verify_predictions(run);data=TemporalDataset(seal['dataset_directory'])
    evaluation_seal=json.loads((run/'evaluation_manifest.json').read_text())
    for name,digest in evaluation_seal['artifacts'].items():
        if sha256(run/name)!=digest:raise ValueError('Evaluation artifact changed')
    if output.exists():raise FileExistsError(output)
    evaluation=json.loads((run/'evaluation.json').read_text());selected=seal['reference_method']
    scores=evaluation['splits']['test'];main=scores[selected]
    table=['| 时点 | 固定共享模型 MAPE | 沿用当前分数 MAPE | 线性外推 MAPE | 活动数 | 案例数 | 观测滞后中位数/min |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for stage in STAGES:
        key=stage['stage_id'];s=main['by_stage'][key]
        age='—' if s['median_observation_age_minutes'] is None else f"{s['median_observation_age_minutes']:.1f}"
        table.append(f"| {stage_name(stage)} | {pct(s['event_macro_mape_pct'])} | {pct(scores['hold_current']['by_stage'][key]['event_macro_mape_pct'])} | {pct(scores['linear']['by_stage'][key]['event_macro_mape_pct'])} | {s['events']} | {s['cases']} | {age} |")
    ranks=list(data.manifest['protocol']['ranks']);remaining=[s for s in STAGES if s['axis']=='remaining']
    matrix=['| 排名 | '+' | '.join(stage_name(s) for s in remaining)+' |','|---|'+'---:|'*len(remaining)]
    by_rank={(r['stage_id'],r['rank']):r for r in main['stage_rank']}
    for rank in ranks:matrix.append(f'| T{rank} | '+' | '.join(pct(by_rank.get((s['stage_id'],rank),{}).get('event_macro_mape_pct')) for s in remaining)+' |')
    panels=[]
    for axis,panel in main['common_panels'].items():
        panels.append(f"### {axis} 共同面板\n\n{panel['events']} 个活动、{panel['event_rank_pairs']} 个活动×排名组合在该轴所有时点均有可用观测。")
        panels+=['\n| 时点 | 同一批案例 MAPE |\n|---|---:|']
        for stage in STAGES:
            if stage['axis']==axis:panels.append(f"| {stage_name(stage)} | {pct(panel['by_stage'][stage['stage_id']]['event_macro_mape_pct'])} |")
    summary=['| 活动组 | 已评测活动数 | 不重复前缀案例 | 时间轴成员关系数 | 固定共享模型 MAPE |','|---|---:|---:|---:|---:|']
    for split in ['train','validation','test']:
        s=data.manifest['splits'][split]
        metric=evaluation['splits'][split][selected]['overall']
        summary.append(f"| {split} | {metric['events']} | {s['cases']:,} | {s['stage_memberships']:,} | {pct(metric['event_macro_mape_pct'])} |")
    fresh=['| 时点 | 全部可用前缀 MAPE | 观测滞后≤15min MAPE | 新鲜前缀活动数 | 新鲜前缀案例数 |','|---|---:|---:|---:|---:|']
    for stage in remaining:
        key=stage['stage_id'];f=main['fresh_15min_by_stage'][key]
        fresh.append(f"| {stage_name(stage)} | {pct(main['by_stage'][key]['event_macro_mape_pct'])} | {pct(f['event_macro_mape_pct'])} | {f['events']} | {f['cases']} |")
    group_sizes='/'.join(str(len(data.manifest['splits'][s]['event_ids'])) for s in ['train','validation','test'])
    text=f'''# 时间评测协议 V2：固定活动留出＋活动内多时点回放

用户要求的V2已单独建立于 `data/temporal_holdout_v2/`。原V1与既有19档数据/模型保持不变；旧目录 `multirank_v2` 只是此前数据处理的修订号，与本次时间评测协议V2不同。

V2保持{group_sizes}活动归属和合格活动×排名队列，在三个活动组使用相同21个请求时点。不同时间轴若落在同一时刻，只生成一个物理案例，多轴通过成员关系引用。训练活动回放不是原训练拟合误差；它在已参与训练的活动上使用新的统一时点。

## 首轮实验的控制条件

固定此前封存的共享排名与元数据CatBoost模型，加入“沿用当前分数”和线性外推；不重训、不选择新模型、不把后续活动终分加入历史。参考模型仍为原验证集选中的 `{selected}`。因此本轮测量时间影响，而非新模型增益。

V2的新训练前缀已准备好，后续训练也能覆盖开局与冲刺时点；本次尚未用它们重新拟合。原模型训练采样大致为10%—98%进度，V2首次明确测量这个范围之外的表现。

{chr(10).join(summary)}

总体按物理案例去重、活动等权，**不能把其数值与V1五时点的整体MAPE直接比较后声称改进**。首要结果是下面的分时点评测，不是一个总分。

## 全部时间点：测试活动

{chr(10).join(table)}

“结束前1h”指请求时刻。若最新观测更早，模型实际拥有的前缀更短；CSV同时保存 requested_remaining_h、observed_remaining_h 和 observation_age_minutes。保留最大60分钟滞后的既有输入规则，不伪造插值后的当前分数。早期不足4个观测或无正分数的情况列为缺失，不给虚构预测。各方法的案例集合完全相同。

## 同一批活动、同一排名的时间对照

每条轴仅保留在该轴全部时点都可用的活动×排名组合。这些共同面板避免把样本构成变化误读为时间带来的改善。共同面板为空时明确报告0，不能换成另一个更容易的集合。

{chr(10).join(panels)}

## 各排名的结束前误差

{chr(10).join(matrix)}

T1500在原训练/验证组无本档样本，仍属于跨排名迁移；V2没有悄悄移动活动以增加其训练数据。每档每时点的活动数与MAE、RMSE、P90见 `../stage_rank_metrics.csv`。

## 观测新鲜度

{chr(10).join(fresh)}

新鲜子集只是事先规定的诊断，不能代替全部案例主结果；不同子集的总体差异也不能单独证明滞后的因果效应。

## 覆盖、日历时间与统计单位

`coverage.csv` 区分无合格曲线和有曲线却缺少可用前缀，含全部19档×21时点×3活动组。数据集的 `unavailable_cases.csv` 冻结每个缺失前缀原因。`../stage_calendar_metrics.csv` 以国服活动开始季度分层，检查跨时期差异；小组的活动数随表附带。

`../paired_bootstrap.csv` 在24h、6h、1h、15min结束前时点，对同一批活动重采样2000次，比较原选中模型与两种简单基线。负的差值表示模型MAPE更低。统计单位是活动，不把排名、快照或重复时间轴当成独立活动；这些多重描述性比较不作为反复择优依据。

## 版本、边界与复现

数据指纹：`{data.manifest['dataset_id']}`。预测与评分分别封存，评分标识：`{evaluation['evaluation_id']}`。所有现有V1数据、模型与历史结果保持原样。

主协议固定历史训练信息，不进行“看完上一测试活动终分后重训”的滚动学习。若以后评估这种生产更新策略，应另建协议，不能与本轮固定历史结果混合。

测试活动队列曾参与早期探索，不能称为全新未见盲测。元数据仍是回溯快照，缺少历史时点存档。保留原有分组可以公平比较同一时间点上的模型，但不能消除这些限制。

```bash
.venv/bin/python -m analysis.temporal_benchmark --dataset data/temporal_holdout_v2 --run analysis/temporal_results/<new-run>
.venv/bin/python -m analysis.temporal_evaluate --run analysis/temporal_results/<new-run>
.venv/bin/python -m analysis.temporal_report --run analysis/temporal_results/<new-run> --output analysis/temporal_results/<new-run>/report
```

方法参考：[Forecasting: Principles and Practice 的时间序列交叉验证](https://otexts.com/fpp3/tscv.html)、[scikit-learn TimeSeriesSplit 文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。本项目的观测不等间隔且同活动相关，因此先按活动分组，再在活动内按预定起点回放；没有把所有快照直接当等间隔独立样本切分。
'''
    output.mkdir(parents=True);(output/'README.md').write_text(text)
    shutil.copyfile(data.root/'coverage.csv',output/'coverage.csv');shutil.copyfile(__file__,output/'report_source.py')
    write_json(output/'manifest.json',{'evaluation_manifest_sha256':sha256(run/'evaluation_manifest.json'),
        'artifacts':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}})
    print(output/'README.md')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();report(a.run,a.output)
