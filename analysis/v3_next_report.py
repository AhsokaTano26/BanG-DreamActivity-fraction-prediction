"""Source-bound report of independently scored V3 next candidates, no retuning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.rolling_benchmark import verify_seal,write_new_json
from analysis.multirank_dataset import identity
from analysis.v3_next_selection import RUNS
from analysis.rolling_dataset import RollingDataset
from analysis.temporal_evaluate import score_temporal

LABELS={'uncorrected':'原Context无校正','ewma_reference':'原冷启动EWMA','ewma_warm':'预序列残差暖启动EWMA','ewma_warm_fast':'暖启动快速EWMA',
        'mape_median':'有界MAPE直接校正','mape_kernel':'有界MAPE＋连续进度','coupled_gap':'T2000锚点＋线间差异',
        'coupled_kalman':'两线共同状态＋差异趋势','hedge_rank10':'按排名在线专家组合η10','hedge_rank30':'按排名在线专家组合η30',
        'hedge_rank_phase10':'按排名/阶段在线专家组合','drift_gate':'漂移触发的保守趋势混合','factorized_mlp':'共同/差异残差神经网络（三种子）'}


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def report(output,selection_path):
    output=Path(output);selection_path=Path(selection_path)
    if output.exists():raise FileExistsError('Preserve prior report')
    frozen=json.loads(selection_path.read_text());winner=frozen['selected'];entries=frozen['runs']
    if {ROOT/e['run'] for e in entries}!=set(RUNS):raise ValueError('Unexpected run inventory')
    expected=min(entries,key=lambda r:(r['validation_mape_pct'],RUNS.index(ROOT/r['run'])))
    if winner!=expected:raise ValueError('Cross-run choice changed')
    rows=[];artifacts=[];evaluations={}
    for run in RUNS:
        item=next(e for e in entries if ROOT/e['run']==run)
        seal=verify_seal(run/'blind_manifest.json');receipt=verify_seal(run/'evaluation_manifest.json')
        if (sha256(run/'blind_manifest.json')!=item['blind_manifest_sha256'] or sha256(run/'selection.json')!=item['selection_sha256']
                or receipt['identity']['blind_manifest_sha256']!=item['blind_manifest_sha256']):raise ValueError('Frozen selection/scoring binding differs')
        e=json.loads((run/'evaluation.json').read_text());evaluations[run.name]=e
        if e['run_id']!=seal['run_id'] or e['evaluation_id']!=receipt['evaluation_id']:raise ValueError('Score run differs')
        family='A 暖启动与自适应校正' if 'warm' in run.name else 'B 基础模型改进'
        for method,test in e['splits']['test'].items():
            val=e['splits']['validation'][method]
            compare=next(c for c in e['paired_comparisons'] if c['method']==method)
            rows.append({'family':family,'run':run.name,'method':method,'description':LABELS.get(method,method),
                'selected_within_run':method==item['selected_method'],'primary':str(run.relative_to(ROOT))==winner['run'] and method==winner['selected_method'],
                'validation_mape_pct':val['overall']['event_macro_mape_pct'],'test_mape_pct':test['overall']['event_macro_mape_pct'],
                'T1000_mape_pct':test['by_rank']['1000']['event_macro_mape_pct'],'T2000_mape_pct':test['by_rank']['2000']['event_macro_mape_pct'],
                'delta_points':compare['method_minus_reference_mape_points'],'improved_events':compare['improved_events'],
                'iid_low':compare['paired_event_95_ci'][0],'iid_high':compare['paired_event_95_ci'][1],
                'block5_low':compare['circular_block_95_ci']['5'][0],'block5_high':compare['circular_block_95_ci']['5'][1],**test['overall']})
        artifacts.append({'run':str(run.relative_to(ROOT)),'run_id':seal['run_id'],'evaluation_id':receipt['evaluation_id'],
            'blind_manifest_sha256':sha256(run/'blind_manifest.json'),'evaluation_manifest_sha256':sha256(run/'evaluation_manifest.json')})
    frame=pd.DataFrame(rows);primary=frame[frame.primary].iloc[0];baseline=frame[(frame.run==primary.run)&(frame.method=='ewma_reference')].iloc[0]
    delta=primary.delta_points;relative=-delta/baseline.test_mape_pct*100
    conclusion=(f'测试前按验证集选中的 `{primary.method}`：验证MAPE {primary.validation_mape_pct:.4f}%，测试MAPE {primary.test_mape_pct:.4f}%。原EWMA为{baseline.test_mape_pct:.4f}%，差值{delta:+.4f}个百分点（相对误差下降{relative:+.2f}%；负数代表增加）。'
        +f'16活动中{primary.improved_events}期改善，活动配对95%区间[{primary.iid_low:+.3f}, {primary.iid_high:+.3f}]，5活动时间块区间[{primary.block5_low:+.3f}, {primary.block5_high:+.3f}]。')
    if delta<0 and primary.iid_high<0 and primary.block5_high<0:conclusion+=' 本队列两种区间均支持下降，但仍属探索，需前瞻确认。'
    else:conclusion+=' 本轮不能据此认定已经稳定超过原模型；不改选测试最低者。'
    sections=['# V3下一轮优化：暖启动残差、在线专家与基础模型（2026-11-16）',conclusion,
       '**边界：测试队列此前已暴露、验证被反复使用，元数据为回溯快照。** 此次固定全部候选后运行，两个实验均封存盲预测，再冻结跨实验验证选择，最后独立评分。所有比较同为16测试活动/560前缀，不把快照当独立活动。',
       '## 全部方法（活动等权MAPE%，越低越好）']
    for family,part in frame.groupby('family',sort=False):
        sections+=[f'### {family}',table(['方法','验证','测试','T1000','T2000','较原EWMA/点'],[[LABELS.get(r.method,r.method)+(' **（事前主结果）**' if r.primary else '（组内预选）' if r.selected_within_run else ''),f'{r.validation_mape_pct:.3f}',f'{r.test_mape_pct:.3f}',f'{r.T1000_mape_pct:.3f}',f'{r.T2000_mape_pct:.3f}',f'{r.delta_points:+.3f}'] for r in part.itertuples()])]
    gap=frame[(frame.run==RUNS[0].name)&(frame.method=='coupled_gap')].iloc[0]
    nn=frame[(frame.run==RUNS[0].name)&(frame.method=='factorized_mlp')].iloc[0]
    warm=frame[(frame.run==RUNS[0].name)&(frame.method=='ewma_warm')].iloc[0]
    gap_relative=(baseline.test_mape_pct-gap.test_mape_pct)/baseline.test_mape_pct*100
    sections+=['## 结果说明与后续证据',
       f'事前主结果未胜出，不能按测试成绩改选。事后观察到的最有潜力候选是“两线误差分开校正”：验证{gap.validation_mape_pct:.3f}%（比原EWMA高），测试{gap.test_mape_pct:.3f}%，较原EWMA相对下降{gap_relative:.2f}%。其中T1000从{baseline.T1000_mape_pct:.3f}%降到{gap.T1000_mape_pct:.3f}%，T2000从{baseline.T2000_mape_pct:.3f}%变为{gap.T2000_mape_pct:.3f}%。它把T2000的平稳偏差当锚点，另以较快速度跟踪T1000相对T2000的历史预测误差差。',
       f'该候选{gap.improved_events}/16活动改善，配对活动95%差值区间[{gap.iid_low:+.3f}, {gap.iid_high:+.3f}]点、5活动时间块区间[{gap.block5_low:+.3f}, {gap.block5_high:+.3f}]点。这些是单个比较的区间，未校正多方法探索与事后关注，不能替代预先选定方法的前瞻验证，也不能外推到其余17排名档。',
       f'神经网络三种子平均测试{nn.test_mape_pct:.3f}%，相较原EWMA只有{-nn.delta_points:+.3f}点的误差下降，其活动配对区间[{nn.iid_low:+.3f}, {nn.iid_high:+.3f}]含零。仅增加因果残差热身的暖EWMA为{warm.test_mape_pct:.3f}%，本轮神经网络也未超过这一匹配控制；没有证据认为增加网络复杂度更有效。',
       '建议后续预先固定“两线误差分开校正”，在尚未结束的新活动上做影子预测，与当前生产模型并行记账比较，待真实终分到达再评分。研究计算继续按用户要求单任务串行、nice=19和系统后台优先级；本轮未启动新的训练菜单，也未部署事后测试胜者。']
    warm_run=RUNS[0]
    warm_seal=verify_seal(warm_run/'blind_manifest.json')
    data=RollingDataset(warm_seal['dataset_directory'])
    warm_predictions=pd.read_csv(warm_run/'warm_predictions.csv')
    warm_ids=set(warm_predictions.case_id)
    warm_cases=data.case_index('train').query('case_id in @warm_ids')
    warm_members=data.memberships('train').query('case_id in @warm_ids')
    labels=data.labels('train');warm_labels={c:labels[c] for c in warm_ids}
    warm_rows=[]
    for method,pred in warm_predictions.groupby('method'):
        score=score_temporal(warm_cases,warm_members,warm_labels,pred,stage_catalog=data.manifest['protocol']['stages'])
        warm_rows.append({'method':method,**score['overall']})
    warm_frame=pd.DataFrame(warm_rows)
    sections+=['## 训练阶段的因果检验',
       '前32活动只作初始训练，不报告拟合精度；随后35训练活动、1282个前缀均先预测并封存，再释放终分。因此下表是训练时期的逐期外推诊断，和后续16测试活动的主成绩分开。冷EWMA在这35期按设计不更新；使用训练期残差的暖启动控制更适合比较校正算法。',
       table(['方法','训练期逐期外推MAPE%'],[[LABELS.get(r.method,r.method),f'{r.event_macro_mape_pct:.3f}'] for r in warm_frame.itertuples()]),
       '## 本轮改了什么',
       '- A：前32训练活动作为基础热身，后35训练活动逐活动预测并封存，再使用真实终分；共1282个训练期盲前缀，未用训练拟合误差替代预测误差。进入14验证/16测试时，基础Context预测与原封存结果相同，暖启动控制用于分辨“额外历史残差”和“算法更换”的收益。\n- A：直接优化有上下界的百分比误差；T2000锚点和两线差异分开跟踪；专家权重只根据先前已结束活动误差更新；分解神经网络按实际有界MAPE训练，42/123/2026三种子平均最终分数。\n- B：每期重新拟合4种固定400树方案，分别考察排名独立、配对MAE、配对MAPE和剩余每小时EP增长。另一条线严格截至本条线当时已观测时刻。每种方案都有原始、EWMA、快速EWMA对照。\n- 两组均保持活动等权、真实时间先后、同案例、先封存后释放结果，基础模型/预处理/权重/每期预测均保存。在线学习规则在测试期固定，不按测试改参数。',
       '评分器沿用原有通用V3独立评分入口，其JSON中的train_scope仍是旧67活动热身的通用文字。A实际残差初始化为32活动基础热身＋35活动因果预测，以本报告、run_manifest与逐期审计为准；该元数据差异不改变任何评分案例或数值。B为原67活动基础热身，不使用训练拟合残差。',
       '## 已训练候选与生产状态',
       '当前生产自动模型继续运行，实验没有修改 production/、其模型权重或定时任务。新候选训练权重保存在各运行的events/目录；候选清单candidate_artifacts.json提供最后一期所用权重、校正状态和来源。末期候选仅拟合其开始前已完成活动，不能把它说成用过本期最终结果的最新生产模型。',
       '## 复核入口',
       '- [完整30组结果](comparison.csv) · [跨实验测试前选择](pretest_selection.json) · [来源身份](provenance.json)\n- 两个原运行的 by_stage_rank.csv、by_type.csv、by_band.csv、paired_comparisons.json 保留时点×排名、类型、乐队和全部区间。\n- 冻结数据仍从2024-01-29 13:00起；不覆盖历史实验，不使用生产数据库中的额外活动改变本轮队列。',
       '## 方法依据',
       'MAPE可以写成按真实值加权的绝对误差，见[原论文](https://arxiv.org/abs/1506.04176)；本轮有界校正使用分段断点候选搜索，再做显式收缩。在线专家组合借鉴[乘法权重方法](https://theoryofcomputing.org/articles/v008a006/v008a006.pdf)，本轮参数固定，表现仍以同案例实验为准。']
    output.mkdir(parents=True);frame.to_csv(output/'comparison.csv',index=False)
    warm_frame.to_csv(output/'warm_prequential_metrics.csv',index=False)
    (output/'README.md').write_text('\n\n'.join(sections)+'\n')
    write_new_json(output/'pretest_selection.json',frozen);write_new_json(output/'provenance.json',artifacts)
    candidate=[]
    for run in RUNS:
        last=max((run/'events').glob('*/event_manifest.json'),key=lambda p:json.loads(p.read_text())['query_start_ms'])
        seal=verify_seal(last);directory=last.parent
        candidate.append({'run':str(run.relative_to(ROOT)),'event':seal['event'],'query_start_ms':seal['query_start_ms'],
            'event_manifest_sha256':sha256(last),'selected_method':next(e['selected_method'] for e in entries if ROOT/e['run']==run),
            'artifacts':{str(p.relative_to(ROOT)):sha256(p) for p in directory.rglob('*') if p.is_file()},
            'deployment':'not promoted; production activity pin preserved'})
        cfg=json.loads((run/'run_manifest.json').read_text())
        base=ROOT/cfg['base_run'] if not Path(cfg['base_run']).is_absolute() else Path(cfg['base_run'])
        if 'warm' in run.name:
            model=base/'events'/str(seal['event'])/'rolling.joblib'
            candidate[-1]['cached_base_model']={'path':str(model.relative_to(ROOT)),'sha256':sha256(model),
                'base_event_manifest_sha256':sha256(model.parent/'event_manifest.json')}
    write_new_json(output/'candidate_artifacts.json',candidate)
    (output/'source.py').write_text(Path(__file__).read_text())
    body={'pretest_selection_sha256':sha256(selection_path),'artifacts':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}}
    body['seal_id']=identity(body);write_new_json(output/'report_manifest.json',body)
    print(frame[['run','method','primary','validation_mape_pct','test_mape_pct','delta_points']].to_string(index=False))
    return frame

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    a=p.parse_args();report(a.output,a.selection)
