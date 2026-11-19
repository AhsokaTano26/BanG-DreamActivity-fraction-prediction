"""Readable result tables for an independently scored V3 residual experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.rolling_benchmark import verify_seal,write_new_json
from analysis.multirank_dataset import identity

NAMES={'uncorrected':'仅滚动重训（无校正）','ewma_reference':'原收缩EWMA','ewma_fast':'快速EWMA','window5':'最近5期均值',
       'kalman_level':'卡尔曼水平跟踪','kalman_trend':'卡尔曼水平＋趋势','kernel_progress':'连续进度核平滑','context_partial_pool':'类型/乐队分组收缩',
       'ridge20':'条件残差岭回归 α20','ridge100':'条件残差岭回归 α100','residual_tree':'条件残差树','residual_mlp':'残差神经网络（三种子均值）',
       **{f'residual_mlp_seed{s}':f'残差神经网络 seed{s}' for s in [42,123,2026]}}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def report(run,output):
    run,output=Path(run),Path(output)
    if output.exists():raise FileExistsError('Preserve existing report')
    seal=verify_seal(run/'blind_manifest.json');receipt=verify_seal(run/'evaluation_manifest.json')
    if receipt['identity']['blind_manifest_sha256']!=sha256(run/'blind_manifest.json'):raise ValueError('Score not bound to these forecasts')
    result=json.loads((run/'evaluation.json').read_text());selection=json.loads((run/'selection.json').read_text());cfg=json.loads((run/'run_manifest.json').read_text())
    if result['evaluation_id']!=receipt['evaluation_id'] or result['run_id']!=seal['run_id'] or result['selection']!=selection:raise ValueError('Scoring/selection identities differ')
    selected=selection['selected_recipe'];v=result['splits']['validation'];t=result['splits']['test'];comparisons={c['method']:c for c in result['paired_comparisons']}
    rows=[]
    for name in cfg['methods']:
        val,test=v[name]['overall'],t[name]['overall'];delta=comparisons[name]
        row={'method':name,'description':NAMES[name],'eligible_for_selection':name in cfg['candidates'],'selected':name==selected,
             'validation_mape_pct':val['event_macro_mape_pct'],'test_mape_pct':test['event_macro_mape_pct'],
             'T1000_mape_pct':t[name]['by_rank']['1000']['event_macro_mape_pct'],'T2000_mape_pct':t[name]['by_rank']['2000']['event_macro_mape_pct'],
             'test_minus_reference_points':delta['method_minus_reference_mape_points'],'iid_ci_low':delta['paired_event_95_ci'][0],'iid_ci_high':delta['paired_event_95_ci'][1],
             'block5_ci_low':delta['circular_block_95_ci']['5'][0],'block5_ci_high':delta['circular_block_95_ci']['5'][1],**{k:test[k] for k in ['case_mape_pct','mae_ep','rmse_ep','p90_ape_pct','events','cases']}}
        rows.append(row)
    frame=pd.DataFrame(rows);row=frame[frame.selected].iloc[0];ref=frame[frame.method=='ewma_reference'].iloc[0];c=comparisons[selected]
    changed=row.test_minus_reference_points;rel=(ref.test_mape_pct-row.test_mape_pct)/ref.test_mape_pct*100
    if changed<0:
        conclusion=f'验证预选的{NAMES[selected]}将测试MAPE从{ref.test_mape_pct:.4f}%降到{row.test_mape_pct:.4f}%，减少{-changed:.4f}个百分点（相对{rel:.2f}%）。'
    else:
        conclusion=f'验证预选的{NAMES[selected]}测试MAPE为{row.test_mape_pct:.4f}%，原EWMA为{ref.test_mape_pct:.4f}%；本轮没有验证出预选方案的测试提升。'
    iid=c['paired_event_95_ci'];block=c['circular_block_95_ci']['5']
    conclusion+=f' 16期中{c["improved_events"]}期改善；活动配对差值95%区间[{iid[0]:+.3f}, {iid[1]:+.3f}]个百分点，5期时间块区间[{block[0]:+.3f}, {block[1]:+.3f}]。'
    if iid[0]<=0<=iid[1] or block[0]<=0<=block[1]:conclusion+=' 区间跨0，证据不足以认定稳定提升。'
    sections=['# 以V3滚动校正为基线的残差实验（2026-11-15）',conclusion,
       '**这是在已暴露测试队列上的后续探索。** 候选菜单和参数在本轮运行前固定，14期验证选方法，16期测试只作独立评分；不按测试最低者改选。仍需新活动的前瞻验证。元数据为回溯快照，缺少历史时点存档。',
       '## 全部结果',table(['方法','验证MAPE','测试MAPE','测试T1000','测试T2000','较原EWMA/百分点'],[[NAMES[r.method]+(' **（预选）**' if r.selected else ''),f'{r.validation_mape_pct:.3f}',f'{r.test_mape_pct:.3f}',f'{r.T1000_mape_pct:.3f}',f'{r.T2000_mape_pct:.3f}',f'{r.test_minus_reference_points:+.3f}'] for r in frame.itertuples()]),
       '单位均为活动等权MAPE（%，越低越好）；同一测试560前缀、16活动，所有方法使用相同名单。单种子仅诊断，不参与选择；神经网络主结果为三个种子的最终分数平均。主指标之外的case-MAPE、MAE、RMSE、P90保留在完整CSV。',
       '## 实验实际做了什么',
       '- 复用前次封存的400树滚动Context盲预测，保持基础预测完全相同。本轮只重训/更新校正器，便于把差异归因于校正。30期逐活动预测与历史释放均重新封存。\n- 只使用此前已结束、预测已封存活动的真实误差；本期开始时冻结历史。本期终分不得提前使用。密集采样下限仍为2024-01-29，V3的67个初始训练活动只用于基础模型预热，不用其拟合残差训练校正器。\n- 统计候选涵盖不同记忆长度、状态空间水平/趋势、连续进度平滑和类型/乐队分组收缩。统计组按活动/排名/阶段先汇总，避免快照数多的活动主导更新。\n- 学习型候选使用排名、阶段、进度、活动类型、乐队、基础预测与前缀速率；只用已释放历史拟合编码、插补、标准化和模型，每活动总权重相同。至少8期真实预测误差后才拟合，之前回退到原EWMA。\n- 岭回归α20/100；残差树200轮、深度2；残差MLP为32/16隐藏层、固定80轮、42/123/2026三个种子，强正则与额外n/(n+20)收缩。没有利用测试早停或调参数。\n- 学习目标严格为`log(base/actual) − requested_EWMA_bias`，是最终分数下限/上限应用之前的代数残差；不声称它等于触及边界时的实际评分残差。全部最终预测保持相同分数边界。\n- 原EWMA在所有30期与之前封存结果精确重放；完整运行、模型、预处理、历史与独立评分都有指纹。',
       '## 主要时间点（预选方法 vs 原EWMA）']
    stages=[]
    for stage in ['elapsed_6h','elapsed_24h','progress_25pct','progress_50pct','progress_75pct','remaining_24h','remaining_6h','remaining_1h','remaining_0.25h']:
        a=t[selected]['by_stage'].get(stage,{});b=t['ewma_reference']['by_stage'].get(stage,{})
        if a.get('cases',0):stages.append([stage,f"{a['event_macro_mape_pct']:.3f}",f"{b['event_macro_mape_pct']:.3f}",a['events'],a['cases']])
    sections += [table(['时点','预选MAPE','原EWMA MAPE','活动','前缀'],stages),
       '## 可复核材料',
       '- [15组完整结果](comparison.csv) · [验证时已封存的选择](selection.json) · [配对区间](../paired_comparisons.json)\n- 原运行目录中的by_stage_rank.csv提供时间点×排名明细；by_type.csv、by_band.csv分别提供类型和乐队明细。\n- 原始逐期盲预测、校正器权重、预处理、训练活动权重及反馈时间审计位于上级events/目录。',
       '## 方法参考',
       '卡尔曼候选按递归状态估计构造，参数为本轮预登记选择：[Welch与Bishop技术报告](https://www.cs.utexas.edu/~pstone/Courses/393Rfall15/readings/Welch%2BBishop-TR-95.pdf)。岭回归使用活动权重和L2正则：[scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)。残差网络使用稳健分段损失：[PyTorch HuberLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)。这些文献支持方法机制，性能结论来自本实验。']
    output.mkdir(parents=True);frame.to_csv(output/'comparison.csv',index=False)
    (output/'README.md').write_text('\n\n'.join(sections)+'\n');write_new_json(output/'selection.json',selection)
    (output/'source.py').write_text(Path(__file__).read_text())
    fields={'evaluation_id':receipt['evaluation_id'],'evaluation_manifest_sha256':sha256(run/'evaluation_manifest.json'),'blind_manifest_sha256':sha256(run/'blind_manifest.json'),
            'artifacts':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    fields['seal_id']=identity(fields);write_new_json(output/'report_manifest.json',fields)
    print(frame[['method','selected','validation_mape_pct','test_mape_pct','test_minus_reference_points']].to_string(index=False))
    return frame

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.run,a.output)
