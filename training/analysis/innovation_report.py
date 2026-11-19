"""Render sealed innovation experiment results without parsing answer files."""
from pathlib import Path
import argparse
import json
import pandas as pd
from analysis.holdout_dataset import sha256, write_json


def checked_comparison(run):
    root=run/'comparison'
    seal=json.loads((root/'evaluation_manifest.json').read_text())
    for name,digest in seal['artifacts'].items():
        if sha256(root/name)!=digest:raise ValueError('Changed comparison output')
    return json.loads((root/'comparison.json').read_text())


def report(runs, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    text=['# 固定数据集创新方法实验','',
        '结论：本轮完成了正式对照，但两套数据均未超过对应的既有已选模型，不替换现有主模型。新的组合优于本轮曲线控制组；双时间轴解码在验证和测试汇总上优于同编码器的直接输出，但首个内部时间折退步。上述局部信号不等于已获得稳定的整体最优方法。','',
        '本轮实施历史检索特征/先验残差、类型与乐队分层收缩、正增量双时间轴神经网络，使用冻结 V1 和 V2 两档数据。所有分数为活动等权 MAPE，单位 %。训练拟合误差不代表泛化。','',
        '每协议预先固定9个配方及2个最终分数组合。神经网络每配方三个种子42/123/2026，各80轮；仅三种子最终分数均值可参与选择。训练内部3折用于稳定性诊断，随后完整训练；验证选择后不重训，独立评分器最后读取测试答案。','']
    rows=[];inputs={}
    for protocol,path in runs.items():
        run=Path(path);cfg=json.loads((run/'run_manifest.json').read_text())
        sel=json.loads((run/'selection.json').read_text());comp=checked_comparison(run)
        dev=pd.read_csv(run/'development_metrics.csv')
        cv=pd.read_csv(run/'cv/metrics.csv');cv=cv[cv.axis=='overall']
        selected=sel['selected_recipe'];scores=comp['scores']
        val=sel['validation_scores'][selected];test=scores[selected]['overall']['event_macro_mape_pct']
        train_count=dev[(dev.split=='train')&(dev.axis=='overall')&(dev.method==selected)].iloc[0]
        valid_count=dev[(dev.split=='validation')&(dev.axis=='overall')&(dev.method==selected)].iloc[0]
        test_count=scores[selected]['overall']
        members=sel['combination_specs'].get(selected,{}).get('members',[])
        text += [f'## {protocol.upper()}','',f'验证选择 **{selected}**：验证 {val:.3f}%，测试 {test:.3f}%。',
            f"实际案例覆盖：训练 {int(train_count.events)} 活动/{int(train_count.cases)} 前缀，验证 {int(valid_count.events)} 活动/{int(valid_count.cases)} 前缀，测试 {test_count['events']} 活动/{test_count['cases']} 前缀。",
            '所选组合按最终分数等权平均：'+', '.join(members)+'。' if members else '所选方法为单一固定配方。','',
            '| 方法 | 训练拟合 | 验证 | 测试 | 内部折1 | 内部折2 | 内部折3 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name in sorted(sel['candidates'],key=sel['validation_scores'].get):
            t=dev[(dev.split=='train')&(dev.axis=='overall')&(dev.method==name)].iloc[0].event_macro_mape_pct
            v=sel['validation_scores'][name];s=scores[name]['overall']['event_macro_mape_pct']
            folds={int(r.fold):r.event_macro_mape_pct for r in cv[cv.method==name].itertuples()}
            text.append(f'| {name} | {t:.3f} | {v:.3f} | {s:.3f} | '+' | '.join(f'{folds[f]:.3f}' if f in folds else '—' for f in [1,2,3])+' |')
            rows.append({'protocol':protocol,'method':name,'train_fit_mape':t,'validation_mape':v,'test_mape':s,
                         **{f'cv_fold_{k}':v for k,v in folds.items()}})
        text += ['', '组合成员由固定验证集选定，因此不把选择后组合回填到内部折并称为未见表现。内部折采用固定的较早训练历史；这不同于上一轮按查询逐步扩充训练历史的检索 pilot。','',
            '### 同案例历史对照','', '| 方法 | 测试 MAPE | T1000 | T2000 |','|---|---:|---:|---:|']
        for name in [selected,*comp['identity']['references']]:
            sc=scores[name]
            text.append(f"| {name} | {sc['overall']['event_macro_mape_pct']:.3f} | {sc['by_rank']['1000']['event_macro_mape_pct']:.3f} | {sc['by_rank']['2000']['event_macro_mape_pct']:.3f} |")
        text+=['','| 对照 | 新减旧/百分点 | 配对活动95%区间 | 改善/变差 |','|---|---:|---|---|']
        for name,c in comp['comparisons'].items():
            lo,hi=c['paired_activity_bootstrap_95ci']
            text.append(f"| {name} | {c['delta_mape_points']:.3f} | [{lo:.3f}, {hi:.3f}] | {c['improved_events']}/{c['worse_events']} |")
        text+=['','区间按冻结案例归属对活动配对重采样5000次，未作多重比较校正；只作描述，不用于测试后重新挑选模型。','',
            '### 神经网络逐种子','', '| 配方/种子 | 验证 MAPE | 测试 MAPE |','|---|---:|---:|']
        for name in scores:
            if not name.startswith('growth__'):continue
            values=dev[(dev.split=='validation')&(dev.axis=='overall')&(dev.method==name)]
            if len(values):text.append(f"| {name} | {values.iloc[0].event_macro_mape_pct:.3f} | {scores[name]['overall']['event_macro_mape_pct']:.3f} |")
        if protocol=='v2':
            text+=['','### V2 全部时间点','', '| 时点 | 本轮选择 | 上轮选择 | 活动/案例 |','|---|---:|---:|---|']
            for stage,value in scores[selected]['by_stage'].items():
                a=value['event_macro_mape_pct'];b=scores['old_selected']['by_stage'][stage]['event_macro_mape_pct']
                text.append(f"| {stage} | {a:.3f}" if a is not None else f'| {stage} | —')
                text[-1]+=f" | {b:.3f}" if b is not None else ' | —'
                text[-1]+=f" | {value['events']}/{value['cases']} |"
            text+=['','缺失早期时点继续保持缺失，时间轴别名不重复计入总体。逐案例实际观测时刻见冻结数据目录的 test_cases.csv（observed_at）；≤15min 新鲜子集、共同活动×排名面板、观测滞后汇总及完整类型/乐队指标见本运行 evaluation.json。','']
        text += ['',f"数据指纹：`{cfg['dataset_id']}`；运行：`{run}`。",'']
        inputs[str(run/'comparison/evaluation_manifest.json')]=sha256(run/'comparison/evaluation_manifest.json')
    text += ['## 设计与解释边界','',
        '- 新控制组只使用51个自身曲线/日历/排名特征，不含活动类型；类型与乐队由命名上下文配方引入。历史参考保留原有输入，不能把战胜新控制组自动解释为战胜旧模型。',
        '- 检索只用该活动开始前已结束且属于当前拟合池的同排名训练曲线，排除本期；保存邻居覆盖与滞后，推理忽略外部传入历史。',
        '- 分层校正使用训练内部前向OOF残差；主效应和阶段交互有固定收缩，不在验证上拟合校正。',
        '- 神经网络只比较解码器：同一个小型MLP编码器，直接输出、进度基函数积分、进度加末24h/6h基函数积分。本轮只用终分监督，不能把分解系数解释为真实玩家行为；尚未验证区间辅助监督、自监督预训练、Neural CDE或19档曲面。',
        '- V1和V2分别拟合/选择，跨协议汇总数字不能直接比较。V2 57、68无可用前缀但保留原分组；198组中实际拟合196活动。',
        '- 所有正式案例保留，训练按活动等权。验证参与多候选选择，存在选择乐观偏差。',
        '', '**限制：测试队列曾参与早期探索，不能称完全未见盲测；元数据为缺少历史时点存档的回溯快照。**','']
    output.mkdir(parents=True)
    (output/'README.md').write_text('\n'.join(text))
    (output/'renderer_source.py').write_text(Path(__file__).read_text())
    pd.DataFrame(rows).to_csv(output/'all_candidates.csv',index=False)
    write_json(output/'report_manifest.json',{'inputs':inputs,'source_sha256':sha256(Path(__file__)),
        'artifacts':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}})
    print(str(output/'README.md'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--v2',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report({'v1':a.v1,'v2':a.v2},a.output)
