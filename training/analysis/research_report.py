"""Human-readable complete candidate ledger from independently sealed scores."""
import argparse, json, shutil
from pathlib import Path
import pandas as pd
from analysis.holdout_dataset import sha256, write_json
from analysis.temporal_dataset import TemporalDataset, STAGES
from analysis.temporal_evaluate import verify_predictions


def report(run):
    run = Path(run); seal = verify_predictions(run)
    for directory in [run, run/'comparison']:
        receipt = json.loads((directory/'evaluation_manifest.json').read_text())
        for n, digest in receipt['artifacts'].items():
            if sha256(directory/n) != digest:
                raise ValueError('Evaluation output changed: '+n)
    evaluation = json.loads((run/'evaluation.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    config = json.loads((run/'run_manifest.json').read_text())
    comparison = json.loads((run/'comparison/comparison.json').read_text())
    data = TemporalDataset(seal['dataset_directory'])
    if evaluation['dataset_id'] != seal['dataset_id'] or data.manifest['dataset_id'] != seal['dataset_id']:
        raise ValueError('Dataset identity differs')
    selected = selection['selected_recipe']; scores = evaluation['splits']
    if selected != seal['reference_method']:
        raise ValueError('Selection differs')
    output = run/'report'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    def pct(v): return '—' if v is None else f'{v:.2f}%'
    methods = list(dict.fromkeys(['reference_single', 'reference_paired', 'reference_tcn']+list(selection['family_best'].values())+[selected]))
    summary = []
    for method in config['methods']:
        row = {'method': method, 'selected_by_validation': method == selected}
        for split in ['train', 'validation', 'test']:
            row[split+'_event_mape_pct'] = scores[split][method]['overall']['event_macro_mape_pct']
        row.update({'test_t1000_mape_pct': scores['test'][method]['by_rank']['1000']['event_macro_mape_pct'],
                    'test_t2000_mape_pct': scores['test'][method]['by_rank']['2000']['event_macro_mape_pct'],
                    'test_p90_ape_pct': scores['test'][method]['overall']['p90_ape_pct']})
        summary.append(row)
    pd.DataFrame(summary).to_csv(output/'all_candidates.csv', index=False)
    lines = ['# V2 多 agent 全方向对照', '',
        f"验证集选中 **{selected}**。验证 MAPE **{pct(scores['validation'][selected]['overall']['event_macro_mape_pct'])}**，测试 MAPE **{pct(scores['test'][selected]['overall']['event_macro_mape_pct'])}**。所有数字为活动等权 event-macro MAPE。", '',
        f"三个 agent 各自开发6个固定配方，统一重新拟合后比较全部{len(config['recipes'])}个新配方、7个验证选择组合、5个历史参考。各 agent 仅访问训练/验证组；主入口在选择及组合参数封存后才打开测试盲前缀，独立评分器最后读取答案。测试成绩未用于改模型或改主方法。", '',
        '## 各方向验证最优及历史参考', '', '| 方法 | 训练 MAPE | 验证 MAPE | 测试 MAPE | T1000 测试 | T2000 测试 |', '|---|---:|---:|---:|---:|---:|']
    lookup = {r['method']: r for r in summary}
    for method in methods:
        r = lookup[method]
        lines.append('| '+method+(' **（验证选中）**' if method == selected else '')+' | '
            +' | '.join(pct(r[k]) for k in ['train_event_mape_pct', 'validation_event_mape_pct', 'test_event_mape_pct', 'test_t1000_mape_pct', 'test_t2000_mape_pct'])+' |')
    lines += ['', '训练组误差仅为拟合表现。旧TCN为上一轮三个固定种子的最终分数均值；本轮没有择优挑选旧种子，也没有新增神经训练。', '',
        '## 全部候选（按验证误差排序）', '', '| 方法 | 训练 MAPE | 验证 MAPE | 测试 MAPE |', '|---|---:|---:|---:|']
    for r in sorted(summary, key=lambda r: r['validation_event_mape_pct']):
        lines.append('| '+r['method']+' | '+' | '.join(pct(r[k]) for k in ['train_event_mape_pct', 'validation_event_mape_pct', 'test_event_mape_pct'])+' |')
    lines += ['', '主方法严格来自验证排序首位，不用测试排序替换。全部配方、权重和源码位于 ../run_manifest.json、../models/、../sources/。全量机器可读汇总见 all_candidates.csv。', '',
        '## 验证阶段选定的组合规则', '', '```json', json.dumps(selection['combination_specs'], ensure_ascii=False, indent=2), '```', '',
        '其中阶段组合边界预先固定为25%/90%；它在三个方向开发发现验证上的阶段互补后、正式统一训练和任何新测试推理前加入，实验计划记录了这一补充。验证集参与多候选选择可能导致乐观偏差，因此同时保留完整队列的独立测试及活动级不确定性。', '',
        '## 配对活动比较', '', '| 方法 | 参考 | 新减旧/百分点 | 95%下界 | 95%上界 | 改善/变差活动数 |', '|---|---|---:|---:|---:|---|']
    for row in comparison['comparison']:
        lines.append(f"| {row['method']} | {row['reference']} | {row['new_minus_reference_mape_points']:.3f} | {row['ci95_low']:.3f} | {row['ci95_high']:.3f} | {row['events_improved']}/{row['events_worsened']} |")
    lines += ['', '按冻结case活动归属先计算活动误差，再对43个活动配对重采样5000次；负差值表示新方法误差更低。区间是未做多重比较校正的描述性边际区间，不能用于测试后择优或直接推断因果。', '',
        '## 全部测试时间点', '', '| 请求时点 | 原单线 | 上轮双线 | 本轮验证选中 | 活动数/案例数 |', '|---|---:|---:|---:|---|']
    for s in STAGES:
        key, axis, v = s['stage_id'], s['axis'], s['value']
        label = f'开始后{v:g}h' if axis == 'elapsed' else f'进度{v:.0%}' if axis == 'progress' else f'结束前{v:g}h'
        metric = scores['test'][selected]['by_stage'][key]
        lines.append('| '+label+' | '+' | '.join(pct(scores['test'][m]['by_stage'][key]['event_macro_mape_pct']) for m in ['reference_single', 'reference_paired', selected])+f" | {metric['events']}/{metric['cases']} |")
    lines += ['', '1h/3h早期没有冻结合格案例，继续报告缺失；请求时点和实际观测时点分别保存。不同方法不能各自删案例；时间轴重叠的物理案例总体只计一次。共同活动×排名面板、滞后≤15min子集、活动类型、乐队、MAE/RMSE/P90等完整诊断保存于 ../evaluation.json；全部时点×排名见 ../stage_rank_metrics.csv。', '',
        '## 数据、运行与限制', '',
        f"数据指纹 `{seal['dataset_id']}`。主评分标识 `{evaluation['evaluation_id']}`。比较评分标识 `{comparison['evaluation_id']}`。", '',
        '训练/验证/测试2229/1548/1514个前缀，原分组198/42/43个活动；训练实际参与196个活动，57/68没有可用前缀，仍保留分组。所有训练活动总权重相同，排名分支切片全局权重；OOF校准只使用训练活动并严格按已结束活动时间组织。', '',
        '双线输入采用此前安全配对规则，训练/验证/测试可用配对为950/1548/1512个；覆盖差异可能影响泛化。缺失保留，掩码与门控显式处理。', '',
        '旧V1/V2、数据库与旧封存结果保留。本轮是用户指定V2范围的比较，不构成V1主标准改进声明。', '',
        '**限制：该测试队列曾参与早期探索，不能称为完全未见盲测；元数据是缺少历史时点存档的回溯快照。**', '',
        '```bash', '.venv/bin/python -m analysis.research_search --run analysis/temporal_results/<new-run>',
        '.venv/bin/python -m analysis.temporal_evaluate --run analysis/temporal_results/<new-run>',
        '.venv/bin/python -m analysis.research_compare --run analysis/temporal_results/<new-run>',
        '.venv/bin/python -m analysis.research_report --run analysis/temporal_results/<new-run>', '```', '']
    (output/'README.md').write_text('\n'.join(lines))
    shutil.copyfile(__file__, output/'report_source.py')
    write_json(output/'manifest.json', {'blind_manifest_sha256': sha256(run/'blind_manifest.json'),
        'evaluation_manifest_sha256': sha256(run/'evaluation_manifest.json'),
        'artifacts': {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return output/'README.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    print(report(parser.parse_args().run))
