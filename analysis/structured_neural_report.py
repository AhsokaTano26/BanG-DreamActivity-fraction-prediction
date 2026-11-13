"""Audit and report sealed predictions; consumes scorer output, never label files."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np
from analysis.holdout_dataset import ROOT, HoldoutDataset, sha256, write_json


def report(run):
    output = run/'paired_comparison_and_verification.json'
    if output.exists(): raise FileExistsError('Report already exists; preserve the audit')
    config = json.loads((run/'run_manifest.json').read_text())
    evaluation = json.loads((run/'evaluation.json').read_text()); selection = evaluation['selection']
    data = HoldoutDataset(config['dataset_directory']); samples = data.inputs('test')
    frozen = {s['case_id']: s['curve'].event for s in samples}
    current = {s['case_id']: s['score'] for s in samples}
    for seal_name in ['development_seal.json', 'blind_test_manifest.json', 'evaluation_manifest.json']:
        seal = json.loads((run/seal_name).read_text())
        for name, digest in seal['artifacts'].items():
            if sha256(run/name) != digest: raise ValueError('Seal mismatch: '+name)
    with (run/'scored_test_predictions.csv').open() as stream: rows = list(csv.DictReader(stream))
    methods = defaultdict(list)
    for row in rows: methods[row['method']].append(row)
    if set(methods) != set(selection['reported_methods']): raise ValueError('Method list changed')
    event_errors = {}; discrepancies = []
    for name, group in methods.items():
        if len(group) != len(frozen) or {r['case_id'] for r in group} != set(frozen): raise ValueError('Case identities changed')
        errors = defaultdict(list)
        for row in group:
            case = row['case_id']; prediction = float(row['predicted']); actual = float(row['actual'])
            if int(row['event']) != frozen[case]: raise ValueError('Event identity changed')
            if prediction < current[case] or not np.isfinite(prediction): raise ValueError('Invalid remaining forecast')
            errors[frozen[case]].append(abs(prediction/actual-1)*100)
        event_errors[name] = {event: statistics.mean(values) for event, values in errors.items()}
        discrepancies.append(abs(statistics.mean(event_errors[name].values())-evaluation['test'][name]['overall']['event_macro_mape_pct']))
    if max(discrepancies) > 1e-10: raise ValueError('Independent metric calculation differs')
    ids = sorted(event_errors['reference']); rng = np.random.default_rng(42)
    draws = rng.integers(0, len(ids), (5000, len(ids))); comparisons = {}
    for name, errors in event_errors.items():
        delta = np.array([errors[e]-event_errors['reference'][e] for e in ids])
        comparisons[name] = {'delta_mape_points': float(delta.mean()),
                             'paired_event_bootstrap_95pct_ci': np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
                             'improved_events': int((delta < 0).sum()), 'worse_events': int((delta > 0).sum())}
    verification = {'dataset_id': data.manifest['dataset_id'], 'run_id': config['run_id'],
                    'evaluation_id': evaluation['evaluation_id'], 'methods': len(methods), 'cases_per_method': len(frozen),
                    'events': len(ids), 'independent_metric_max_difference': max(discrepancies),
                    'all_seals_valid': True, 'event_assignment': 'frozen test inputs, not prediction-supplied IDs',
                    'bootstrap_draws': 5000, 'bootstrap_seed': 42,
                    'comparisons_to_reference': comparisons, 'source_sha256': sha256(Path(__file__)),
                    'source_database_sha256': sha256(ROOT/'data/db.sqlite3')}
    write_json(output, verification)
    records = json.loads((run/'training_records.json').read_text())
    summary_rows = []
    for name in selection['reported_methods']:
        summary_rows.append({'method': name, 'validation_event_mape_pct': selection['validation_scores'][name],
                             **evaluation['test'][name]['overall'], **comparisons[name]})
    with (run/'all_method_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
    selected = selection['selected_recipe']; results = evaluation['test']; ref = results['reference']['overall']; chosen = results[selected]['overall']
    nn = sorted([n for n in selection['selection_candidates'] if '_ensemble' in n], key=lambda n: selection['validation_scores'][n])
    best_nn = nn[0]; top = results[best_nn]['overall']; delta = comparisons[selected]
    lines = ['# 双档结构化神经网络实验', '',
             f'固定 `holdout_v1` 上完成 {len(config["recipes"])} 个配置 × {len(config["seeds"])} 个种子，共 {len(records)} 次训练；每个方法评价相同的 {len(frozen)} 个测试案例、{len(ids)} 个活动。', '',
             f'验证集选定 **`{selected}`**：验证活动 MAPE **{selection["validation_scores"][selected]:.3f}%**，测试 **{chosen["event_macro_mape_pct"]:.3f}%**。对照是上一轮验证选定的组合，测试 **{ref["event_macro_mape_pct"]:.3f}%**。', '',
             f'神经网络中验证最优的三种子集成为 `{best_nn}`：验证 {selection["validation_scores"][best_nn]:.3f}%，测试 {top["event_macro_mape_pct"]:.3f}%。单个种子只作透明报告，不进入最终方案选择。', '',
             '## 设计与预先固定的比较', '',
             '网络学习后续平均增速相对已有平均增速的对数倍率，终分 = 当前分数 × [1 + exp(倍率对数) × (1−进度)/进度]。倍率对数裁剪到 [-6,4]；预测不低于当前分数，剩余时长趋零时回到当前分数。', '',
             '- 共享 MLP 与分排名头 MLP：检验排名专属输出头的作用。',
             '- 双档 GRU、因果 TCN、全程加最近 48 小时 TCN：共享编码器，融合本档、另一档、差和乘积。',
             '- 跨档 patch attention：每 8 个时间位置作为 patch，联合两档 token，并用排名 query 汇总。',
             '- 门控 TCN：三个软门控专家，各输出两档预测头。',
             '- TCN 目标损失对照：0.8 平滑相对终分误差 + 0.2 Huber 对数倍率误差。',
             '- TCN 稀疏化对照：训练时随机将一半前缀观测减少至约四分之一，保留首末观测、当前分数和预测时刻。', '',
             '每档输入为 64 个共享时间位置 × 8 通道：相对累计分数、相对增速、可用标记、观测年龄、插值跨度、活动进度和昼夜正余弦。另一档只取当前已观测时刻以内数据；缺失或超时用 mask 表达。数值汇总包含 39 个紧凑原始特征，去掉绝对分数和乐队特征，以检验前一轮推荐结构；活动类型保留。', '',
             '固定宽度 48、dropout 0.2、AdamW 学习率 0.001/权重衰减 0.005、batch 256、上限 100 轮、至少 30 轮/耐心 20、梯度裁剪 1。普通损失为 beta=0.2 的 Huber。所有样本使用全局活动等权，同一活动两档合计权重不变；预处理只拟合训练数据。每个配置固定种子 42、123、2026，先逐种子选验证最佳检查点，再对最终分数取均值。', '',
             '验证集决定三个神经集成成员及与旧方案 25%/50%/75% 的混合，旧方案本身也列为候选。测试预测全部封存后，原独立评分器才读取答案；没有根据测试结果换模型或重训。', '',
             '## 三种子集成与组合结果', '',
             '| 方法 | 验证活动 MAPE % | 测试活动 MAPE % |', '|---|---:|---:|']
    for name in sorted(selection['selection_candidates'], key=lambda n: selection['validation_scores'][n]):
        lines.append(f'| `{name}` | {selection["validation_scores"][name]:.3f} | {results[name]["overall"]["event_macro_mape_pct"]:.3f} |')
    lines += ['', '验证选出的三个神经成员：'+', '.join(f'`{n}`' for n in selection['top3_members'])+'。', '',
              '## 逐种子结果', '', '| 配置 | 种子 | 最佳轮数 | 验证 MAPE % | 测试 MAPE % |', '|---|---:|---:|---:|---:|']
    for r in records:
        name = f'{r["method"]}_seed{r["seed"]}'
        lines.append(f'| `{r["method"]}` | {r["seed"]} | {r["best_epoch"]} | {r["validation_mape"]:.3f} | {results[name]["overall"]["event_macro_mape_pct"]:.3f} |')
    lines += ['', '## 分排名与预测时点', '',
              f'下表同时给出旧方案、事先选定方案和验证最优神经网络 `{best_nn}`。', '',
              '| 分组 | 旧方案 MAPE % | 选定方案 MAPE % | 验证最优神经 MAPE % |', '|---|---:|---:|---:|']
    for key in ['by_rank', 'by_stage']:
        for group in results['reference'][key]:
            values = [results[n][key][group]['event_macro_mape_pct'] for n in ['reference', selected, best_nn]]
            lines.append(f'| {group} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |')
    ci = delta['paired_event_bootstrap_95pct_ci']
    lines += ['', f'选定方案相对旧方案的活动 MAPE 差值为 {delta["delta_mape_points"]:+.3f} 个百分点，配对活动 bootstrap 95% 区间 [{ci[0]:.3f}, {ci[1]:.3f}]；{delta["improved_events"]}/{len(ids)} 个活动改善。若选定方案就是 reference，差值恒为零，不构成新的改善证据。', '',
              'case-MAPE、MAE、RMSE、P90、活动类型与乐队明细见 [evaluation.json](evaluation.json)；全部方法汇总见 [all_method_comparison.csv](all_method_comparison.csv)。乐队分组按加成角色构成，不等同于活动故事主角。', '',
              '## 本轮决策', '',
              ('验证选择保留上一轮组合。本轮神经网络探索没有形成可替换旧方案的验证证据，不能用某个测试分数较好的网络反过来改选。'
               if selected == 'reference' else
               f'保留验证事先选定的 `{selected}` 作为本轮版本。它在本队列的测试差值为 {delta["delta_mape_points"]:+.3f} 个百分点；区间和分组结果一起判断，不能只看单个种子的最佳值。'), '',
              '后续应优先做训练内部按活动、按时间的验证，检查神经网络能否学习已有结构模型的剩余误差，并保留共享编码与排名专属输出头的对照。对新活动建立前瞻记录，核对 1000 线和早期阶段。增加网络层数本身并不构成预期收益；本轮实验也不能把某种架构的一次胜负推广为普遍规律。', '',
              '## 审计与限制', '',
              f'- 数据指纹：`{config["dataset_id"]}`；训练/验证/测试仍为 198/42/43 个活动，4377/414/412 个前缀。',
              f'- 运行身份：`{config["run_id"]}`；评分身份：`{evaluation["evaluation_id"]}`。',
              f'- 独立逐活动重算 {len(methods)} 组主指标，最大差异 {max(discrepancies):.2g}；数据、开发、盲预测及评分封存均通过校验。',
              '- [paired_comparison_and_verification.json](paired_comparison_and_verification.json) 保存每个方法相对 reference 的配对活动区间。区间为队列内探索比较，没有多重模型比较校正，也没有建模活动间长期时间相关；不能解释成未来预测区间。',
              '- 原数据库未用于模型输入；数据读取使用 HoldoutDataset 校验固定文件。模型、预处理、源代码、依赖版本和预测已归档。',
              '- **测试队列曾被早期探索使用，不能称为完全未见过的盲测；元数据是回溯快照，缺少历史时点存档。** 训练活动只有 198 个，重复前缀不是独立活动；采样密度和跨档可用性存在年代变化。',
              '- 验证集已在多轮研究中复用，存在选择乐观；三种子不能增加独立活动数。稀疏网格仍使用插值近似，倍率裁剪不能精确表达零后续增长。', '',
              '方法借鉴 [PatchTST 的分块思路](https://arxiv.org/abs/2211.14730) 与 [GRU-D 对缺失和观测间隔的处理动机](https://arxiv.org/abs/1606.01865)。这里的跨档 attention、GRU+间隔通道是任务适配实现，未复现这两篇论文的完整架构。', '',
              '```bash', '# 新实验另存目录，保留本轮封存结果',
              'DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib .venv/bin/python -m analysis.structured_neural_benchmark develop --run analysis/holdout_results/new_neural_run --device mps',
              'DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib .venv/bin/python -m analysis.structured_neural_benchmark predict --run analysis/holdout_results/new_neural_run',
              '.venv/bin/python -m analysis.holdout_evaluate --run analysis/holdout_results/new_neural_run',
              '.venv/bin/python -m analysis.structured_neural_report --run analysis/holdout_results/new_neural_run', '```', '']
    (run/'README.md').write_text('\n'.join(lines))
    shutil_source = run/'report_source.py'; shutil_source.write_text(Path(__file__).read_text())
    write_json(run/'report_manifest.json', {'evaluation_id': evaluation['evaluation_id'], 'artifacts':
               {p.name: sha256(p) for p in [output, run/'all_method_comparison.csv', run/'README.md', shutil_source]}})
    print(json.dumps({'selected': selected, 'test_mape': chosen['event_macro_mape_pct'], 'comparison': delta,
                      'best_validation_neural': best_nn, 'neural_test_mape': top['event_macro_mape_pct']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    report(parser.parse_args().run)
