"""Render paired/TCN experiment evidence from sealed scores and blind audit files."""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from analysis.holdout_dataset import SPLITS, sha256, write_json
from analysis.temporal_dataset import TemporalDataset, STAGES
from analysis.temporal_evaluate import verify_predictions

NAMES = {'hold_current': '沿用当前分数', 'linear': '线性外推', 'reference_catboost': '原单线 CatBoost',
    'catboost_paired': '双线 CatBoost', 'catboost_phase': '单线 CatBoost：阶段权重', 'dual_tcn_mean': '双分支 TCN：三种子均值',
    'dual_tcn_seed42': 'TCN seed 42', 'dual_tcn_seed123': 'TCN seed 123', 'dual_tcn_seed2026': 'TCN seed 2026'}


def report(run):
    run = Path(run); seal = verify_predictions(run)
    evaluation_seal = json.loads((run/'evaluation_manifest.json').read_text())
    for name, digest in evaluation_seal['artifacts'].items():
        if sha256(run/name) != digest:
            raise ValueError('Scored artifact changed: '+name)
    config = json.loads((run/'run_manifest.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    evaluation = json.loads((run/'evaluation.json').read_text())
    data = TemporalDataset(seal['dataset_directory'])
    if evaluation['dataset_id'] != seal['dataset_id'] or data.manifest['dataset_id'] != seal['dataset_id']:
        raise ValueError('Dataset identity mismatch')
    if evaluation['evaluation_id'] != evaluation_seal['evaluation_id']:
        raise ValueError('Evaluation identity mismatch')
    selected = selection['selected_recipe']; scores = evaluation['splits']
    if selected != seal['reference_method']:
        raise ValueError('Selection differs from sealed scorer reference')
    output = run/'report'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    def pct(value):
        return '—' if value is None else f'{value:.2f}%'
    lines = ['# V2 两档：双线信息、TCN 与阶段权重对照', '',
        f"验证集选择 **{NAMES[selected]}**；对应测试集 event-macro MAPE 为 **{pct(scores['test'][selected]['overall']['event_macro_mape_pct'])}**。",
        '', '本轮冻结候选与配置后训练，全部盲预测封存后才独立评分。没有根据本轮测试表现重训、调参或挑单个种子。全部方法使用相同的 2,229/1,548/1,514 个训练/验证/测试前缀，沿用 198/42/43 的活动归属；训练实际参与 196 个活动。',
        '', '## 完整方法比较', '', '| 方法 | 训练组 MAPE | 验证组 MAPE | 测试组 MAPE | 测试 P90 APE |', '|---|---:|---:|---:|---:|']
    for method in seal['methods']:
        values = [pct(scores[s][method]['overall']['event_macro_mape_pct']) for s in SPLITS]
        lines.append('| '+NAMES[method]+(' **（验证选中）**' if method == selected else '')+' | '+' | '.join(values)
                     +' | '+pct(scores['test'][method]['overall']['p90_ape_pct'])+' |')
    lines += ['', '训练组数值是拟合误差，不代表泛化。逐种子结果全部公开；候选包含三种子最终分数的算术平均，不包含单独种子。主指标按活动等权，不把快照和两条排名当作独立活动。',
        '', '## 三种子训练记录', '', '| 种子 | 参数量 | 实际训练轮数 | 验证选择轮数 | 验证 MAPE | 测试 MAPE |', '|---|---:|---:|---:|---:|---:|']
    for seed in config['seeds']:
        method = f'dual_tcn_seed{seed}'; trace = json.loads((run/f'models/{method}.json').read_text())
        lines.append(f"| {seed} | {trace['parameters']:,} | {trace['epochs_run']} | {trace['best_epoch']} | {pct(scores['validation'][method]['overall']['event_macro_mape_pct'])} | {pct(scores['test'][method]['overall']['event_macro_mape_pct'])} |")
    methods = ['reference_catboost', 'catboost_paired', 'catboost_phase', 'dual_tcn_mean']
    lines += ['', '## 各排名的完整测试结果', '', '| 方法 | T1000 MAPE | T2000 MAPE |', '|---|---:|---:|']
    for method in methods:
        lines.append('| '+NAMES[method]+' | '+' | '.join(pct(scores['test'][method]['by_rank'][str(r)]['event_macro_mape_pct']) for r in [1000, 2000])+' |')
    lines += ['', '## 全部测试时点', '', '| 请求时点 | 原单线 | 双线 CatBoost | 阶段权重单线 | TCN 均值 | 活动数/案例数 |', '|---|---:|---:|---:|---:|---|']
    for stage in STAGES:
        key, axis, value = stage['stage_id'], stage['axis'], stage['value']
        label = f'开始后 {value:g}h' if axis == 'elapsed' else f'进度 {value:.0%}' if axis == 'progress' else f'结束前 {value:g}h'
        metric = scores['test'][selected]['by_stage'][key]
        lines.append('| '+label+' | '+' | '.join(pct(scores['test'][m]['by_stage'][key]['event_macro_mape_pct']) for m in methods)
                     +f" | {metric['events']}/{metric['cases']} |")
    lines += ['', '“结束前”是请求时点，实际最后观测可能更早。开始后 1h、3h 的冻结队列没有合格案例，继续报告缺失。各方法不能各自剔除难例。',
        '', '## 双线可用性与实验边界', '', '| 分组 | 全部案例 | 配对可用 | 可用率 | 无同请求时点配对 | 截断后不足质量条件 |', '|---|---:|---:|---:|---:|---:|']
    coverage = []
    for split in SPLITS:
        audit = pd.read_csv(run/f'features/{split}_pair_audit.csv')
        available = int(audit.peer_available.sum()); missing = int((audit.reason == 'no same-origin peer case').sum())
        invalid = int((audit.reason == 'peer invalid after as-of truncation').sum())
        coverage.append({'split': split, 'cases': len(audit), 'available': available, 'missing_origin': missing, 'invalid_after_truncation': invalid})
        lines.append(f'| {split} | {len(audit)} | {available} | {available/len(audit):.1%} | {missing} | {invalid} |')
    pd.DataFrame(coverage).to_csv(output/'paired_coverage.csv', index=False)
    lines += ['', '仅配对同活动同请求时点的盲前缀，并将另一档截到当前线的实际已观测时刻。截断后为空、不足4点或过旧则保留为缺失。不同分组的配对覆盖差异会影响迁移，不能把双线模型的差异只归因于网络优劣。双线 CatBoost 与 TCN 看到相同原始特征信息；与原基线相比，新增内容包括双线特征及显式观测掩码。',
        '', '阶段权重对照保持原单线特征、模型参数和案例不变，只在每个活动内部按 <25%、25%–90%、>=90% 的互斥进度阶段均分权重，再保证每活动总权重相等。它不能创造缺失的早期观测。',
        '', 'TCN 采用两条3层因果卷积分支、全前缀均值与末端状态汇总、表格分支及排名专用输出头。与树模型相同，目标是线性外推的 log 增长残差，按活动等权平方误差训练；本轮没有同时加入新的乐队/卡牌特征。',
        '', '## 同活动同排名面板与审计', '',
        '`../evaluation.json` 保存每种方法的共同面板、观测滞后≤15min子集、排名、活动类型、乐队细分，以及 case-MAPE、MAE、RMSE、P90。`../stage_rank_metrics.csv` 给出每个时点×排名的误差及样本量；`../stage_calendar_metrics.csv` 保留季度差异。`../paired_bootstrap.csv` 对验证选中方法与简单基线进行活动级配对 bootstrap，属于描述性比较。',
        '', f"数据指纹：`{seal['dataset_id']}`。", '', f"评分标识：`{evaluation['evaluation_id']}`。", '',
        '模型、训练预处理、逐轮记录、特征缓存、配对审计、权重、源代码、依赖版本与预测都已封存。原 V1/V2 和上一轮结果不变；这不是 V1 主标准的改进声明。', '',
        '**限制：测试活动曾参与早期探索，不能称为完全未见盲测；元数据仍是回溯快照，没有历史时点存档。**', '',
        'TCN 方法参考：[Bai 等，2018](https://arxiv.org/abs/1803.01271)。本项目是前缀到终分的适配模型，不声称复现论文任务。', '',
        '复现时使用新运行目录：', '', '```bash',
        'DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib .venv/bin/python -m analysis.temporal_paired_benchmark --run analysis/temporal_results/<new-run>',
        '.venv/bin/python -m analysis.temporal_evaluate --run analysis/temporal_results/<new-run>',
        '.venv/bin/python -m analysis.temporal_paired_report --run analysis/temporal_results/<new-run>', '```', '']
    (output/'README.md').write_text('\n'.join(lines))
    shutil.copyfile(__file__, output/'report_source.py')
    write_json(output/'manifest.json', {'blind_manifest_sha256': sha256(run/'blind_manifest.json'),
        'evaluation_manifest_sha256': sha256(run/'evaluation_manifest.json'),
        'artifacts': {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return output/'README.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(report(parser.parse_args().run))
