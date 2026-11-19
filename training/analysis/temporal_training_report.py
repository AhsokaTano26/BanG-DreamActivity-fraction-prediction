"""Report a sealed Temporal V2 training run without reading target labels."""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from analysis.holdout_dataset import SPLITS, sha256, write_json
from analysis.temporal_dataset import TemporalDataset, STAGES
from analysis.temporal_evaluate import verify_predictions


def report(run):
    run = Path(run)
    seal = verify_predictions(run)
    data = TemporalDataset(seal['dataset_directory'])
    if data.manifest['dataset_id'] != seal['dataset_id'] or data.manifest['protocol']['ranks'] != [1000, 2000]:
        raise ValueError('This report requires the sealed T1000/T2000 dataset')
    config = json.loads((run / 'run_manifest.json').read_text())
    scoring = json.loads((run / 'evaluation_manifest.json').read_text())
    for name, digest in scoring['artifacts'].items():
        if sha256(run / name) != digest:
            raise ValueError('Evaluation artifact changed: ' + name)
    evaluation = json.loads((run / 'evaluation.json').read_text())
    if evaluation['dataset_id'] != seal['dataset_id'] or evaluation['evaluation_id'] != scoring['evaluation_id']:
        raise ValueError('Evaluation identity differs')
    selection = json.loads((run / 'selection.json').read_text())
    selected = selection['selected_recipe']
    scores = evaluation['splits']
    output = run / 'report'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    pct = lambda v: '—' if v is None else f'{v:.2f}%'
    names = {'hold_current': '沿用当前分数', 'linear': '线性外推',
             'catboost_rank': '新训 CatBoost：曲线', 'catboost_context': '新训 CatBoost：曲线＋活动元数据',
             'frozen_catboost_rank': '旧冻结 19 档 CatBoost：曲线',
             'frozen_catboost_context': '旧冻结 19 档 CatBoost：曲线＋活动元数据'}
    lines = ['# T1000 / T2000 完整 V2 训练与评测', '',
        f"验证集选择的主方法为 **{names.get(selected, selected)}**，全部测试时点的 event-macro MAPE 为 **{pct(scores['test'][selected]['overall']['event_macro_mape_pct'])}**。",
        '', '本次使用两档全部可用 V2 训练前缀重新拟合固定 CatBoost 配方。验证集只选择主方法，测试评分前已完成模型及预测封存；没有把验证/测试活动终分加入训练。',
        '', '## 样本范围', '', '| 分组 | 固定分组活动数 | 有可用前缀的活动数 | 前缀数 | 无可用前缀的活动 ID |', '|---|---:|---:|---:|---|']
    for split in SPLITS:
        s = data.manifest['splits'][split]
        missing = ', '.join(map(str, s['events_without_cases'])) or '无'
        lines.append(f"| {split} | {len(s['event_ids'])} | {s['events_with_cases']} | {s['cases']:,} | {missing} |")
    lines += ['', '保持 198/42/43 的活动归属、21 个预设时点及原有质量条件。开始后 1h、3h 无合格两档前缀，按缺失报告；不插入虚构样本。每个训练活动总权重相同，总体指标先对活动内部案例平均、再对活动等权平均。',
        '', '## 所有固定方法', '', '| 方法 | 训练组 MAPE | 验证组 MAPE | 测试组 MAPE | 测试 P90 APE |', '|---|---:|---:|---:|---:|']
    for method in seal['methods']:
        values = [pct(scores[s][method]['overall']['event_macro_mape_pct']) for s in SPLITS]
        lines.append('| ' + names.get(method, method) + (' **（验证选中）**' if method == selected else '') + ' | '
                     + ' | '.join(values) + ' | ' + pct(scores['test'][method]['overall']['p90_ape_pct']) + ' |')
    lines += ['', '新模型的训练组数值是拟合误差，不代表泛化。旧冻结模型一栏是在已参与旧模型训练的活动上回放 V2 时点，并非旧模型原训练案例的拟合误差。旧模型使用 19 档训练，新模型使用两档及 V2 时点，因此同案例对照同时包含排名范围和训练采样变化，不能单独归因于其中一项。',
        '', f"固定配方：{config['iterations']} 棵树、深度 {config['tree_depth']}、学习率 {config['learning_rate']}、L2 {config['l2_leaf_reg']}、种子 {config['seed']}；未依据测试分数搜索参数。元数据版包含活动类型、乐队/角色与按国服发布时间过滤的卡牌特征。本轮没有新增神经网络训练。",
        '', '## 主方法的全部测试时点', '',
        '| 请求时点 | T1000 MAPE | T1000 活动数 | T2000 MAPE | T2000 活动数 | 两档合计 MAPE | 合计活动数/案例数 | 线性外推 MAPE | 沿用当前 MAPE |',
        '|---|---:|---:|---:|---:|---:|---|---:|---:|']
    main = scores['test'][selected]
    stage_rank = {(r['stage_id'], r['rank']): r for r in main['stage_rank']}
    stage_rows = []
    for stage in STAGES:
        key, axis, value = stage['stage_id'], stage['axis'], stage['value']
        if axis == 'elapsed':
            label = f'开始后 {value:g}h'
        elif axis == 'progress':
            label = f'进度 {value:.0%}'
        else:
            label = f'结束前 {value:g}h' if value >= 1 else f'结束前 {value * 60:g}min'
        a, b = (stage_rank.get((key, rank), {}) for rank in (1000, 2000))
        combined = main['by_stage'][key]
        linear = scores['test']['linear']['by_stage'][key]['event_macro_mape_pct']
        current = scores['test']['hold_current']['by_stage'][key]['event_macro_mape_pct']
        lines.append(f"| {label} | {pct(a.get('event_macro_mape_pct'))} | {a.get('events', 0)} | {pct(b.get('event_macro_mape_pct'))} | {b.get('events', 0)} | {pct(combined['event_macro_mape_pct'])} | {combined['events']}/{combined['cases']} | {pct(linear)} | {pct(current)} |")
        stage_rows.append({'stage_id': key, 'label': label, 't1000_mape_pct': a.get('event_macro_mape_pct'),
            't1000_events': a.get('events', 0), 't2000_mape_pct': b.get('event_macro_mape_pct'),
            't2000_events': b.get('events', 0), 'combined_mape_pct': combined['event_macro_mape_pct'],
            'events': combined['events'], 'cases': combined['cases'], 'linear_mape_pct': linear, 'hold_current_mape_pct': current})
    pd.DataFrame(stage_rows).to_csv(output / 'selected_stage_rank.csv', index=False)
    lines += ['', '“请求时点”与“最后观测时点”分别保存。前缀最多允许滞后 60 分钟，所以上表的结束前 15min 不保证模型已见到距结束恰好 15min 的观测。全部方法使用完全相同的冻结案例。',
        '', '## 相同活动×排名的共同面板', '', '| 时间轴 | 共同活动数 | 共同活动×排名组合数 |', '|---|---:|---:|']
    for axis, panel in main['common_panels'].items():
        lines.append(f"| {axis} | {panel['events']} | {panel['event_rank_pairs']} |")
    lines += ['', 'elapsed 共同面板因 1h/3h 无数据而为空。其余面板固定同一批活动和排名，用于检查时间走势是否由样本构成变化造成。',
        '', '| 结束前 | 共同面板主方法 MAPE | 共同面板线性 MAPE | 全部案例中滞后≤15min的主方法 MAPE | 新鲜案例活动数/案例数 |', '|---|---:|---:|---:|---|']
    for stage in STAGES:
        if stage['axis'] != 'remaining':
            continue
        key = stage['stage_id']
        p = main['common_panels']['remaining']['by_stage'][key]
        b = scores['test']['linear']['common_panels']['remaining']['by_stage'][key]
        f = main['fresh_15min_by_stage'][key]
        lines.append(f"| {stage['value']:g}h | {pct(p['event_macro_mape_pct'])} | {pct(b['event_macro_mape_pct'])} | {pct(f['event_macro_mape_pct'])} | {f['events']}/{f['cases']} |")
    lines += ['', '## 审计和复现', '',
        'MAE、RMSE、P90、case-MAPE、活动类型和乐队分层均保存于 `../evaluation.json`。`../stage_metrics.csv`、`../stage_rank_metrics.csv`、`../stage_calendar_metrics.csv` 包含三个数据组的时间细分结果。`../paired_bootstrap.csv` 对冻结活动归属进行 2,000 次配对活动重采样；区间属于描述性比较，不用于测试集择优。',
        '', f"数据指纹：`{data.manifest['dataset_id']}`。", '', f"评分指纹：`{evaluation['evaluation_id']}`。", '',
        '原 V1、父 V2、旧预测与数据库均保留。不同时间网格的 V1/V2 总分不能直接作为模型改进证据。', '',
        '**限制：测试活动队列曾参与早期探索，不能称为完全未见盲测；活动元数据是回溯快照，缺少历史时点存档。**', '',
        '重跑必须使用新的运行目录；已封存预测拒绝覆盖。', '', '```bash',
        '.venv/bin/python -m analysis.temporal_train --run analysis/temporal_results/<new-run> --reference-run analysis/temporal_results/v2_frozen_reference',
        '.venv/bin/python -m analysis.temporal_evaluate --run analysis/temporal_results/<new-run>',
        '.venv/bin/python -m analysis.temporal_training_report --run analysis/temporal_results/<new-run>', '```', '']
    (output / 'README.md').write_text('\n'.join(lines))
    shutil.copyfile(__file__, output / 'report_source.py')
    write_json(output / 'manifest.json', {'evaluation_manifest_sha256': sha256(run / 'evaluation_manifest.json'),
        'blind_manifest_sha256': sha256(run / 'blind_manifest.json'),
        'artifacts': {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return output / 'README.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(report(parser.parse_args().run))
