"""Independent post-seal V2 comparisons and reports; never fits or selects."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256, write_json
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_evaluate import aligned_frame, verify_predictions
from analysis.innovation_common import local_sources


def ordered_errors(cases, labels, predictions):
    frame = aligned_frame(cases, labels, predictions)
    order = cases[['event', 'start_ms']].drop_duplicates().sort_values(['start_ms', 'event']).event
    if order.duplicated().any():
        raise ValueError('One actual start time per frozen activity is required')
    frame['ape'] = abs(frame.predicted/frame.actual-1)*100
    return frame.groupby('event').ape.mean().reindex(order)


def paired_intervals(delta, *, draws=5000):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Finite chronological paired activity errors required')
    n = len(delta)
    intervals = {}
    for length in [1, 3, 5]:
        rng = np.random.default_rng(42)
        starts = rng.integers(0, n, (draws, int(np.ceil(n/length))))
        indices = ((starts[:, :, None]+np.arange(length)) % n).reshape(draws, -1)[:, :n]
        means = delta[indices].mean(axis=1)
        intervals['iid' if length == 1 else f'block_{length}'] = np.quantile(means, [.025, .975]).tolist()
    return {'events': n, 'mean_difference': float(delta.mean()),
            'improved': int((delta < 0).sum()), 'worsened': int((delta > 0).sum()),
            'equal': int((delta == 0).sum()), 'intervals': intervals}


def _verify(path):
    receipt = json.loads(path.read_text())
    for name, digest in receipt['artifacts'].items():
        if sha256(path.parent/name) != digest:
            raise ValueError('Sealed result changed: '+str(path.parent/name))
    return receipt


def comparison_pairs(selection, recipes):
    pairs = {'selected_vs_old': [selection['selected_recipe'], 'old_selected'],
             'validation_best_new_vs_old': [selection['best_new'], 'old_selected']}
    by_name = {spec['config']['name']: name for name, spec in recipes.items()}
    for name, spec in recipes.items():
        config = spec['config']; family = spec['family']
        if family == 'adaptation':
            control = by_name['adaptation_context_control']
        elif family == 'growth':
            control = by_name['next_growth_'+config['head']] if config['auxiliary_lambda'] else by_name['next_growth_direct']
        else:
            control = 'old_selected'
        if name != control:
            pairs[name+'_vs_control'] = [name, control]
    return pairs


def evaluate_comparison(run, output=None):
    """Only this independent final scorer reads test labels, after both seals."""
    run = Path(run); seal = verify_predictions(run)
    _verify(run/'evaluation_manifest.json')
    cfg = json.loads((run/'run_manifest.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    sources = local_sources([Path(__file__)])
    ident = {'blind_manifest_sha256': sha256(run/'blind_manifest.json'),
             'evaluation_manifest_sha256': sha256(run/'evaluation_manifest.json'),
             'source_hashes': {n: sha256(ROOT/n) for n in sources},
             'software': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas']},
             'pairs': comparison_pairs(selection, cfg['recipes']),
             'bootstrap': {'seed': 42, 'draws': 5000, 'circular_block_lengths': [1, 3, 5]}}
    evaluation_id = identity(ident); output = Path(output) if output is not None else run/'comparison'
    if (output/'evaluation_manifest.json').exists():
        saved = _verify(output/'evaluation_manifest.json')
        if saved['evaluation_id'] != evaluation_id:
            raise FileExistsError('Different comparison scorer; preserve the existing audit')
        return json.loads((output/'comparison.json').read_text())
    if output.exists():
        raise FileExistsError('Comparison directory already exists without a complete seal')
    data = TemporalDataset(seal['dataset_directory'])
    if data.manifest['dataset_id'] != seal['dataset_id']:
        raise ValueError('Dataset changed')
    cases = data.case_index('test')
    labels = data.labels('test', allow_test=True)
    frame = pd.read_csv(run/'test_predictions.csv')
    errors = {n: ordered_errors(cases, labels, g) for n, g in frame.groupby('method')}
    scores = json.loads((run/'evaluation.json').read_text())['splits']['test']
    differences = [abs(v.mean()-scores[n]['overall']['event_macro_mape_pct']) for n, v in errors.items()]
    if max(differences) > 1e-9:
        raise ValueError('Independent event-macro MAPE did not reconcile')
    result = {'evaluation_id': evaluation_id,
              'independent_metric_max_difference': float(max(differences)),
              'pairs': {key: {'method': a, 'reference': b, **paired_intervals(errors[a]-errors[b])}
                        for key, (a, b) in ident['pairs'].items()},
              'limits': 'Paired old-cohort diagnostics; circular blocks3/5 sensitivity, '
                        'not a prospective guarantee and not multiple-comparison adjusted.'}
    output.mkdir(parents=True)
    write_json(output/'comparison.json', result)
    pd.DataFrame(errors).rename_axis('event').to_csv(output/'event_errors.csv')
    for name in sources:
        path = output/'sources'/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, path)
    write_json(output/'evaluation_manifest.json', {'evaluation_id': evaluation_id, 'identity': ident,
        'artifacts': {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}})
    return result


def _number(value):
    return '—' if value is None or not np.isfinite(value) else f'{value:.3f}'


def report(run, output=None, *, comparison=None):
    run = Path(run); verify_predictions(run)
    comparison = Path(comparison) if comparison is not None else run/'comparison'
    _verify(run/'evaluation_manifest.json'); _verify(comparison/'evaluation_manifest.json')
    result = json.loads((run/'evaluation.json').read_text())
    comparisons = json.loads((comparison/'comparison.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    cfg = json.loads((run/'run_manifest.json').read_text())
    output = Path(output) if output is not None else run/'report'
    inputs = {str(p.resolve()): sha256(p) for p in [run/'evaluation_manifest.json',
        comparison/'evaluation_manifest.json', run/'selection.json', run/'run_manifest.json']}
    renderer = sha256(Path(__file__))
    if (output/'report_manifest.json').exists():
        receipt = _verify(output/'report_manifest.json')
        if receipt['inputs'] != inputs or receipt['renderer_sha256'] != renderer:
            raise FileExistsError('Different renderer or inputs; preserve the report')
        return output/'README.md'
    if output.exists():
        raise FileExistsError(output)
    selected = selection['selected_recipe']; scores = result['splits']
    counts = {s: scores[s][selected]['overall'] for s in ['train', 'validation', 'test']}
    main = scores['test'][selected]; old = scores['test']['old_selected']
    delta = main['overall']['event_macro_mape_pct']-old['overall']['event_macro_mape_pct']
    status = ('验证保留既有模型；没有选出替代方案。' if selected == 'old_selected' else
              '新方案在本测试队列误差更低；稳定性仍需结合区间和前瞻验证。' if delta < 0 else
              '验证选择的新方案未在本测试队列超过既有模型。')
    lines = ['# V2 两档后续全面实验', '', status, '',
        f"验证选择 `{selected}`。训练{counts['train']['cases']}/验证{counts['validation']['cases']}/测试{counts['test']['cases']}个冻结案例；训练实际{counts['train']['events']}活动，验证{counts['validation']['events']}、测试{counts['test']['events']}活动。",
        '所有指标为活动等权 MAPE（%，越低越好）。训练拟合不代表泛化；TabPFN训练拟合含上下文自匹配，不用于校正。', '',
        '## 固定候选与消融', '', '| 方法 | 训练拟合 | 验证 | 测试 |', '|---|---:|---:|---:|']
    for name in selection['candidates']:
        values = [_number(scores[s][name]['overall']['event_macro_mape_pct']) for s in ['train', 'validation', 'test']]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', '## 逐种子', '', '神经候选按种子42/123/2026的最终分数平均参与选择。基础模型种子控制推理/上下文，零样本确定性模型的三个推理种子输出相同，不是三次独立预训练。', '',
              '| 方法 | 验证 | 测试 |', '|---|---:|---:|']
    for name in selection['reported_methods']:
        if '__seed_' in name:
            lines.append(f"| {name} | {_number(scores['validation'][name]['overall']['event_macro_mape_pct'])} | {_number(scores['test'][name]['overall']['event_macro_mape_pct'])} |")
    lines += ['', '## 基础模型执行状态', '']
    availability = cfg.get('family_provenance', {}).get('foundation', {}).get('availability', {})
    for name, status_row in availability.items():
        lines.append(f"- {name}：{'本地权重/依赖校验通过' if status_row['available'] else '未运行'}；"+
                     ('无缺失依赖或权重' if not status_row['reasons'] else '; '.join(status_row['reasons'])))
    if not availability:
        lines.append('本运行未登记基础模型（合成夹具或限定家族运行）。')
    lines += ['', '## 全部时点与排名', '',
              '| 时点 | 本轮选择 | 既有模型 | T1000 | T2000 | 活动/案例 |',
              '|---|---:|---:|---:|---:|---:|']
    ranks = {(row['stage_id'], row['rank']): row for row in main['stage_rank']}
    for stage, metric in main['by_stage'].items():
        rank_scores = [_number(ranks.get((stage, rank), {}).get('event_macro_mape_pct')) for rank in [1000, 2000]]
        lines.append(f"| {stage} | {_number(metric['event_macro_mape_pct'])} | {_number(old['by_stage'][stage]['event_macro_mape_pct'])} | {' | '.join(rank_scores)} | {metric['events']}/{metric['cases']} |")
    lines += ['', '## 配对不确定性', '',
              '| 对照（前者减后者） | 差值/百分点 | 活动IID 95% | 连续3活动块95% | 连续5活动块95% | 改善/变差 |',
              '|---|---:|---|---|---|---|']
    for name, row in comparisons['pairs'].items():
        ci = [f"[{_number(a)}, {_number(b)}]" for a, b in row['intervals'].values()]
        lines.append(f"| {row['method']} − {row['reference']} | {_number(row['mean_difference'])} | {' | '.join(ci)} | {row['improved']}/{row['worsened']} |")
    lines += ['', '区间基于冻结案例的活动归属及实际开始时间。连续块是时间相关敏感性检查，未作多重比较校正，不能证明未来收益。', '',
              '## 完整性与解释边界', '',
              '- 全部候选在读取测试答案前登记并封存；仅验证选择，不回灌验证或测试终分、不重新拟合。',
              '- 历史校正是训练内OOF残差的冻结校正，不能称为滚动在线校正；近期统计不改变监督活动等权。',
              '- 中间时距答案仅来自训练完整曲线，稀疏缺口不补造；缺少辅助目标的活动仍保留终分训练。',
              '- 基础模型使用公开预训练权重，本地离线推理；上下文均衡重复行不增加独立活动数量。权重与全部依赖指纹见run_manifest.json。',
              '- 内部三折见cv/metrics.csv；零样本模型不做任务训练，未执行训练内部CV。跨时间折缺失和失败不删正式案例。',
              '- MAE、RMSE、P90、类型/乐队、观测滞后、15分钟新鲜面板及共同活动面板保留于evaluation.json和stage_rank_metrics.csv。',
              '- 固定测试队列曾参与早期探索，不能称完全未见盲测；元数据为缺少历史时点存档的回溯快照；固定验证也经历多轮开发。',
              f"- 独立重算全部测试主指标的最大差异：{comparisons['independent_metric_max_difference']:.3g}。", '',
              '## 复现', '', '```bash',
              'DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib .venv/bin/python -m analysis.v2_next_runner --run analysis/temporal_results/<new-run>',
              '.venv/bin/python -m analysis.temporal_evaluate --run analysis/temporal_results/<new-run>',
              '.venv/bin/python -m analysis.v2_next_report --run analysis/temporal_results/<new-run>', '```', '',
              '同代码、配置、依赖和权重才能使用--resume；已有封存结果不得覆盖。']
    output.mkdir(parents=True); (output/'README.md').write_text('\n'.join(lines)+'\n')
    rows = [{'method': n, **{s: scores[s][n]['overall']['event_macro_mape_pct'] for s in ['train', 'validation', 'test']}}
            for n in selection['reported_methods']]
    pd.DataFrame(rows).to_csv(output/'all_methods.csv', index=False)
    shutil.copyfile(Path(__file__), output/'renderer_source.py')
    write_json(output/'report_manifest.json', {'inputs': inputs, 'renderer_sha256': renderer,
        'artifacts': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    return output/'README.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Separate audit root for a changed scorer/renderer')
    args = parser.parse_args()
    comparison = args.output/'comparison' if args.output is not None else None
    destination = args.output/'report' if args.output is not None else None
    evaluate_comparison(args.run, comparison)
    print(report(args.run, destination, comparison=comparison))
