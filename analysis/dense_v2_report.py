"""Summarize sealed dense two-rank V2 retraining; never perform model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.dense_v2 import DATASET_ID, RUNS, STAGES
from analysis.holdout_dataset import sha256, write_json
from analysis.temporal_evaluate import verify_predictions


LIMITATIONS = ('本队列来自曾用于旧探索的活动，验证集也被反复使用；不能称为完全未见盲测。'
               '元数据是回溯快照，缺少历史时点存档。这里只比较相同密集时期队列和相同时间网格，'
               '不把旧队列与新队列分数差称为模型改进。训练拟合误差不作为泛化成绩。')


def coverage(run, stage):
    config = json.loads((run/'run_manifest.json').read_text())
    seal = verify_predictions(run)
    if seal['dataset_id'] != DATASET_ID:
        raise ValueError('Coverage must contain the same dense two-rank cohort')
    if stage == 'base':
        recipes = {name: {'family': 'baseline', 'config': {'iterations': config['iterations']}}
                   for name in ['catboost_rank', 'catboost_context']}
    elif stage == 'paired':
        recipes = {name: {'family': 'paired', 'config': config['tree']} for name in ['catboost_paired', 'catboost_phase']}
        recipes['dual_tcn_mean'] = {'family': 'dual_tcn', 'config': config['neural'] | {'seeds': config['seeds']}}
    else:
        recipes = config['recipes']
    rows = []
    for name, spec in recipes.items():
        recipe = spec['config']
        zero = recipe.get('zero_shot', False)
        neural = 'seeds' in recipe
        kind = ('zero-shot, no task fitting' if zero else
                'training-only in-context conditioning' if recipe.get('backend') == 'tabpfn_v2' else
                'task parameter fitting')
        if stage == 'next':
            path = run/'models'/name/'model.joblib'
            folds = len(list((run/'cv').glob(f'fold_*/{name}/checkpoint.json')))
        elif stage == 'paired' and name == 'dual_tcn_mean':
            path = run/'models/dual_tcn_seed42.pt'; folds = 0
        else:
            path = run/'models'/f'{name}.joblib'
            folds = len(list((run/'cv').glob(f'fold_*/{name}.joblib')))
        if not path.is_file():
            raise ValueError('Missing full-cohort artifact: '+str(path))
        rows.append({'stage': stage, 'family': spec['family'], 'method': name,
                     'fit_kind': kind, 'seeds': ','.join(map(str, recipe.get('seeds', []))),
                     'epochs': recipe.get('epochs', recipe.get('max_epochs')),
                     'iterations': recipe.get('iterations'), 'internal_cv_folds': folds,
                     'model_artifact': str(path), 'model_sha256': sha256(path),
                     'blind_manifest_sha256': sha256(run/'blind_manifest.json'),
                     'configuration': json.dumps(recipe, ensure_ascii=False, sort_keys=True)})
    return rows


def report(output, *, require_evaluated=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = []; runs = []; all_metrics = []; rank_metrics = []; stage_metrics = []
    for stage in STAGES:
        run = RUNS[stage]
        rows.extend(coverage(run, stage))
        seal = verify_predictions(run)
        selection = json.loads((run/'selection.json').read_text())
        selected = selection['selected_recipe']
        item = {'stage': stage, 'run': str(run), 'selected_recipe': selected,
                'reported_methods': len(seal['methods']),
                'blind_manifest_sha256': sha256(run/'blind_manifest.json'),
                'selection_sha256': sha256(run/'selection.json')}
        evaluated = run/'evaluation.json'
        if require_evaluated and not evaluated.exists():
            raise ValueError('Independent scoring required before report')
        if evaluated.exists():
            result = json.loads(evaluated.read_text())
            item['evaluation_id'] = result['evaluation_id']
            for split, methods in result['splits'].items():
                for method, scores in methods.items():
                    tags = {'stage': stage, 'split': split, 'method': method, 'selected': method == selected}
                    all_metrics.append(tags | scores['overall'])
                    rank_metrics.extend(tags | {'rank': rank} | metrics for rank, metrics in scores['by_rank'].items())
                    stage_metrics.extend(tags | row for row in scores['stage_rank'])
            for split in ['train', 'validation', 'test']:
                item[split+'_mape'] = result['splits'][split][selected]['overall']['event_macro_mape_pct']
        runs.append(item)
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output/'recipe_coverage.csv', index=False)
    for name, records in [('all_metrics', all_metrics), ('rank_metrics', rank_metrics), ('stage_rank_metrics', stage_metrics)]:
        if records:
            pd.DataFrame(records).to_csv(output/(name+'.csv'), index=False)
    write_json(output/'runs.json', runs)
    lines = ['# 密集时期 V2 两档全部固定配方重训', '',
             '固定数据：67 训练、14 验证、16 测试活动；2452 / 488 / 560 个前缀。'
             '实际国服开始时间不早于 2024-01-29 13:00（UTC+8）。', '',
             f'完整覆盖 {len(rows)} 个已有配方，其中 2 个时间序列基础模型为零样本推理，'
             '2 个 TabPFN 配方重新建立训练集上下文，其余重新拟合任务参数。'
             '所有神经网络成员均使用 42、123、2026；保留全部逐种子预测。'
             'Chronos2/TimesFM 的三种子是确定性重复推理，不是三个重新训练的模型。', '',
             '研究扩展、创新模型和后续模型的已有组合菜单全部重新由验证集选择。'
             '创新与后续模型保留三折训练期时间 CV（零样本配方不做 CV）；'
             '训练过程只读取训练/验证答案，参考预测全部来自本轮相同队列重训。', '',
             '| 阶段 | 验证选中的方法 | 训练拟合 MAPE | 验证 MAPE | 测试 MAPE |',
             '|---|---|---:|---:|---:|']
    for run in runs:
        values = [f"{run[k+'_mape']:.3f}%" if k+'_mape' in run else '待独立评分' for k in ['train', 'validation', 'test']]
        lines.append(f"| {run['stage']} | {run['selected_recipe']} | {' | '.join(values)} |")
    lines += ['', '主指标均为活动等权 event-macro MAPE，越低越好。表中每阶段方法在测试答案释放前由验证选择；'
              '跨阶段测试最低项仅可作为描述性结果，不能作为新一轮模型选择。', '', LIMITATIONS, '',
              '详细配方与模型指纹见 `recipe_coverage.csv`；独立评分后可见全部方法、排名和时间点明细。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    write_json(output/'report_manifest.json', {'dataset_id': DATASET_ID,
        'renderer_sha256': sha256(Path(__file__)), 'runs': runs,
        'artifacts': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    return output/'README.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-evaluated', action='store_true')
    args = parser.parse_args()
    print(report(args.output, require_evaluated=args.require_evaluated))
