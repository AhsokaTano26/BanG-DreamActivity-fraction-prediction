"""In-sample diagnostics from sealed checkpoints; no retraining or model selection."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import importlib.util
import json
import statistics
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
from analysis import structured_neural_benchmark as nn
from analysis.relationship_benchmark import decode, rows
from analysis.relationship_features import relationship_frame, select_features
from analysis.holdout_dataset import HoldoutDataset, sha256, write_json
from analysis.holdout_evaluate import case_metadata, score_predictions
from analysis.event_features import EventFeatures
import torch


def main():
    output = Path(__file__).resolve().parent
    if (output/'manifest.json').exists(): raise FileExistsError('Preserve the existing audit')
    run = ROOT/'analysis/holdout_results/structured_neural_v1'
    replay_path = ROOT/'analysis/live_predictions/20261111T112141Z/calculate.py'
    spec = importlib.util.spec_from_file_location('archived_forecast_replay', replay_path)
    replay = importlib.util.module_from_spec(spec); spec.loader.exec_module(replay)
    torch.set_num_threads(4); data = HoldoutDataset(); models = replay.verify(data)
    config = json.loads((run/'run_manifest.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    receipt = json.loads((run/'evaluation_manifest.json').read_text())
    for name, digest in receipt['artifacts'].items(): assert sha256(run/name) == digest
    evaluation = json.loads((run/'evaluation.json').read_text())
    original_labels = data.labels
    def only_training(split, **kwargs):
        if split != 'train': raise AssertionError('No validation/test label reads in this diagnostic')
        return original_labels(split, **kwargs)
    data.labels = only_training
    samples = data.inputs('train'); labels = data.labels('train')
    metadata = EventFeatures.load(data.metadata_directory)
    print('Replaying reference ensemble on 4377 frozen training prefixes.', flush=True)
    previous = replay.predict(samples, metadata, data)
    values = {'reference': previous['relationships_v3']}
    frame = select_features(relationship_frame(samples, metadata), 'invariant')
    inputs = nn.tensor_arrays(nn.arrays(samples, frame, joblib.load(run/'preprocessor.joblib')), config['device'])
    for name in config['recipes']:
        for seed in config['seeds']:
            model = nn.load_model(run/'models'/f'{name}_{seed}.pt', config)
            values[f'{name}_seed{seed}'] = decode(samples, nn.infer(model, inputs), 'remaining')
        print('Replayed '+name, flush=True)
    nn.ensembles(values, config, selection['top3_members'])
    assert set(values) == set(selection['reported_methods'])
    assert np.allclose(values[selection['selected_recipe']], previous['structured_neural_v1'], rtol=1e-7, atol=.01)
    for name in selection['top3_members']:
        assert np.allclose(values[name], previous[name], rtol=1e-7, atol=.01)
    cases = case_metadata(samples, metadata); scores = {}; summary = []; predictions = []; differences = []
    for name, prediction in values.items():
        records = rows(samples, name, prediction)
        scores[name] = score_predictions(cases, labels, records)
        # Independent event-macro calculation, separate from the scorer's pandas aggregation.
        grouped = defaultdict(list)
        for sample, value in zip(samples, prediction):
            grouped[sample['curve'].event].append(abs(float(value)/labels[sample['case_id']]-1)*100)
        independent = statistics.mean(statistics.mean(group) for group in grouped.values())
        differences.append(abs(independent-scores[name]['overall']['event_macro_mape_pct']))
        assert differences[-1] < 1e-10
        summary.append({'method': name,
                        'train_event_macro_mape_pct': scores[name]['overall']['event_macro_mape_pct'],
                        'validation_event_macro_mape_pct': selection['validation_scores'][name],
                        'test_event_macro_mape_pct': evaluation['test'][name]['overall']['event_macro_mape_pct'],
                        **{'train_'+key: value for key, value in scores[name]['overall'].items() if key != 'event_macro_mape_pct'}})
        predictions.extend(row | {'actual': labels[row['case_id']]} for row in records)
    pd.DataFrame(summary).to_csv(output/'train_validation_test.csv', index=False)
    pd.DataFrame(predictions).to_csv(output/'training_predictions.csv.gz', index=False, compression='gzip')
    write_json(output/'training_metrics.json', scores)
    names = {'shared_mlp_ensemble3': '共享 MLP', 'rank_mlp_ensemble3': '分排名头 MLP',
             'dual_gru_ensemble3': '双档 GRU', 'dual_tcn_ensemble3': '双档 TCN',
             'multiscale_tcn_ensemble3': '多时间尺度 TCN', 'cross_patch_ensemble3': '跨档 Patch Attention',
             'gated_tcn_ensemble3': '门控专家 TCN', 'dual_tcn_metric_ensemble3': '相对终分损失 TCN',
             'dual_tcn_thin_ensemble3': '稀疏增强 TCN', 'neural_top3': '三个神经配置等权组合',
             'reference': '上一轮组合', 'blend_neural_0.25': '当前选定混合方案'}
    lines = ['# 已封存模型的训练集拟合表现', '',
             '使用训练 198 个活动/4377 个前缀、验证 42 个活动/414 个前缀、测试 43 个活动/412 个前缀。', '',
             '主指标：event-macro MAPE，先在每个活动内平均各案例绝对百分比误差，再让活动等权。单位 %，越低越好。', '',
             '训练列为验证集选定检查点在原始训练前缀上的样本内拟合误差：推理模式关闭 dropout，不做稀疏增强。没有重新训练、重新拟合预处理、重新选择模型或读取验证/测试答案；后两列取已封存评分。神经配置为三种子最终分数的算术平均，不是三次 MAPE 的平均。', '',
             '| 模型 | 训练拟合 MAPE % | 验证 MAPE % | 测试 MAPE % |', '|---|---:|---:|---:|']
    lookup = {row['method']: row for row in summary}
    for method, label in names.items():
        r = lookup[method]
        lines.append(f'| {label} | {r["train_event_macro_mape_pct"]:.3f} | {r["validation_event_macro_mape_pct"]:.3f} | {r["test_event_macro_mape_pct"]:.3f} |')
    lines += ['', '**训练拟合误差不能解释成对新活动的预测能力。** 三个集合年代、活动构成、采样密度不同，训练截断点约为 10%–98% 的多个进度位置，验证/测试是五个固定阶段；跨列差异不能全部归因于过拟合，也不能据训练列重新挑选本轮模型。', '',
              '测试活动曾在早期探索中使用，不是完全未见的盲测；元数据是当前回溯快照，缺少历史时点存档。验证集还存在多轮复用限制。', '',
              '全部 41 组（含 27 个单种子）的主指标见 [train_validation_test.csv](train_validation_test.csv)，训练 MAE/RMSE/P90 与排名/时点/类型/乐队细分见 [training_metrics.json](training_metrics.json)。', '',
              f'独立重算训练主指标最大差异 {max(differences):.2g}；现有三个运行的模型、预处理和推理代码哈希均校验，通过两条组合推理路径核对当前方案。', '']
    (output/'README.md').write_text('\n'.join(lines))
    write_json(output/'manifest.json', {'dataset_id': data.manifest['dataset_id'], 'run_id': config['run_id'],
               'evaluation_id': evaluation['evaluation_id'], 'purpose': 'in-sample fit diagnostics; no selection or retraining',
               'train_events': len({s['curve'].event for s in samples}), 'train_cases': len(samples), 'methods': len(values),
               'independent_metric_max_difference': max(differences), 'models': models,
               'inference_helper_sha256': sha256(replay_path),
               'source_database_sha256': sha256(ROOT/'data/db.sqlite3'),
               'artifacts': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    print('\n'.join(lines), flush=True)


if __name__ == '__main__': main()
