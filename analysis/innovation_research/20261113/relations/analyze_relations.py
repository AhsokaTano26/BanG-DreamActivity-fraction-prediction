"""Training-only, event-unit diagnostics for frozen temporal V2 two-rank data.

Run from the repository root:
  .venv/bin/python analysis/innovation_research/20261113/relations/analyze_relations.py

Only TemporalDataset.train_curves(), inputs('train'), and memberships('train')
are parsed. The constructor hashes all frozen artifacts but does not parse
validation/test labels. These are descriptive diagnostics, not model scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.backtest import TZ, snapshot
from analysis.holdout_dataset import prefix_curve
from analysis.temporal_dataset import TemporalDataset

DATASET_ID = '1f76f39af8151bba185496411bbcee064b16a4c5aca5c515c1d957a5056fe854'
HOUR = 3600000
STAGES = ('progress_50pct', 'progress_90pct', 'remaining_6h')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def year_of(curve):
    return datetime.fromtimestamp(curve.start / 1000, TZ).year


def distribution(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                q10=float(np.quantile(a, .1)), q25=float(np.quantile(a, .25)),
                q75=float(np.quantile(a, .75)), q90=float(np.quantile(a, .9)),
                minimum=float(a.min()), maximum=float(a.max()))


def correlation(frame, x, y):
    """Rows must represent separate activities, with one value per activity."""
    data = frame[['event', x, y]].dropna()
    if data.event.duplicated().any():
        raise ValueError('Correlation cannot count repeated snapshots as activities')
    if len(data) < 4:
        return {'n_events': len(data)}
    a, b = data[x].to_numpy(), data[y].to_numpy()
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(a), len(a))
        if np.std(a[idx]) > 0 and np.std(b[idx]) > 0:
            boot.append(float(np.corrcoef(a[idx], b[idx])[0, 1]))
    return {'n_events': len(data), 'pearson': float(np.corrcoef(a, b)[0, 1]),
            'spearman': float(spearmanr(a, b).statistic),
            'pearson_event_bootstrap_95ci': [float(v) for v in np.quantile(boot, [.025, .975])]}


def grouped_distributions(frame, keys, value):
    out = []
    for names, g in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(names, tuple):
            names = (names,)
        # General pooled summaries give each event equal weight across its ranks.
        ev = g.groupby('event')[value].mean()
        out.append(dict(zip(keys, names)) | {'n_events': len(ev), 'n_rows': len(g),
            'event_equal_mean': float(ev.mean())} | distribution(g[value]))
    return pd.DataFrame(out)


def boundary(curve, when):
    """Descriptive target ONLY: interpolate complete training curve locally.

    Refuse extrapolation and require each bracketing observation within 60 min.
    No value from this routine is used to form a blind prefix feature.
    """
    right = int(np.searchsorted(curve.times, when, side='left'))
    if right < len(curve.times) and curve.times[right] == when:
        return float(curve.scores[right]), 0., 0.
    if right == 0 or right == len(curve.times):
        return None, None, None
    before = (when - curve.times[right - 1]) / 60000
    after = (curve.times[right] - when) / 60000
    if before > 60 or after > 60:
        return None, float(before), float(after)
    return float(np.interp(when, curve.times, curve.scores)), float(before), float(after)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT / 'data/temporal_holdout_v2_t1000_t2000')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    dataset = TemporalDataset(args.dataset)  # Full frozen artifact hash verification.
    if dataset.manifest['dataset_id'] != DATASET_ID:
        raise ValueError('Unexpected dataset; this analysis is fixed to V2 two ranks')
    curves = list(dataset.train_curves())
    samples = list(dataset.inputs('train'))
    memberships = dataset.memberships('train')
    curve_map = {(c.event, c.rank): c for c in curves}
    sample_map = {(s['curve'].event, s['curve'].rank, s['cutoff']): s for s in samples}
    by_id = {s['case_id']: s for s in samples}
    train_ids = set(dataset.manifest['splits']['train']['event_ids'])
    assert all(c.event in train_ids for c in curves)
    assert all(s['curve'].event in train_ids and s['times'][-1] <= s['cutoff'] for s in samples)

    curve_rows = []
    for c in curves:
        curve_rows.append({'event': c.event, 'rank': c.rank, 'event_type': c.kind,
            'year': year_of(c), 'duration_h': c.duration/HOUR, 'observations': len(c.times),
            'first_progress': float((c.times[0]-c.start)/c.duration),
            'median_gap_minutes': float(np.median(np.diff(c.times))/60000),
            'max_gap_hours': float(np.max(np.diff(c.times))/HOUR), 'final_score': c.final,
            'has_other_rank_curve': int((c.event, 3000-c.rank) in curve_map)})
    curve_frame = pd.DataFrame(curve_rows)
    curve_frame.to_csv(out/'train_curve_coverage.csv', index=False)
    curve_frame.groupby('year').agg(n_events=('event','nunique'), n_curves=('event','size'),
        median_observations=('observations','median'),
        median_gap_minutes=('median_gap_minutes','median'),
        median_max_gap_hours=('max_gap_hours','median')).to_csv(out/'train_curve_sampling_by_year.csv')

    final_rows = []
    for c in curves:
        peer = curve_map.get((c.event, 2000))
        if c.rank == 1000 and peer is not None:
            assert c.start == peer.start and c.end == peer.end
            final_rows.append({'event': c.event, 'event_type': c.kind, 'year': year_of(c),
                'duration_h': c.duration/HOUR, 'final_1000': c.final, 'final_2000': peer.final,
                'final_ratio': c.final/peer.final, 'log_final_ratio': np.log(c.final/peer.final)})
    finals = pd.DataFrame(final_rows)
    finals.to_csv(out/'final_rank_ratios.csv', index=False)
    grouped_distributions(finals, ['event_type'], 'final_ratio').to_csv(out/'final_ratio_by_type.csv', index=False)
    grouped_distributions(finals, ['year'], 'final_ratio').to_csv(out/'final_ratio_by_year.csv', index=False)

    # Same availability semantics as prepare_paired(), without creating model features.
    audit = []
    for s in samples:
        c = s['curve']
        other = sample_map.get((c.event, 3000-c.rank, s['cutoff']))
        paired = snapshot(prefix_curve(other['curve'], s['now']), s['now']) if other else None
        if paired is not None:
            reason = 'available'
        elif other is not None:
            reason = 'invalid_after_truncation'
        elif (c.event, 3000-c.rank) not in curve_map:
            reason = 'no_same_origin_missing_rank_curve'
        else:
            reason = 'no_same_origin_rank_curve_exists'
        audit.append({'case_id': s['case_id'], 'event': c.event, 'rank': c.rank,
            'event_type': c.kind, 'year': year_of(c), 'reason': reason,
            'available': int(paired is not None),
            'observation_age_minutes': (s['cutoff']-s['now'])/60000})
    audit = pd.DataFrame(audit)
    audit.to_csv(out/'peer_availability_cases.csv', index=False)

    def coverage_by(frame, keys):
        result = []
        for names, g in frame.groupby(keys, sort=True):
            if not isinstance(names, tuple): names = (names,)
            counts = Counter(g.reason)
            result.append(dict(zip(keys, names)) | {'n_cases': len(g),
                'n_events': g.event.nunique(), 'available_cases': int(g.available.sum()),
                'case_availability_pct': 100*float(g.available.mean()),
                'event_equal_availability_pct': 100*float(g.groupby('event').available.mean().mean()),
                **{reason: counts.get(reason, 0) for reason in (
                    'no_same_origin_missing_rank_curve', 'no_same_origin_rank_curve_exists',
                    'invalid_after_truncation')}})
        return pd.DataFrame(result)

    coverage_by(audit, ['year']).to_csv(out/'peer_availability_by_year.csv', index=False)
    coverage_by(audit, ['event_type']).to_csv(out/'peer_availability_by_type.csv', index=False)
    coverage_by(audit.merge(memberships[['case_id','stage_id']], on='case_id'), ['stage_id']).to_csv(
        out/'peer_availability_by_stage.csv', index=False)

    pair_rows, pair_coverage = [], []
    for stage in STAGES:
        selected = [by_id[k] for k in memberships.loc[memberships.stage_id == stage, 'case_id']]
        stage_map = {(s['curve'].event, s['curve'].rank): s for s in selected}
        count = Counter(n_cases=len(selected), n_events=len({s['curve'].event for s in selected}))
        for (event, rank), left in sorted(stage_map.items()):
            right = stage_map.get((event, 2000))
            if rank != 1000 or right is None: continue
            count['same_origin_pairs'] += 1
            # Scores are causal last-observation-carried-forward at the earlier
            # observed time. Neither branch can see a sample after that time.
            sync = min(left['now'], right['now'])
            a = snapshot(prefix_curve(left['curve'], sync), sync)
            b = snapshot(prefix_curve(right['curve'], sync), sync)
            if a is None or b is None:
                count['invalid_after_synchronization'] += 1
                continue
            assert a['now'] <= sync and b['now'] <= sync
            ca, cb = curve_map[event,1000], curve_map[event,2000]
            ratio = a['score']/b['score']
            final_ratio = ca.final/cb.final
            pair_rows.append({'stage_id': stage, 'event': event, 'event_type': ca.kind,
                'year': year_of(ca), 'duration_h': ca.duration/HOUR,
                'requested_cutoff': left['cutoff'], 'sync_at': sync,
                'sync_age_minutes': (left['cutoff']-sync)/60000,
                'rank_observation_gap_minutes': abs(a['now']-b['now'])/60000,
                'prefix_ratio': ratio, 'final_ratio': final_ratio,
                'log_prefix_ratio': np.log(ratio), 'log_final_ratio': np.log(final_ratio),
                'ratio_relative_change_pct': 100*(final_ratio/ratio-1),
                'growth_log_1000': np.log(ca.final/a['score']),
                'growth_log_2000': np.log(cb.final/b['score'])})
            count['usable_synchronized_pairs'] += 1
        pair_coverage.append({'stage_id': stage, **count})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out/'synchronized_stage_pairs.csv', index=False)
    pd.DataFrame(pair_coverage).to_csv(out/'synchronized_pair_coverage.csv', index=False)
    pair_stats = {}
    for stage, g in pairs.groupby('stage_id', sort=True):
        stats = {'prefix_final_log_ratio': correlation(g, 'log_prefix_ratio', 'log_final_ratio'),
            'remaining_log_growth_crossrank': correlation(g, 'growth_log_1000', 'growth_log_2000'),
            'ratio_relative_change_pct': distribution(g.ratio_relative_change_pct),
            'median_absolute_ratio_change_pct': float(g.ratio_relative_change_pct.abs().median()),
            'rank_observation_gap_minutes': distribution(g.rank_observation_gap_minutes),
            'year_counts': {str(k):int(v) for k,v in g.year.value_counts().sort_index().items()},
            'type_counts': {str(k):int(v) for k,v in g.event_type.value_counts().sort_index().items()}}
        # Controls are descriptive and unstable at small n, not fitted predictors.
        for category in ['year', 'event_type']:
            usable = g[g.groupby(category).event.transform('count') >= 3].copy()
            for col in ['log_prefix_ratio', 'log_final_ratio']:
                usable[col+'_centered'] = usable[col] - usable.groupby(category)[col].transform('mean')
            stats['within_'+category+'_centered_log_ratio'] = correlation(
                usable, 'log_prefix_ratio_centered', 'log_final_ratio_centered')
            # Group means are estimated in this diagnostic sample. Report only
            # the descriptive centered correlations, not a naive bootstrap CI.
            stats['within_'+category+'_centered_log_ratio'].pop('pearson_event_bootstrap_95ci', None)
        pair_stats[stage] = stats

    growth_rows = []
    for c in curves:
        for h in [6, 24]:
            cutoff = c.end - h*HOUR
            score, lag, lead = boundary(c, cutoff)
            previous, previous_lag, previous_lead = boundary(c, cutoff-h*HOUR)
            sample = sample_map.get((c.event,c.rank,cutoff))
            share = (c.final-score)/c.final if score is not None else None
            prior_share = (score-previous)/c.final if score is not None and previous is not None else None
            growth_rows.append({'event': c.event, 'rank': c.rank, 'event_type': c.kind,
                'year': year_of(c), 'window_h': h, 'duration_h': c.duration/HOUR,
                'time_fraction_pct': 100*h/(c.duration/HOUR),
                'exact_boundary_available': int(score is not None),
                'boundary_before_minutes': lag, 'boundary_after_minutes': lead,
                'diagnostic_final_window_share_pct': 100*share if share is not None else None,
                'diagnostic_previous_window_share_pct': 100*prior_share if prior_share is not None else None,
                'diagnostic_acceleration_ratio': share/prior_share if prior_share is not None and prior_share > 0 else None,
                'diagnostic_share_vs_uniform_rate': share/(h/(c.duration/HOUR)) if share is not None else None,
                'frozen_prefix_available': int(sample is not None),
                'frozen_observation_age_minutes': (cutoff-sample['now'])/60000 if sample else None,
                'frozen_observed_remaining_share_pct': 100*(c.final-sample['score'])/c.final if sample else None})
    growth = pd.DataFrame(growth_rows)
    growth.to_csv(out/'deadline_growth_target_diagnostics.csv', index=False)
    good = growth[growth.exact_boundary_available == 1]
    grouped_distributions(good, ['window_h','rank'], 'diagnostic_final_window_share_pct').to_csv(
        out/'deadline_growth_by_rank.csv', index=False)
    grouped_distributions(good, ['window_h','rank','event_type'], 'diagnostic_final_window_share_pct').to_csv(
        out/'deadline_growth_by_type.csv', index=False)
    grouped_distributions(good, ['window_h','rank','year'], 'diagnostic_final_window_share_pct').to_csv(
        out/'deadline_growth_by_year.csv', index=False)
    coverage = []
    for (h, rank, year), g in growth.groupby(['window_h','rank','year'], sort=True):
        coverage.append({'window_h':h,'rank':rank,'year':year,'eligible_events':len(g),
            'bracket_covered_events':int(g.exact_boundary_available.sum()),
            'frozen_prefix_events':int(g.frozen_prefix_available.sum()),
            'bracket_coverage_pct':100*float(g.exact_boundary_available.mean())})
    pd.DataFrame(coverage).to_csv(out/'deadline_coverage_by_year.csv', index=False)

    growth_stats = {}
    for (h, rank), g in growth.groupby(['window_h','rank'], sort=True):
        v = g[g.exact_boundary_available == 1]
        acceleration = v.diagnostic_acceleration_ratio.dropna()
        growth_stats[f'{h}h_T{rank}'] = {'eligible_events':len(g),'covered_events':len(v),
            'frozen_prefix_events':int(g.frozen_prefix_available.sum()),
            'final_window_share_pct':distribution(v.diagnostic_final_window_share_pct),
            'share_vs_uniform_rate':distribution(v.diagnostic_share_vs_uniform_rate),
            'acceleration_ratio':distribution(acceleration),
            'fraction_accelerating':float((acceleration>1).mean()) if len(acceleration) else None,
            'frozen_observed_remaining_share_pct':distribution(g.frozen_observed_remaining_share_pct),
            'frozen_age_minutes':distribution(g.frozen_observation_age_minutes)}

    summary = {'dataset_id':DATASET_ID,'training_events':len(train_ids),'training_curves':len(curves),
        'training_input_cases':len(samples),'training_events_with_inputs':audit.event.nunique(),
        'events_with_both_rank_curves':len(finals),'final_ratio':distribution(finals.final_ratio),
        'peer_case_reasons':dict(Counter(audit.reason)),
        'peer_case_availability_pct':100*float(audit.available.mean()),
        'peer_event_equal_availability_pct':100*float(audit.groupby('event').available.mean().mean()),
        'synchronized_stage_statistics':pair_stats,'deadline_growth':growth_stats,
        'limitations':[
            'Training-only descriptive evidence; neither generalization scores nor proof of causal relationships.',
            'Training availability strongly varies with calendar year, rank, stage, and event type; available panels are selected subsets.',
            'Crossrank scores use causal carry-forward at the earlier observation; residual timestamp gaps are recorded.',
            'Exact deadline shares interpolate only full training curves within 60-minute brackets on each side, for target diagnostics only.',
            'Snapshot counts are not independent sample sizes; correlations use one row per event per stage and bootstrap activities.',
            'Validation/test labels and complete curves, SQLite, and existing test results are not parsed.',
            'The fixed test cohort appeared in early exploratory experiments and cannot be described as wholly unseen.',
            'Metadata is a present retrospective snapshot without historical point-in-time archives.'
        ]}
    write_json(out/'summary.json', summary)

    lines = ['# 冻结 V2 两档训练组关系诊断', '',
        f'数据指纹 `{DATASET_ID}`。仅解析训练完整曲线、训练盲前缀和训练时点成员表；构造器校验全部冻结文件哈希。',
        f'训练组 {len(train_ids)} 个活动、{len(curves)} 条排名曲线、{len(samples)} 个冻结前缀；其中 {len(finals)} 个活动有两档完整曲线。', '',
        '## 1. 两档终分与可观测比值', '',
        f'T1000/T2000 终分比中位数 {finals.final_ratio.median():.3f}，P10–P90 为 {finals.final_ratio.quantile(.1):.3f}–{finals.final_ratio.quantile(.9):.3f}（n={len(finals)} 活动）。', '',
        '| 时点 | 同步可用活动 n | 前缀/终分 log 比值 Pearson r（活动 bootstrap 95% CI） | Spearman ρ | 剩余 log 增长跨档 r | 比值变化绝对值中位数 |',
        '|---|---:|---|---:|---:|---:|']
    for stage, stats in pair_stats.items():
        r = stats['prefix_final_log_ratio']; ci = r['pearson_event_bootstrap_95ci']
        lines.append(f"| {stage} | {r['n_events']} | {r['pearson']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {r['spearman']:.3f} | {stats['remaining_log_growth_crossrank']['pearson']:.3f} | {stats['median_absolute_ratio_change_pct']:.2f}% |")
    lines += ['', '每个时点每活动只保留一对。两档先截到较早观测时刻，再用各自最后已知分数；单档截断后仍须满足原快照有效性条件。残余时间差、年份/类型分布和组内中心化相关见 CSV/JSON。小样本相关及其区间不能视为泛化效果。', '',
        '**覆盖限制很强：50% 面板的 12 个活动全来自 2024 年；90% 和剩余 6h 面板也各有 12 个来自 2024 年。** 完整两档队列的终分比中位数从 2019 年的 1.127（n=22）到 2024 年的 1.397（n=14），但年代、活动类型与采样覆盖混杂，不能解释为单一时间趋势。', '',
        '## 2. 最后 6h / 24h 的增长', '',
        '下表使用训练完整曲线对准确边界作描述性插值，两侧观测各距边界不超过 60 分钟；这些完整曲线量仅是诊断目标，不能进入当期输入。加速比为最后窗口增长 / 紧邻的前一同长窗口增长。', '',
        '| 窗口/排名 | 有效活动/合格曲线 | 终分占比中位数 | P10–P90 | 相对匀速倍数中位数 | 加速比中位数（n） |',
        '|---|---:|---:|---:|---:|---:|']
    for key, stats in growth_stats.items():
        share=stats['final_window_share_pct']; acceleration=stats['acceleration_ratio']
        lines.append(f"| {key} | {stats['covered_events']}/{stats['eligible_events']} | {share['median']:.2f}% | {share['q10']:.2f}–{share['q90']:.2f}% | {stats['share_vs_uniform_rate']['median']:.2f} | {acceleration['median']:.2f}（{acceleration['n']}） |")
    lines += ['', '按活动类型、年份和排名的分布见 `deadline_growth_by_type.csv`、`deadline_growth_by_year.csv`；覆盖分母见 `deadline_coverage_by_year.csv`。冻结前缀口径的尾段占比和滞后另存于 JSON/明细，以便区分真实末段增长与采样滞后。', '',
        '## 3. 配对缺失机制', '',
        f"生产配对逻辑可用 {int(audit.available.sum())}/{len(audit)} 个训练案例（{100*audit.available.mean():.2f}%）；在 {audit.event.nunique()} 个至少有一个输入的活动中，按每活动等权后的可用率为 {100*audit.groupby('event').available.mean().mean():.2f}%。",
        f"无同源时点对档案例：对档曲线本身缺失 {int((audit.reason=='no_same_origin_missing_rank_curve').sum())}；对档曲线存在但该时点无合格前缀 {int((audit.reason=='no_same_origin_rank_curve_exists').sum())}；存在对档时点案例但截至自身 now 后失效 {int((audit.reason=='invalid_after_truncation').sum())}。", '',
        '2024 年配对可用 442/464（95.26%，15 个有输入活动），2019–2023 年各年仅 14.94%–35.29%。采样密度也不同：2024 年每曲线观测数中位数 330、采样间隔中位数 30 分钟；2019–2023 年分别仅 28–35 次和 123–218 分钟。这说明缺失主要与历史采样/时点覆盖有关，不能仅解释为另一档完全没有曲线。', '',
        '按年、类型、阶段详见 `peer_availability_by_*.csv`，采样密度见 `train_curve_sampling_by_year.csv`。阶段表包含多轴成员引用，只能在单阶段内看比例，不能加总为独立案例。', '',
        '## 优先验证的机制', '',
        '1. **共享热度 + 排名差距动态。** 把两档终分建模为一个共同规模与一个有约束的相对比值，利用截至同步时刻的比值和变化率；缺失对档时切换到单档状态并显式记录采样年龄。强相关只是结构依据，必须在冻结验证队列验证增益。',
        '2. **距结束时间驱动的末段增长。** 用可解释的倒计时增长项/分段累积增量来刻画冲线，把活动类型、年代和已观测增长速度作为条件；训练目标可由训练曲线构造，但任何当期输入仅由盲前缀计算。', '',
        '## 限制', '',
        '这是训练组描述性研究，没有读取验证/测试答案、SQLite 或既有测试评分，也没有训练/选择新模型。较早年份和中段前缀稀疏，样本覆盖、活动类型和年代互相混杂；不能从可用面板外推全体活动，也不能把活动内快照当成独立样本。',
        '固定测试队列曾参与早期探索，不能称为完全未见过的盲测；元数据是当前回溯快照，缺少历史时点存档。', '',
        '重跑：`.venv/bin/python analysis/innovation_research/20261113/relations/analyze_relations.py`。脚本固定数据集指纹、bootstrap 种子 42、2000 次按活动重采样。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    manifest = {'dataset_id':DATASET_ID,'parsed_splits':['train'],
        'parsed_dataset_artifacts':['train_curves.jsonl.gz','train_inputs.jsonl.gz','train_memberships.csv'],
        'bootstrap_seed':42,'bootstrap_replicates':2000,
        'python':platform.python_version(),
        'software':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy']},
        'code_sha256':digest(__file__),
        'imported_source_sha256':{name:digest(ROOT/name) for name in [
            'analysis/temporal_dataset.py','analysis/multirank_dataset.py',
            'analysis/holdout_dataset.py','analysis/backtest.py']},
        'artifacts':{f.name:digest(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!='analysis_manifest.json'}}
    write_json(out/'analysis_manifest.json', manifest)
    print(json.dumps({'dataset_id':DATASET_ID,'output':str(out),'events':len(train_ids),
        'paired_final_events':len(finals),'peer_reasons':summary['peer_case_reasons'],
        'stage_pair_counts':{k:v['prefix_final_log_ratio']['n_events'] for k,v in pair_stats.items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()
