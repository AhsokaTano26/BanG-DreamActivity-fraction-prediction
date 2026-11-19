"""Read sealed results for manuscript tables; never train, tune or rescore models."""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
DATA = OUT / 'data'
SOURCES = {}


def source(name):
    p = ROOT / name
    SOURCES[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def frame(name):
    return pd.read_csv(source(name), float_precision='round_trip')


def load(name):
    return json.loads(source(name).read_text())


def export(name, f):
    f.to_csv(DATA / name, index=False)


def main():
    DATA.mkdir(exist_ok=True)
    facts = {'scope': 'Existing frozen descriptive results only; no new selection, model fit, or label-file access.'}
    datasets = {}
    for key, path in {
        'v1_two': 'data/holdout_v1_dense_20261115',
        'v2_all': 'data/temporal_holdout_v2_dense_20261115',
        'v2_two': 'data/temporal_holdout_v2_dense_t1000_t2000_20261115',
        'v3_all': 'data/rolling_holdout_v3_dense_19rank_20261117',
        'v3_two': 'data/rolling_holdout_v3_dense_t1000_t2000_20261115',
    }.items():
        x = load(path + '/manifest.json')
        datasets[key] = {'dataset_id': x['dataset_id'], 'path': path,
                         'cases': {s: x['splits'][s]['cases'] for s in ['train', 'validation', 'test']},
                         'event_ids': {s: x['splits'][s]['event_ids'] for s in ['train', 'validation', 'test']}}
    ids = datasets['v2_all']['event_ids']
    for x in datasets.values():
        assert x['event_ids'] == ids
    assert [len(ids[s]) for s in ['train', 'validation', 'test']] == [67, 14, 16]
    facts['datasets'] = datasets
    p = frame('data/temporal_holdout_v2_dense_20261115/partitions.csv')
    tz = ZoneInfo('Asia/Shanghai')
    date = lambda n: datetime.fromtimestamp(n/1000, tz).strftime('%Y-%m-%d')
    boundaries = []
    for s in ['train', 'validation', 'test']:
        f = p[p.split == s]
        boundaries.append({'split': s, 'events': f.event.nunique(), 'first_start': date(f.start_ms.min()),
                           'last_start': date(f.start_ms.max()), 'last_end': date(f.end_ms.max())})
    export('cohort_boundaries.csv', pd.DataFrame(boundaries))
    facts['cohort_dates'] = boundaries
    facts['density'] = load('analysis/data_quality/20261115_sampling_density/summary.json')

    base = 'analysis/task_learning_records/20261118/report/'
    f = frame(base + 'comparison.csv')
    export('all19_comparison.csv', f)
    methods = ['uncorrected', 'ewma_reference', 'legacy_pair', 'previous_neural',
               'previous_selected_graph', 'shared_single', 'rank_heads', 'related_static', 'related_phase']
    main = f.set_index('method').loc[methods].reset_index()
    assert (main.cases == 5220).all() and (main.events == 16).all()
    assert abs(main.set_index('method').loc['legacy_pair', 'test_mape_pct'] - 6.540280738457424) < 1e-10
    export('main_models.csv', main)
    facts['main_models'] = main[['method', 'validation_mape_pct', 'test_mape_pct', 'original2_test_mape_pct',
                                'case_mape_pct', 'p90_ape_pct', 'mae_ep', 'rmse_ep']].to_dict('records')
    raw, warm, pair = (float(main.set_index('method').loc[m, 'test_mape_pct'])
                       for m in ['uncorrected', 'ewma_reference', 'legacy_pair'])
    facts['derived_relative_reductions_pct'] = {'warm_vs_raw': 100*(raw-warm)/raw,
                                                'pair_vs_raw': 100*(raw-pair)/raw,
                                                'pair_vs_warm': 100*(warm-pair)/warm}
    facts['task_matched_comparisons'] = load(base + 'matched_comparisons.json')
    facts['multiline_matched_comparisons'] = load('analysis/multiline_records/20261117/report_v2/matched_comparisons.json')
    facts['two_rank_historical_comparisons'] = load('analysis/rolling_results/v3_next_warm_20261116/paired_comparisons.json')

    run = 'analysis/rolling_results/v3_multiline_calibration_20261117/'
    rank = frame(run + 'by_rank.csv')
    keep = rank[(rank.split == 'test') & rank.method.isin(['uncorrected', 'ewma_reference', 'legacy_pair'])]
    table = keep.pivot(index='rank', columns='method', values='event_macro_mape_pct').reset_index()
    support = frame('analysis/task_learning_records/20261118/report/rank_comparison.csv')
    table = support[['rank', 'train_curves', 'validation_curves', 'test_curves', 'test_cases']].merge(table, on='rank')
    assert len(table) == 19 and table.loc[table['rank'] == 1500, 'train_curves'].iloc[0] == 0
    export('rank_results.csv', table)
    stage = frame(run + 'by_stage.csv')
    stage = stage[(stage.split == 'test') & stage.method.isin(['uncorrected', 'ewma_reference', 'legacy_pair'])]
    export('stage_results.csv', stage)
    type_rows = frame(run + 'by_type.csv')
    band_rows = frame(run + 'by_band.csv')
    export('type_results.csv', type_rows[type_rows['split'].eq('test') & type_rows.method.isin(['uncorrected','ewma_reference','legacy_pair'])])
    export('band_results.csv', band_rows[band_rows['split'].eq('test') & band_rows.method.isin(['uncorrected','ewma_reference','legacy_pair'])])

    static = frame('analysis/dense_retrain_records/20261115/report/all_models.csv')
    assert len(static) == 329
    export('supplement_all_329_results.csv', static)
    selected = static[static.selected_within_run.astype(str).str.lower().eq('true')]
    export('supplement_historical_selections.csv', selected)
    v2 = static[static.protocol.eq('V2 two-rank')]
    picks = ['catboost_rank', 'dual_tcn_mean', 'phase_route', 'blend_new_25',
             'retrieval__retrieval_residual', 'growth__growth_dual_clock', 'growth__growth_progress',
             'foundation__foundation_tabpfn_v2', 'foundation__foundation_tabpfn_v2_context',
             'foundation__foundation_chronos2', 'foundation__foundation_timesfm25',
             'growth__next_growth_progress_aux025', 'growth__next_growth_progress_aux100']
    export('static_v2_models.csv', v2[v2.method.isin(picks)])
    facts['static_v2_model_names'] = v2.method.tolist()

    stats = 'analysis/cross_rank_records/20261117/results/'
    export('correlation_summary.csv', frame(stats + 'panel_summary.csv'))
    export('rank_coverage.csv', frame(stats + 'rank_coverage.csv'))
    export('pca_variance.csv', frame(stats + 'pca_variance.csv'))
    for name in ['train_log_final', 'train_remaining_growth_p50', 'train_error_raw_phase0', 'train_error_raw_phase1_lag1']:
        shutil.copyfile(source(stats + f'matrices/{name}.csv'), DATA / f'{name}.csv')
    facts['statistics'] = load(stats + 'results.json')
    # Register explanatory source documents and implementation; never modify them.
    for name in ['AGENTS.md', 'analysis/v3_next_calibration.py', 'analysis/rolling_residual_stats.py',
                 'analysis/task_learning_models.py', 'analysis/task_peer_data.py', 'production/selected_model.py',
                 'production/model.py', 'analysis/rolling_benchmark.py', 'analysis/multirank_model.py',
                 'analysis/dense_retrain_records/20261115/report/README.md',
                 'analysis/v3_next_records/20261116/report/README.md',
                 'analysis/v3_coupled_records/20261117/report/README.md',
                 'analysis/multiline_records/20261117/report_v2/README.md',
                 'analysis/task_learning_records/20261118/report/README.md',
                 'analysis/cross_rank_records/20261117/results/README.md',
                 'production_records/20261118_best/activation_receipt.json',
                 'production_records/20261118_best/method_manifest_v3.json',
                 'production_records/20261118_best/rollout_final_tests_receipt.json',
                 'production_records/20261118_best/parity.json']:
        source(name)
    facts['production'] = load('production_records/20261118_best/activation_receipt.json')
    (DATA / 'facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2, default=str) + '\n')
    (DATA / 'source_manifest.json').write_text(json.dumps(SOURCES, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status':'pass','sources':len(SOURCES),'frozen_results':329,'main_models':len(main),
                      'rank_rows':len(table),'relative_changes':facts['derived_relative_reductions_pct'],
                      'dates':boundaries},ensure_ascii=False))


if __name__ == '__main__':
    main()
