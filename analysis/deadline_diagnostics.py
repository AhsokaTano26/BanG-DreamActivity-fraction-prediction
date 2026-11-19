"""Training-only deadline dynamics; event bootstrap and interpolation sensitivity."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import TZ
from analysis.holdout_dataset import HoldoutDataset, sha256, write_json
from analysis.event_features import EventFeatures
from analysis.relationship_statistics import design, clustered_ols, association_tests


def at_time(curve, time):
    if time < curve.times[0] or time > curve.times[-1]: return np.nan, np.nan
    right = int(np.searchsorted(curve.times, time, side='left'))
    if curve.times[right] == time: return float(curve.scores[right]), 0.
    left = right-1
    return float(np.interp(time, curve.times, curve.scores)), float((curve.times[right]-curve.times[left])/3600000)


def paired_summary(frame, horizon, dense=False):
    f = frame[frame.horizon_h == horizon].copy()
    if dense: f = f[f.interpolation_span_h <= 2]
    pairs = f.pivot(index='event', columns='rank', values=['log_rate_multiplier', 'share_pct', 'fraction_before', 'final'])
    pairs = pairs.dropna()
    if not len(pairs): return {'paired_events': 0}
    difference = (pairs['log_rate_multiplier'][1000]-pairs['log_rate_multiplier'][2000]).to_numpy()
    # Change in log rank gap from the cutoff to the endpoint.
    gap_change = np.log(pairs['fraction_before'][2000]/pairs['fraction_before'][1000]).to_numpy()
    rng = np.random.default_rng(42); draws = rng.integers(0, len(pairs), (5000, len(pairs)))
    return {'paired_events': len(pairs), 'geometric_rate_multiplier_ratio_1000_to_2000': float(np.exp(difference.mean())),
            'ratio_event_bootstrap_95ci': np.exp(np.quantile(difference[draws].mean(1), [.025, .975])).tolist(),
            'median_share_pct_1000': float(pairs['share_pct'][1000].median()),
            'median_share_pct_2000': float(pairs['share_pct'][2000].median()),
            'mean_log_gap_change': float(gap_change.mean()),
            'mean_rank_gap_increase_pct': float(np.expm1(gap_change.mean())*100),
            'gap_increase_pct_event_bootstrap_95ci': (np.expm1(np.quantile(gap_change[draws].mean(1), [.025, .975]))*100).tolist(),
            'gap_widens_events': int((gap_change > 0).sum())}


def run(output):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); data = HoldoutDataset(); curves = data.train_curves(); meta = EventFeatures.load(data.metadata_directory)
    rows = []
    for c in curves:
        for h in [3, 6, 12, 24, 48]:
            value, span = at_time(c, c.end-h*3600000)
            if not np.isfinite(value) or value <= 0 or c.duration/3600000 <= h: continue
            k = max(1e-12, (c.final/value-1)*(c.duration/3600000-h)/h)
            rows.append({'event': c.event, 'rank': c.rank, 'type': c.kind, 'band': meta.band_only(c)['band_group'],
                         'year': str(datetime.fromtimestamp(c.start/1000, TZ).year),
                         'duration_h': c.duration/3600000, 'log_duration': np.log(c.duration/3600000),
                         'horizon_h': h, 'interpolation_span_h': span, 'final': c.final,
                         'fraction_before': value/c.final, 'share_pct': (1-value/c.final)*100,
                         'log_rate_multiplier': np.log(k), 'rate_multiplier': k})
    frame = pd.DataFrame(rows); frame.to_csv(output/'training_window_statistics.csv', index=False)
    summaries = []
    for dense in [False, True]:
        f = frame[frame.interpolation_span_h <= 2] if dense else frame
        for keys, g in f.groupby(['horizon_h', 'rank']):
            summaries.append({'dense_interpolation_only': dense, 'horizon_h': keys[0], 'rank': keys[1],
                              'events': len(g), 'median_final_share_pct': float(g.share_pct.median()),
                              'median_rate_multiplier': float(g.rate_multiplier.median()),
                              'median_interpolation_span_h': float(g.interpolation_span_h.median())})
    pd.DataFrame(summaries).to_csv(output/'window_summary.csv', index=False)
    regressions = {}
    for dense in [False, True]:
        f = frame[frame.horizon_h == 6].copy()
        if dense: f = f[f.interpolation_span_h <= 2].copy()
        f = f.reset_index(drop=True); x = design(f, ['log_duration'], ['rank', 'type', 'year', 'band'])
        weight = 1/f.groupby('event').event.transform('size').to_numpy()
        coef, cov, info = clustered_ols(x, f.log_rate_multiplier, f.event.to_numpy(), weight)
        name = 'dense_6h' if dense else 'all_6h'; coef.to_csv(output/f'{name}_coefficients.csv', index=False)
        regressions[name] = {'fit': info, 'factor_tests': association_tests(coef, cov,
              {'rank': ['rank_'], 'type': ['type_'], 'year': ['year_'], 'band': ['band_'], 'duration': ['log_duration']}, info['events'])}
    type_summary = frame[frame.horizon_h == 6].groupby(['type', 'rank']).agg(events=('event', 'size'),
                        median_share_pct=('share_pct', 'median'), median_rate_multiplier=('rate_multiplier', 'median')).reset_index()
    type_summary.to_csv(output/'type_rank_6h.csv', index=False)
    paired = {str(h): {'all': paired_summary(frame, h), 'span_le_2h': paired_summary(frame, h, True)} for h in [3, 6, 12, 24, 48]}
    result = {'scope': 'training only', 'dataset_id': data.manifest['dataset_id'], 'train_events': len({c.event for c in curves}),
              'train_curves': len(curves), 'paired': paired, 'regressions': regressions,
              'nonpositive_remaining_increments': int((frame.fraction_before >= 1).sum()),
              'duration_h': {'min': min(c.duration/3600000 for c in curves), 'max': max(c.duration/3600000 for c in curves),
                             'median': float(np.median([c.duration/3600000 for c in curves]))},
              'limits': ['observational, not causal', 'interpolation uncertainty', 'dense subset has different eras',
                         'old test exposure', 'retrospective metadata', 'event bootstrap ignores serial event dependence']}
    write_json(output/'diagnostics.json', result)
    (output/'source.py').write_text(Path(__file__).read_text())
    write_json(output/'manifest.json', {'dataset_id': data.manifest['dataset_id'],
               'source_hashes': {str(p): sha256(p) for p in [Path(__file__), Path('analysis/relationship_statistics.py'), Path('analysis/backtest.py')]},
               'artifacts': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'six_hour_pairs': paired['6'], 'regressions': regressions, 'duration_h': result['duration_h']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); run(p.parse_args().output)
