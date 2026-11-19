"""Read-only, chronological cutoff forecasting experiment.

Run: .venv/bin/python analysis/backtest.py
The database is opened in read-only mode. All test-event snapshots are kept
out of training, including the other ranking tier of that same event.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'analysis' / 'results'
STAGES = ['25%', '50%', '75%', '24h remaining', '6h remaining']
METHODS = ['linear', 'recent_24h', 'template', 'same_type_template',
           'shape_analog', 'ridge_growth', 'xgb_growth', 'template_xgb_blend']
CAT = ['event_type', 'rank']
TZ = timezone(timedelta(hours=8))


@dataclass
class Curve:
    event: int
    rank: int
    kind: str
    start: int
    end: int
    times: np.ndarray
    scores: np.ndarray

    @property
    def duration(self):
        return self.end - self.start

    @property
    def final(self):
        return float(self.scores[-1])


def load_curves(database=None, *, now_ms=None):
    database = Path(database) if database is not None else ROOT / 'data/db.sqlite3'
    conn = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    curves, excluded = [], []
    now = int(datetime.now(timezone.utc).timestamp() * 1000) if now_ms is None else now_ms
    for r in conn.execute("SELECT * FROM event WHERE Country='中国大陆' AND EventID>=226 ORDER BY StartAt,Rank"):
        points = ast.literal_eval(r['PointRank'])
        times = np.array([p['time'] for p in points], dtype=np.int64)
        scores = np.array([p['ep'] for p in points], dtype=float)
        reason = None
        if r['EndAt'] >= now:
            reason = 'ongoing'
        elif not len(points) or scores[-1] <= 0:
            reason = 'no usable final score'
        elif r['EndAt'] - times[-1] > 15 * 60000 or times[-1] - r['EndAt'] > 5 * 60000:
            reason = 'final observation not near event end'
        elif len(points) < 20 or np.any(np.diff(times) <= 0):
            reason = 'sparse or invalid timestamps'
        if reason:
            excluded.append({'event': r['EventID'], 'rank': r['Rank'], 'reason': reason})
            continue
        curves.append(Curve(r['EventID'], r['Rank'], r['EventType'], r['StartAt'], r['EndAt'], times, scores))
    conn.close()
    return curves, excluded


def snapshot(curve, cutoff):
    """Feature extraction can see only the prefix ending at cutoff."""
    stop = np.searchsorted(curve.times, cutoff, side='right')
    times, scores = curve.times[:stop], curve.scores[:stop]
    if len(times) < 4 or scores[-1] <= 0 or cutoff - times[-1] > 3600000:
        return None
    now, score = int(times[-1]), float(scores[-1])
    elapsed = (now - curve.start) / 3600000
    if elapsed <= 0:
        return None
    duration = curve.duration / 3600000
    p = elapsed / duration
    mean_rate = score / elapsed
    local = datetime.fromtimestamp(now / 1000, TZ)
    features = {
        'event_type': curve.kind, 'rank': str(curve.rank),
        'log_current': np.log1p(score), 'progress': p,
        'elapsed_h': elapsed, 'remaining_h': duration - elapsed,
        'duration_h': duration,
        'hour_sin': np.sin(2 * np.pi * local.hour / 24),
        'hour_cos': np.cos(2 * np.pi * local.hour / 24),
        'weekday_sin': np.sin(2 * np.pi * local.weekday() / 7),
        'weekday_cos': np.cos(2 * np.pi * local.weekday() / 7),
    }
    for hours in [1, 3, 6, 12, 24]:
        earlier = now - hours * 3600000
        if earlier >= times[0]:
            before = float(np.interp(earlier, times, scores))
            rate = max(0, score - before) / hours
            features[f'rate_{hours}h_relative'] = rate / mean_rate
        else:
            features[f'rate_{hours}h_relative'] = np.nan
    for fraction in [.25, .5, .75]:
        earlier = curve.start + fraction * (now - curve.start)
        features[f'prefix_ratio_{fraction}'] = (
            float(np.interp(earlier, times, scores)) / score if earlier >= times[0] else np.nan
        )
    return {'features': features, 'curve': curve, 'cutoff': int(cutoff),
            'now': now, 'score': score, 'progress': p,
            'times': times, 'scores': scores}


def make_samples(curves):
    samples, unavailable = [], []
    for curve in curves:
        cutoffs = [curve.start + curve.duration * p for p in [.25, .5, .75]]
        cutoffs += [curve.end - h * 3600000 for h in [24, 6]]
        for stage, cutoff in zip(STAGES, cutoffs):
            sample = snapshot(curve, cutoff)
            if sample is None:
                unavailable.append({'event': curve.event, 'rank': curve.rank, 'stage': stage})
                continue
            sample['stage'] = stage
            sample['target'] = np.log(max(1., curve.final / sample['score']))
            samples.append(sample)
    return samples, unavailable


def weighted_quantile(values, weights, q=.5):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    return float(np.interp(q, (np.cumsum(weights) - weights / 2) / weights.sum(), values))


def templates(sample, history):
    curve, p, score = sample['curve'], sample['progress'], sample['score']
    candidates = []
    # All candidate finals are known before this event begins.
    for past in history:
        if past.rank != curve.rank:
            continue
        t = past.start + p * past.duration
        if t < past.times[0] or t > past.times[-1]:
            continue
        historical_current = float(np.interp(t, past.times, past.scores))
        if historical_current <= 0:
            continue
        fraction = historical_current / past.final
        duration_distance = abs(np.log(past.duration / curve.duration))
        days = (curve.start - past.end) / 86400000
        weight = np.exp(-duration_distance / .35 - days / 240)
        predicted = max(score, score / fraction)
        shape_differences = []
        for f in [.2, .4, .6, .8]:
            current_t = curve.start + f * (sample['now'] - curve.start)
            past_t = past.start + f * p * past.duration
            if current_t >= sample['times'][0] and past_t >= past.times[0]:
                current_ratio = float(np.interp(current_t, sample['times'], sample['scores'])) / score
                past_ratio = float(np.interp(past_t, past.times, past.scores)) / historical_current
                shape_differences.append((current_ratio - past_ratio) ** 2)
        shape_distance = np.mean(shape_differences) if len(shape_differences) >= 2 else .1
        candidates.append((predicted, weight, past.kind == curve.kind, shape_distance))
    assert candidates, 'No historical templates available'
    all_predictions = np.array([c[0] for c in candidates])
    weights = np.array([c[1] for c in candidates])
    template = weighted_quantile(all_predictions, weights)
    same = np.array([c[2] for c in candidates])
    # Shrink type matching toward pooled templates when the type is rare.
    type_weights = weights * np.where(same, 1., .15)
    typed = weighted_quantile(all_predictions, type_weights)
    shape_weights = type_weights * np.exp(-np.array([c[3] for c in candidates]) / .0125)
    analog = weighted_quantile(all_predictions, shape_weights)
    return template, typed, analog


def fit_models(samples):
    x = pd.DataFrame([s['features'] for s in samples])
    y = np.array([s['target'] for s in samples])
    numerical = [c for c in x.columns if c not in CAT]
    prep = ColumnTransformer([
        ('numerical', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), numerical),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT),
    ])
    ridge = make_pipeline(prep, Ridge(alpha=20))
    ridge.fit(x, y)
    # Use a distinct preprocessor; fitted transformers are never shared.
    prep_xgb = ColumnTransformer([
        ('numerical', SimpleImputer(strategy='median', keep_empty_features=True), numerical),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT),
    ])
    xgb = make_pipeline(prep_xgb, XGBRegressor(
        n_estimators=180, max_depth=2, learning_rate=.04,
        min_child_weight=12, reg_lambda=10, subsample=.9,
        colsample_bytree=.9, random_state=42, n_jobs=2,
    ))
    xgb.fit(x, y)
    return ridge, xgb


def model_signature():
    sources = '\n'.join(inspect.getsource(function) for function in (snapshot, make_samples, fit_models))
    configuration = json.dumps({'categorical_features': CAT, 'stages': STAGES,
                                'timezone': str(TZ)}, sort_keys=True)
    return hashlib.sha256((configuration + sources).encode()).hexdigest()


def summarize(frame):
    summaries = []
    for (split, method, stage, rank), group in frame.groupby(['split', 'method', 'stage', 'rank']):
        ape = abs(group['predicted'] / group['actual'] - 1) * 100
        signed = (group['predicted'] / group['actual'] - 1) * 100
        summaries.append({'split': split, 'method': method, 'stage': stage, 'rank': int(rank),
                          'n': len(group), 'mape': float(ape.mean()), 'median_ape': float(ape.median()),
                          'p90_ape': float(ape.quantile(.9)), 'bias_pct': float(signed.mean()),
                          'mae_ep': float(abs(group['predicted'] - group['actual']).mean())})
    return summaries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves, excluded = load_curves()
    samples, unavailable = make_samples(curves)
    rows, audit = [], []
    # 266..291: development diagnostics; 292..327: held-out later events.
    starts = {c.event: c.start for c in curves}
    # CN has rescheduled events, so IDs do not give chronological order.
    evaluation_events = sorted({c.event for c in curves if 266 <= c.event <= 327}, key=starts.get)
    for event in evaluation_events:
        current = [s for s in samples if s['curve'].event == event]
        if not current:
            continue
        start = current[0]['curve'].start
        training_curves = [c for c in curves if c.event != event and c.end + 15 * 60000 < start]
        training_keys = {(c.event, c.rank) for c in training_curves}
        training_samples = [s for s in samples if (s['curve'].event, s['curve'].rank) in training_keys]
        assert all(s['curve'].end + 15 * 60000 < start for s in training_samples)
        ridge, xgb = fit_models(training_samples)
        x = pd.DataFrame([s['features'] for s in current])
        ridge_growth = ridge.predict(x)
        xgb_growth = xgb.predict(x)
        split = 'development' if event < 292 else 'test'
        for i, s in enumerate(current):
            template, typed, analog = templates(s, training_curves)
            score, p = s['score'], s['progress']
            naive = score / p
            rate = s['features']['rate_24h_relative']
            recent = score + (max(0., rate) * score / s['features']['elapsed_h']) * s['features']['remaining_h'] if np.isfinite(rate) else naive
            ridge_pred = score * np.exp(np.clip(ridge_growth[i], 0, 4))
            xgb_pred = score * np.exp(np.clip(xgb_growth[i], 0, 4))
            predictions = [naive, recent, template, typed, analog, ridge_pred, xgb_pred, .5 * typed + .5 * xgb_pred]
            for method, predicted in zip(METHODS, predictions):
                rows.append({'split': split, 'method': method, 'event': event, 'rank': s['curve'].rank, 'stage': s['stage'],
                             'event_type': s['curve'].kind, 'cutoff': s['cutoff'], 'observed_at': s['now'],
                             'current': score, 'actual': s['curve'].final, 'predicted': float(predicted),
                             'training_events': len({c.event for c in training_curves})})
        audit.append({'event': event, 'split': split, 'training_events': sorted({c.event for c in training_curves}),
                      'train_latest_end': max(c.end for c in training_curves), 'test_start': start,
                      'test_snapshots': len(current)})
        if event % 5 == 0:
            print(f'Completed event {event}; training samples {len(training_samples)}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'predictions.csv', index=False)
    summary = summarize(frame)
    pd.DataFrame(summary).to_csv(OUT / 'metrics.csv', index=False)
    result = {'created_at': datetime.now(timezone.utc).isoformat(),
              'database_sha256': hashlib.sha256((ROOT / 'data/db.sqlite3').read_bytes()).hexdigest(),
              'model_signature': model_signature(),
              'training_start_event': 226, 'development_events': [266, 291], 'test_events': [292, 327],
              'final_tolerance_minutes': 15, 'maximum_snapshot_staleness_minutes': 60,
              'eligible_curves': len(curves), 'excluded': excluded, 'unavailable_snapshots': unavailable,
              'methods': METHODS, 'stages': STAGES, 'audit': audit, 'metrics': summary}
    (OUT / 'backtest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for split in ['development', 'test']:
        part = frame[frame.split == split].copy()
        part['ape'] = abs(part.predicted / part.actual - 1) * 100
        print('\n' + split.upper() + ' MAPE (%)')
        print(part.pivot_table(index='method', columns='stage', values='ape').reindex(columns=STAGES).round(2).to_string())
        print('Distinct events', part.event.nunique(), 'snapshots', len(part) // len(METHODS))
    print('Saved', OUT)


if __name__ == '__main__':
    main()
