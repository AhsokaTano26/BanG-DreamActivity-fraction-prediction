"""Pure, causal learned corrections for frozen dense V3 base forecasts.

Only detached released-history and blind-target frames enter this module. It
never opens a dataset or an answer vault. Positive total bias corrects a base
forecast downward: the coordinator applies ``base * exp(-bias)`` and bounds
that final score. The coordinator also averages the three bounded NN scores.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from analysis.data_policy import MIN_EVENT_START_MS


NUMERIC_COLUMNS = (
    'progress', 'current', 'base_prediction', 'ewma_bias', 'duration_h',
    'remaining_h', 'log_base_growth', 'rate_24h_relative', 'rate_6h_relative',
)
CATEGORICAL_COLUMNS = ('rank', 'phase', 'event_type', 'band_group')
INPUT_COLUMNS = ('case_id', 'event', 'start_ms', 'end_ms') + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
SEEDS = (42, 123, 2026)
MIN_HISTORY_EVENTS = 8
LEARNED_SHRINKAGE_EVENTS = 20
BIAS_CLIP = .3

# This fixed menu is archived by the coordinator before any forecast release.
CONFIGS = {
    'ridge20': {'kind': 'ridge', 'alpha': 20., 'fit_intercept': True},
    'ridge100': {'kind': 'ridge', 'alpha': 100., 'fit_intercept': True},
    'residual_tree': {'kind': 'catboost', 'iterations': 200, 'depth': 2,
        'learning_rate': .03, 'l2_leaf_reg': 25, 'random_seed': 42,
        'thread_count': 2, 'loss_function': 'RMSE', 'allow_const_label': True,
        'allow_writing_files': False, 'verbose': False},
    'residual_mlp': {'kind': 'mlp', 'hidden': [32, 16], 'normalization': 'LayerNorm',
        'activation': 'SiLU', 'dropout': .15, 'epochs': 80, 'optimizer': 'AdamW',
        'learning_rate': .001, 'weight_decay': .05, 'huber_beta': .1,
        'batch': 'full', 'seeds': list(SEEDS), 'device': 'cpu', 'threads': 2},
}


def _category(value):
    if pd.isna(value):
        return '__missing__'
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isfinite(value) and value == int(value):
            return str(int(value))
    return str(value)


def _keys(series):
    return series.map(_category)


def _integers(series, label):
    try:
        values = pd.to_numeric(series, errors='raise').to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f'{label} must contain finite integer timestamps') from error
    if not np.isfinite(values).all() or np.any(values != np.floor(values)):
        raise ValueError(f'{label} must contain finite integer timestamps')
    return values


def _finite(series, label):
    try:
        values = pd.to_numeric(series, errors='raise').to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f'{label} must be finite') from error
    if not np.isfinite(values).all():
        raise ValueError(f'{label} must be finite')
    return values


def _validate_target(target, query_start_ms):
    if not isinstance(target, pd.DataFrame):
        raise ValueError('Target must be a detached blind DataFrame')
    missing = set(INPUT_COLUMNS) - set(target.columns)
    forbidden = {'actual', 'target', 'final', 'log_error'} & set(target.columns)
    if missing or forbidden:
        raise ValueError(f'Blind target schema mismatch: missing={sorted(missing)}, '
                         f'answer_columns={sorted(forbidden)}')
    if target.empty:
        raise ValueError('A query requires a nonempty target event')
    if (target[['case_id', 'event']].isna().any().any()
            or _keys(target.case_id).duplicated().any()):
        raise ValueError('Target requires unique case IDs and a known event')
    if len(set(_keys(target.event))) != 1:
        raise ValueError('All target rows must belong to one query event')
    clock = _integers(pd.Series([query_start_ms]), 'Query clock')[0]
    starts = _integers(target.start_ms, 'Target start')
    ends = _integers(target.end_ms, 'Target end')
    if clock < MIN_EVENT_START_MS or np.any(starts < MIN_EVENT_START_MS):
        raise ValueError('Target violates the dense actual-start date policy')
    if np.any(starts != clock) or len(set(ends)) != 1 or np.any(ends <= starts):
        raise ValueError('Target dates disagree with the query event start clock')
    _finite(target.ewma_bias, 'Target EWMA bias')
    return int(clock)


def _validate(history, target, query_start_ms):
    """Run before the fallback branch, so small pools cannot bypass the clock."""
    clock = _validate_target(target, query_start_ms)
    if not isinstance(history, pd.DataFrame):
        raise ValueError('History must be a detached released DataFrame')
    missing = set(INPUT_COLUMNS + ('log_error',)) - set(history.columns)
    if missing:
        raise ValueError(f'Released history schema mismatch: missing={sorted(missing)}')
    if (history[['case_id', 'event']].isna().any().any()
            or _keys(history.case_id).duplicated().any()):
        raise ValueError('History requires unique case IDs and known events')
    if set(_keys(history.event)) & set(_keys(target.event)):
        raise ValueError('Current event cannot supervise its own correction')
    if set(_keys(history.case_id)) & set(_keys(target.case_id)):
        raise ValueError('Current target cases cannot occur in released history')
    starts = _integers(history.start_ms, 'History start')
    ends = _integers(history.end_ms, 'History end')
    if np.any(starts < MIN_EVENT_START_MS):
        raise ValueError('History violates the dense actual-start date policy')
    if np.any(ends <= starts) or np.any(ends >= clock):
        raise ValueError('All history events must be completed strictly before query start')
    for _, group in history.groupby(_keys(history.event), sort=False):
        if group.start_ms.nunique() != 1 or group.end_ms.nunique() != 1:
            raise ValueError('History event has inconsistent start/end dates')
    _finite(history.log_error, 'History log error')
    _finite(history.ewma_bias, 'Historical causal EWMA bias')
    return clock


def _event_weights(history):
    events = _keys(history.event)
    counts = events.value_counts()
    return events.map(lambda event: 1./counts[event]).to_numpy(dtype=float)


def _weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, weights.sum()/2, side='left')
    return float(values[order[min(index, len(order)-1)]])


@dataclass
class ResidualPreprocessor:
    """History-only numeric imputation/scaling and unknown-safe one-hot codes.

    Fitting both numeric summaries and supervised models uses one total weight
    per activity; additional snapshots do not increase an activity's influence.
    Unknown target categories map to zeros and cannot grow the fitted schema.
    """
    medians: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    categories: dict[str, list[str]]

    @classmethod
    def fit(cls, history, weights):
        raw = cls._numeric(history)
        medians = []
        for column in raw.T:
            observed = np.isfinite(column)
            medians.append(_weighted_median(column[observed], weights[observed])
                           if observed.any() else 0.)
        medians = np.asarray(medians)
        filled = np.where(np.isfinite(raw), raw, medians)
        centers = np.average(filled, axis=0, weights=weights)
        variance = np.average((filled-centers)**2, axis=0, weights=weights)
        scales = np.sqrt(variance)
        scales[scales < 1e-12] = 1.
        categories = {column: sorted(set(_keys(history[column])))
                      for column in CATEGORICAL_COLUMNS}
        return cls(medians, centers, scales, categories)

    @staticmethod
    def _numeric(frame):
        return frame[list(NUMERIC_COLUMNS)].apply(pd.to_numeric,
            errors='coerce').to_numpy(dtype=float)

    def transform(self, frame):
        raw = self._numeric(frame)
        numeric = (np.where(np.isfinite(raw), raw, self.medians)-self.centers)/self.scales
        encoded = [numeric]
        for column, categories in self.categories.items():
            values = _keys(frame[column]).to_numpy()
            encoded.append(np.column_stack([values == category for category in categories]))
        result = np.column_stack(encoded).astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError('Residual preprocessing produced nonfinite features')
        return result

    def state(self):
        return {'numeric_columns': list(NUMERIC_COLUMNS),
            'categorical_columns': list(CATEGORICAL_COLUMNS),
            'medians': self.medians.tolist(), 'centers': self.centers.tolist(),
            'scales': self.scales.tolist(), 'categories': self.categories,
            'unseen_categories': 'all-zero encoding',
            'numeric_fitting': 'event-balanced weighted median/mean/population-variance'}


def _json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _methods(name):
    return [f'residual_mlp_seed{seed}' for seed in SEEDS] if name == 'residual_mlp' else [name]


def _total_bias(target, correction, shrinkage):
    values = _finite(target.ewma_bias, 'Target EWMA bias') + shrinkage*np.asarray(correction)
    if not np.isfinite(values).all():
        raise ValueError('Model returned a nonfinite residual correction')
    return np.clip(values, -BIAS_CLIP, BIAS_CLIP)


def _make_mlp(input_features, recipe):
    # Lazy import keeps PyTorch out of tree-only parent test processes.
    from torch import nn
    modules = []
    width = input_features
    for hidden in recipe['hidden']:
        modules.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                        nn.Dropout(recipe['dropout'])])
        width = hidden
    modules.append(nn.Linear(width, 1))
    return nn.Sequential(*modules)


def _fit_mlp(x, y, weights, query_x, recipe, directory):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(recipe['threads'])
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    w_tensor = torch.tensor(weights, dtype=torch.float32)
    query_tensor = torch.tensor(query_x, dtype=torch.float32)
    predictions, training = {}, {}
    # Forking restores the caller's RNG; every seed gets its own initialization
    # and full-batch dropout stream, with no validation or test early stopping.
    with torch.random.fork_rng(devices=[]):
        for seed in recipe['seeds']:
            torch.manual_seed(seed)
            model = _make_mlp(x.shape[1], recipe)
            optimizer = torch.optim.AdamW(model.parameters(),
                lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
            losses = []
            model.train()
            for _ in range(recipe['epochs']):
                optimizer.zero_grad(set_to_none=True)
                output = model(x_tensor).squeeze(-1)
                per_case = F.smooth_l1_loss(output, y_tensor,
                    beta=recipe['huber_beta'], reduction='none')
                loss = (per_case*w_tensor).sum()/w_tensor.sum()
                if not torch.isfinite(loss):
                    raise ValueError('Residual MLP training loss is nonfinite')
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                prediction = model(query_tensor).squeeze(-1).numpy().astype(float)
            member = f'residual_mlp_seed{seed}'
            predictions[member] = prediction
            training[member] = {'seed': seed, 'epochs': recipe['epochs'],
                'initial_weighted_huber': losses[0], 'last_weighted_huber': losses[-1]}
            torch.save({'seed': seed, 'epochs': recipe['epochs'], 'recipe': recipe,
                'input_features': x.shape[1], 'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_weighted_huber': losses}, directory/f'seed{seed}.pt')
    return predictions, training


def fit_predict(name, history, target, *, query_start_ms, model_directory):
    """Return total log biases and JSON diagnostics; write new replay artifacts.

    Each history row's supervised target is its original log error minus the
    EWMA bias saved when that historical event was predicted. The current
    target's bias is used only when composing the returned total correction.
    """
    if name not in CONFIGS:
        raise ValueError(f'Unknown learned residual recipe: {name}')
    clock = _validate(history, target, query_start_ms)
    directory = Path(model_directory)
    if directory.exists():
        raise FileExistsError('Residual model directory already exists; artifacts cannot be overwritten')
    recipe = deepcopy(CONFIGS[name])
    events = sorted(set(_keys(history.event)))
    n_events = len(events)
    shrinkage = n_events/(n_events+LEARNED_SHRINKAGE_EVENTS)
    config = {'method': name, 'recipe': recipe, 'query_start_ms': clock,
        'query_event': _category(target.event.iloc[0]), 'numeric_columns': list(NUMERIC_COLUMNS),
        'categorical_columns': list(CATEGORICAL_COLUMNS), 'minimum_history_events': MIN_HISTORY_EVENTS,
        'additional_shrinkage': 'n_history_events/(n_history_events+20)',
        'total_bias_clip': [-BIAS_CLIP, BIAS_CLIP],
        'supervised_target': 'historical log_error - historical causal ewma_bias',
        'weighting': 'each completed forecast event has total supervised weight 1',
        'dense_minimum_actual_start_ms': MIN_EVENT_START_MS}
    audit = {'method': name, 'query_start_ms': clock, 'query_event': config['query_event'],
        'history_events': events, 'history_event_count': n_events, 'history_case_count': len(history),
        'target_case_count': len(target), 'learned_shrinkage': shrinkage,
        'status': 'fitted' if n_events >= MIN_HISTORY_EVENTS else 'ewma_fallback',
        'history_max_end_ms': int(history.end_ms.max()) if len(history) else None,
        'methods': _methods(name)}
    weights = _event_weights(history)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+directory.name+'-', dir=directory.parent) as temporary:
        staging = Path(temporary)/'artifacts'
        staging.mkdir()
        _json(staging/'config.json', config)
        pd.DataFrame({'case_id': history.case_id.to_numpy(), 'event': history.event.to_numpy(),
                      'weight': weights}).to_csv(staging/'training_weights.csv', index=False)
        if n_events < MIN_HISTORY_EVENTS:
            predictions = {member: _total_bias(target, np.zeros(len(target)), 0.)
                           for member in _methods(name)}
        else:
            preprocessor = ResidualPreprocessor.fit(history, weights)
            x, query_x = preprocessor.transform(history), preprocessor.transform(target)
            y = _finite(history.log_error, 'History log error') - _finite(history.ewma_bias, 'History EWMA')
            joblib.dump(preprocessor, staging/'preprocessor.joblib', compress=3)
            _json(staging/'preprocessor.json', preprocessor.state())
            audit['feature_count'] = x.shape[1]
            if name == 'residual_mlp':
                corrections, training = _fit_mlp(x, y, weights, query_x, recipe, staging)
                audit['training'] = training
            else:
                if recipe['kind'] == 'ridge':
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=recipe['alpha'], fit_intercept=recipe['fit_intercept'])
                else:
                    from catboost import CatBoostRegressor
                    model = CatBoostRegressor(**{key: value for key, value in recipe.items()
                                                if key != 'kind'})
                model.fit(x, y, sample_weight=weights)
                corrections = {name: np.asarray(model.predict(query_x), dtype=float)}
                joblib.dump(model, staging/'model.joblib', compress=3)
            predictions = {member: _total_bias(target, correction, shrinkage)
                           for member, correction in corrections.items()}
        _json(staging/'diagnostics.json', audit)
        if directory.exists():
            raise FileExistsError('Residual model directory appeared during fit; refusing overwrite')
        staging.rename(directory)
    return predictions, audit


def replay_predict(model_directory, target, *, query_start_ms):
    """Replay saved artifacts on the same blind event without refitting anything."""
    clock = _validate_target(target, query_start_ms)
    directory = Path(model_directory)
    config = json.loads((directory/'config.json').read_text())
    audit = json.loads((directory/'diagnostics.json').read_text())
    if (config['query_start_ms'] != clock
            or config['query_event'] != _category(target.event.iloc[0])):
        raise ValueError('Saved correction model belongs to a different query event clock')
    name = config['method']
    if audit['status'] == 'ewma_fallback':
        return {member: _total_bias(target, np.zeros(len(target)), 0.) for member in _methods(name)}
    preprocessor = joblib.load(directory/'preprocessor.joblib')
    x = preprocessor.transform(target)
    if name == 'residual_mlp':
        import torch
        recipe = config['recipe']
        torch.set_num_threads(recipe['threads'])
        corrections = {}
        with torch.random.fork_rng(devices=[]):
            for seed in recipe['seeds']:
                checkpoint = torch.load(directory/f'seed{seed}.pt', map_location='cpu', weights_only=False)
                model = _make_mlp(checkpoint['input_features'], recipe)
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                with torch.no_grad():
                    corrections[f'residual_mlp_seed{seed}'] = model(torch.tensor(x,
                        dtype=torch.float32)).squeeze(-1).numpy().astype(float)
    else:
        corrections = {name: joblib.load(directory/'model.joblib').predict(x)}
    return {member: _total_bias(target, correction, audit['learned_shrinkage'])
            for member, correction in corrections.items()}
