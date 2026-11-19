"""Fixed causal warm-residual menu for the dense two-rank V3 experiment.

Only detached, already released forecast frames enter this module. The runner
owns dataset access and prediction seals. Positive bias means base*exp(-bias).
No target outcomes or current expert evidence are accepted, even on fallback.
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

from analysis import rolling_residual_stats as stats

SEEDS = (42, 123, 2026)
EXPERT_NAMES = ('ewma_warm', 'ewma_warm_fast', 'kalman_level', 'kalman_trend', 'kernel_progress')
BIAS_CLIP = .3
FEATURES = ('progress', 'log_remaining_h', 'log_base_growth', 'rate_6h_relative',
            'rate_24h_relative', 'rank_flag', 'peer_log_ratio')
_COMMON = {'total_bias_clip': [-.3, .3], 'final_bounds': 'current <= score <= current*exp(4)',
    'clock': 'actual start >= 1706504400000; history end strictly < query start',
    'order': 'event end, actual start, event ID', 'phase_edges': [.25, .9],
    'initial_history': 'sealed prequential forecasts only; never train fit residuals'}
_M = {'temporal_decay': .85, 'shrinkage': 10.,
    'objective': 'event-equal actual bounded final-score MAPE conditional on rank/phase or rank/progress',
    'optimizer': 'exact breakpoint ERM before shrinkage, not plain log-error median',
    'breakpoints': 'unique clip({0,-.3,.3,e,g,g-4},-.3,.3); e=log(base/actual),g=log(base/current)',
    'tie_break': 'smallest absolute bias, then smallest bias',
    'event_mass': 'normalize prefix weights within event, then temporal_decay**event_age',
    'shrinkage_order': 'unshrunk bounded objective optimum; multiply n/(n+10); clip'}
EXPERT_CONFIGS = {
    'ewma_warm': deepcopy(stats.CONFIGS['ewma_reference']) | {'initial_residuals': 'completed sealed prequential forecast events'},
    'ewma_warm_fast': deepcopy(stats.CONFIGS['ewma_fast']) | {'initial_residuals': 'completed sealed prequential forecast events'},
    'kalman_level': deepcopy(stats.CONFIGS['kalman_level']) | {'initial_residuals': 'completed sealed prequential forecast events'},
    'kalman_trend': deepcopy(stats.CONFIGS['kalman_trend']) | {'initial_residuals': 'completed sealed prequential forecast events'},
    'kernel_progress': deepcopy(stats.CONFIGS['kernel_progress']) | {'initial_residuals': 'completed sealed prequential forecast events'},
}
_H = {'gamma': .9, 'loss_update': 'first event loss initializes; L=.9*L+.1*event_MAPE',
    'prior': {'ewma_warm': .7, **{name: .075 for name in EXPERT_NAMES[1:]}},
    'expert_recipes': EXPERT_CONFIGS, 'mixture': 'arithmetic mean of bounded expert final scores',
    'evidence': 'previous sealed expert score columns only; complete expert panel required per event group',
    'weight_formula': 'softmax(log(prior)-eta*discounted_event_mean_APE)',
    'loss_unit': 'fraction, not percent', 'missing_panel': 'skip whole event group; report coverage'}
CONFIGS = {
    'ewma_warm': _COMMON | {'alpha': .3, 'shrinkage': 10., 'initial': 'first event mean',
        'aggregation': 'one observation mean per event/rank/phase'},
    'ewma_warm_fast': _COMMON | {'alpha': .6, 'shrinkage': 10., 'initial': 'first event mean',
        'aggregation': 'one observation mean per event/rank/phase'},
    'mape_median': _COMMON | _M | {'conditioning': 'same rank and phase'},
    'mape_kernel': _COMMON | _M | {'conditioning': 'same rank', 'progress_gaussian_bandwidth': .15},
    'coupled_gap': _COMMON | {'anchor_rank': 2000, 'anchor_alpha': .2, 'anchor_shrinkage': 10.,
        'contrast': 'same-event rank1000 mean log error minus rank2000 mean log error, by phase',
        'contrast_alpha': .6, 'contrast_shrinkage': 5., 'minimum_pairs': 1,
        'missing_pair': 'rank1000 warm EWMA; rank2000 uses available anchor or warm EWMA'},
    'coupled_kalman': _COMMON | {'common': 'paired-rank average log error per event/phase',
        'common_q': .0002, 'common_r': .01, 'common_initial_mean': 0., 'common_initial_variance': .1,
        'contrast': 'paired rank1000 minus rank2000 event/phase mean log error',
        'contrast_level_q': .002, 'contrast_trend_q': .0002, 'contrast_r': .02,
        'contrast_initial_state': [0., 0.], 'contrast_initial_covariance_diagonal': [.1, .01],
        'transition': [[1., 1.], [0., 1.]], 'forecast': 'common level +/- half(next contrast level)',
        'minimum_pairs': 1, 'missing_pair': 'own rank/phase warm EWMA'},
    'hedge_rank10': _COMMON | _H | {'eta': 10., 'conditioning': 'rank'},
    'hedge_rank30': _COMMON | _H | {'eta': 30., 'conditioning': 'rank'},
    'hedge_rank_phase10': _COMMON | _H | {'eta': 10., 'conditioning': 'rank/phase'},
    'drift_gate': _COMMON | {'minimum_events': 8, 'recent_events': 3, 'prior_events': 5,
        'threshold_log_error': .04, 'trend_weight': .25,
        'gate': 'abs(recent3 mean-prior5 mean)>.04 and same sign as recent3 endpoint slope',
        'mixture': '.75 warm EWMA score + .25 Kalman trend score if gate; otherwise warm EWMA'},
    'factorized_mlp': _COMMON | {'hidden': [16, 8], 'activation': 'SiLU', 'dropout': .1,
        'heads': 'shared trunk, common head + rank_sign*contrast head/2',
        'head_initialization': 'both heads zeros', 'normalization': 'event-weighted history numeric standardization',
        'numeric_columns': list(FEATURES), 'feature_clip': [-10., 10.],
        'minimum_history_events': 8, 'shrinkage': 20., 'epochs': 80, 'optimizer': 'AdamW',
        'learning_rate': .001, 'weight_decay': .05, 'batch': 'full', 'seeds': list(SEEDS),
        'device': 'cpu', 'threads': 2, 'loss': 'event-equal actual bounded MAPE', 'loss_unit': 'fraction',
        'baseline': 'historical sealed expert__ewma_warm bias; current recomputed warm EWMA bias',
        'loss_shrinkage': 'apply same n/(n+20) delta in training and forecasting',
        'missing_warm_panel': 'warm EWMA fallback, no fit on imputed historical predictions'},
}


def _json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _finite(series, label):
    try:
        values = pd.to_numeric(series, errors='raise').to_numpy(float)
    except (ValueError, TypeError) as error:
        raise ValueError(label+' must be numeric and finite') from error
    if not np.isfinite(values).all():
        raise ValueError(label+' must be numeric and finite')
    return values


def _validate(history, target, query):
    past, current, clock = stats._frames(history, target, query, context=False)
    if current.empty:
        raise ValueError('One nonempty blind event is required')
    if any(str(name).startswith('expert__') for name in current.columns):
        raise ValueError('Current target cannot carry expert evidence')
    for frame in (past, current):
        for name in ('current', 'base_prediction'):
            if name not in frame:
                raise ValueError('Missing '+name)
            frame[name] = _finite(frame[name], name)
        if (frame.current <= 0).any() or (frame.base_prediction <= 0).any():
            raise ValueError('Scores must be positive')
        growth = np.log(frame.base_prediction.to_numpy()/frame.current.to_numpy())
        if np.any(growth < -1e-10) or np.any(growth > 4.+1e-10):
            raise ValueError('Base forecasts must obey the current/exp4 bounds')
        if not set(frame['rank']).issubset({1000, 2000}):
            raise ValueError('The frozen warm menu supports ranks 1000/2000')
        for name in ('cutoff', 'observed_at'):
            if name not in frame:
                continue
            values = _finite(frame[name], name)
            if np.any(values != np.floor(values)) or np.any(values < frame.start_ms) or np.any(values > frame.end_ms):
                raise ValueError('Observed/cutoff clock lies outside its event')
            if name == 'observed_at' and 'cutoff' in frame and np.any(values > frame.cutoff):
                raise ValueError('Observed clock exceeds the actual prefix cutoff')
    if len(past):
        log_actual = np.log(past.base_prediction.to_numpy())-past.log_error.to_numpy()
        if np.any(log_actual < np.log(np.finfo(float).tiny)) or np.any(log_actual > np.log(np.finfo(float).max)):
            raise ValueError('Historical error implies an invalid actual final score')
        for name in EXPERT_NAMES:
            column = 'expert__'+name
            if column not in past:
                continue
            values = pd.to_numeric(past[column], errors='raise').to_numpy(float)
            present = ~np.isnan(values)
            if (not np.isfinite(values[present]).all() or np.any(values[present] <= 0)
                    or np.any(values[present] < past.current.to_numpy()[present]*(1-1e-10))
                    or np.any(values[present] > past.current.to_numpy()[present]*np.exp(4)*(1+1e-10))):
                raise ValueError('Released expert scores violate the final score bounds')
            past[column] = values
    return past, current, clock


def _final(target, bias):
    bias = np.asarray(bias, dtype=float)
    if bias.shape != (len(target),) or not np.isfinite(bias).all():
        raise ValueError('Invalid log correction')
    bias = np.clip(bias, -BIAS_CLIP, BIAS_CLIP)
    return np.clip(target.base_prediction.to_numpy()*np.exp(-bias),
                   target.current.to_numpy(), target.current.to_numpy()*np.exp(4))


def _weighted_median(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (values.ndim != 1 or weights.shape != values.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('A weighted median needs finite values and nonnegative positive-mass weights')
    order = np.argsort(values, kind='stable')
    position = np.searchsorted(np.cumsum(weights[order]), weights.sum()/2, side='left')
    return float(values[order[min(position, len(order)-1)]])


def mape_log_median(errors, weights):
    """Unbounded conditional MAPE optimum (not the bounded estimator below).

    For a shared bias b and e=log(base/actual), APE=|exp(e-b)-1|.
    In z=exp(-b) this is exp(e)*|z-exp(-e)|, giving an exp(e)-weighted
    median of e. Event/kernel/decay masses enter before the exp(e) factor.
    Score bounds and later regularizing shrinkage change this objective.
    """
    errors = np.asarray(errors, float)
    return _weighted_median(errors, np.asarray(weights)*np.exp(errors-errors.max()))


def bounded_mape_bias(frame, weights):
    """Exact bounded empirical MAPE minimizer via all piecewise-linear kinks.

    In q=exp(-b), each bounded APE is piecewise linear. Its kinks are
    b=e (zero error), b=g (current floor), b=g-4 (growth ceiling).
    The domain endpoints therefore complete the candidate set.
    """
    e = frame.log_error.to_numpy(float)
    g = np.log(frame.base_prediction.to_numpy()/frame.current.to_numpy())
    weights = np.asarray(weights, float)
    if weights.shape != e.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Invalid bounded-MAPE empirical weights')
    candidates = np.unique(np.clip(np.r_[0., -.3, .3, e, g, g-4.], -.3, .3))
    loss = []
    # Chunks bound working memory as prequential history grows.
    for chunk in np.array_split(candidates, max(1, int(np.ceil(len(candidates)/256)))):
        log_growth = np.clip(g[None, :]-chunk[:, None], 0., 4.)
        ape = np.abs(np.expm1(log_growth+e[None, :]-g[None, :]))
        loss.extend((ape @ weights/weights.sum()).tolist())
    loss = np.asarray(loss)
    best = np.flatnonzero(loss <= loss.min()+1e-14)
    index = min(best, key=lambda i: (abs(candidates[i]), candidates[i]))
    return float(candidates[index]), float(loss[index])


def _conditional_mape(past, target, *, kernel):
    config = CONFIGS['mape_kernel' if kernel else 'mape_median']
    values, detail, cache = [], [], {}
    for row in target.itertuples(index=False):
        key = (int(row.rank), float(row.progress) if kernel else int(row.phase))
        if key not in cache:
            group = past[past['rank'] == row.rank]
            if not kernel:
                group = group[group.phase == row.phase]
            events = sorted([(int(p.end_ms.iloc[0]), int(p.start_ms.iloc[0]), int(event), p)
                             for event, p in group.groupby('event', sort=False)], key=lambda x: x[:3])
            panels, mass = [], []
            for index, (_, _, _, panel) in enumerate(events):
                if kernel:
                    logw = -.5*((panel.progress.to_numpy()-row.progress)/config['progress_gaussian_bandwidth'])**2
                    w = np.exp(logw-logw.max())
                else:
                    w = np.ones(len(panel))
                panels.append(panel)
                mass.append(w/w.sum()*config['temporal_decay']**(len(events)-1-index))
            if events:
                panel = pd.concat(panels, ignore_index=True)
                raw, loss = bounded_mape_bias(panel, np.concatenate(mass))
                value = raw*len(events)/(len(events)+config['shrinkage'])
            else:
                raw, value, loss = 0., 0., None
            cache[key] = (value, {'n_events': len(events), 'unshrunk_bias': raw,
                'unshrunk_weighted_bounded_mape': loss,
                'events': [x[2] for x in events]})
        value, state = cache[key]
        values.append(value)
        detail.append({'case_id': str(row.case_id), **state, 'bias': float(value)})
    return np.asarray(values), {'target_states': detail}


def _coupled(past, target, warm, *, kalman):
    means = stats._event_means(past)
    values, details = [], []
    for index, row in enumerate(target.itertuples(index=False)):
        part = means[means.phase == row.phase]
        table = part.pivot(index='event', columns='rank', values='error').reindex(columns=[1000, 2000])
        paired = table.dropna().join(part.drop_duplicates('event').set_index('event')[['end_ms','start_ms']])
        paired = paired.sort_values(['end_ms','start_ms']).reset_index().sort_values(['end_ms','start_ms','event'])
        n = len(paired)
        state = {'case_id': str(row.case_id), 'paired_events': n, 'fallback': n == 0}
        if kalman:
            value = float(warm[index])
            if n:
                config = CONFIGS['coupled_kalman']
                common, common_state = stats._kalman((paired[1000]+paired[2000]).to_numpy()/2,
                    {'initial_mean': 0., 'initial_variance': .1, 'q': config['common_q'], 'r': config['common_r']}, trend=False)
                contrast, contrast_state = stats._kalman((paired[1000]-paired[2000]).to_numpy(),
                    {'initial_state': [0.,0.], 'initial_covariance_diagonal': [.1,.01],
                     'transition': [[1.,1.],[0.,1.]], 'level_q': config['contrast_level_q'],
                     'trend_q': config['contrast_trend_q'], 'r': config['contrast_r']}, trend=True)
                value = common+(.5 if row.rank == 1000 else -.5)*contrast
                state.update(common_state=common_state, contrast_state=contrast_state)
        else:
            anchor = part[part['rank'] == 2000]
            a, na = stats._ewma(anchor.error.to_numpy(), .2)
            a *= na/(na+10.)
            if row.rank == 2000:
                value = a if na else float(warm[index])
                state['fallback'] = na == 0
            elif n and na:
                contrast, _ = stats._ewma((paired[1000]-paired[2000]).to_numpy(), .6)
                value = a+contrast*n/(n+5.)
                state['contrast'] = float(contrast)
            else:
                value = float(warm[index])
            state.update(anchor_events=na, anchor_bias=float(a))
        values.append(float(np.clip(value, -.3, .3)))
        details.append(state | {'bias': values[-1]})
    return np.asarray(values), {'target_states': details}


def _hedge(name, past, target, experts):
    config = CONFIGS[name]
    phase = config['conditioning'] == 'rank/phase'
    grouping = ['rank','phase'] if phase else ['rank']
    columns = ['expert__'+name for name in EXPERT_NAMES]
    groups, states = {}, []
    keys = target[grouping].drop_duplicates().itertuples(index=False, name=None)
    for key in keys:
        panel = past[past['rank'] == key[0]]
        if phase:
            panel = panel[panel.phase == key[1]]
        events = sorted([(int(p.end_ms.iloc[0]), int(p.start_ms.iloc[0]), int(event), p)
                         for event,p in panel.groupby('event', sort=False)], key=lambda x:x[:3])
        losses, used, observations = np.zeros(len(EXPERT_NAMES)), 0, []
        for _, _, event, event_frame in events:
            if set(columns)-set(event_frame) or event_frame[columns].isna().any().any():
                continue
            logratio = (np.log(event_frame[columns].to_numpy())
                        -np.log(event_frame.base_prediction.to_numpy())[:,None]
                        +event_frame.log_error.to_numpy()[:,None])
            current = np.abs(np.expm1(logratio)).mean(axis=0)
            losses = config['gamma']*losses+(1-config['gamma'])*current if used else current
            used += 1
            observations.append({'event': event, 'mean_ape': dict(zip(EXPERT_NAMES, map(float,current)))})
        prior = np.asarray([config['prior'][expert] for expert in EXPERT_NAMES])
        logits = np.log(prior)-config['eta']*losses
        probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
        groups[key] = probabilities
        states.append({'rank': int(key[0]), **({'phase': int(key[1])} if phase else {}),
            'n_events': used, 'available_events': len(events), 'skipped_events': len(events)-used,
            'weights': dict(zip(EXPERT_NAMES, map(float,probabilities))),
            'losses': dict(zip(EXPERT_NAMES, map(float,losses))), 'observations': observations})
    matrix = np.column_stack([experts[expert] for expert in EXPERT_NAMES])
    result = np.asarray([np.dot(matrix[i], groups[tuple(int(row[name]) for name in grouping)])
                         for i, (_,row) in enumerate(target.iterrows())])
    return result, {'groups': states}


def _drift(past, target, experts):
    config = CONFIGS['drift_gate']
    means = stats._event_means(past)
    values, details = [], []
    for i, row in enumerate(target.itertuples(index=False)):
        e = means[(means['rank'] == row.rank)&(means.phase == row.phase)].error.to_numpy()
        delta = float(e[-3:].mean()-e[-8:-3].mean()) if len(e) >= 8 else 0.
        slope = float(e[-1]-e[-3]) if len(e) >= 3 else 0.
        gate = len(e) >= config['minimum_events'] and abs(delta) > config['threshold_log_error'] and delta*slope > 0
        weight = config['trend_weight'] if gate else 0.
        values.append((1-weight)*experts['ewma_warm'][i]+weight*experts['kalman_trend'][i])
        details.append({'case_id': str(row.case_id), 'n_events': len(e), 'recent_prior_difference': delta,
            'recent_endpoint_slope': slope, 'trend_weight': weight})
    return np.asarray(values), {'target_states': details}


def _event_weights(frame):
    counts = frame.event.value_counts()
    return frame.event.map(lambda event: 1./counts[event]).to_numpy(float)


@dataclass
class NumericPreprocessor:
    medians: np.ndarray
    centers: np.ndarray
    scales: np.ndarray

    @staticmethod
    def raw(frame):
        def number(name):
            return pd.to_numeric(frame[name], errors='coerce').to_numpy(float) if name in frame else np.full(len(frame), np.nan)
        return np.column_stack([number('progress'), np.log1p(np.maximum(number('remaining_h'),0)),
            np.log(number('base_prediction')/number('current')), number('rate_6h_relative'),
            number('rate_24h_relative'), np.where(number('rank') == 1000,1.,-1.), number('peer_log_ratio')])

    @classmethod
    def fit(cls, frame, weights):
        x = cls.raw(frame)
        medians = np.asarray([_weighted_median(col[np.isfinite(col)], weights[np.isfinite(col)])
            if np.isfinite(col).any() else 0. for col in x.T])
        filled = np.where(np.isfinite(x),x,medians)
        centers = np.average(filled,axis=0,weights=weights)
        scales = np.sqrt(np.average((filled-centers)**2,axis=0,weights=weights))
        scales[scales < 1e-12] = 1.
        return cls(medians,centers,scales)

    def transform(self, frame):
        x = self.raw(frame)
        return np.clip((np.where(np.isfinite(x),x,self.medians)-self.centers)/self.scales,-10.,10.)

    def state(self):
        return {'numeric_columns': list(FEATURES), 'medians': self.medians.tolist(),
            'centers': self.centers.tolist(), 'scales': self.scales.tolist(),
            'fit_scope': 'released event-equal history only', 'categorical_features': []}


def _make_model(width, recipe):
    from torch import nn
    class Factorized(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(width,16),nn.SiLU(),nn.Dropout(recipe['dropout']),
                nn.Linear(16,8),nn.SiLU(),nn.Dropout(recipe['dropout']))
            self.common = nn.Linear(8,1)
            self.contrast = nn.Linear(8,1)
            for head in (self.common,self.contrast):
                nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        def forward(self, x, sign):
            hidden = self.trunk(x)
            return self.common(hidden).squeeze(-1)+.5*sign*self.contrast(hidden).squeeze(-1)
    return Factorized()


def bounded_mape_loss(current, base, bias, actual):
    """Per-case differentiable actual score APE, including floor/ceiling."""
    import torch
    prediction = torch.minimum(torch.maximum(base*torch.exp(-torch.clamp(bias,-.3,.3)),current),current*np.exp(4))
    return torch.abs(prediction/actual-1.)


def _neural(past, target, warm, directory, query):
    recipe = deepcopy(CONFIGS['factorized_mlp'])
    n = int(past.event.nunique())
    shrink = n/(n+recipe['shrinkage'])
    column = 'expert__ewma_warm'
    complete = column in past and not past[column].isna().any()
    fitted = n >= recipe['minimum_history_events'] and complete
    audit = {'status': 'fitted' if fitted else 'ewma_fallback', 'history_event_count': n,
        'history_case_count': len(past), 'learned_shrinkage': shrink, 'query_start_ms': query,
        'baseline_source': 'sealed expert__ewma_warm final score', 'warm_panel_complete': bool(complete)}
    config = {'recipe':recipe, 'query_start_ms':query, 'query_event':int(target.event.iloc[0]),
        'case_warm_bias':dict(zip(target.case_id.astype(str),map(float,warm))), 'audit':audit}
    directory.mkdir()
    _json(directory/'config.json',config)
    weights = _event_weights(past)
    pd.DataFrame({'case_id':past.case_id,'event':past.event,'weight':weights}).to_csv(directory/'training_weights.csv',index=False)
    if not fitted:
        result = {f'factorized_mlp_seed{seed}':_final(target,warm) for seed in SEEDS}
    else:
        import torch
        torch.set_num_threads(recipe['threads'])
        prep = NumericPreprocessor.fit(past,weights)
        joblib.dump(prep,directory/'preprocessor.joblib',compress=3)
        _json(directory/'preprocessor.json',prep.state())
        x = torch.tensor(prep.transform(past),dtype=torch.float32)
        qx = torch.tensor(prep.transform(target),dtype=torch.float32)
        sign = torch.tensor(np.where(past['rank'].to_numpy()==1000,1.,-1.),dtype=torch.float32)
        qsign = torch.tensor(np.where(target['rank'].to_numpy()==1000,1.,-1.),dtype=torch.float32)
        baseline = torch.tensor(np.log(past.base_prediction.to_numpy()/past[column].to_numpy()),dtype=torch.float32)
        current = torch.tensor(past.current.to_numpy(),dtype=torch.float32)
        base = torch.tensor(past.base_prediction.to_numpy(),dtype=torch.float32)
        actual = torch.tensor(np.exp(np.log(past.base_prediction.to_numpy())-past.log_error.to_numpy()),dtype=torch.float32)
        w = torch.tensor(weights,dtype=torch.float32)
        result, training = {}, {}
        with torch.random.fork_rng(devices=[]):
            for seed in SEEDS:
                torch.manual_seed(seed)
                model = _make_model(len(FEATURES),recipe)
                optimizer = torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
                losses = []
                model.train()
                for _ in range(recipe['epochs']):
                    optimizer.zero_grad(set_to_none=True)
                    bias = baseline+shrink*model(x,sign)
                    loss = (bounded_mape_loss(current,base,bias,actual)*w).sum()/w.sum()
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite bounded MAPE neural objective')
                    loss.backward();optimizer.step();losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():
                    delta = model(qx,qsign).numpy().astype(float)
                name = f'factorized_mlp_seed{seed}'
                result[name] = _final(target,np.asarray(warm)+shrink*delta)
                training[name] = {'seed':seed,'epochs':recipe['epochs'],
                    'first_event_macro_mape':losses[0],'last_event_macro_mape':losses[-1]}
                torch.save({'seed':seed,'epochs':recipe['epochs'],'recipe':recipe,'input_features':len(FEATURES),
                    'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'training_event_macro_mape':losses},directory/f'seed{seed}.pt')
        audit['training'] = training
    result['factorized_mlp'] = np.mean([result[f'factorized_mlp_seed{seed}'] for seed in SEEDS],axis=0)
    _json(directory/'diagnostics.json',audit)
    return result,audit


def replay_neural(model_directory, target, *, query_start_ms):
    """Replay saved neural members and arithmetic final-score ensemble."""
    empty = target.iloc[:0].copy()
    empty['log_error'] = pd.Series(dtype=float)
    _, target, query = _validate(empty,target,query_start_ms)
    directory = Path(model_directory)
    config = json.loads((directory/'config.json').read_text())
    if config['query_start_ms'] != query or config['query_event'] != int(target.event.iloc[0]):
        raise ValueError('Saved calibrator belongs to a different query clock/event')
    if set(config['case_warm_bias']) != set(target.case_id.astype(str)):
        raise ValueError('Replay requires the sealed target case panel')
    warm = np.asarray([config['case_warm_bias'][str(case)] for case in target.case_id])
    audit = json.loads((directory/'diagnostics.json').read_text())
    if audit['status'] == 'ewma_fallback':
        result = {f'factorized_mlp_seed{seed}':_final(target,warm) for seed in SEEDS}
    else:
        import torch
        recipe = config['recipe'];torch.set_num_threads(recipe['threads'])
        prep = joblib.load(directory/'preprocessor.joblib')
        x = torch.tensor(prep.transform(target),dtype=torch.float32)
        sign = torch.tensor(np.where(target['rank'].to_numpy()==1000,1.,-1.),dtype=torch.float32)
        result = {}
        with torch.random.fork_rng(devices=[]):
            for seed in SEEDS:
                checkpoint = torch.load(directory/f'seed{seed}.pt',map_location='cpu',weights_only=False)
                model = _make_model(checkpoint['input_features'],recipe)
                model.load_state_dict(checkpoint['state_dict']);model.eval()
                with torch.no_grad():
                    delta = model(x,sign).numpy().astype(float)
                result[f'factorized_mlp_seed{seed}'] = _final(target,warm+audit['learned_shrinkage']*delta)
    result['factorized_mlp'] = np.mean([result[f'factorized_mlp_seed{seed}'] for seed in SEEDS],axis=0)
    return result


def predict_all(history, target, *, query_start_ms, model_directory):
    """Return (final scores by method, JSON diagnostics, sealed expert scores).

    All recipes are fixed before run registration. Artifacts are written once
    in a new directory. The caller must save the returned experts with the
    current forecasts and only release them after this event completes.
    """
    past, target, query = _validate(history,target,query_start_ms)
    directory = Path(model_directory)
    if directory.exists():
        raise FileExistsError('Preserve sealed model artifacts; use a new directory')
    directory.parent.mkdir(parents=True,exist_ok=True)
    scores, diagnostics, experts, expert_bias = {}, {}, {}, {}
    aliases = {'ewma_warm':'ewma_reference','ewma_warm_fast':'ewma_fast'}
    for name in EXPERT_NAMES:
        bias, diagnostic = stats.predict_bias(aliases.get(name,name),past,target,query_start_ms=query)
        expert_bias[name] = bias
        experts[name] = _final(target,bias)
        diagnostic['method'] = name
        diagnostics[name] = diagnostic
        if name in CONFIGS:
            scores[name] = experts[name].copy()
    for name,kernel in [('mape_median',False),('mape_kernel',True)]:
        bias, diagnostics[name] = _conditional_mape(past,target,kernel=kernel)
        scores[name] = _final(target,bias)
    for name,kalman in [('coupled_gap',False),('coupled_kalman',True)]:
        bias, diagnostics[name] = _coupled(past,target,expert_bias['ewma_warm'],kalman=kalman)
        scores[name] = _final(target,bias)
    for name in ('hedge_rank10','hedge_rank30','hedge_rank_phase10'):
        scores[name],diagnostics[name] = _hedge(name,past,target,experts)
    scores['drift_gate'],diagnostics['drift_gate'] = _drift(past,target,experts)
    with tempfile.TemporaryDirectory(prefix='.'+directory.name+'-',dir=directory.parent) as temporary:
        staging = Path(temporary)/'artifacts';staging.mkdir()
        _json(staging/'config.json',{'methods':CONFIGS,'experts':EXPERT_CONFIGS,'query_start_ms':query})
        neural,diagnostics['factorized_mlp'] = _neural(past,target,expert_bias['ewma_warm'],staging/'factorized_mlp',query)
        scores.update(neural)
        diagnostics['_boundary'] = {'query_start_ms':query,'history_events':int(past.event.nunique()),
            'history_cases':len(past),'latest_history_end_ms':int(past.end_ms.max()) if len(past) else None,
            'history_event_ids':sorted(map(int,past.event.unique())),'target_outcomes_accepted':False}
        _json(staging/'diagnostics.json',diagnostics)
        if directory.exists():
            raise FileExistsError('Model directory appeared during fit; refusing overwrite')
        staging.rename(directory)
    return scores,diagnostics,experts
