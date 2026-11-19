"""Sealed, small synthetic mechanism study; never imports real data or models.

Run only after PI authorization, with a frozen B_synthetic_plan.json:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m \
    analysis.paper_upgrade_synthetic --plan <sealed-plan> --output <new-directory>

This module reports state risk and simulation APE separately. Its oracle knows
the generated covariance, so it is a diagnostic rather than a selectable model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform

import numpy as np


ANCHOR_TRANSFORM = np.array([[0., 1.], [1., -1.]])
METHODS = (
    'uncorrected', 'independent_ewma', 'fully_shared_ewma',
    'equal_filter_anchor_gap', 'anchor_gap', 'independent_kalman_fixed',
    'dlm_diagonal_R_fixed', 'dlm_fullcov_fixed', 'dlm_oracle',
)
METRICS = ('state_mse', 'raw_state_mse', 'synthetic_unbounded_ape_pct')
GRID_KEYS = ('q_common', 'q_private', 'observation_variance', 'mismatch_loading')


def _weights(length, alpha):
    if length < 1 or not 0. < alpha <= 1.:
        raise ValueError('A nonempty history and alpha in (0,1] are required')
    weights = alpha * (1. - alpha) ** np.arange(length - 1, -1, -1)
    weights[0] = (1. - alpha) ** (length - 1)
    return weights


def equal_filter_bias(observations, *, alpha=.3, shrinkage=10.):
    """Independent same-weight EWMA; final axis is original [T1000,T2000]."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim not in (2, 3) or observations.shape[-1] != 2:
        raise ValueError('Expected history shape (n,2) or (MC,n,2)')
    n = observations.shape[-2]
    return np.tensordot(observations, _weights(n, alpha), axes=([-2], [0])) * n / (n + shrinkage)


def equal_filter_anchor_bias(observations, *, alpha=.3, shrinkage=10.):
    """Algebraic negative control: transform, same filter, inverse transform."""
    changed = np.asarray(observations, dtype=float) @ ANCHOR_TRANSFORM.T
    filtered = equal_filter_bias(changed, alpha=alpha, shrinkage=shrinkage)
    return filtered @ np.linalg.inv(ANCHOR_TRANSFORM).T


def anchor_gap_bias(observations, *, anchor_alpha=.2, anchor_shrinkage=10.,
                    contrast_alpha=.6, contrast_shrinkage=5.):
    observations = np.asarray(observations, dtype=float)
    n = observations.shape[-2]
    anchor = np.tensordot(observations[..., 1], _weights(n, anchor_alpha), axes=([-1], [0]))
    contrast = np.tensordot(observations[..., 0] - observations[..., 1],
                           _weights(n, contrast_alpha), axes=([-1], [0]))
    anchor *= n / (n + anchor_shrinkage)
    contrast *= n / (n + contrast_shrinkage)
    return np.stack([anchor + contrast, anchor], axis=-1)


def kalman_level_bias(observations, process, noise, prior):
    """I-transition/I-observation Gaussian local-level filter, complete panel.

    The process covariance is added before every completed observation, matching
    existing scalar code. The one-step-ahead mean is the last filtered mean.
    Covariance is shared across MC replicas, while their filtered means differ.
    """
    observations = np.asarray(observations, dtype=float)
    process, noise, covariance = (np.array(item, dtype=float, copy=True)
                                  for item in (process, noise, prior))
    if observations.ndim not in (2, 3) or observations.shape[-1] != 2:
        raise ValueError('Expected history shape (n,2) or (MC,n,2)')
    for matrix in (process, noise, covariance):
        if matrix.shape != (2, 2) or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-14):
            raise ValueError('Covariance matrices must be symmetric 2 by 2')
        if not np.isfinite(matrix).all() or np.linalg.eigvalsh(matrix).min() < -1e-12:
            raise ValueError('Covariance matrices must be finite positive semidefinite')
    mean = np.zeros(observations.shape[:-2] + (2,))
    identity = np.eye(2)
    for index in range(observations.shape[-2]):
        predictive = covariance + process
        gain = np.linalg.solve((predictive + noise).T, predictive.T).T
        mean += (observations[..., index, :] - mean) @ gain.T
        remainder = identity - gain
        covariance = remainder @ predictive @ remainder.T + gain @ noise @ gain.T
        covariance = (covariance + covariance.T) / 2
    return mean


def sharing_risk_difference(delta, variance1, variance2, covariance, weight):
    """Exact task-1 squared-risk change, under the report's unbiased assumptions."""
    return (-2 * weight * (variance1 - covariance)
            + weight ** 2 * (delta ** 2 + variance1 + variance2 - 2 * covariance))


def scores_for_bias(state, observed, bias, *, clip):
    """Return one task-mean score per MC path, preserving estimand names."""
    state, observed, bias = (np.asarray(value, float) for value in (state, observed, bias))
    capped = np.clip(bias, -clip, clip)
    return {
        'state_mse': np.mean((state - capped) ** 2, axis=-1),
        'raw_state_mse': np.mean((state - bias) ** 2, axis=-1),
        'synthetic_unbounded_ape_pct': 100 * np.mean(np.abs(np.expm1(observed - capped)), axis=-1),
    }


def plan_digest(plan):
    detached = dict(plan)
    detached['source_seal'] = None
    payload = json.dumps(detached, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_plan(plan, root):
    """Fail before generating any random path if frozen code/config changed."""
    seal = plan.get('source_seal')
    if not isinstance(seal, dict) or seal.get('plan_sha256') != plan_digest(plan):
        raise ValueError('Plan is unsealed or its configuration changed')
    for name, expected in seal['source_sha256'].items():
        if _sha256(root / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    if tuple(plan['methods']) != METHODS or set(plan['grid']) != set(GRID_KEYS):
        raise ValueError('Fixed method/grid contract differs')
    if any(len(plan['grid'][key]) != 2 for key in GRID_KEYS):
        raise ValueError('The fixed design must contain exactly 16 cells')
    histories = plan['history_lengths']
    if histories != [8, 16, 32, 64] or plan['sequence_length'] != 65:
        raise ValueError('Expected fixed histories and one-step-ahead horizon')
    if plan['seeds'] != [42, 123, 2026] or plan['mc_sequences_per_seed'] != 100:
        raise ValueError('Expected sealed seeds and 100 MC sequences per seed')
    if plan['real_data_access'] != 'none':
        raise ValueError('This module supports synthetic data only')


def prepare_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def _method_biases(history, cell, parameters):
    n = history.shape[-2]
    independent = equal_filter_bias(history, alpha=parameters['independent_alpha'],
                                    shrinkage=parameters['independent_shrinkage'])
    pooled = np.mean(independent, axis=-1, keepdims=True)
    prior = parameters['fixed_kalman_initial_original_state_variance'] * np.eye(2)
    transform = ANCHOR_TRANSFORM
    changed = history @ transform.T
    changed_prior = transform @ prior @ transform.T
    process = np.diag([parameters['fixed_dlm_anchor_q'], parameters['fixed_dlm_contrast_q']])
    original_noise = parameters['fixed_dlm_original_observation_r'] * np.eye(2)
    changed_noise = transform @ original_noise @ transform.T
    q_common, q_private, r, mismatch = (cell[key] for key in GRID_KEYS)
    loadings = np.array([[1., 1. - mismatch], [1., -mismatch]])
    true_process = loadings @ np.diag([q_common, q_private]) @ loadings.T
    result = {
        'uncorrected': np.zeros(history.shape[:-2] + (2,)),
        'independent_ewma': independent,
        'fully_shared_ewma': np.repeat(pooled, 2, axis=-1),
        'equal_filter_anchor_gap': equal_filter_anchor_bias(
            history, alpha=parameters['independent_alpha'], shrinkage=parameters['independent_shrinkage']),
        'anchor_gap': anchor_gap_bias(
            history, anchor_alpha=parameters['anchor_alpha'], anchor_shrinkage=parameters['anchor_shrinkage'],
            contrast_alpha=parameters['contrast_alpha'], contrast_shrinkage=parameters['contrast_shrinkage']),
        'independent_kalman_fixed': kalman_level_bias(
            history, parameters['independent_kalman_q'] * np.eye(2),
            parameters['independent_kalman_r'] * np.eye(2), prior),
        'dlm_oracle': kalman_level_bias(history, true_process, r * np.eye(2), np.zeros((2, 2))),
    }
    for name, noise in [('dlm_diagonal_R_fixed', np.diag(np.diag(changed_noise))),
                        ('dlm_fullcov_fixed', changed_noise)]:
        result[name] = kalman_level_bias(changed, process, noise, changed_prior) @ np.linalg.inv(transform).T
    if not np.allclose(result['equal_filter_anchor_gap'], independent, rtol=0., atol=1e-12):
        raise AssertionError('The same-filter algebraic negative control failed')
    return result


def _describe(values):
    values = np.asarray(values, float)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return {'mc_sequences': len(values), 'mean': mean, 'mc_standard_error': standard_error,
            'mc_normal_95_low': mean - 1.96 * standard_error,
            'mc_normal_95_high': mean + 1.96 * standard_error}


def _write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(plan_path, output):
    """Execute only the frozen design, after external approval of its seal."""
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(Path(plan_path).read_text())
    verify_plan(plan, root)
    prepare_output(output)
    output = Path(output)
    cells = [dict(zip(GRID_KEYS, values)) for values in itertools.product(
        *(plan['grid'][key] for key in GRID_KEYS))]
    gathered, paired, per_seed = {}, {}, []
    for seed in plan['seeds']:
        generator = np.random.default_rng(seed)
        shape = (plan['mc_sequences_per_seed'], plan['sequence_length'])
        z_common = generator.standard_normal(shape)
        z_private = generator.standard_normal(shape)
        z_noise = generator.standard_normal(shape + (2,))
        for cell_index, cell in enumerate(cells):
            common = np.cumsum(np.sqrt(cell['q_common']) * z_common, axis=1)
            private = np.cumsum(np.sqrt(cell['q_private']) * z_private, axis=1)
            mismatch = cell['mismatch_loading']
            state = np.stack([common + (1 - mismatch) * private, common - mismatch * private], axis=-1)
            observed = state + np.sqrt(cell['observation_variance']) * z_noise
            for history_length in plan['history_lengths']:
                history = observed[:, :history_length, :]
                biases = _method_biases(history, cell, plan['fixed_parameters'])
                scores = {name: scores_for_bias(
                    state[:, history_length, :], observed[:, history_length, :], biases[name],
                    clip=plan['fixed_parameters']['bias_clip']) for name in METHODS}
                for method in METHODS:
                    for metric in METRICS:
                        key = (cell_index, history_length, method, metric)
                        values = scores[method][metric]
                        gathered.setdefault(key, []).append(values)
                        paired.setdefault(key, []).append(values - scores['independent_ewma'][metric])
                        per_seed.append({'cell': cell_index, **cell, 'history_length': history_length,
                                         'seed': seed, 'method': method, 'metric': metric, **_describe(values)})
    summary, comparisons = [], []
    for key, arrays in gathered.items():
        cell_index, history_length, method, metric = key
        identity = {'cell': cell_index, **cells[cell_index], 'history_length': history_length,
                    'method': method, 'metric': metric, 'oracle_diagnostic': method == 'dlm_oracle'}
        summary.append(identity | _describe(np.concatenate(arrays)))
        comparisons.append(identity | {'reference': 'independent_ewma'}
                           | _describe(np.concatenate(paired[key])))
    _write_csv(output / 'summary.csv', summary)
    _write_csv(output / 'per_seed.csv', per_seed)
    _write_csv(output / 'paired_comparisons.csv', comparisons)
    manifest = {
        'experiment': plan['experiment'], 'plan': plan,
        'execution_time_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'numpy': np.__version__,
        'real_data_access': False, 'real_method_selection': False,
        'oracle_uses_generating_parameters': True,
        'thread_environment': {name: os.environ.get(name) for name in
                               ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        'grid_cells': len(cells), 'methods': len(METHODS),
        'summary_rows': len(summary), 'per_seed_rows': len(per_seed),
        'artifacts_sha256': {name: _sha256(output / name) for name in
                             ['summary.csv', 'per_seed.csv', 'paired_comparisons.csv']},
    }
    with (output / 'run_manifest.json').open('x') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = run(args.plan, args.output)
    print(json.dumps({key: manifest[key] for key in ['experiment', 'grid_cells', 'methods', 'summary_rows']},
                     sort_keys=True))


if __name__ == '__main__':
    main()
