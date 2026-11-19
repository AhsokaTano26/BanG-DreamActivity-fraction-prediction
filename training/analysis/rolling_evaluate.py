"""Independent cached scoring of sealed rolling V3 validation/test forecasts."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd

from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import identity
from analysis.rolling_dataset import RollingDataset
from analysis.rolling_benchmark import verify_seal, write_new_json
from analysis.temporal_evaluate import aligned_frame, score_temporal


def paired_static_bootstrap(cases, members, labels, predictions, *, draws=2000):
    """Paired activity bootstrap; aliases never duplicate the overall panel."""
    rows = []; rng = np.random.default_rng(42)
    scopes = [('overall', set(cases.case_id))]
    scopes += [(stage, set(group.case_id)) for stage, group in members.groupby('stage_id')]
    for stage, ids in scopes:
        subset = cases[cases.case_id.isin(ids)]
        if subset.empty:
            continue
        actual = {case: labels[case] for case in ids}
        errors = {}
        for method, group in predictions.groupby('method'):
            frame = aligned_frame(subset, actual, group[group.case_id.isin(ids)])
            frame['ape'] = np.abs(frame.predicted/frame.actual-1)*100
            errors[method] = frame.groupby('event').ape.mean().sort_index()
        reference = errors['static_context']
        for method, values in errors.items():
            if method == 'static_context':
                continue
            if not values.index.equals(reference.index):
                raise ValueError('Paired bootstrap activity identities differ')
            delta = (values-reference).to_numpy(); n = len(delta)
            boot = delta[rng.integers(0, n, (draws, n))].mean(axis=1) if n >= 2 else None
            rows.append({'stage_id': stage, 'method': method, 'reference': 'static_context',
                'events': n, 'method_minus_reference_mape_points': float(delta.mean()),
                'ci_low': float(np.quantile(boot, .025)) if boot is not None else None,
                'ci_high': float(np.quantile(boot, .975)) if boot is not None else None})
    return rows


def evaluate(run, output=None):
    run = Path(run); seal = verify_seal(run/'blind_manifest.json')
    if seal.get('protocol') != 'rolling_holdout_v3' or seal.get('scored_splits') != ['validation', 'test']:
        raise ValueError('Expected final sealed V3 validation/test forecasts')
    data = RollingDataset(seal['dataset_directory'])
    if data.manifest['dataset_id'] != seal['dataset_id']:
        raise ValueError('Dataset differs from frozen V3 forecasts')
    output = Path(output or run); receipt = output/'evaluation_manifest.json'
    sources = local_sources([Path(__file__)])
    ident = {'blind_manifest_sha256': sha256(run/'blind_manifest.json'),
        'dataset_id': data.manifest['dataset_id'], 'sources': {n: sha256(ROOT/n) for n in sources},
        'python': platform.python_version(),
        'software': {n: importlib.metadata.version(n) for n in
                     ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost', 'joblib', 'httpx']},
        'bootstrap_seed': 42, 'bootstrap_draws': 2000, 'bootstrap_unit': 'paired event',
        'bootstrap_reference': 'static_context', 'scored_splits': ['validation', 'test']}
    evaluation_id = identity(ident)
    if receipt.exists():
        saved = verify_seal(receipt)
        if saved['evaluation_id'] != evaluation_id:
            raise FileExistsError('Different evaluator; choose a new --output audit directory')
        return json.loads((output/'evaluation.json').read_text())
    names = ['evaluation.json', 'stage_metrics.csv', 'stage_rank_metrics.csv',
             'stage_calendar_metrics.csv', 'paired_bootstrap.csv', 'evaluator_sources']
    if any((output/name).exists() for name in names):
        raise FileExistsError('Scoring destination already contains outputs; preserve and choose new --output')
    selection = json.loads((run/'selection.json').read_text())
    result = {'evaluation_id': evaluation_id, 'dataset_id': data.manifest['dataset_id'],
        'protocol': 'rolling_holdout_v3', 'splits': {}, 'selection': selection,
        'training_scope': 'Initial train group is warmup; no train resubstitution scores',
        'exposure': data.manifest['protocol']['exposure'],
        'metadata_caveat': data.manifest['protocol']['metadata_caveat'],
        'comparison_caveat': 'Only compare methods on these identical dense V3 cases; old V1/V2 totals differ in cohort/protocol'}
    stage_rows, rank_rows, calendar_rows, bootstrap = [], [], [], []
    for split in seal['scored_splits']:
        cases, members = data.case_index(split), data.memberships(split)
        labels = data.labels(split, allow_test=split == 'test')
        predictions = pd.read_csv(run/f'{split}_predictions.csv')
        if set(predictions.method) != set(seal['methods']):
            raise ValueError('Method set differs from sealed configuration')
        scores = {}
        for method, group in predictions.groupby('method'):
            score = score_temporal(cases, members, labels, group,
                                   stage_catalog=data.manifest['protocol']['stages'])
            scores[method] = score
            stage_rows.extend({'split': split, 'method': method, 'stage_id': stage, **metrics}
                              for stage, metrics in score['by_stage'].items())
            rank_rows.extend({'split': split, 'method': method, **row} for row in score['stage_rank'])
            calendar_rows.extend({'split': split, 'method': method, **row} for row in score['stage_calendar'])
        result['splits'][split] = scores
        if split == 'test':
            bootstrap = paired_static_bootstrap(cases, members, labels, predictions)
    result['paired_bootstrap'] = bootstrap
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.rolling-score-', dir=output.parent))
    try:
        write_new_json(staging/'evaluation.json', result)
        for name, rows in [('stage_metrics.csv', stage_rows), ('stage_rank_metrics.csv', rank_rows),
                           ('stage_calendar_metrics.csv', calendar_rows), ('paired_bootstrap.csv', bootstrap)]:
            pd.DataFrame(rows).to_csv(staging/name, index=False)
        for source in sources:
            dest = staging/'evaluator_sources'/source; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/source, dest)
        saved = {'evaluation_id': evaluation_id, 'identity': ident,
            'artifacts': {str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}}
        saved['seal_id'] = identity(saved)
        write_new_json(staging/'evaluation_manifest.json', saved)
        output.mkdir(parents=True, exist_ok=True)
        # Install receipt last. A failure leaves preserved partial output and
        # the next invocation refuses it; no existing result is overwritten.
        for name in names+['evaluation_manifest.json']:
            if (output/name).exists():
                raise FileExistsError('Scoring output appeared during execution: '+name)
            (staging/name).rename(output/name)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    print(json.dumps({'evaluation_id': evaluation_id, 'test': {
        method: score['overall'] for method, score in result['splits']['test'].items()}}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); evaluate(args.run, args.output)
