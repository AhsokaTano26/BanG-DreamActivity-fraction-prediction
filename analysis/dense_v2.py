"""Retrain every existing two-rank V2 recipe on the frozen dense-era cohort.

The legacy runners remain byte-for-byte unchanged. This adapter narrowly binds
those runners to the audited new dataset and records its own source and resource
policy in their manifests. No recipe budget, seed, or selection rule changes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
from pathlib import Path

import pandas as pd

from analysis.data_policy import DENSE_V2_TWO_RANK, MIN_EVENT_START_MS, POLICY_ID
from analysis.holdout_dataset import sha256
from analysis.temporal_dataset import TemporalDataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = 'd057beca6597dd28b59fde7da08f665ae74af41d96f940da04268c881864a215'
STAGES = ('base', 'paired', 'research', 'innovation', 'next')
MODULES = dict(base='analysis.temporal_train', paired='analysis.temporal_paired_benchmark',
               research='analysis.research_search', innovation='analysis.innovation_runner',
               next='analysis.v2_next_runner')
RUNS = {stage: ROOT / 'analysis/temporal_results' / f'dense_{stage}_20261115' for stage in STAGES}
ADAPTER_SOURCES = ['analysis/dense_v2.py', 'analysis/data_policy.py']


class DenseTrainingData:
    """Policy-checked input facade that cannot release test answers to a model."""
    def __init__(self, data):
        spec = data.manifest['protocol']
        if data.manifest['dataset_id'] != DATASET_ID:
            raise ValueError('Dense two-rank dataset fingerprint differs')
        if (spec.get('data_policy_id') != POLICY_ID or
                spec.get('minimum_event_start_ms') != MIN_EVENT_START_MS or
                spec.get('ranks') != [1000, 2000]):
            raise ValueError('Dense two-rank population policy differs')
        parts = pd.read_csv(data.root / 'partitions.csv')
        if parts.empty or parts.start_ms.isna().any() or (parts.start_ms < MIN_EVENT_START_MS).any():
            raise ValueError('Pre-dense activity in frozen partitions')
        self._data = data
        train_ids = set(data.manifest['splits']['train']['event_ids'])
        for curve in data.train_curves():
            if curve.start < MIN_EVENT_START_MS or curve.event not in train_ids:
                raise ValueError('Training history violates the dense cohort')

    def __getattr__(self, name):
        return getattr(self._data, name)

    def inputs(self, split):
        for sample in self._data.inputs(split):
            if sample['curve'].start < MIN_EVENT_START_MS:
                raise ValueError('Prefix violates the actual-start cutoff')
            yield sample

    def labels(self, split, **kwargs):
        if split == 'test':
            raise PermissionError('Test labels are reserved for the independent scorer')
        return self._data.labels(split, **kwargs)


@contextmanager
def adapted_runner(stage):
    """Bind only dataset identity, source provenance and the two-CPU resource cap."""
    module = importlib.import_module(MODULES[stage])
    changes = []
    def replace(name, value):
        changes.append((name, getattr(module, name)))
        setattr(module, name, value)
    try:
        if stage == 'innovation':
            replace('DATASET_IDS', dict(module.DATASET_IDS, v2=DATASET_ID))
        if stage == 'next':
            replace('DATASET_ID', DATASET_ID)
        if hasattr(module, 'SOURCES'):
            replace('SOURCES', list(dict.fromkeys(module.SOURCES + ADAPTER_SOURCES)))
        if hasattr(module, 'local_sources'):
            original_sources = module.local_sources
            replace('local_sources', lambda paths: original_sources([*paths, *[ROOT / p for p in ADAPTER_SOURCES]]))
        if hasattr(module, 'sources_for'):
            original_sources_for = module.sources_for
            replace('sources_for', lambda modules: sorted(set(original_sources_for(modules)) | set(ADAPTER_SOURCES)))
        original_identity = module.identity
        def identity(config):
            config['execution_adapter'] = {
                'module': 'analysis.dense_v2', 'stage': stage, 'data_policy_id': POLICY_ID,
                'minimum_event_start_ms': MIN_EVENT_START_MS, 'cpu_threads': 2,
                'recipe_policy': 'all existing registered recipes; unchanged full budgets and selection rules',
                'reference_policy': 'only freshly retrained same-cohort sealed predictions',
                'test_labels': 'blocked in model facade; independent scorer only',
                'source_hashes': {name: sha256(ROOT / name) for name in ADAPTER_SOURCES}}
            return original_identity(config)
        replace('identity', identity)
        if hasattr(module, 'CatBoostRegressor'):
            original_tree = module.CatBoostRegressor
            def tree(*args, **kwargs):
                kwargs['thread_count'] = 2
                return original_tree(*args, **kwargs)
            replace('CatBoostRegressor', tree)
        yield module
    finally:
        for name, value in reversed(changes):
            setattr(module, name, value)


def run_stage(stage, run=None, reference=None, *, dataset=DENSE_V2_TWO_RANK,
              phase='all', resume=False):
    if stage not in STAGES:
        raise ValueError('Unknown dense V2 stage')
    if stage != 'next' and (phase != 'all' or resume):
        raise ValueError('Only the checkpointed next stage supports phases or resume')
    data = DenseTrainingData(TemporalDataset(dataset))
    run = Path(run) if run is not None else RUNS[stage]
    with adapted_runner(stage) as module:
        if stage == 'base':
            return module.run_training(data, run)
        if stage in ('paired', 'research'):
            parent = 'base' if stage == 'paired' else 'paired'
            return module.run_experiment(data, reference or RUNS[parent], run)
        if stage == 'innovation':
            return module.run_experiment(data, run, protocol='v2')
        return module.run_experiment(data, run, reference=reference or RUNS['research'],
                                     phase=phase, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--dataset', type=Path, default=DENSE_V2_TWO_RANK)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--phase', choices=['all', 'develop', 'blind'], default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_stage(args.stage, args.run, args.reference, dataset=args.dataset,
              phase=args.phase, resume=args.resume)


if __name__ == '__main__':
    main()
