"""Freeze a rank-only view of Temporal V2 without rebuilding or moving cases."""
from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import platform
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from analysis.backtest import ROOT
from analysis.holdout_dataset import SPLITS, sha256, write_json
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import DEFAULT as PARENT, TemporalDataset

DEFAULT = ROOT / 'data/temporal_holdout_v2_t1000_t2000'
SOURCES = ['training/analysis/temporal_rank_subset.py', 'training/analysis/temporal_dataset.py',
           'training/analysis/multirank_dataset.py', 'training/analysis/holdout_dataset.py', 'training/analysis/backtest.py']


def build_subset(parent, destination, *, ranks=(1000, 2000)):
    """Trusted data preparation only; the training entry point never calls this."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    ranks = tuple(ranks)
    if (not ranks or len(set(ranks)) != len(ranks)
            or not set(ranks).issubset(parent.manifest['protocol']['ranks'])):
        raise ValueError('Ranks must be unique nonempty members of the parent protocol')
    ranks = sorted(ranks)
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.temporal-ranks-', dir=destination.parent))
    try:
        shutil.copytree(parent.metadata_directory, work / 'metadata')
        shutil.copyfile(parent.root / 'manifest.json', work / 'parent_manifest.json')
        splits = {}
        for split in SPLITS:
            cases = parent.case_index(split)
            cases = cases[cases['rank'].isin(ranks)]
            ids = set(cases.case_id)
            if not ids:
                raise ValueError('No selected cases in split ' + split)
            members = parent.memberships(split)
            members = members[members.case_id.isin(ids)]
            cases.to_csv(work / f'{split}_cases.csv', index=False)
            members.to_csv(work / f'{split}_memberships.csv', index=False)
            labels = pd.read_csv(parent.root / f'{split}_labels.csv')
            labels = labels[labels.case_id.isin(ids)]
            if set(labels.case_id) != ids or labels.case_id.duplicated().any():
                raise ValueError('Parent labels do not match selected cases')
            labels.to_csv(work / f'{split}_labels.csv', index=False)
            seen = set()
            with gzip.open(parent.root / f'{split}_inputs.jsonl.gz', 'rt') as source, \
                 gzip.open(work / f'{split}_inputs.jsonl.gz', 'wt', compresslevel=1) as target:
                for line in source:
                    record = json.loads(line)
                    if record['rank'] not in ranks:
                        continue
                    case = record['case_id']
                    if case not in ids or case in seen:
                        raise ValueError('Parent prefixes do not match selected cases')
                    target.write(line)
                    seen.add(case)
            if seen != ids:
                raise ValueError('Missing selected prefix')
            events = parent.manifest['splits'][split]['event_ids']
            splits[split] = {'event_ids': events, 'cases': len(cases),
                'stage_memberships': len(members), 'events_with_cases': int(cases.event.nunique()),
                'events_without_cases': sorted(set(events) - set(cases.event))}
        with gzip.open(parent.root / 'train_curves.jsonl.gz', 'rt') as source, \
             gzip.open(work / 'train_curves.jsonl.gz', 'wt', compresslevel=1) as target:
            for line in source:
                if json.loads(line)['rank'] in ranks:
                    target.write(line)
        for name in ('partitions.csv', 'coverage.csv', 'unavailable_cases.csv'):
            frame = pd.read_csv(parent.root / name)
            frame[frame['rank'].isin(ranks)].to_csv(work / name, index=False)
        (work / 'builder_sources').mkdir()
        for name in SOURCES:
            shutil.copyfile(ROOT / name, work / 'builder_sources' / Path(name).name)
        manifest = {'protocol': parent.manifest['protocol'] | {
                'ranks': ranks, 'scope': 'Rank subset of Temporal V2; same groups, cases and time grid'},
            'parent_dataset_id': parent.manifest['dataset_id'],
            'parent_manifest_sha256': sha256(parent.root / 'manifest.json'),
            'base_dataset_id': parent.manifest['base_dataset_id'],
            'original_v1_dataset_id': parent.manifest['original_v1_dataset_id'],
            'source_identity': {'parent_directory': str(parent.root.resolve()),
                                'operation': 'rank filter only; no new split or curve extraction'},
            'python': platform.python_version(),
            'build_environment': {'source_hashes': {n: sha256(ROOT / n) for n in SOURCES},
                                  'software': {'pandas': importlib.metadata.version('pandas')}},
            'splits': splits,
            'artifacts': {str(p.relative_to(work)): sha256(p)
                          for p in sorted(work.rglob('*')) if p.is_file()}}
        manifest['dataset_id'] = identity(manifest)
        write_json(work / 'manifest.json', manifest)
        work.rename(destination)
        return manifest
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--destination', type=Path, default=DEFAULT)
    parser.add_argument('--ranks', type=int, nargs='+', default=[1000, 2000])
    args = parser.parse_args()
    result = build_subset(TemporalDataset(args.parent), args.destination, ranks=args.ranks)
    print(json.dumps({'dataset_id': result['dataset_id'], 'splits': result['splits']}))
