"""Trusted V3 preparation and strictly completed-before-start history release.

The history vault is only for the trusted coordinator. Model functions receive
released Curve copies and blind arrays, never this object or a filesystem path.
V1/V2 data and their train-only access contracts are unchanged.
"""
from __future__ import annotations

import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd

from analysis.backtest import ROOT
from analysis.data_policy import MIN_EVENT_START_MS, MIN_EVENT_START_ISO
from analysis.holdout_dataset import SPLITS, curve_record, from_record, sha256, write_json
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import TemporalDataset

DEFAULT = ROOT/'data/rolling_holdout_v3_dense_t1000_t2000_20261115'


def _cohort(parent):
    partitions = pd.read_csv(parent.root/'partitions.csv')
    if partitions[['event', 'rank']].duplicated().any():
        raise ValueError('Duplicate parent cohort curve')
    groups = {}
    for split in SPLITS:
        for event in parent.manifest['splits'][split]['event_ids']:
            if int(event) in groups:
                raise ValueError('Overlapping parent event groups')
            groups[int(event)] = split
    events = {}
    for row in partitions.to_dict('records'):
        event, rank = int(row['event']), int(row['rank'])
        if rank not in (1000, 2000) or groups.get(event) != row['split']:
            raise ValueError('Parent cohort requires two ranks and consistent event groups')
        start, end = int(row['start_ms']), int(row['end_ms'])
        if start < MIN_EVENT_START_MS or end <= start:
            raise ValueError('Parent cohort violates minimum start/date policy')
        previous = events.setdefault(event, {'event': event, 'split': row['split'],
            'start_ms': start, 'end_ms': end, 'ranks': []})
        if (previous['start_ms'], previous['end_ms']) != (start, end):
            raise ValueError('Parent ranks disagree on event dates')
        previous['ranks'].append(rank)
    if set(events) != set(groups):
        raise ValueError('Parent groups and eligible curve cohort differ')
    ordered = sorted(events.values(), key=lambda r: (r['start_ms'], r['event']))
    split_order = [SPLITS.index(r['split']) for r in ordered]
    if split_order != sorted(split_order):
        raise ValueError('Parent groups must be chronological by actual start')
    return partitions, ordered


def build_rolling(parent, destination, curves, *, source_identity):
    """Copy exact V2 cases; add complete, independently guarded event files."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    partitions, events = _cohort(parent)
    expected = {(int(r['event']), int(r['rank'])): r for r in partitions.to_dict('records')}
    saved = {}
    for c in curves:
        key = (c.event, c.rank)
        if key not in expected or key in saved:
            raise ValueError('Unexpected or duplicate history cohort curve')
        row = expected[key]
        if (c.start < MIN_EVENT_START_MS or c.start != int(row['start_ms'])
                or c.end != int(row['end_ms']) or len(c.times) != int(row['observations'])):
            raise ValueError('History curve dates/count differ from frozen cohort')
        if (c.end <= c.start or len(c.times) < 4 or len(c.scores) != len(c.times)
                or np.any(np.diff(c.times) <= 0) or not np.isfinite(c.scores).all()
                or np.any(c.scores < 0) or c.final <= 0
                or not -300000 <= c.end-int(c.times[-1]) <= 60000):
            raise ValueError('Invalid complete historical curve')
        saved[key] = c
    if set(saved) != set(expected):
        raise ValueError('Missing history cohort curves')
    # Preparation is trusted: it verifies the vault agrees with every frozen
    # answer and prefix. This is never invoked by the prediction coordinator.
    for split in SPLITS:
        labels = parent.labels(split, allow_test=split == 'test')
        label_rows = pd.read_csv(parent.root/f'{split}_labels.csv')
        cases = parent.case_index(split)
        if (label_rows.case_id.duplicated().any() or cases.case_id.duplicated().any()
                or set(labels) != set(cases.case_id)):
            raise ValueError('Duplicate or mismatched frozen case/label IDs')
        case_lookup = cases.set_index('case_id').to_dict('index')
        seen = set()
        for sample in parent.inputs(split):
            prefix = sample['curve']; c = saved[(prefix.event, prefix.rank)]
            case = case_lookup.get(sample['case_id'])
            if (case is None or any(int(case[key]) != value for key, value in
                    [('event', c.event), ('rank', c.rank), ('start_ms', c.start),
                     ('end_ms', c.end), ('cutoff', sample['cutoff'])])
                    or (prefix.start, prefix.end) != (c.start, c.end)
                    or expected[(c.event, c.rank)]['split'] != split
                    or sample['case_id'] in seen):
                raise ValueError('Frozen case identity disagrees with blind prefix')
            stop = np.searchsorted(c.times, sample['cutoff'], side='right')
            if (prefix.kind != c.kind or not np.array_equal(c.times[:stop], prefix.times)
                    or not np.array_equal(c.scores[:stop], prefix.scores)
                    or not np.isclose(c.final, labels[sample['case_id']], rtol=1e-12, atol=0)):
                raise ValueError('Vault curve disagrees with frozen prefix or outcome')
            seen.add(sample['case_id'])
        if (cases.case_id.duplicated().any() or seen != set(cases.case_id) or seen != set(labels)
                or not set(cases.event).issubset(set(parent.manifest['splits'][split]['event_ids']))):
            raise ValueError('Frozen case/input/label cohort mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.rolling-v3-', dir=destination.parent))
    try:
        shutil.copytree(parent.root, work, dirs_exist_ok=True)
        (work/'manifest.json').unlink()
        shutil.copyfile(parent.root/'manifest.json', work/'rolling_parent_manifest.json')
        (work/'history_vault').mkdir()
        for row in events:
            name = f"history_vault/{row['event']}.jsonl.gz"
            with gzip.open(work/name, 'wt', compresslevel=1) as stream:
                for rank in sorted(row['ranks']):
                    stream.write(json.dumps(curve_record(saved[(row['event'], rank)]), separators=(',', ':'))+'\n')
            row['file'] = name
            row['sha256'] = sha256(work/name)
            row['ranks'] = sorted(row['ranks'])
        (work/'rolling_builder_sources').mkdir()
        sources = ['rolling_dataset.py', 'data_policy.py', 'temporal_dataset.py',
                   'multirank_dataset.py', 'holdout_dataset.py', 'backtest.py', 'event_features.py']
        for name in sources:
            shutil.copyfile(ROOT/'analysis'/name, work/'rolling_builder_sources'/name)
        manifest = {'protocol': parent.manifest['protocol'] | {
                'version': 'rolling_holdout_v3', 'ranks': [1000, 2000],
                'minimum_event_start_ms': MIN_EVENT_START_MS,
                'minimum_event_start_iso': MIN_EVENT_START_ISO,
                'scope': 'Same dense V2 cases; expanding completed-event history at event start',
                'history_boundary': 'start >= minimum and end strictly < current event start',
                'feedback': 'Earlier validation/test finals permitted only after their prediction seal and completion',
                'training_evaluation': 'Initial training group is warmup; validation/test only are scored'},
            'parent_dataset_id': parent.manifest['dataset_id'],
            'parent_manifest_sha256': sha256(parent.root/'manifest.json'),
            'source_identity': source_identity, 'splits': parent.manifest['splits'],
            'python': platform.python_version(),
            'build_environment': {'source_hashes': {'analysis/'+name: sha256(ROOT/'analysis'/name) for name in sources},
                'software': {name: importlib.metadata.version(name) for name in
                             ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'httpx']}},
            'history_index': events,
            'artifacts': {str(p.relative_to(work)): sha256(p) for p in sorted(work.rglob('*')) if p.is_file()}}
        manifest['dataset_id'] = identity(manifest)
        write_json(work/'manifest.json', manifest)
        work.rename(destination)
        return manifest
    finally:
        if work.exists(): shutil.rmtree(work)


class RollingDataset(TemporalDataset):
    def __init__(self, root=DEFAULT):
        super().__init__(root)
        if self.manifest['protocol'].get('version') != 'rolling_holdout_v3':
            raise ValueError('Expected rolling V3 dataset')
        _, expected = _cohort(self)
        rows = self.manifest['history_index']
        self.event_index = {int(row['event']): dict(row) for row in rows}
        if len(self.event_index) != len(rows) or set(self.event_index) != {r['event'] for r in expected}:
            raise ValueError('History release index differs from cohort')
        for row in expected:
            actual = self.event_index[row['event']]
            if any(actual[k] != row[k] for k in ['event', 'split', 'start_ms', 'end_ms']):
                raise ValueError('History release timing differs from cohort')
            if sorted(actual['ranks']) != sorted(row['ranks']):
                raise ValueError('History release ranks differ from cohort')
            if actual['sha256'] != self.manifest['artifacts'].get(actual['file']):
                raise ValueError('History release file lacks frozen identity')

    def reveal_event(self, event, *, before_ms):
        """Trusted release: no file is decoded until the strict guard passes."""
        if not np.isfinite(before_ms) or int(before_ms) != before_ms:
            raise ValueError('A finite integer release boundary is required')
        row = self.event_index[int(event)]
        if row['start_ms'] < MIN_EVENT_START_MS or row['end_ms'] >= before_ms:
            raise ValueError('Event is not completed before the release boundary')
        path = self.root/row['file']
        if sha256(path) != row['sha256']:
            raise ValueError('Sealed history artifact changed')
        with gzip.open(path, 'rt') as stream:
            curves = [from_record(json.loads(line)) for line in stream]
        if (sorted(c.rank for c in curves) != row['ranks'] or any(
                c.event != event or c.start != row['start_ms'] or c.end != row['end_ms'] for c in curves)):
            raise ValueError('Released history content differs from guarded index')
        return curves

    def history_before(self, start_ms):
        result = []
        for row in sorted(self.event_index.values(), key=lambda r: (r['end_ms'], r['event'])):
            if row['end_ms'] < start_ms:
                result.extend(self.reveal_event(row['event'], before_ms=start_ms))
        return result
