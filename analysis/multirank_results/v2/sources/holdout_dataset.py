"""Immutable event-grouped chronological benchmark; never follow the live DB.

Build once: python -m analysis.holdout_dataset
Only the evaluator may request test labels. Model inputs contain prefixes only.
"""
from __future__ import annotations

import argparse
import ast
import csv
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np

from analysis.backtest import ROOT, STAGES, Curve, snapshot

DEFAULT_DATASET = ROOT / 'data/holdout_v1'
SPLITS = ('train', 'validation', 'test')
HOUR = 3600000
PROTOCOL = {
    'version': 'holdout_v1', 'country': '中国大陆', 'ranks': [1000, 2000],
    'group': 'EventID (all ranks and snapshots together)',
    'order': 'real CN StartAt, then EventID; never shuffled',
    'fractions': [0.70, 0.15, 0.15], 'rounding': 'floor train, floor validation, remainder test',
    'minimum_curve_points': 5, 'final_before_end_max_ms': 60000,
    'final_after_end_max_ms': 300000, 'minimum_prefix_points': 4,
    'prefix_max_age_ms': HOUR, 'evaluation_stages': STAGES,
    'training_sampling': 'one observed prefix per 5% bin from 10% through 98%; closest to bin midpoint',
    'primary_metric': 'event_macro_mape_pct', 'training_seeds': [42, 123, 2026],
    'fit_policy': 'train only; validation selects checkpoints/recipes; no refit or heat update on validation/test',
    'exposure': 'Test events overlap previous exploratory analyses; not a pristine unseen cohort.',
    'metadata_caveat': 'Retrospective Bestdori snapshot; no archived point-in-time metadata proof. CN card release times are filtered.',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def curve_record(curve):
    return {'event': curve.event, 'rank': curve.rank, 'kind': curve.kind,
            'start': curve.start, 'end': curve.end,
            'times': curve.times.tolist(), 'scores': curve.scores.tolist()}


def from_record(record):
    return Curve(record['event'], record['rank'], record['kind'], record['start'], record['end'],
                 np.asarray(record['times'], dtype=np.int64), np.asarray(record['scores'], dtype=float))


def prefix_curve(curve, cutoff):
    stop = int(np.searchsorted(curve.times, cutoff, side='right'))
    if stop == 0:
        return None
    return Curve(curve.event, curve.rank, curve.kind, curve.start, curve.end,
                 curve.times[:stop], curve.scores[:stop])


def candidate_cutoffs(curve, split):
    if split != 'train':
        cutoffs = [curve.start + int(curve.duration * p) for p in [.25, .50, .75]]
        cutoffs += [curve.end - h * HOUR for h in [24, 6]]
        return list(zip(STAGES, cutoffs))
    # Observed timestamps retain sparse older curves; the last/future score
    # never determines which prefix is chosen.
    bins = {}
    for i, timestamp in enumerate(curve.times):
        progress = (int(timestamp) - curve.start) / curve.duration
        if i < 3 or not .10 <= progress <= .98:
            continue
        bucket = int(progress * 20)
        distance = abs(progress - (bucket + .5) / 20)
        if bucket not in bins or distance < bins[bucket][0]:
            bins[bucket] = (distance, int(timestamp))
    return [(f'observed_bin_{bucket:02}', value[1]) for bucket, value in sorted(bins.items())]


def build_dataset(destination=DEFAULT_DATASET, source_db=ROOT/'data/db.sqlite3',
                  metadata_directory=ROOT/'data/metadata', *, as_of_ms=None):
    destination, source_db, metadata_directory = map(Path, (destination, source_db, metadata_directory))
    if destination.exists():
        raise FileExistsError(f'Frozen dataset already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    as_of_ms = int(datetime.now(timezone.utc).timestamp() * 1000) if as_of_ms is None else int(as_of_ms)
    work = Path(tempfile.mkdtemp(prefix='.holdout-build-', dir=destination.parent))
    try:
        # SQLite backup reads a consistent database even if its source uses WAL.
        with closing(sqlite3.connect(source_db.resolve().as_uri()+'?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(work/'snapshot.sqlite3')) as target:
                source.backup(target)
        (work/'metadata').mkdir()
        for name in ('events', 'characters', 'bands', 'cards'):
            shutil.copyfile(metadata_directory/f'{name}.json', work/f'metadata/{name}.json')
        for name in ('sources', 'event_contexts'):
            if (metadata_directory/f'{name}.json').exists():
                shutil.copyfile(metadata_directory/f'{name}.json', work/f'metadata/{name}.json')
        eligible, all_curves, excluded = [], {}, []
        with closing(sqlite3.connect((work/'snapshot.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM event WHERE Country='中国大陆' AND Rank IN (1000,2000) ORDER BY StartAt,EventID,Rank").fetchall()
        seen = set()
        for row in rows:
            event, rank = int(row['EventID']), int(row['Rank'])
            if (event, rank) in seen:
                raise ValueError(f'Duplicate event/rank grain: {event}/{rank}')
            seen.add((event, rank)); reason = None
            try:
                points = ast.literal_eval(row['PointRank'])
                times = np.array([p['time'] for p in points], dtype=np.int64)
                scores = np.array([p['ep'] for p in points], dtype=float)
                curve = Curve(event, rank, row['EventType'], int(row['StartAt']), int(row['EndAt']), times, scores)
                if curve.duration <= 0 or not len(times) or np.any(np.diff(times) <= 0) or np.any(~np.isfinite(scores)) or np.any(scores < 0):
                    reason = 'invalid timestamps, duration or scores'
                else:
                    all_curves[(event, rank)] = curve
                    if curve.end >= as_of_ms: reason = 'ongoing'
                    elif len(times) < 5: reason = 'fewer than 5 observations'
                    elif scores[-1] <= 0: reason = 'nonpositive terminal score'
                    elif not -300000 <= curve.end - int(times[-1]) <= 60000:
                        reason = 'terminal observation outside [-300s,+60s] end-minus-observation'
            except (ValueError, TypeError, KeyError, SyntaxError, OverflowError):
                reason = 'unparseable observation data'
            if reason:
                excluded.append({'event': event, 'rank': rank, 'reason': reason})
            else:
                eligible.append(curve)
        starts = {c.event: c.start for c in eligible}
        event_ids = sorted(starts, key=lambda e: (starts[e], e))
        ntrain, nval = int(len(event_ids)*.70), int(len(event_ids)*.15)
        if ntrain < 1 or nval < 1 or len(event_ids)-ntrain-nval < 1:
            raise ValueError('Too few eligible event groups for a three-way holdout')
        groups = dict(zip(SPLITS, (event_ids[:ntrain], event_ids[ntrain:ntrain+nval], event_ids[ntrain+nval:])))
        partition, unavailable, manifest_splits = [], [], {}
        for split, ids in groups.items():
            curves = [c for c in eligible if c.event in set(ids)]
            samples, labels = [], []
            for curve in curves:
                partition.append({'split': split, 'event': curve.event, 'rank': curve.rank,
                                  'start_ms': curve.start, 'end_ms': curve.end, 'observations': len(curve.times),
                                  'score_decreases': int(np.sum(np.diff(curve.scores) < 0))})
                for stage, cutoff in candidate_cutoffs(curve, split):
                    sample = snapshot(curve, cutoff)
                    case_id = f'{split}:{curve.event}:{curve.rank}:{stage}'
                    if sample is None:
                        unavailable.append({'split': split, 'event': curve.event, 'rank': curve.rank,
                                            'stage': stage, 'cutoff_ms': cutoff, 'reason': 'prefix needs >=4 points, positive score, <=60min age and elapsed>0'})
                        continue
                    prefix = prefix_curve(curve, cutoff)
                    peer = all_curves.get((curve.event, 2000 if curve.rank == 1000 else 1000))
                    peer = prefix_curve(peer, sample['now']) if peer is not None else None
                    record = curve_record(prefix) | {'case_id': case_id, 'stage': stage, 'cutoff': cutoff,
                                                    'peer': curve_record(peer) if peer is not None else None}
                    samples.append(record)
                    labels.append({'case_id': case_id, 'actual': curve.final})
            with (work/f'{split}_inputs.jsonl').open('w') as f:
                for record in samples: f.write(json.dumps(record, separators=(',', ':'))+'\n')
            write_csv(work/f'{split}_labels.csv', labels, ['case_id', 'actual'])
            manifest_splits[split] = {'event_ids': ids, 'event_count': len(ids), 'curves': len(curves),
                'cases': len(samples), 'events_with_cases': len({s['event'] for s in samples}),
                'earliest_start_ms': min(c.start for c in curves), 'latest_end_ms': max(c.end for c in curves)}
            if split == 'train':
                with (work/'train_curves.jsonl').open('w') as f:
                    for curve in curves: f.write(json.dumps(curve_record(curve), separators=(',', ':'))+'\n')
        for left, right in zip(SPLITS, SPLITS[1:]):
            if manifest_splits[left]['latest_end_ms'] >= manifest_splits[right]['earliest_start_ms']:
                raise ValueError('Chronological groups overlap in time; require a versioned purged split')
        write_csv(work/'partitions.csv', partition, ['split','event','rank','start_ms','end_ms','observations','score_decreases'])
        write_csv(work/'excluded_curves.csv', excluded, ['event','rank','reason'])
        write_csv(work/'unavailable_cases.csv', unavailable, ['split','event','rank','stage','cutoff_ms','reason'])
        hashes = {str(p.relative_to(work)): sha256(p) for p in sorted(work.rglob('*')) if p.is_file()}
        manifest = {'protocol': PROTOCOL, 'as_of_ms': as_of_ms,
                    'source_db_sha256': sha256(source_db), 'source_rows': len(rows),
                    'eligible_events': len(event_ids), 'eligible_curves': len(eligible),
                    'excluded_curves': len(excluded), 'unavailable_cases': len(unavailable),
                    'splits': manifest_splits, 'artifacts': hashes}
        manifest['dataset_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        write_json(work/'manifest.json', manifest)
        work.rename(destination)
        return manifest
    finally:
        if work.exists(): shutil.rmtree(work)


class HoldoutDataset:
    def __init__(self, root=DEFAULT_DATASET):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        identity = {k:v for k,v in self.manifest.items() if k != 'dataset_id'}
        expected = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if expected != self.manifest['dataset_id']:
            raise ValueError('Dataset manifest hash 校验失败')
        for name, digest in self.manifest['artifacts'].items():
            if sha256(self.root/name) != digest:
                raise ValueError(f'Dataset artifact hash 校验失败: {name}')

    @property
    def metadata_directory(self):
        return self.root/'metadata'

    def inputs(self, split):
        if split not in SPLITS: raise ValueError(split)
        values = []
        with (self.root/f'{split}_inputs.jsonl').open() as f:
            for line in f:
                record = json.loads(line); curve = from_record(record)
                value = snapshot(curve, record['cutoff'])
                if value is None: raise ValueError(f'Frozen case invalid: {record["case_id"]}')
                value.update({'case_id': record['case_id'], 'stage': record['stage'],
                              'peer_curve': from_record(record['peer']) if record['peer'] is not None else None})
                values.append(value)
        return values

    def labels(self, split, *, allow_test=False):
        if split not in SPLITS: raise ValueError(split)
        if split == 'test' and not allow_test:
            raise ValueError('test labels are restricted to final evaluation')
        with (self.root/f'{split}_labels.csv').open() as f:
            return {r['case_id']: float(r['actual']) for r in csv.DictReader(f)}

    def train_curves(self):
        with (self.root/'train_curves.jsonl').open() as f:
            return [from_record(json.loads(line)) for line in f]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    result = build_dataset(args.destination)
    print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
