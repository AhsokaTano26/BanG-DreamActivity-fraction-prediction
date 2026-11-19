"""Trusted preparation of dense-era V1/V2; old datasets remain immutable.

This module may read frozen source curves and labels to WRITE datasets. Model
entry points must use the ordinary blind-prefix loaders, never this module.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, closing
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
import pandas as pd

from analysis.backtest import ROOT, Curve
from analysis.data_policy import MIN_EVENT_START_MS, MIN_EVENT_START_ISO, POLICY, POLICY_ID
from analysis.holdout_dataset import (HoldoutDataset, SPLITS, build_dataset, curve_record,
                                     sha256, write_json)
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import MultirankDataset, identity
from analysis.temporal_dataset import TemporalDataset, trusted_curves
from analysis.temporal_rank_subset import build_subset

V1 = ROOT/'data/holdout_v1_dense_20261115'
V2 = ROOT/'data/temporal_holdout_v2_dense_20261115'
V2_TWO = ROOT/'data/temporal_holdout_v2_dense_t1000_t2000_20261115'


def _new_work(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='.dense-prepare-', dir=destination.parent))


def _cohort(base):
    parts = pd.read_csv(base.root/'partitions.csv')
    if parts.duplicated(['event', 'rank']).any():
        raise ValueError('Duplicate frozen curve cohort')
    events = parts[['event', 'start_ms', 'end_ms']].drop_duplicates()
    if events.event.duplicated().any():
        raise ValueError('Ranks disagree on actual event dates')
    events = events.sort_values(['start_ms', 'event'])
    kept = events[events.start_ms >= MIN_EVENT_START_MS]
    ntrain, nval = int(len(kept)*.70), int(len(kept)*.15)
    if min(ntrain, nval, len(kept)-ntrain-nval) < 1:
        raise ValueError('Too few dense events for chronological three-way split')
    blocks = [kept.iloc[:ntrain], kept.iloc[ntrain:ntrain+nval], kept.iloc[ntrain+nval:]]
    for left, right in zip(blocks, blocks[1:]):
        if left.end_ms.max() >= right.start_ms.min():
            raise ValueError('Chronological event groups overlap; purging requires a new protocol')
    groups = {s: [int(e) for e in block.event] for s, block in zip(SPLITS, blocks)}
    membership = {e: s for s, ids in groups.items() for e in ids}
    old = {int(e): s for s in SPLITS for e in base.manifest['splits'][s]['event_ids']}
    events['original_split'] = events.event.map(old)
    events['new_split'] = events.event.map(membership).fillna('excluded_before_dense_start')
    return groups, parts[parts.event.isin(membership)].copy(), events


def _metadata(source, target, events):
    shutil.copytree(source, target)
    allowed = {str(e) for e in events}
    path = target/'events.json'
    values = json.loads(path.read_text())
    write_json(path, {k: v for k, v in values.items() if k in allowed})
    path = target/'event_contexts.json'
    if path.exists():
        values = json.loads(path.read_text())
        if isinstance(values, list):
            values = [v for v in values if str(v['event_id']) in allowed]
        else:
            values = {k: v for k, v in values.items() if k in allowed}
        write_json(path, values)


def _seal(work, manifest):
    sources = local_sources([Path(__file__)])
    archive = work/'dense_builder_sources'; archive.mkdir()
    for name in sources:
        target = archive/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    result = {k: v for k, v in manifest.items() if k not in ('dataset_id', 'artifacts')}
    result['dense_build_environment'] = {
        'source_hashes': {n: sha256(ROOT/n) for n in sources},
        'software': {n: importlib.metadata.version(n) for n in ('numpy', 'pandas')},
    }
    result['artifacts'] = {str(p.relative_to(work)): sha256(p) for p in sorted(work.rglob('*'))
                           if p.is_file() and p != work/'manifest.json'}
    result['dataset_id'] = identity(result)
    write_json(work/'manifest.json', result)
    return result


def _protocol(original, version):
    return original | {
        'version': version, 'population_version': 'dense_20261115',
        'minimum_event_start_ms': MIN_EVENT_START_MS,
        'minimum_event_start_iso': MIN_EVENT_START_ISO,
        'data_policy_id': POLICY_ID,
        'collection_context': 'User reports earlier data were manually submitted; excluded by actual event start.',
        'scope': 'New dense-era cohort; original V1/V2 retained as historical references',
        'group': 'One shared chronological 70%/15%/remainder event split; all ranks/prefixes grouped',
        'exposure': 'Retained events overlap previously explored training/validation/test cohorts; not unseen data.',
    }


def _prefix_digest(times, scores):
    value = hashlib.sha256(np.asarray(times, dtype='<i8').tobytes())
    value.update(np.asarray(scores, dtype='<f8').tobytes())
    return value.hexdigest()


def build_dense_v1(original, destination):
    """Rebuild V1 sampling on the cutoff-filtered FROZEN source, not live DB."""
    destination = Path(destination); work = _new_work(destination)
    try:
        groups, expected, migration = _cohort(original)
        events = [e for ids in groups.values() for e in ids]
        filtered = work/'filtered.sqlite3'
        with closing(sqlite3.connect((original.root/'snapshot.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(filtered)) as target:
                source.backup(target)
                target.execute('CREATE TEMP TABLE kept_events(event INTEGER PRIMARY KEY)')
                target.executemany('INSERT INTO kept_events VALUES (?)', [(e,) for e in events])
                target.execute("DELETE FROM event WHERE Country IS NOT '中国大陆' OR Rank IS NULL OR Rank NOT IN (1000,2000) "
                               'OR StartAt IS NULL OR StartAt < ? OR EventID NOT IN (SELECT event FROM kept_events)',
                               (MIN_EVENT_START_MS,))
                target.commit(); target.execute('VACUUM')
        _metadata(original.metadata_directory, work/'metadata', events)
        built = work/'dataset'
        manifest = build_dataset(built, filtered, work/'metadata', as_of_ms=original.manifest['as_of_ms'])
        actual = pd.read_csv(built/'partitions.csv')
        fields = ['event','rank','start_ms','end_ms','observations','score_decreases']
        a = actual[fields].sort_values(['event','rank']).reset_index(drop=True)
        b = expected[fields].sort_values(['event','rank']).reset_index(drop=True)
        if not a.equals(b) or any(manifest['splits'][s]['event_ids'] != groups[s] for s in SPLITS):
            raise ValueError('Rebuilt V1 differs from the original eligible dense cohort')
        migration.to_csv(built/'cohort_migration.csv', index=False)
        shutil.copyfile(original.root/'manifest.json', built/'original_manifest.json')
        manifest['protocol'] = _protocol(manifest['protocol'], 'holdout_v1_dense_20261115')
        manifest['data_policy'] = POLICY
        manifest['source_identity'] = {
            'original_dataset_id': original.manifest['dataset_id'],
            'original_manifest_sha256': sha256(original.root/'manifest.json'),
            'original_snapshot_sha256': sha256(original.root/'snapshot.sqlite3'),
            'operation': 'actual-start cutoff; preserve original eligible curves; chronological regroup',
        }
        manifest['removed_events'] = int((migration.new_split == 'excluded_before_dense_start').sum())
        result = _seal(built, manifest)
        HoldoutDataset(built)
        built.rename(destination)
        return result
    finally:
        shutil.rmtree(work)


def snapshot_curves(data):
    """Trusted builder-only reading of eligible two-rank complete curves."""
    wanted = {(int(r.event), int(r['rank'])) for _, r in pd.read_csv(data.root/'partitions.csv').iterrows()}
    with closing(sqlite3.connect((data.root/'snapshot.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute("SELECT * FROM event WHERE Country='中国大陆' ORDER BY StartAt,EventID,Rank"):
            if (row['EventID'], row['Rank']) not in wanted:
                continue
            points = ast.literal_eval(row['PointRank'])
            yield Curve(row['EventID'], row['Rank'], row['EventType'], row['StartAt'], row['EndAt'],
                        np.array([p['time'] for p in points], dtype=np.int64),
                        np.array([p['ep'] for p in points], dtype=float))


def build_dense_temporal(parent, dense_v1, destination, curves, *, source_identity):
    """Keep frozen V2 origins/availability exactly; only filter/regroup events."""
    destination = Path(destination); work = _new_work(destination)
    try:
        if parent.manifest['original_v1_dataset_id'] != dense_v1.manifest['source_identity']['original_dataset_id']:
            raise ValueError('V2 parent and dense V1 have different original cohorts')
        groups = {s: dense_v1.manifest['splits'][s]['event_ids'] for s in SPLITS}
        membership = {e: s for s, events in groups.items() for e in events}
        partitions = pd.read_csv(parent.root/'partitions.csv')
        partitions = partitions[partitions.event.isin(membership)].copy()
        if (partitions.start_ms < MIN_EVENT_START_MS).any() or partitions.duplicated(['event','rank']).any():
            raise ValueError('Invalid dense temporal cohort')
        partitions['split'] = partitions.event.map(membership)
        expected = {(int(r.event), int(r['rank'])): r for _, r in partitions.iterrows()}
        v1parts = pd.read_csv(dense_v1.root/'partitions.csv')
        two = partitions[partitions['rank'].isin([1000,2000])]
        cols = ['split','event','rank','start_ms','end_ms','observations']
        if not two[cols].sort_values(['event','rank']).reset_index(drop=True).equals(
                v1parts[cols].sort_values(['event','rank']).reset_index(drop=True)):
            raise ValueError('Dense V1/V2 two-rank cohorts differ')
        _metadata(parent.metadata_directory, work/'metadata', membership)
        partitions.to_csv(work/'partitions.csv', index=False)
        shutil.copyfile(dense_v1.root/'cohort_migration.csv', work/'cohort_migration.csv')
        shutil.copyfile(parent.root/'manifest.json', work/'original_manifest.json')
        frames = {s: [] for s in SPLITS}; members = {s: [] for s in SPLITS}; answers = {s: [] for s in SPLITS}
        remap = {}; target_split = {}
        for old_split in SPLITS:
            old_cases = parent.case_index(old_split)
            selected = old_cases[old_cases.event.isin(membership)].copy()
            for row in selected.itertuples():
                new_split = membership[row.event]
                remap[row.case_id] = new_split+':'+row.case_id.split(':',1)[1]
                target_split[row.case_id] = new_split
            old_members = parent.memberships(old_split)
            # Trusted data preparation: answers are copied, never used to choose groups/cases.
            old_labels = pd.read_csv(parent.root/f'{old_split}_labels.csv')
            if old_labels.case_id.duplicated().any() or set(old_labels.case_id) != set(old_cases.case_id):
                raise ValueError('Parent labels differ from frozen cases')
            for split in SPLITS:
                part = selected[selected.event.isin(groups[split])].copy(); ids = set(part.case_id)
                mem = old_members[old_members.case_id.isin(ids)].copy()
                lab = old_labels[old_labels.case_id.isin(ids)].copy()
                for table in (part,mem,lab): table['case_id'] = table.case_id.map(remap)
                frames[split].append(part); members[split].append(mem); answers[split].append(lab)
        seen = set(); prefix_checks = {}
        with ExitStack() as stack:
            outputs = {s: stack.enter_context(gzip.open(work/f'{s}_inputs.jsonl.gz','wt',compresslevel=1)) for s in SPLITS}
            for old_split in SPLITS:
                with gzip.open(parent.root/f'{old_split}_inputs.jsonl.gz','rt') as source:
                    for line in source:
                        record = json.loads(line); old_id = record['case_id']
                        if old_id not in remap: continue
                        if old_id in seen or record['start'] < MIN_EVENT_START_MS:
                            raise ValueError('Duplicate or pre-cutoff prefix')
                        key = (record['event'],record['rank'])
                        prefix_checks.setdefault(key,[]).append((record['cutoff'],record['kind'],
                            _prefix_digest(record['times'],record['scores'])))
                        seen.add(old_id); record['case_id'] = remap[old_id]
                        outputs[target_split[old_id]].write(json.dumps(record,separators=(',',':'))+'\n')
        if seen != set(remap): raise ValueError('Missing frozen temporal prefix')
        curve_seen = set(); finals = {}
        with gzip.open(work/'train_curves.jsonl.gz','wt',compresslevel=1) as history:
            for c in curves:
                if c.event not in membership: continue
                key = (c.event,c.rank)
                if key not in expected or key in curve_seen: raise ValueError('Unexpected/duplicate curve cohort member')
                row = expected[key]
                if (c.start != int(row.start_ms) or c.end != int(row.end_ms) or len(c.times) != int(row.observations)
                    or len(c.scores) != len(c.times) or c.start < MIN_EVENT_START_MS or np.any(np.diff(c.times)<=0)
                    or not np.isfinite(c.scores).all() or np.any(c.scores<0) or c.final<=0):
                    raise ValueError('Complete history curve differs from frozen cohort')
                for cutoff, kind, digest in prefix_checks.get(key,[]):
                    stop = np.searchsorted(c.times,cutoff,side='right')
                    if c.kind != kind or _prefix_digest(c.times[:stop],c.scores[:stop]) != digest:
                        raise ValueError('Complete history curve differs from frozen prefix')
                curve_seen.add(key); finals[key] = c.final
                if membership[c.event] == 'train':
                    history.write(json.dumps(curve_record(c),separators=(',',':'))+'\n')
        if curve_seen != set(expected): raise ValueError('Missing complete cohort history curves')
        split_stats = {}; coverage = []
        for split in SPLITS:
            cases = pd.concat(frames[split],ignore_index=True)
            aliases = pd.concat(members[split],ignore_index=True)
            labels = pd.concat(answers[split],ignore_index=True)
            if cases.empty or cases.case_id.duplicated().any() or set(aliases.case_id) != set(cases.case_id):
                raise ValueError('Incomplete or duplicate regrouped temporal cases')
            check = cases.merge(labels,on='case_id',validate='one_to_one')
            expected_finals = np.array([finals[(r.event,r['rank'])] for _,r in check.iterrows()])
            if not np.allclose(check.actual,expected_finals,rtol=0,atol=1e-8):
                raise ValueError('Complete curve outcomes differ from frozen labels')
            cases.to_csv(work/f'{split}_cases.csv',index=False)
            aliases.to_csv(work/f'{split}_memberships.csv',index=False)
            labels.to_csv(work/f'{split}_labels.csv',index=False)
            split_stats[split] = {'event_ids':groups[split],'cases':len(cases),'stage_memberships':len(aliases),
                'events_with_cases':int(cases.event.nunique()),'events_without_cases':sorted(set(groups[split])-set(cases.event))}
            joined = aliases.merge(cases[['case_id','rank']],on='case_id',validate='many_to_one')
            for rank in parent.manifest['protocol']['ranks']:
                eligible = int(((partitions['split']==split)&(partitions['rank']==rank)).sum())
                for stage in parent.manifest['protocol']['stages']:
                    n = int(((joined['rank']==rank)&(joined.stage_id==stage['stage_id'])).sum())
                    coverage.append({'split':split,'rank':rank,**stage,'group_events':len(groups[split]),
                        'eligible_curves':eligible,'missing_curves':len(groups[split])-eligible,
                        'available_cases':n,'missing_prefixes':eligible-n})
        pd.DataFrame(coverage).to_csv(work/'coverage.csv',index=False)
        unavailable = pd.read_csv(parent.root/'unavailable_cases.csv')
        unavailable = unavailable[unavailable.event.isin(membership)].copy()
        unavailable['split'] = unavailable.event.map(membership)
        unavailable.to_csv(work/'unavailable_cases.csv',index=False)
        protocol = _protocol(parent.manifest['protocol'],'temporal_holdout_v2_dense_20261115')
        protocol['first_experiment'] = 'New cohort requires training from its own train group; old model predictions are not transferable'
        manifest = {'protocol':protocol,'data_policy':POLICY,'base_dataset_id':dense_v1.manifest['dataset_id'],
            'original_v1_dataset_id':dense_v1.manifest['source_identity']['original_dataset_id'],
            'source_identity':source_identity|{'parent_dataset_id':parent.manifest['dataset_id'],
                'parent_manifest_sha256':sha256(parent.root/'manifest.json'),
                'dense_v1_manifest_sha256':sha256(dense_v1.root/'manifest.json'),
                'operation':'filter/regroup frozen V2 cases; time grid and availability unchanged'},
            'splits':split_stats}
        result = _seal(work,manifest); TemporalDataset(work)
        work.rename(destination)
        return result
    finally:
        if work.exists(): shutil.rmtree(work)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1',type=Path,default=V1);p.add_argument('--v2',type=Path,default=V2)
    p.add_argument('--two-rank',type=Path,default=V2_TWO)
    args=p.parse_args()
    for destination in (args.v1,args.v2,args.two_rank):
        if destination.exists(): raise FileExistsError(destination)
    original=HoldoutDataset();parent=TemporalDataset()
    base=MultirankDataset(ROOT/'data/multirank_v2');archive=ROOT/'data/downloads/cn_19_ranks_v1'
    if parent.manifest['base_dataset_id'] != base.manifest['dataset_id']:
        raise ValueError('Temporal parent has a different frozen multirank source')
    first=build_dense_v1(original,args.v1);dense=HoldoutDataset(args.v1)
    second=build_dense_temporal(parent,dense,args.v2,trusted_curves(base,original,archive),source_identity={
        'original_snapshot_sha256':sha256(original.root/'snapshot.sqlite3'),
        'download_manifest_sha256':sha256(archive/'download_manifest.json')})
    third=build_subset(TemporalDataset(args.v2),args.two_rank)
    print(json.dumps({'v1':first['dataset_id'],'v2':second['dataset_id'],'v2_two_rank':third['dataset_id'],
                      'splits':third['splits']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
