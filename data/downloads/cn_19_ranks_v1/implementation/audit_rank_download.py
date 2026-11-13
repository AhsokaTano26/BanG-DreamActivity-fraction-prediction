"""Verify downloaded raw cutoff histories and their exact SQLite merge."""
import argparse
import ast
from collections import Counter,defaultdict
from contextlib import closing
import csv
from datetime import datetime,timedelta,timezone
import gzip
import heapq
import json
from pathlib import Path
import shutil
import sqlite3
from download_ranks import COUNTRY,RANKS,sha256,write_json,verify_download,latest_records,database_identity,check_database_path
from analysis.holdout_dataset import HoldoutDataset

EXPECTED_HOLDOUT='d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb'
TZ=timezone(timedelta(hours=8))


def read_rows(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)) as con:
        con.row_factory=sqlite3.Row;con.execute('BEGIN')
        if con.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity failure')
        rows={}
        for row in con.execute('SELECT * FROM event'):
            key=(row['EventID'],row['Rank'],row['Country'])
            if key in rows:raise ValueError('Duplicate logical key')
            rows[key]=dict(row)
    return rows


def independent_raw_curve(payload,rank):
    if rank==10:
        groups=defaultdict(dict)
        for p in payload['points']:
            t,uid,value=int(p['time']),int(p['uid']),int(p['value'])
            if uid in groups[t] and groups[t][uid]!=value:raise ValueError('Ambiguous top snapshot')
            groups[t][uid]=value
        return {t:heapq.nlargest(10,values.values())[-1] for t,values in groups.items() if len(values)>=10}
    if payload.get('result') is not True:raise ValueError('Invalid tracker result')
    result={}
    for p in payload['cutoffs']:
        t,ep=int(p['time']),int(p['ep'])
        if t in result and result[t]!=ep:raise ValueError('Ambiguous cutoff timestamp')
        result[t]=ep
    return result


def audit(run,database,output):
    output=check_database_path(output)
    if output.exists():raise FileExistsError(output)
    summary=verify_download(run);config=json.loads((run/'config.json').read_text())
    receipt=json.loads((run/'database_receipt.json').read_text());records=latest_records(run)
    frozen=HoldoutDataset()
    if frozen.manifest['dataset_id']!=EXPECTED_HOLDOUT:raise ValueError('Frozen dataset changed')
    if database_identity(database)!=receipt['database_after_logical_sha256']:raise ValueError('Imported database identity changed')
    if sha256(receipt['backup'])!=receipt['backup_sha256']:raise ValueError('Backup changed')
    old=read_rows(receipt['backup']);current=read_rows(database);expected_keys=set(old)
    data_quality=[];preserved=0;raw_points_verified=0;top_snapshots_verified=0
    for (event,rank),record in sorted(records.items()):
        key=(event,rank,COUNTRY)
        if record['status'] in ['ok','empty']:
            payload=json.loads(gzip.decompress((run/record['raw']).read_bytes()))
            raw=independent_raw_curve(payload,rank)
            if record['status']=='empty':
                if raw:raise ValueError('Nonempty source recorded as empty')
                continue
            normalized=json.loads((run/record['curve']).read_text())
            incoming={p['time']:p['ep'] for p in normalized['points']}
            if len(incoming)!=len(normalized['points']) or raw!=incoming:raise ValueError('Raw-to-curve conversion differs')
            raw_points_verified+=len(raw)
            if rank==10:top_snapshots_verified+=len(raw)
            expected_keys.add(key)
            existing=ast.literal_eval(old[key]['PointRank']) if key in old else []
            expected={p['time']:p['ep'] for p in existing};expected.update(incoming)
            stored=ast.literal_eval(current[key]['PointRank']);actual={p['time']:p['ep'] for p in stored}
            if len(actual)!=len(stored) or actual!=expected:raise ValueError('Database merge differs from raw source and backup')
            times=[p['time'] for p in stored];scores=[p['ep'] for p in stored]
            if times!=sorted(times) or any(v<0 for v in scores):raise ValueError('Invalid stored curve')
            end=normalized['end'];start=normalized['start']
            if current[key]['StartAt']!=start or current[key]['EndAt']!=end:raise ValueError('Metadata dates differ')
            data_quality.append({'event':event,'rank':rank,'points':len(stored),'first_time':times[0],'last_time':times[-1],
                'zero_score_points':sum(v==0 for v in scores),'score_decreases':sum(b<a for a,b in zip(scores,scores[1:])),
                'before_start_points':sum(t<start for t in times),'after_end_points':sum(t>end for t in times),
                'near_end_points':sum(end-60000<=t<=end+300000 for t in times),
                'completed_at_download':end<=config['as_of_ms']})
    if set(current)!=expected_keys:raise ValueError('Unexpected database keys after import')
    for key,row in old.items():
        r=records.get(key[:2]) if key[2]==COUNTRY else None
        if r is None or r['status']!='ok':
            if current[key]!=row:raise ValueError('Unrequested/empty/invalid source erased an existing row')
            preserved+=1
    coverage=[]
    for rank in config['ranks']:
        rows=[(e,r) for (e,t,c),r in current.items() if c==COUNTRY and t==rank]
        nonempty=[(e,r) for e,r in rows if ast.literal_eval(r['PointRank'])]
        downloaded=summary['coverage'][config['ranks'].index(rank)]
        ranked_quality=[q for q in data_quality if q['rank']==rank]
        coverage.append(downloaded|{'database_rows':len(rows),'database_nonempty_curves':len(nonempty),
            'database_points':sum(len(ast.literal_eval(r['PointRank'])) for _,r in rows),
            'first_cn_start':datetime.fromtimestamp(min(r['StartAt'] for _,r in nonempty)/1000,TZ).isoformat() if nonempty else None,
            'latest_activity':max(nonempty,key=lambda x:x[1]['StartAt'])[0] if nonempty else None,
            'imported_completed_curves_with_near_end_observation':sum(q['completed_at_download'] and q['near_end_points']>0 for q in ranked_quality)})
    output.mkdir(parents=True);shutil.copyfile(__file__,output/'audit_source.py')
    for name,rows in [('coverage.csv',coverage),('curve_quality.csv',data_quality)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'frozen_dataset_id':frozen.manifest['dataset_id'],'download_requested_pairs':summary['requested_pairs'],
        'download_status_counts':summary['status_counts'],'normalized_points_verified_against_raw':raw_points_verified,
        'top10_complete_snapshots_independently_verified':top_snapshots_verified,
        'database_rows':len(current),'database_nonempty_curves':sum(r['database_nonempty_curves'] for r in coverage),
        'database_points':sum(r['database_points'] for r in coverage),'unchanged_existing_rows':preserved,
        'sqlite_integrity':'ok','logical_keys_unique':True,'all_imported_curves_match_source_plus_backup':True,
        'zero_score_points':sum(q['zero_score_points'] for q in data_quality),
        'curves_with_score_decreases':sum(q['score_decreases']>0 for q in data_quality),
        'source_sha256':sha256(__file__),'database_logical_sha256':database_identity(database)}
    write_json(output/'verification.json',result)
    write_json(output/'manifest.json',{'artifacts':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--database',type=Path,default=Path('data/db.sqlite3'));parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.run.resolve(),args.database.resolve(),args.output.resolve())
