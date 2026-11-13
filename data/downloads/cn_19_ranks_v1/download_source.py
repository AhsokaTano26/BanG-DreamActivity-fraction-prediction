"""Archive CN cutoff histories for nineteen ranks, then import with a backup.

fetch: bounded public API reads, resumable raw archives and explicit missingness.
apply: verify the archive, back up SQLite, merge in one transaction.
Neither command modifies or rebuilds a frozen evaluation dataset.
"""
from __future__ import annotations
import argparse
import ast
import asyncio
from collections import Counter,defaultdict
from contextlib import closing
import csv
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sqlite3
import time
import httpx

ROOT=Path(__file__).resolve().parent
RANKS=[10,20,30,40,50,100,200,300,400,500,1000,1500,2000,3000,4000,5000,10000,20000,30000]
SERVER=3
COUNTRY='中国大陆'
EVENTS_URL='https://bestdori.com/api/events/all.5.json'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def now_ms():return int(time.time()*1000)
def utc_now():return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_bytes(value);temp.replace(path)


def write_json(path,value):atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def integer(value,name,minimum=0):
    if isinstance(value,bool):raise ValueError(f'{name} is boolean')
    try:number=float(value);result=int(value)
    except (TypeError,ValueError,OverflowError):raise ValueError(f'Invalid {name}') from None
    if not math.isfinite(number) or number!=result or result<minimum:raise ValueError(f'Invalid {name}')
    return result


def normalize_points(points):
    if not isinstance(points,list):raise ValueError('Points must be a list')
    indexed={};duplicates=0
    for point in points:
        if not isinstance(point,dict):raise ValueError('Invalid point object')
        t=integer(point.get('time'),'time',1);ep=integer(point.get('ep'),'ep')
        if t in indexed:
            if indexed[t]!=ep:raise ValueError('Conflicting scores at the same timestamp')
            duplicates+=1
        indexed[t]=ep
    curve=[{'time':t,'ep':ep} for t,ep in sorted(indexed.items())]
    return curve,{'raw_points':len(points),'points':len(curve),'duplicate_points':duplicates,
                  'score_decreases':sum(b['ep']<a['ep'] for a,b in zip(curve,curve[1:]))}


def normalize_cutoffs(payload):
    if not isinstance(payload,dict) or payload.get('result') is not True:
        raise ValueError('Tracker result is not true: '+str(payload.get('code') if isinstance(payload,dict) else 'invalid object'))
    return normalize_points(payload.get('cutoffs'))


def normalize_top(payload):
    if not isinstance(payload,dict) or payload.get('result') is False or not isinstance(payload.get('points'),list):
        raise ValueError('Invalid top10 response')
    grouped=defaultdict(dict);duplicates=0
    for point in payload['points']:
        if not isinstance(point,dict):raise ValueError('Invalid top10 point')
        t=integer(point.get('time'),'time',1);uid=integer(point.get('uid'),'uid',1);value=integer(point.get('value'),'value')
        if uid in grouped[t]:
            if grouped[t][uid]!=value:raise ValueError('Conflicting same-user top10 snapshot')
            duplicates+=1
        grouped[t][uid]=value
    points=[{'time':t,'ep':sorted(users.values(),reverse=True)[9]} for t,users in grouped.items() if len(users)>=10]
    curve,quality=normalize_points(points)
    quality.update(raw_points=len(payload['points']),duplicate_points=duplicates,top_snapshots=len(grouped),
                   incomplete_top_snapshots=sum(len(g)<10 for g in grouped.values()),
                   overcomplete_top_snapshots=sum(len(g)>10 for g in grouped.values()))
    return curve,quality


def started_events(metadata,as_of):
    if not isinstance(metadata,dict):raise ValueError('Invalid event metadata')
    selected=[]
    for key,info in metadata.items():
        if not isinstance(info,dict):raise ValueError('Invalid event metadata record')
        starts=info.get('startAt',[]);ends=info.get('endAt',[])
        if len(starts)<=SERVER or starts[SERVER] is None:continue
        if len(ends)<=SERVER or ends[SERVER] is None:raise ValueError('CN start without end')
        start=integer(starts[SERVER],'start',1);end=integer(ends[SERVER],'end',1)
        if end<=start:raise ValueError('Invalid CN event duration')
        if start<=as_of:selected.append((start,integer(key,'event',1)))
    return [event for _,event in sorted(selected)]


def check_database_path(path):
    path=Path(path).resolve()
    if any(part.startswith('holdout_') for part in path.parts) or path.name=='snapshot.sqlite3':
        raise ValueError('Frozen datasets cannot be download/import destinations')
    return path


def merge_points(old,new):
    old,_=normalize_points(old);new,_=normalize_points(new)
    a={p['time']:p['ep'] for p in old};b={p['time']:p['ep'] for p in new}
    changes={'added_points':len(b.keys()-a.keys()),
             'corrected_points':sum(a[t]!=b[t] for t in a.keys()&b.keys()),
             'retained_old_points':len(a.keys()-b.keys())}
    a.update(b)
    return [{'time':t,'ep':v} for t,v in sorted(a.items())],changes


class Limiter:
    def __init__(self,rate):self.rate=rate;self.next_at=0.;self.lock=asyncio.Lock()
    async def wait(self):
        async with self.lock:
            delay=max(0.,self.next_at-time.monotonic())
            if delay:await asyncio.sleep(delay)
            self.next_at=time.monotonic()+1/self.rate
    async def cooldown(self,seconds):
        async with self.lock:self.next_at=max(self.next_at,time.monotonic()+seconds)


async def request(client,url,params,limiter,attempts=4):
    for attempt in range(attempts):
        await limiter.wait()
        try:
            response=await client.get(url,params=params)
            if response.status_code==429:
                retry=response.headers.get('retry-after','2')
                try:delay=float(retry)
                except ValueError:
                    try:delay=parsedate_to_datetime(retry).timestamp()-time.time()
                    except (TypeError,ValueError):delay=2.
                await limiter.cooldown(max(1.,delay))
            response.raise_for_status()
            return response,attempt+1
        except (httpx.TransportError,httpx.HTTPStatusError) as exc:
            retryable=not isinstance(exc,httpx.HTTPStatusError) or exc.response.status_code in [408,429] or exc.response.status_code>=500
            if not retryable or attempt==attempts-1:raise
            await asyncio.sleep(min(2**attempt,8))
    raise RuntimeError('Unreachable retry state')


def latest_records(run):
    path=run/'records.jsonl';records={}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record=json.loads(line);records[(record['event'],record['rank'])]=record
    return records


def verify_record(run,record):
    for field in ['raw','curve']:
        if field in record and sha256(run/record[field])!=record[field+'_sha256']:
            raise ValueError('Changed download artifact: '+record[field])


def summary(run,config,records):
    rows=[]
    for rank in config['ranks']:
        entries=[v for (e,r),v in records.items() if r==rank];counts=Counter(v['status'] for v in entries)
        rows.append({'rank':rank,'requested_events':len(config['events']),'downloaded_curves':counts['ok'],
                     'empty_events':counts['empty'],'invalid_events':counts['invalid'],'failed_requests':counts['failed'],
                     'pending':len(config['events'])-len(entries),
                     'points':sum(v.get('quality',{}).get('points',0) for v in entries),
                     'curves_with_decreases':sum(v.get('quality',{}).get('score_decreases',0)>0 for v in entries)})
    with (run/'coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    detail=[]
    for key in sorted(records):
        r=records[key];detail.append({k:r.get(k) for k in ['event','rank','status','fetched_at','error']}|r.get('quality',{}))
    fields=list(dict.fromkeys(k for row in detail for k in row))
    with (run/'quality.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(detail)
    result={'events':len(config['events']),'ranks':config['ranks'],'requested_pairs':len(config['events'])*len(config['ranks']),
            'status_counts':dict(Counter(r['status'] for r in records.values())),
            'points':sum(row['points'] for row in rows),'coverage':rows}
    write_json(run/'summary.json',result);return result


async def download(run,database,*,ranks=None,rate=8.,workers=8,resume=False,transport=None):
    run=Path(run).resolve();database=check_database_path(database);ranks=list(RANKS if ranks is None else ranks)
    if not ranks or len(set(ranks))!=len(ranks) or not set(ranks)<=set(RANKS):raise ValueError('Invalid requested ranks')
    if rate<=0 or workers<1 or workers>16:raise ValueError('Invalid worker/rate limits')
    if not database.exists():raise FileNotFoundError(database)
    limiter=Limiter(rate)
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.,connect=15.),limits=httpx.Limits(max_connections=workers),
                                 follow_redirects=True,transport=transport) as client:
        if run.exists():
            if not resume:raise FileExistsError(run)
            config=json.loads((run/'config.json').read_text())
            if config['database']!=str(database) or config['ranks']!=ranks:raise ValueError('Resume scope differs')
            if config['code_sha256']!=sha256(__file__) or config['python']!=platform.python_version() or config['httpx']!=importlib.metadata.version('httpx'):
                raise ValueError('Resume code/environment differs')
            if sha256(run/'events.json')!=config['metadata_sha256']:raise ValueError('Metadata changed')
            if (run/'download_manifest.json').exists():return verify_download(run)
            metadata=json.loads((run/'events.json').read_text())
        else:
            response,_=await request(client,EVENTS_URL,None,limiter);metadata=response.json();as_of=now_ms()
            events=started_events(metadata,as_of)
            if not events:raise ValueError('No started CN events')
            run.mkdir(parents=True);(run/'raw').mkdir();(run/'curves').mkdir()
            atomic_bytes(run/'events.json',response.content);shutil.copyfile(__file__,run/'download_source.py')
            config={'created_at':utc_now(),'as_of_ms':as_of,'server':SERVER,'country':COUNTRY,'ranks':ranks,'events':events,
                    'database':str(database),'initial_database_sha256':sha256(database),'metadata_url':EVENTS_URL,
                    'metadata_sha256':sha256(run/'events.json'),'code_sha256':sha256(__file__),
                    'python':platform.python_version(),'httpx':importlib.metadata.version('httpx'),'rate':rate,'workers':workers,
                    'top10':'tenth highest distinct-user score per timestamp with at least ten users; interval=0',
                    'merge_policy':'fresh overlapping timestamps replace old; old-only observations retained; empty never erases'}
            write_json(run/'config.json',config)
        records=latest_records(run);queue=asyncio.Queue()
        for event in reversed(config['events']):
            for rank in ranks:
                previous=records.get((event,rank))
                if previous is not None and previous['status']!='failed':verify_record(run,previous);continue
                queue.put_nowait((event,rank))
        total=len(config['events'])*len(ranks);started=time.monotonic();count=0
        print(json.dumps({'run':str(run),'events':len(config['events']),'ranks':ranks,'jobs_to_fetch':queue.qsize()}),flush=True)
        async def worker():
            nonlocal count
            while True:
                try:event,rank=queue.get_nowait()
                except asyncio.QueueEmpty:return
                params={'server':SERVER,'event':event}
                if rank==10:api='eventtop';params.update(mid=0,interval=0)
                else:api='tracker';params['tier']=rank
                url=f'https://bestdori.com/api/{api}/data';record={'event':event,'rank':rank,'url':url,'params':params}
                try:
                    response,attempts=await request(client,url,params,limiter)
                    record.update(fetched_at=utc_now(),attempts=attempts,http_status=response.status_code)
                    raw=f'raw/{event}_{rank}.json.gz';atomic_bytes(run/raw,gzip.compress(response.content,compresslevel=6,mtime=0))
                    record.update(raw=raw,raw_sha256=sha256(run/raw))
                    try:
                        payload=response.json();points,quality=normalize_top(payload) if rank==10 else normalize_cutoffs(payload)
                        start=integer(metadata[str(event)]['startAt'][SERVER],'start');end=integer(metadata[str(event)]['endAt'][SERVER],'end')
                        quality.update(first_time=points[0]['time'] if points else None,last_time=points[-1]['time'] if points else None,
                            before_start_points=sum(p['time']<start for p in points),after_end_points=sum(p['time']>end for p in points),
                            near_end_points=sum(end-60000<=p['time']<=end+300000 for p in points))
                        if any(p['time']>now_ms()+300000 for p in points):raise ValueError('Observation timestamp is in the future')
                        record.update(status='ok' if points else 'empty',quality=quality)
                        if points:
                            path=f'curves/{event}_{rank}.json'
                            write_json(run/path,{'event':event,'rank':rank,'start':start,'end':end,'points':points})
                            record.update(curve=path,curve_sha256=sha256(run/path))
                    except (ValueError,KeyError,TypeError,IndexError) as exc:record.update(status='invalid',error=str(exc))
                except (httpx.HTTPError,OSError) as exc:record.update(status='failed',fetched_at=utc_now(),error=f'{type(exc).__name__}: {exc}')
                with (run/'records.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
                records[(event,rank)]=record;count+=1;queue.task_done()
                if count%40==0 or queue.empty():
                    progress={'recorded':len(records),'total':total,'fetched_this_session':count,
                              'status_counts':dict(Counter(r['status'] for r in records.values())),
                              'seconds':round(time.monotonic()-started,1)}
                    write_json(run/'progress.json',progress);print(json.dumps(progress),flush=True)
        await asyncio.gather(*(worker() for _ in range(workers)))
    result=summary(run,config,records)
    if any(r['status']=='failed' for r in records.values()):
        print('Download has retryable failures; use fetch --resume.',flush=True);return result
    write_json(run/'download_manifest.json',{'created_at':utc_now(),'requested_pairs':total,
        'artifacts':{str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file() and p.name!='download_manifest.json'}})
    verify_download(run);print('DOWNLOAD_COMPLETE '+json.dumps(result,ensure_ascii=False),flush=True);return result


def verify_download(run):
    run=Path(run);seal=json.loads((run/'download_manifest.json').read_text())
    for name,digest in seal['artifacts'].items():
        if sha256(run/name)!=digest:raise ValueError('Download artifact changed: '+name)
    config=json.loads((run/'config.json').read_text());records=latest_records(run)
    expected={(e,r) for e in config['events'] for r in config['ranks']}
    if set(records)!=expected or any(r['status']=='failed' for r in records.values()):raise ValueError('Incomplete request coverage')
    for record in records.values():verify_record(run,record)
    return json.loads((run/'summary.json').read_text())


def apply_download(run,database):
    run=Path(run).resolve();database=check_database_path(database);verify_download(run)
    receipt_path=run/'database_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['database']!=str(database) or receipt['database_after_sha256']!=sha256(database):raise ValueError('Database changed since this import')
        return receipt
    config=json.loads((run/'config.json').read_text())
    if config['database']!=str(database) or config['initial_database_sha256']!=sha256(database):raise ValueError('Database changed since download began; refuse stale import')
    metadata=json.loads((run/'events.json').read_text());records=latest_records(run)
    backup=database.parent/'backups'/f'{database.stem}_before_{run.name}.sqlite3'
    if backup.exists():raise FileExistsError(backup)
    backup.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as source:
        with closing(sqlite3.connect(backup)) as target:source.backup(target)
    changes=[];inserted=0;updated=0
    with closing(sqlite3.connect(database)) as con:
        con.row_factory=sqlite3.Row
        if con.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Database integrity failed before import')
        duplicates=con.execute('SELECT EventID,Rank,Country,COUNT(*) n FROM event GROUP BY EventID,Rank,Country HAVING n>1').fetchall()
        if duplicates:raise ValueError('Duplicate event/rank/country keys')
        before_rows=con.execute('SELECT COUNT(*) FROM event').fetchone()[0]
        con.execute('BEGIN IMMEDIATE')
        try:
            for (event,rank),record in sorted(records.items()):
                if record['status']!='ok':continue
                incoming=json.loads((run/record['curve']).read_text());info=metadata[str(event)]
                old=con.execute('SELECT * FROM event WHERE EventID=? AND Rank=? AND Country=?',(event,rank,COUNTRY)).fetchone()
                old_points=ast.literal_eval(old['PointRank']) if old is not None else []
                merged,delta=merge_points(old_points,incoming['points'])
                name=info.get('eventName',[None]*5);name=name[SERVER] if len(name)>SERVER else None
                name=name or (old['EventName'] if old is not None else str(event))
                band=old['EventBand'] if old is not None else None
                identity=old['ID'] if old is not None else hashlib.md5(f'{event}{SERVER}{rank}'.encode()).hexdigest()
                values=(event,band,name,info['eventType'],incoming['start'],incoming['end'],json.dumps(merged,separators=(',',':')),identity,rank,COUNTRY)
                if old is None:
                    con.execute('INSERT INTO event(EventID,EventBand,EventName,EventType,StartAt,EndAt,PointRank,ID,Rank,Country) VALUES(?,?,?,?,?,?,?,?,?,?)',values);inserted+=1
                else:
                    con.execute('UPDATE event SET EventID=?,EventBand=?,EventName=?,EventType=?,StartAt=?,EndAt=?,PointRank=?,ID=?,Rank=?,Country=? WHERE ID=?',values+(identity,));updated+=1
                changes.append({'event':event,'rank':rank,'operation':'insert' if old is None else 'merge',
                                'previous_points':len(old_points),'incoming_points':len(incoming['points']),'stored_points':len(merged)}|delta)
            if con.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Database integrity failed after import')
            con.commit()
        except BaseException:con.rollback();raise
        after_rows=con.execute('SELECT COUNT(*) FROM event').fetchone()[0]
    with (run/'database_changes.csv').open('w',newline='') as f:
        fields=['event','rank','operation','previous_points','incoming_points','stored_points','added_points','corrected_points','retained_old_points']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(changes)
    receipt={'applied_at':utc_now(),'database':str(database),'database_before_sha256':config['initial_database_sha256'],
             'database_after_sha256':sha256(database),'backup':str(backup),'backup_sha256':sha256(backup),
             'download_manifest_sha256':sha256(run/'download_manifest.json'),'before_rows':before_rows,'after_rows':after_rows,
             'inserted_curves':inserted,'merged_curves':updated,'added_points':sum(c['added_points'] for c in changes),
             'corrected_points':sum(c['corrected_points'] for c in changes),'changes_sha256':sha256(run/'database_changes.csv'),
             'apply_code_sha256':sha256(__file__)}
    write_json(receipt_path,receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['fetch','verify','apply'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--database',type=Path,default=ROOT/'data/db.sqlite3')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--workers',type=int,default=8);parser.add_argument('--rate',type=float,default=8.)
    args=parser.parse_args()
    if args.command=='fetch':asyncio.run(download(args.run,args.database,resume=args.resume,workers=args.workers,rate=args.rate))
    elif args.command=='verify':print(json.dumps(verify_download(args.run),ensure_ascii=False))
    else:print(json.dumps(apply_download(args.run,args.database),ensure_ascii=False))


if __name__=='__main__':main()
