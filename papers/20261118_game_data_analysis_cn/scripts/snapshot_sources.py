"""Export a read-only, time-bounded source snapshot for a descriptive study."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import sqlite3
from collections import Counter

P = Path(__file__).resolve().parents[1]
ROOT = P.parents[1]
LOWER = 1706504400000
RANKS = [10,20,30,40,50,100,200,300,400,500,1000,1500,2000,3000,4000,5000,10000,20000,30000]


def main():
    target = P / 'data/source_curves.jsonl.gz'
    if target.exists():
        raise FileExistsError('Source snapshot is immutable: ' + str(target))
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    meta = {}
    hashes = {}
    operational = ROOT / 'runtime/live'
    meta_times = {}
    for name in ['events','characters','bands','cards']:
        f = operational / f'metadata/{name}.json'
        raw = f.read_bytes()
        hashes[str(f.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        meta[name] = json.loads(raw)
        meta_times[name] = datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat()
    contexts=[]
    for key,event in meta['events'].items():
        chars=sorted(set(int(x['characterId']) for x in event.get('characters',[])))
        bands=Counter(str(meta['characters'].get(str(cid),{}).get('bandId',0)) for cid in chars)
        total=sum(bands.values())
        contexts.append({'event_id':int(key),'characters':chars,
            'band_shares':{bid:n/total for bid,n in bands.items()} if total else {},
            'band_names':{bid:meta['bands'].get(bid,{}).get('bandName',['']*4)[3] for bid in bands},
            'band_group':'single' if len(bands)==1 else 'mixed'})
    meta['event_contexts']=contexts
    meta['sources']={'latest_local_file_mtime':max(meta_times.values()),'file_mtimes':meta_times,
                     'context_method':'Character membership and band IDs from the same retrospective metadata snapshot.'}
    with gzip.open(P / 'data/source_metadata.json.gz', 'wt', encoding='utf8') as f:
        json.dump(meta, f, ensure_ascii=False, separators=(',', ':'))
    con = sqlite3.connect('file:' + str(operational / 'data.sqlite3') + '?mode=ro', uri=True)
    con.execute('BEGIN')
    total = (con.execute('SELECT COUNT(*) FROM curves').fetchone()[0],con.execute('SELECT COUNT(*) FROM events').fetchone()[0])
    metadata_state=con.execute('SELECT name,refreshed_ms,source_sha256 FROM metadata_state').fetchall()
    query = '''SELECT e.event,c.rank,e.start_ms,e.end_ms,e.kind,e.name,c.points_json,c.updated_ms,c.verified_closed,c.quality_json
               FROM events e JOIN curves c USING(event) WHERE e.start_ms>=? AND e.end_ms<?
               ORDER BY e.start_ms,e.event,c.rank'''
    digest = hashlib.sha256()
    events = set()
    count = points = 0
    with gzip.open(target, 'wt', encoding='utf8') as f:
        for eid, rank, start, end, kind, name, raw, updated, verified, quality in con.execute(query, (LOWER, now)):
            if rank not in RANKS:
                continue
            event = meta['events'][str(eid)]
            assert int(event['startAt'][3]) == start and int(event['endAt'][3]) == end
            record = {'event':eid,'rank':rank,'country':'中国大陆','start_ms':start,'end_ms':end,
                      'kind':kind,'name':name,'points':json.loads(raw),'upstream_updated_ms':updated,
                      'upstream_verified_closed':bool(verified),'upstream_quality':json.loads(quality)}
            line = json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n'
            digest.update(line.encode())
            f.write(line)
            count += 1
            points += len(record['points'])
            events.add(eid)
    con.close()
    partition = ROOT / 'data/temporal_holdout_v2_dense_20261115/partitions.csv'
    (P / 'data/original_partitions.csv').write_bytes(partition.read_bytes())
    hashes[str(partition.relative_to(ROOT))] = hashlib.sha256(partition.read_bytes()).hexdigest()
    result = {'captured_at':datetime.now(timezone.utc).isoformat(),'cutoff_ms':now,'lower_start_ms':LOWER,
              'query':query,'parameters':[LOWER,now], 'database_path':'runtime/live/data.sqlite3','country_server_index':3,
              'metadata_file_mtimes':meta_times,'metadata_state':metadata_state,
              'database_total_curves':total[0],'database_total_events':total[1],
              'selected_events':len(events),'selected_curves':count,'raw_points':points,
              'event_ids':sorted(events),'canonical_selected_rows_sha256':digest.hexdigest(),
              'source_file_hashes':hashes,
              'snapshot_hashes':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [target,P/'data/source_metadata.json.gz',P/'data/original_partitions.csv']}}
    (P / 'data/source_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['selected_events','selected_curves','raw_points','canonical_selected_rows_sha256']}))


if __name__ == '__main__':
    main()
