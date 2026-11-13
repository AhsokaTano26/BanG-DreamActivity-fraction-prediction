"""Read-only primary-source archive for new CN activities; no model access."""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import httpx
from download_ranks import Limiter, normalize_cutoffs, request

OUT = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def iso(ms):
    return datetime.fromtimestamp(ms / 1000, TZ).isoformat() if ms is not None else None

async def main():
    if (OUT / 'events.json').exists():
        raise FileExistsError('Never overwrite archived primary-source responses')
    started = datetime.now(timezone.utc).isoformat()
    limiter = Limiter(2.)
    receipts = []
    async def fetch(client, name, url, params):
        began = datetime.now(timezone.utc).isoformat()
        response, attempts = await request(client, url, params, limiter, attempts=4)
        fetched = datetime.now(timezone.utc).isoformat()
        path = OUT / name
        path.write_bytes(response.content)
        receipt = {'file': name, 'url': str(response.url), 'request_started_at_utc': began,
            'fetched_at_utc': fetched, 'status_code': response.status_code, 'attempts': attempts,
            'headers': dict(response.headers), 'sha256': sha(path), 'bytes': len(response.content)}
        receipts.append(receipt)
        write(OUT / (name + '.http.json'), receipt)
        return response.json(), receipt
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        events, events_receipt = await fetch(client, 'events.json',
            'https://bestdori.com/api/events/all.5.json', {})
        now_ms = int(time.time() * 1000)
        current = []
        for key, info in events.items():
            starts, ends = info.get('startAt', []), info.get('endAt', [])
            if len(starts) <= 3 or starts[3] is None:
                continue
            start, end = int(starts[3]), int(ends[3])
            current.append({'event': int(key), 'name': info.get('eventName', [None] * 4)[3],
                'event_type': info.get('eventType'), 'start_ms': start, 'end_ms': end,
                'start_cn': iso(start), 'end_cn': iso(end),
                'status_at_fetch': 'unstarted' if now_ms < start else 'ongoing' if now_ms <= end else 'ended'})
        rows = []
        for event in (327, 328):
            info = next(r for r in current if r['event'] == event)
            for rank in (1000, 2000):
                payload, receipt = await fetch(client, f'event_{event}_rank_{rank}.json',
                    'https://bestdori.com/api/tracker/data', {'server': 3, 'event': event, 'tier': rank})
                points, quality = normalize_cutoffs(payload)
                near_end = [p for p in points if info['end_ms'] - 60000 <= p['time'] <= info['end_ms'] + 300000]
                positive = all(p['ep'] > 0 for p in points)
                first, last = (points[0] if points else None), (points[-1] if points else None)
                row = {'event': event, 'rank': rank, 'event_info': info,
                    'raw_file': receipt['file'], 'raw_sha256': receipt['sha256'],
                    'fetched_at_utc': receipt['fetched_at_utc'], 'source_url': receipt['url'],
                    'quality': quality | {'positive_scores': positive,
                        'strictly_increasing_timestamps': all(a['time'] < b['time'] for a,b in zip(points, points[1:])),
                        'before_start_points': sum(p['time'] < info['start_ms'] for p in points),
                        'after_end_points': sum(p['time'] > info['end_ms'] for p in points)},
                    'point_count': len(points), 'first_observation': first, 'last_observation': last,
                    'first_observation_cn': iso(first['time']) if first else None,
                    'last_observation_cn': iso(last['time']) if last else None,
                    'last_minus_end_seconds': (last['time'] - info['end_ms']) / 1000 if last else None,
                    'near_end_window_ms': [info['end_ms'] - 60000, info['end_ms'] + 300000],
                    'near_end_point_count': len(near_end),
                    'final_status': 'available_near_end' if near_end else 'missing_near_end',
                    'near_end_final': near_end[-1] if near_end else None,
                    'observations': points}
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k != 'observations'}, ensure_ascii=False), flush=True)
    frozen = json.loads((ROOT / 'data/temporal_holdout_v2_t1000_t2000/manifest.json').read_text())
    frozen_events = sorted({e for group in frozen['splits'].values() for e in group['event_ids']})
    metadata = {'protocol': 'additional forward validation source audit; no fitting, model selection or predictions',
        'server': 3, 'fetch_started_at_utc': started, 'as_of_utc': datetime.fromtimestamp(now_ms / 1000, timezone.utc).isoformat(),
        'as_of_cn': iso(now_ms), 'latest_cn_activities': sorted(current, key=lambda r:r['start_ms'])[-6:],
        'ongoing_activities': [r for r in current if r['status_at_fetch'] == 'ongoing'],
        'unstarted_announced_activities': [r for r in current if r['status_at_fetch'] == 'unstarted'],
        'frozen_temporal_subset_dataset_id': frozen['dataset_id'], 'frozen_max_event_id': max(frozen_events),
        'new_event_overlap_with_frozen_cohort': sorted(set((327,328)) & set(frozen_events)),
        'frozen_manifest_sha256': sha(ROOT / 'data/temporal_holdout_v2_t1000_t2000/manifest.json'),
        'final_definition': 'latest observation in [EndAt-60000, EndAt+300000]; latest pre-end observation outside window is not a final label',
        'rows': rows}
    write(OUT / 'metadata.json', metadata)
    write(OUT / 'http_receipts.json', receipts)
    write(OUT / 'environment.json', {'python': platform.python_version(), 'httpx': importlib.metadata.version('httpx')})
    write(OUT / 'manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'artifacts': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}})
    print(json.dumps({k:v for k,v in metadata.items() if k != 'rows'}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    asyncio.run(main())
