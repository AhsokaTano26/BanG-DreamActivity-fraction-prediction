"""Local chronological demo over a physically isolated, growing prefix store.

ReplaySource is the trusted source coordinator. Only its published store is
visible to the normal inference and web paths; model fitting is a separate step.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from contextlib import closing
import json
from pathlib import Path
import signal
import sqlite3
import threading
import time

from analysis.data_policy import MIN_EVENT_START_MS
from analysis.holdout_dataset import sha256
from production.data import Store, _clock_points, _complete, normalize_points
from production.state import Ledger, atomic_json


class ReplaySource:
    def __init__(self, source_root, event):
        self.root=Path(source_root).resolve();self.last_ms=None;self.counts={}
        with closing(sqlite3.connect((self.root/'data.sqlite3').as_uri()+'?mode=ro',uri=True)) as con:
            con.row_factory=sqlite3.Row
            row=con.execute('SELECT * FROM events WHERE event=? AND start_ms>=?',
                            (int(event),MIN_EVENT_START_MS)).fetchone()
            if row is None:raise ValueError('Replay activity is missing or outside the dense policy')
            self.event=dict(row)
            self.curves={}
            for r in con.execute('SELECT rank,points_json FROM curves WHERE event=?',(event,)):
                points,_=normalize_points(json.loads(r['points_json']))
                points=_clock_points(points,row['start_ms'],row['end_ms'],row['end_ms']+300001)
                self.curves[r['rank']]=(points,[p['time'] for p in points])
        self.metadata={name:json.loads((self.root/'metadata'/(name+'.json')).read_text())
                       for name in ('events','characters','bands','cards')}
        self.card_count=None;self.initialized_metadata=False

    def publish(self, store, clock):
        if store.root==self.root:raise ValueError('Replay must not replace the production store')
        if self.last_ms is not None and clock<self.last_ms:raise ValueError('Replay clock cannot move backwards')
        self.last_ms=clock;maximum=None;counts={}
        with store._connect() as con:
            store._upsert_events(con,[self.event])
            for rank,(points,times) in self.curves.items():
                n=bisect_right(times,clock);visible=points[:n]
                closed=bool(clock>self.event['end_ms']+600000 and _complete(visible,self.event['end_ms'],clock))
                counts[str(rank)]=n
                if n:maximum=max(maximum or 0,visible[-1]['time'])
                if self.counts.get(rank)!=(n,closed):
                    con.execute('INSERT INTO curves(event,rank,points_json,updated_ms,verified_closed) VALUES(?,?,?,?,?) '
                        'ON CONFLICT(event,rank) DO UPDATE SET points_json=excluded.points_json,'
                        'updated_ms=excluded.updated_ms,verified_closed=excluded.verified_closed',
                        (self.event['event'],rank,json.dumps(visible,separators=(',',':')),clock,int(closed)))
                    self.counts[rank]=(n,closed)
        store.metadata_directory.mkdir(exist_ok=True)
        if not self.initialized_metadata:
            events={}
            for key,value in self.metadata['events'].items():
                dates=value.get('startAt',[])
                if len(dates)>3 and dates[3] is not None and int(dates[3])<=self.event['start_ms']:
                    events[key]=value
            atomic_json(store.metadata_directory/'events.json',events)
            for name in ('characters','bands'):atomic_json(store.metadata_directory/(name+'.json'),self.metadata[name])
            self.initialized_metadata=True
        cards={}
        for key,value in self.metadata['cards'].items():
            dates=value.get('releasedAt',[])
            if len(dates)>3 and dates[3] is not None and int(dates[3])<=clock:cards[key]=value
        if self.card_count!=len(cards):
            atomic_json(store.metadata_directory/'cards.json',cards);self.card_count=len(cards)
        assert maximum is None or maximum<=clock
        return {'max_visible_ms':maximum,'visible_points_by_rank':counts,'visible_cards':self.card_count}


class ReplayDriver:
    def __init__(self, source, root, model, *, wall_seconds=1.5):
        from production.model import verify_pack
        from production.runtime import _activate
        self.source=source;self.root=Path(root).resolve();self.model=Path(model).resolve()
        if (self.root/'config.json').exists():raise FileExistsError('Use a new replay directory')
        manifest=verify_pack(self.model);start=source.event['start_ms']
        if (not manifest['production_ready'] or manifest['training_before_ms']>start
            or manifest['history_max_end_ms']>=start or source.event['event'] in manifest['history_event_ids']):
            raise ValueError('Replay model must exclude this activity and all later outcomes')
        self.lock=threading.RLock();self.done=threading.Event();self.clock_ms=start
        self.frames=list(range(start,source.event['end_ms'],900000))+[source.event['end_ms']+600001]
        self.index=0;self.state='starting';self.error=None;self.wall_seconds=wall_seconds
        self.config=dict(root=str(self.root),ranks=manifest['ranks'],interval_seconds=900,
            forecast_method='legacy_pair',automatic_training=False,selected_seed_run=None,feedback_grace_ms=600000)
        self.store=Store(self.root);Ledger(self.root).fact('bootstrap_complete',True)
        _activate(self.root,self.model,manifest);atomic_json(self.root/'config.json',self.config)
        self.manifest=manifest
        # No feedback receipt predates this activity in the existing installation.
        # Start the online correction cold rather than importing later-created errors.
        atomic_json(self.root/'replay_manifest.json',dict(event=source.event,model_id=manifest['model_id'],
            model_manifest_sha256=sha256(self.model/'manifest.json'),training_events=manifest['training_events'],
            training_before_ms=manifest['training_before_ms'],history_max_end_ms=manifest['history_max_end_ms'],
            source_database_sha256=sha256(source.root/'data.sqlite3'),step_ms=900000,
            calibration='cold start; no later-created feedback imported',
            limitation='Recorded observation clocks and retrospective metadata; original API arrival times are unavailable'))
        self.source.publish(self.store,start)
        self.thread=threading.Thread(target=self.run,name='historical-replay',daemon=True)

    def clock(self):return self.clock_ms

    def snapshot(self):
        with self.lock:
            return dict(state=self.state,mode='replay',event=self.source.event['event'],clock_ms=self.clock_ms,
                frames_completed=self.index,total_frames=len(self.frames),refresh_ms=1000,
                training_events=self.manifest['training_events'],error=self.error,
                note='按观测记录时间释放数据；元数据为回溯快照，缺少当时 API 到达时间存档。误差校正从零开始。')

    def run(self):
        from production.model import InferenceCache,inference_session
        from production.selected_runtime import run_cycle
        cache=InferenceCache()
        try:
            for index,clock in enumerate(self.frames,1):
                if self.done.is_set():break
                with self.lock,inference_session(cache):
                    self.state='updating'
                    publication=self.source.publish(self.store,clock)
                    result=run_cycle(self.config,now_ms=clock,sync=False)
                    if result['errors']:raise ValueError(result['errors'])
                    if any(r.get('observed_at',clock)>clock for r in result['predictions']):
                        raise ValueError('Future observation entered a replay prediction')
                    self.clock_ms=clock;self.index=index;self.state='complete' if index==len(self.frames) else 'idle'
                    record=dict(frame=index,clock_ms=clock,predictions=sum(bool(r.get('available')) for r in result['predictions']),
                                model_id=result['model_id'],**publication)
                    with (self.root/'replay_audit.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
                    atomic_json(self.root/'replay_status.json',self.snapshot())
                    if index%24==0 or index==len(self.frames):print(json.dumps(record),flush=True)
                if self.done.wait(self.wall_seconds):break
        except Exception as exc:
            with self.lock:
                self.error=f'{type(exc).__name__}: {exc}';self.state='error'
                atomic_json(self.root/'replay_status.json',self.snapshot())
            print(self.error,flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runtime/live'))
    parser.add_argument('--event',type=int,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--wall-seconds',type=float,default=1.5)
    args=parser.parse_args(argv)
    if not 1<=args.port<=65535 or args.wall_seconds<0:parser.error('Invalid replay port or wall interval')
    from production.web import make_server
    source=ReplaySource(args.source,args.event)
    driver=ReplayDriver(source,args.state,args.model,wall_seconds=args.wall_seconds)
    server=make_server(driver.config,host='127.0.0.1',port=args.port,worker=driver,
                       clock=driver.clock,read_guard=driver.lock)
    def stop(*_):
        driver.done.set();threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(f'Historical replay: http://127.0.0.1:{server.server_port}',flush=True)
    driver.thread.start()
    try:server.serve_forever()
    finally:driver.done.set();server.server_close();driver.thread.join(timeout=10)
    return 0


if __name__=='__main__':raise SystemExit(main())
