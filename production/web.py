"""Read-only local dashboard with one explicitly owned forecasting process."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

from download_ranks import RANKS
from analysis.data_policy import MIN_EVENT_START_MS
from production.web_history import event_catalog, event_rows, curve_points, terminal_score

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name('web_assets')
ROW_FIELDS = ('event', 'rank', 'start_ms', 'end_ms', 'cutoff', 'observed_at',
              'current_ep', 'predicted_final_ep', 'raw_predicted_final_ep', 'progress',
              'available', 'reason', 'provisional', 'forecast_method', 'rank_training_events')


def _read_json(path, fallback):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return fallback


def event_details(root, payload, *, now_ms=None, ranks=RANKS):
    now=int(time.time()*1000) if now_ms is None else int(now_ms)
    metadata = _read_json(Path(root)/'metadata/events.json', {})
    catalog={row['event']:row for row in event_catalog(root,ranks,now)}
    types = {'story':'一般活动', 'challenge':'挑战LIVE', 'versus':'对邦LIVE',
             'live_try':'LIVE试炼', 'mission_live':'任务LIVE', 'medley':'组曲LIVE',
             'festival':'团队LIVE'}
    ids = dict.fromkeys([*payload.get('active_events', []),
                         *(row['event'] for row in payload.get('predictions', [])),*catalog])
    events = []
    for event in ids:
        entry = metadata.get(str(event), {})
        fallback = catalog.get(event) or next((row for row in payload.get('predictions', [])
                         if row['event'] == event and row.get('start_ms')), {})
        def date(key, region):
            values = entry.get(key, [])
            try:
                return int(values[region]) if values[region] else None
            except (IndexError, ValueError, TypeError):
                return None
        names = entry.get('eventName', [])
        name = next((names[i] for i in (3,0) if len(names)>i and names[i]), fallback.get('name',f'活动 {event}'))
        kind = entry.get('eventType', fallback.get('kind',''))
        start = date('startAt',3) or fallback.get('start_ms')
        end = date('endAt',3) or fallback.get('end_ms')
        if start is not None and (start < MIN_EVENT_START_MS or start>now):
            continue
        banner = entry.get('bannerAssetBundleName', '')
        events.append({'event':event, 'name':name, 'kind':kind,
            'type_label':types.get(kind, kind or '活动类型待更新'),
            'start_ms':start, 'end_ms':end, 'jp_start_ms':date('startAt',0),
            'status':'finished' if end and end<=now else 'active',
            'curve_ranks':catalog.get(event,{}).get('curve_ranks',0),
            'has_live_predictions':bool(catalog.get(event,{}).get('live_forecasts',0)),
            'banner_url':f'https://bestdori.com/assets/cn/homebanner_rip/{banner}.png'
                if isinstance(banner,str) and re.fullmatch(r'[a-zA-Z0-9_]+',banner) else None})
    return sorted(events,key=lambda event:(event['status']!='active',-(event.get('start_ms') or 0),-event['event']))


def public_snapshot(config, *, now_ms=None, service=None, initial_rank=1000, event_id=None):
    if event_id is not None and (isinstance(event_id,bool) or not isinstance(event_id,int) or event_id<=0):
        raise ValueError('Invalid activity identifier')
    now = int(time.time()*1000) if now_ms is None else int(now_ms)
    payload = _read_json(Path(config['root'])/'latest.json', {})
    events=event_details(config['root'],payload,now_ms=now,ranks=config['ranks'])
    selected_event=event_id or next(iter(payload.get('active_events',[])),None) or next(
        (row['event'] for row in payload.get('predictions',[])),None)
    selected=next((event for event in events if event['event']==selected_event),None)
    if event_id is not None and selected is None:
        raise ValueError('Unknown or disallowed activity')
    historical=bool(event_id is not None and selected and selected['status']=='finished')
    as_of = payload.get('as_of_ms')
    stale = not as_of or now-as_of > max(120000, 2*config['interval_seconds']*1000)
    rows = []
    for original in payload.get('predictions', []):
        if event_id is not None and original['event']!=event_id:
            continue
        row = {key: original[key] for key in ROW_FIELDS if key in original}
        if row.get('end_ms', now+1) <= now:
            row.update(available=False, reason='活动已结束，等待下期活动')
        elif row.get('observed_at', now) < now-3600000:
            row.update(available=False, reason='最新观测已超过60分钟')
        row['observation_available']=bool(row.get('current_ep') is not None
            and row.get('end_ms',now+1)>now and 0<=now-row.get('observed_at',now)<=3600000)
        rows.append(row)
    if historical or (event_id is not None and not rows):
        rows=event_rows(config['root'],selected,config['ranks'],now)
    return {'status': 'ok' if historical else payload.get('status', 'waiting_for_data'), 'as_of_ms': as_of,
            'server_time_ms': now, 'stale': False if historical else bool(stale), 'ranks': config['ranks'],
            'initial_rank': initial_rank, 'interval_seconds': config['interval_seconds'],
            'events':events,'selected_event':selected_event,'scope':'historical' if historical else 'live',
            'active_events': payload.get('active_events', []), 'predictions': rows,
            'data_update_failed': bool(payload.get('errors')), 'service': service or {'state': 'idle'}}


def forecast_history(root, event, rank):
    database = Path(root).resolve()/'state.sqlite3'
    if not database.is_file():
        return []
    con = sqlite3.connect(database.as_uri()+'?mode=ro', uri=True, timeout=2)
    con.row_factory = sqlite3.Row
    try:
        # Replay the last actually issued revision of each physical observation.
        rows = con.execute('''SELECT observed_at,issued_ms,current,prediction,raw_prediction,0 AS revision,rowid AS seq
            FROM forecasts WHERE event=? AND rank=? UNION ALL
            SELECT observed_at,issued_ms,current,prediction,raw_prediction,1 AS revision,rowid AS seq
            FROM forecast_revisions WHERE event=? AND rank=?
            ORDER BY issued_ms,revision,seq''', (event, rank, event, rank)).fetchall()
    finally:
        con.close()
    points = {}
    for r in rows:
        points[r['observed_at']] = {'observed_at': r['observed_at'], 'issued_at': r['issued_ms'],
            'current_ep': r['current'], 'predicted_final_ep': r['prediction']}
    return [points[key] for key in sorted(points)[-1000:]]


def observed_curve(root, event, rank, *, now_ms=None):
    """Read real tracker observations without promoting history or future data."""
    now = int(time.time()*1000) if now_ms is None else int(now_ms)
    result = {'event':event, 'rank':rank, 'as_of_ms':now, 'points':[]}
    database = Path(root).resolve()/'data.sqlite3'
    if not database.is_file():
        return result
    con = sqlite3.connect(database.as_uri()+'?mode=ro', uri=True, timeout=2)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute('''SELECT e.start_ms,e.end_ms,c.* FROM events e
            LEFT JOIN curves c ON c.event=e.event AND c.rank=?
            WHERE e.event=? AND e.start_ms>=? AND e.start_ms<=?''',
            (rank,event,MIN_EVENT_START_MS,now)).fetchone()
    finally:
        con.close()
    if row is None:
        return result
    row=dict(row)
    result.update(start_ms=row['start_ms'], end_ms=row['end_ms'])
    points=curve_points(row,now)
    result['points']=[point for point in points if point['time']<=row['end_ms']]
    final=terminal_score(row,points,now)
    if final:result['terminal_score']=final
    return result


class CycleWorker:
    """Own at most one bounded cycle subprocess, including interruption cleanup."""
    def __init__(self, config, config_path, *, interval=900, offline=False):
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.interval = interval
        self.offline = offline
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.process = None
        self.state = {'state': 'starting', 'offline': offline, 'next_update_ms': None}
        self.thread = threading.Thread(target=self._run, name='forecast-updates', daemon=True)

    def cycle_command(self):
        return [sys.executable, '-m', 'production.selected_cli', 'cycle', '--config', str(self.config_path)] + (['--offline'] if self.offline else [])

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.done.is_set():
            try:
                with self.lock:
                    if self.done.is_set():
                        break
                    env = dict(os.environ)
                    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                        env[key] = '1'
                    self.process = subprocess.Popen(self.cycle_command(), cwd=PROJECT, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                        start_new_session=True)
                    process = self.process
                    self.state.update(state='updating', pid=process.pid, next_update_ms=None)
                stdout, stderr = process.communicate()
                status = 'error' if process.returncode else _read_worker_result(stdout)
                if not self.done.is_set():
                    print(f"预测更新：{status}", flush=True)
                    if process.returncode:
                        log = Path(self.config['root'])/'web_worker_errors.log'
                        with log.open('a') as stream:
                            stream.write(time.strftime('%Y-%m-%d %H:%M:%S')+'\n'+stdout[-8000:]+stderr[-8000:]+'\n')
                with self.lock:
                    self.process = None
                    self.state.update(state='error' if status == 'error' else 'idle', pid=None,
                        last_status=status, next_update_ms=int((time.time()+self.interval)*1000))
            except Exception as exc:
                print(f'预测更新失败：{exc}', file=sys.stderr, flush=True)
                with self.lock:
                    self.process = None
                    self.state.update(state='error', pid=None,
                        next_update_ms=int((time.time()+self.interval)*1000))
            if self.done.wait(self.interval):
                break
        with self.lock:
            self.state.update(state='stopped', pid=None, next_update_ms=None)

    def stop(self):
        self.done.set()
        with self.lock:
            process = self.process
        if process is not None and process.poll() is None:
            try:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            except ProcessLookupError:
                pass
        if self.thread.ident is not None:
            self.thread.join(timeout=6)


def _read_worker_result(stdout):
    try:
        return json.loads(stdout).get('status', 'error')
    except ValueError:
        return 'error'


def make_server(config, *, host='127.0.0.1', port=8765, worker=None, initial_rank=1000):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, body, content_type='application/json; charset=utf-8'):
            raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: https://bestdori.com; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = urlsplit(self.path)
            try:
                if path.path == '/healthz':
                    return self.send(200, {'status':'ok'})
                if path.path == '/readyz':
                    latest = _read_json(Path(config['root'])/'latest.json', {})
                    status = _read_json(Path(config['root'])/'status.json', {})
                    age = int(time.time()*1000)-latest.get('as_of_ms', 0)
                    ready = bool(latest.get('model_id') and status.get('status') != 'error'
                                 and age <= max(3600000, config['interval_seconds']*3000))
                    return self.send(200 if ready else 503, {'ready':ready,
                        'model_id':latest.get('model_id'), 'status':status.get('status','waiting_for_model')})
                if path.path == '/api/predictions':
                    query=parse_qs(path.query)
                    event_id=int(query['event'][0]) if 'event' in query else None
                    return self.send(200, public_snapshot(config, service=worker.snapshot() if worker else None, initial_rank=initial_rank,event_id=event_id))
                if path.path in ('/api/history', '/api/curve'):
                    query = parse_qs(path.query)
                    event, rank = int(query['event'][0]), int(query['rank'][0])
                    if event <= 0 or rank not in config['ranks']:
                        raise ValueError('Invalid event/rank')
                    if path.path == '/api/curve':
                        return self.send(200, observed_curve(config['root'], event, rank))
                    return self.send(200, {'event':event, 'rank':rank,
                        'points':forecast_history(config['root'], event, rank)})
                files = {'/':('index.html', 'text/html; charset=utf-8'),
                         '/favicon.ico':('favicon.ico', 'image/x-icon'),
                         '/bangdream-icon.png':('bangdream-icon.png', 'image/png'),
                         '/components.js':('components.js', 'text/javascript; charset=utf-8'),
                         '/app.js':('app.js', 'text/javascript; charset=utf-8'),
                         '/style.css':('style.css', 'text/css; charset=utf-8')}
                if path.path in files:
                    name, mime = files[path.path]
                    return self.send(200, (ASSETS/name).read_bytes(), mime)
                return self.send(404, {'error':'页面不存在'})
            except (KeyError, ValueError, IndexError):
                return self.send(400, {'error':'活动或分数线参数无效'})
            except (OSError, sqlite3.Error):
                return self.send(503, {'error':'暂时无法读取预测记录，请稍后刷新'})

        def do_POST(self):
            self.send(405, {'error':'此网页仅提供读取结果的功能'})

    return ThreadingHTTPServer((host, port), Handler)


@contextmanager
def _service_lock(root):
    with (Path(root)/'foreground.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('此数据目录已有持续网页服务运行') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def serve(config, config_path, *, host='127.0.0.1', port=8765, interval=None, offline=False, initial_rank=1000, worker_factory=CycleWorker):
    interval = config['interval_seconds'] if interval is None else interval
    if interval < 60:
        raise ValueError('更新间隔至少60秒')
    Path(config['root']).mkdir(parents=True, exist_ok=True)
    with _service_lock(config['root']):
        worker = worker_factory(config, config_path, interval=interval, offline=offline)
        server = make_server(config | {'interval_seconds':interval}, host=host, port=port,
                             worker=worker, initial_rank=initial_rank)
        prior_signals = {}
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGHUP):
                prior_signals[sig] = signal.signal(sig, interrupt)
        try:
            print(f'网页：http://{host}:{server.server_port}', flush=True)
            print(f'每{interval}秒串行更新；按 Ctrl+C 停止。', flush=True)
            worker.start()
            server.serve_forever(poll_interval=.25)
        except KeyboardInterrupt:
            print('\n正在停止预测和网页服务……', flush=True)
        finally:
            worker.stop()
            server.server_close()
            for sig, handler in prior_signals.items():
                signal.signal(sig, handler)
        print('已停止，无后台自动任务。', flush=True)
    return 0
