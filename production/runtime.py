"""One resumable production cycle: synchronize, learn completed outcomes, predict."""
from __future__ import annotations
import asyncio
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import io
import csv
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd
from analysis.backtest import ROOT,snapshot,TZ
from analysis.holdout_dataset import prefix_curve,sha256
from analysis.rolling_benchmark import verify_seal
from download_ranks import RANKS
from production.data import Store,sync_data
from production.model import train_pack,verify_pack,predict_pack,build_bias
from production.state import Ledger,atomic_json

DEFAULT_ROOT=ROOT/'runtime/live'
DEFAULT_SEED=ROOT/'analysis/rolling_results/v3_dense_full_retrain_20261115'


def validate_root(root):
    path=Path(root).expanduser().resolve()
    if path==ROOT or path in ROOT.parents or any(path==ROOT/d or ROOT/d in path.parents for d in ['data','analysis','training','production','tests','docs']):
        raise ValueError('Runtime storage cannot replace source or frozen research directories')
    return path


def initialize(root=DEFAULT_ROOT,*,ranks=None,interval_seconds=900):
    root=validate_root(root);ranks=list(RANKS if ranks is None else ranks)
    if not ranks or len(set(ranks))!=len(ranks) or not set(ranks)<=set(RANKS):raise ValueError('Invalid rank scope')
    if int(interval_seconds)!=interval_seconds or interval_seconds<60:raise ValueError('Schedule interval must be at least60seconds')
    config={'schema':1,'root':str(root),'ranks':ranks,'interval_seconds':int(interval_seconds),
        'bootstrap_database':str(ROOT/'data/db.sqlite3'),'bootstrap_metadata':str(ROOT/'data/metadata'),
        'seed_run':str(DEFAULT_SEED) if (DEFAULT_SEED/'blind_manifest.json').exists() else None,
        'feedback_grace_ms':600000}
    path=root/'config.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['root']!=str(root) or old['ranks']!=ranks or old['interval_seconds']!=int(interval_seconds):
            raise FileExistsError('Configuration already exists; edit the existing config explicitly')
        return old
    atomic_json(path,config);return config


def read_config(path):
    cfg=json.loads(Path(path).read_text());validate_root(cfg['root'])
    if not cfg['ranks'] or len(set(cfg['ranks']))!=len(cfg['ranks']) or not set(cfg['ranks'])<=set(RANKS):raise ValueError('Invalid rank configuration')
    if cfg['interval_seconds']<60:raise ValueError('Invalid schedule interval')
    return cfg


@contextmanager
def cycle_lock(root):
    root=validate_root(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'cycle.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:yield False;return
        try:
            f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush();yield True
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def _append_log(root,record):
    path=root/'cycles.jsonl'
    with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')


def _activate(root,path,manifest):
    if not manifest.get('production_ready',False):raise ValueError('Fixture models cannot become production active')
    pointer={'model_id':manifest['model_id'],'path':str(Path(path).resolve()),'manifest_sha256':sha256(Path(path)/'manifest.json')}
    atomic_json(root/'models/current.json',pointer)
    return pointer


def _previous(root,*,before_ms):
    candidates=[]
    pointer=root/'models/current.json'
    if pointer.exists():
        try:candidates.append(Path(json.loads(pointer.read_text())['path']))
        except (OSError,ValueError,KeyError,TypeError):pass
    candidates += sorted((root/'models/versions').glob('*'),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in dict.fromkeys(candidates):
        try:
            manifest=verify_pack(path)
            if manifest.get('production_ready') and manifest['history_max_end_ms']<before_ms:return path,manifest
        except (ValueError,OSError,KeyError):continue
    raise ValueError('No valid causal model version is available')


def _seed_feedback(store,ledger,config,*,before_ms,now_ms):
    seed=config.get('seed_run')
    if not seed or ledger.fact('seed_complete'):return {'added':0,'status':'not_needed'}
    run=Path(seed);seal=verify_seal(run/'blind_manifest.json')
    if seal.get('protocol')!='rolling_holdout_v3':raise ValueError('Seed must be sealed causal V3 forecasts')
    curves={(c.event,c.rank):c for c in store.load_curves(before_ms,ranks=[1000,2000])}
    rows=[];missing=[]
    for split in ['validation','test']:
        frame=pd.read_csv(run/f'{split}_predictions.csv');frame=frame[frame.method=='rolling_context']
        for (event,rank,phase),part in frame.groupby(['event','rank','phase']):
            c=curves.get((int(event),int(rank)))
            if c is None:missing.append([int(event),int(rank)]);continue
            rows.append({'event':c.event,'rank':c.rank,'phase':int(phase),'start_ms':c.start,'end_ms':c.end,
                'mean_log_error':float(np.log(part.predicted.to_numpy(float)/c.final).mean()),'cases':len(part),
                'actual_final':c.final,'origin':'retrospective_v3_seed','source':str(run.resolve())+'#'+sha256(run/'blind_manifest.json')})
    added=ledger.seed(rows,now_ms=now_ms)
    result={'added':added,'missing_curves':sorted(set(tuple(p) for p in missing)),'base_run_id':seal['run_id'],
            'origin':'retrospective_v3_seed','status':'complete' if not missing else 'partial'}
    ledger.fact('seed_complete',result if not missing else False)
    atomic_json(Path(config['root'])/'calibration_seed.json',result)
    return result


def run_cycle(config,*,now_ms=None,sync=True,transport=None):
    root=validate_root(config['root']);now=int(time.time()*1000) if now_ms is None else int(now_ms)
    with cycle_lock(root) as acquired:
        if not acquired:return {'status':'busy','as_of_ms':now,'predictions':[]}
        errors=[];start=time.monotonic();store=Store(root);ledger=Ledger(root)
        try:
            if not ledger.fact('bootstrap_complete'):
                store.bootstrap(database=Path(config['bootstrap_database']),metadata_directory=Path(config['bootstrap_metadata']),now_ms=now)
                ledger.fact('bootstrap_complete',True)
            atomic_json(root/'status.json',{'status':'running','phase':'sync','as_of_ms':now,'ledger':ledger.summary()})
            sync_result={'status':'offline','errors':[]}
            if sync:
                try:sync_result=asyncio.run(sync_data(store,now_ms=now,ranks=config['ranks'],transport=transport))
                except Exception as exc:errors.append({'stage':'sync','error':f'{type(exc).__name__}: {exc}'})
                errors.extend({'stage':'sync','error':str(e)} for e in sync_result.get('errors',[]))
            events=store.events(now)
            active=[e for e in events if e['start_ms']<=now<e['end_ms']]
            before=min((e['start_ms'] for e in active),default=now)
            completed=store.load_curves(now,ranks=config['ranks'])
            feedback_updates=ledger.learn_completed(completed,now_ms=now,grace_ms=config.get('feedback_grace_ms',600000))
            try:seed_result=_seed_feedback(store,ledger,config,before_ms=before,now_ms=now)
            except Exception as exc:
                errors.append({'stage':'calibration_seed','error':f'{type(exc).__name__}: {exc}'});seed_result={'status':'failed'}
            history=store.load_curves(before,ranks=config['ranks'])
            atomic_json(root/'status.json',{'status':'running','phase':'training','as_of_ms':now,'eligible_curves':len(history),'ledger':ledger.summary()})
            override=root/'models/manual_override.json'
            if override.exists():
                chosen=json.loads(override.read_text());model_path=Path(chosen['path']);manifest=verify_pack(model_path)
                if manifest['history_max_end_ms']>=before:raise ValueError('Rollback model is not causal for this event')
            else:
                try:
                    model_path=train_pack(history,store.metadata_directory,root/'models/versions',before_ms=before,ranks=config['ranks'])
                    manifest=verify_pack(model_path);_activate(root,model_path,manifest)
                except Exception as exc:
                    errors.append({'stage':'training','error':f'{type(exc).__name__}: {exc}'})
                    model_path,manifest=_previous(root,before_ms=before)
            curve_map={(c.event,c.rank):c for c in store.load_curves(now,ranks=config['ranks'],complete_only=False)}
            output=[];added=0;pinned_models={}
            for event in active:
                pin=ledger.pin(event['event'])
                if pin is None:
                    calibration=build_bias(ledger.feedback_rows(before_ms=event['start_ms']),before_ms=event['start_ms'],exclude_event_id=event['event'])
                    pin=ledger.create_pin(event,model_path,manifest['model_id'],calibration,now_ms=now)
                pinned=Path(pin['model_path'])
                try:pin_manifest=verify_pack(pinned)
                except (ValueError,OSError,KeyError) as exc:
                    replacement,replacement_manifest=_previous(root,before_ms=event['start_ms'])
                    errors.append({'stage':'pin_recovery','event':event['event'],'error':str(exc),'replacement':replacement_manifest['model_id']})
                    pin=ledger.replace_unusable_pin(event['event'],replacement,replacement_manifest['model_id'],reason=str(exc),now_ms=now)
                    pinned=replacement;pin_manifest=replacement_manifest
                if pin_manifest['history_max_end_ms']>=event['start_ms']:raise ValueError('Pinned model uses current or future event data')
                samples=[];unavailable=[]
                for rank in config['ranks']:
                    c=curve_map.get((event['event'],rank))
                    prefix=None if c is None else prefix_curve(c,now)
                    sample=None if prefix is None else snapshot(prefix,now)
                    if sample is None:
                        unavailable.append({'event':event['event'],'rank':rank,'available':False,'reason':'missing, insufficient, or observations older than60minutes'});continue
                    sample.update(case_id=f"live:{event['event']}:{rank}:{now}",stage='live');samples.append(sample)
                rows=predict_pack(pinned,samples,metadata_directory=store.metadata_directory,calibration=pin['calibration']) if samples else []
                added+=ledger.record(event,rows,cutoff_ms=now,issued_ms=now,model_id=pin['model_id'])
                output.extend(rows+unavailable)
                pinned_models[str(event['event'])]={'model_id':pin['model_id'],'path':str(pinned)}
            result={'status':'degraded' if errors else ('ok' if active else 'waiting_for_event'),'as_of_ms':now,
                'as_of':datetime.fromtimestamp(now/1000,TZ).isoformat(),'ranks':config['ranks'],
                'model_id':manifest['model_id'],'model_path':str(Path(model_path).resolve()),
                'automatic_model_updates':not override.exists(),
                'active_event_models':pinned_models,
                'active_events':[e['event'] for e in active],'predictions':sorted(output,key=lambda r:(r['event'],r['rank'])),
                'new_issued_forecasts':added,'feedback_updates':feedback_updates,'calibration_seed':seed_result,
                'ledger':ledger.summary(),'sync':sync_result,'errors':errors,'duration_seconds':round(time.monotonic()-start,3),
                'scope':'Production on all eligible completed dense histories; benchmark test metrics do not describe this latest fit.'}
            atomic_json(root/'latest.json',result);atomic_json(root/'status.json',{k:v for k,v in result.items() if k!='predictions'})
            buffer=io.StringIO();fields=list(dict.fromkeys(k for r in result['predictions'] for k in r))
            if fields:
                writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader();writer.writerows(result['predictions'])
            path=root/'latest.csv';fd,tmp=tempfile.mkstemp(prefix='.latest-',dir=root)
            with os.fdopen(fd,'w') as f:f.write(buffer.getvalue())
            Path(tmp).replace(path)
            _append_log(root,{'as_of_ms':now,'status':result['status'],'model_id':manifest['model_id'],'predictions':len(output),'new_forecasts':added,'feedback_updates':feedback_updates,'errors':errors})
            return result
        except Exception as exc:
            failure={'status':'error','as_of_ms':now,'error':f'{type(exc).__name__}: {exc}','errors':errors,'ledger':ledger.summary()}
            atomic_json(root/'status.json',failure);_append_log(root,failure)
            raise


def rollback(config,model_id):
    root=validate_root(config['root']);path=root/'models/versions'/model_id
    if path.parent!=root/'models/versions' or path.name!=model_id:raise ValueError('Invalid version ID')
    manifest=verify_pack(path)
    if not manifest.get('production_ready'):raise ValueError('Cannot roll back to fixture model')
    with cycle_lock(root) as acquired:
        if not acquired:raise RuntimeError('Cycle is running; retry rollback after it completes')
        pointer=_activate(root,path,manifest);atomic_json(root/'models/manual_override.json',pointer)
        # Make rollback effective immediately at the next prediction cycle.
        with Ledger(root).connect() as con:con.execute('DELETE FROM pins WHERE end_ms>?',(int(time.time()*1000),))
    return pointer
