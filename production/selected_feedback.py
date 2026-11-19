"""Trusted migration of sealed out-of-sample RAW errors into production history."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.data_policy import MIN_EVENT_START_MS
from analysis.holdout_dataset import sha256
from analysis.rolling_benchmark import verify_seal
from production.state import atomic_json

REFERENCE=ROOT/'analysis/rolling_results/v3_multiline_calibration_20261117'
REFERENCE_ID='14888ceeff945c1fd387a0e03265ddde1283901a654a678b9b81ac4a71699f5d'
REFERENCE_MANIFEST_SHA256='97c4ef8043fc27fe4bdbbdb72919b3b301814ec801caac8417eaf222a6dea119'

def ensure_archive(ledger):
    """Archive seed-to-live replacements in the same SQLite write transaction.

    The original ledger uses INSERT OR REPLACE, so a BEFORE INSERT trigger
    captures the prior seed before SQLite removes it. No legacy source changes.
    """
    with ledger.connect() as con:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS selected_feedback_history(
            id INTEGER PRIMARY KEY,event INTEGER,rank INTEGER,phase INTEGER,
            old_row TEXT NOT NULL,replaced_ms INTEGER NOT NULL);
        CREATE TRIGGER IF NOT EXISTS selected_archive_seed_to_live
        BEFORE INSERT ON feedback
        WHEN NEW.origin='live' AND EXISTS(
            SELECT 1 FROM feedback WHERE event=NEW.event AND rank=NEW.rank
                AND phase=NEW.phase AND origin='retrospective_v3_seed')
        BEGIN
            INSERT INTO selected_feedback_history(event,rank,phase,old_row,replaced_ms)
            SELECT event,rank,phase,json_object(
                'event',event,'rank',rank,'phase',phase,'start_ms',start_ms,'end_ms',end_ms,
                'mean_log_error',mean_log_error,'cases',cases,'actual_final',actual_final,
                'origin',origin,'source',source,'updated_ms',updated_ms),NEW.updated_ms
            FROM feedback WHERE event=NEW.event AND rank=NEW.rank AND phase=NEW.phase;
        END;
        ''')

def merge_seed(ledger,rows,*,now_ms):
    unique={}
    for row in rows:
        r=dict(row);key=tuple(int(r[k]) for k in ['event','rank','phase'])
        if (r['origin']!='retrospective_v3_seed' or r['start_ms']<MIN_EVENT_START_MS
            or not r['start_ms']<r['end_ms']<now_ms or r['cases']<1 or key[1]<=0 or key[2] not in [0,1,2]
            or not math.isfinite(r['mean_log_error']) or not math.isfinite(r['actual_final']) or r['actual_final']<=0):
            raise ValueError('Invalid selected retrospective seed')
        if key in unique and unique[key]!=r:raise ValueError('Conflicting selected seed group')
        unique[key]=r
    result={'added':0,'replaced':0,'skipped_live':0}
    ensure_archive(ledger)
    with ledger.connect() as con:
        for key,r in unique.items():
            previous=con.execute('SELECT * FROM feedback WHERE event=? AND rank=? AND phase=?',key).fetchone()
            if previous and previous['origin']=='live':result['skipped_live']+=1;continue
            fields=['event','rank','phase','start_ms','end_ms','mean_log_error','cases','actual_final','origin','source']
            if previous and all(previous[k]==r.get(k) for k in fields):continue
            if previous:
                con.execute('INSERT INTO selected_feedback_history(event,rank,phase,old_row,replaced_ms) VALUES(?,?,?,?,?)',(*key,json.dumps(dict(previous),sort_keys=True),now_ms));result['replaced']+=1
            else:result['added']+=1
            con.execute('INSERT OR REPLACE INTO feedback VALUES(?,?,?,?,?,?,?,?,?,?,?)',tuple(r.get(k) for k in fields)+(now_ms,))
    return result

def seed_feedback(store,ledger,config,*,before_ms,now_ms):
    location=config.get('selected_seed_run')
    if location is None:return {'status':'not_configured','added':0}
    key='selected_seed:'+REFERENCE_ID+':'+REFERENCE_MANIFEST_SHA256
    previous=ledger.fact(key)
    if previous and previous['status']=='complete':
        receipt=Path(config['root'])/'selected_seed.json'
        if not receipt.exists():atomic_json(receipt,previous)
        return {'status':'not_needed','source_run_id':REFERENCE_ID,'added':0}
    run=Path(location);seal=verify_seal(run/'blind_manifest.json')
    if (sha256(run/'blind_manifest.json')!=REFERENCE_MANIFEST_SHA256 or seal['run_id']!=REFERENCE_ID
        or seal['protocol']!='rolling_multiline_calibration_v3'):raise ValueError('Wrong selected seed source')
    curves={(c.event,c.rank):c for c in store.load_curves(before_ms,ranks=config['ranks'])}
    rows=[];missing=[];source=str(run.resolve())+'#'+sha256(run/'blind_manifest.json')
    members=sorted(n for n in seal['artifacts'] if len(Path(n).parts)==3 and Path(n).parts[0]=='events'
                   and Path(n).parts[1].isdigit() and Path(n).name=='event_manifest.json')
    if not members:raise ValueError('Frozen seed has no activity members')
    for name in members:
        path=run/name
        eventseal=verify_seal(path)
        if eventseal['run_id']!=REFERENCE_ID:raise ValueError('Seed event belongs to another run')
        if eventseal['end_ms']>=before_ms:missing.append([eventseal['event'],'not_completed_before_start']);continue
        feature_path=path.parent/'features.csv'
        if seal['artifacts'].get(str(feature_path.relative_to(run)))!=sha256(feature_path):raise ValueError('Feature file is not a frozen seed member')
        f=pd.read_csv(feature_path,float_precision='round_trip')
        if f.case_id.duplicated().any() or set(f.event)!={eventseal['event']}:raise ValueError('Invalid sealed seed cases')
        for (event,rank,phase),part in f.groupby(['event','rank','phase']):
            if int(rank) not in config['ranks']:continue
            c=curves.get((int(event),int(rank)))
            if c is None:missing.append([int(event),int(rank)]);continue
            if (c.start<MIN_EVENT_START_MS or c.end>=before_ms or not (part.start_ms==c.start).all()
                or not (part.end_ms==c.end).all() or not np.isfinite(part.base_prediction).all()
                or (part.base_prediction<=0).any()):raise ValueError('Seed curve clock or raw prediction differs')
            rows.append(dict(event=c.event,rank=c.rank,phase=int(phase),start_ms=c.start,end_ms=c.end,
                mean_log_error=float(np.log(part.base_prediction.to_numpy(float)/c.final).mean()),cases=len(part),
                actual_final=c.final,origin='retrospective_v3_seed',source=source))
    result=merge_seed(ledger,rows,now_ms=now_ms)|{'status':'partial' if missing else 'complete',
        'source_run_id':REFERENCE_ID,'source_manifest_sha256':REFERENCE_MANIFEST_SHA256,'source_event_count':len(members),
        'source':source,'missing_curves':missing,'origin':'retrospective_v3_seed'}
    ledger.fact(key,result);atomic_json(Path(config['root'])/'selected_seed.json',result);return result
