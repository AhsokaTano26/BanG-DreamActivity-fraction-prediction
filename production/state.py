"""Idempotent live forecast/feedback ledger and immutable activity pins."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import numpy as np


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.'+path.name,delete=False) as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno());temporary=Path(f.name)
    temporary.replace(path)


class Ledger:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.database=self.root/'state.sqlite3'
        with self.connect() as con:
            con.executescript('''
            CREATE TABLE IF NOT EXISTS pins(event INTEGER PRIMARY KEY,start_ms INTEGER NOT NULL,end_ms INTEGER NOT NULL,model_path TEXT NOT NULL,model_id TEXT NOT NULL,calibration TEXT NOT NULL,created_ms INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS forecasts(id TEXT PRIMARY KEY,event INTEGER NOT NULL,rank INTEGER NOT NULL,phase INTEGER NOT NULL,start_ms INTEGER NOT NULL,end_ms INTEGER NOT NULL,cutoff_ms INTEGER NOT NULL,observed_at INTEGER NOT NULL,issued_ms INTEGER NOT NULL,model_id TEXT NOT NULL,raw_prediction REAL NOT NULL,prediction REAL NOT NULL,current REAL NOT NULL,UNIQUE(event,rank,observed_at));
            CREATE TABLE IF NOT EXISTS forecast_revisions(id TEXT PRIMARY KEY,event INTEGER NOT NULL,rank INTEGER NOT NULL,phase INTEGER NOT NULL,start_ms INTEGER NOT NULL,end_ms INTEGER NOT NULL,cutoff_ms INTEGER NOT NULL,observed_at INTEGER NOT NULL,issued_ms INTEGER NOT NULL,model_id TEXT NOT NULL,raw_prediction REAL NOT NULL,prediction REAL NOT NULL,current REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS feedback(event INTEGER NOT NULL,rank INTEGER NOT NULL,phase INTEGER NOT NULL,start_ms INTEGER NOT NULL,end_ms INTEGER NOT NULL,mean_log_error REAL NOT NULL,cases INTEGER NOT NULL,actual_final REAL NOT NULL,origin TEXT NOT NULL,source TEXT,updated_ms INTEGER NOT NULL,PRIMARY KEY(event,rank,phase));
            CREATE TABLE IF NOT EXISTS facts(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pin_history(id INTEGER PRIMARY KEY,event INTEGER NOT NULL,old_pin TEXT NOT NULL,reason TEXT NOT NULL,replaced_ms INTEGER NOT NULL);
            ''')
            schema=con.execute("SELECT sql FROM sqlite_master WHERE name='forecast_revisions'").fetchone()[0]
            if 'UNIQUE(event,rank,observed_at,model_id' in schema:
                con.execute('BEGIN')
                con.execute('ALTER TABLE forecast_revisions RENAME TO forecast_revisions_old')
                con.execute('CREATE TABLE forecast_revisions AS SELECT * FROM forecast_revisions_old WHERE 0')
                con.execute('CREATE UNIQUE INDEX forecast_revision_id ON forecast_revisions(id)')
                con.execute('INSERT INTO forecast_revisions SELECT * FROM forecast_revisions_old')
                con.execute('DROP TABLE forecast_revisions_old')

    @contextmanager
    def connect(self):
        con=sqlite3.connect(self.database,timeout=30);con.row_factory=sqlite3.Row
        try:
            with con:yield con
        finally:con.close()

    def pin(self,event):
        with self.connect() as con:r=con.execute('SELECT * FROM pins WHERE event=?',(int(event),)).fetchone()
        return None if r is None else dict(r)|{'calibration':json.loads(r['calibration'])}

    def create_pin(self,event,model_path,model_id,calibration,*,now_ms):
        with self.connect() as con:
            con.execute('INSERT INTO pins VALUES(?,?,?,?,?,?,?)',(event['event'],event['start_ms'],event['end_ms'],str(Path(model_path).resolve()),model_id,json.dumps(calibration,sort_keys=True),now_ms))
        return self.pin(event['event'])

    def replace_unusable_pin(self,event,model_path,model_id,*,reason,now_ms):
        """Keep the original record; replace only an explicitly unusable pin."""
        prior=self.pin(event)
        if prior is None:raise ValueError('Missing activity pin')
        with self.connect() as con:
            con.execute('INSERT INTO pin_history(event,old_pin,reason,replaced_ms) VALUES(?,?,?,?)',(event,json.dumps(prior,sort_keys=True),reason,now_ms))
            con.execute('UPDATE pins SET model_path=?,model_id=? WHERE event=?',(str(Path(model_path).resolve()),model_id,event))
        return self.pin(event)

    def record(self,event,rows,*,cutoff_ms,issued_ms,model_id):
        added=0
        with self.connect() as con:
            for row in rows:
                if row.get('available',True) is False or 'raw_predicted_final_ep' not in row:continue
                rank=int(row['rank']);observed=int(row['observed_at']);raw=float(row['raw_predicted_final_ep']);pred=float(row['predicted_final_ep']);current=float(row['current_ep'])
                if not event['start_ms']<observed<=cutoff_ms<event['end_ms'] or issued_ms<cutoff_ms or int(row['phase']) not in (0,1,2) or not np.isfinite([raw,pred,current]).all() or min(raw,pred,current)<=0 or pred<current:raise ValueError('Invalid issued live forecast')
                key=hashlib.sha256(f"{event['event']}:{rank}:{observed}".encode()).hexdigest()
                prior=con.execute('SELECT * FROM forecast_revisions WHERE event=? AND rank=? AND observed_at=? ORDER BY issued_ms DESC,rowid DESC LIMIT 1',(event['event'],rank,observed)).fetchone()
                if prior is None:prior=con.execute('SELECT * FROM forecasts WHERE id=?',(key,)).fetchone()
                if prior and all(prior[k]==v for k,v in [('model_id',model_id),('raw_prediction',raw),('prediction',pred),('current',current)]):continue
                table='forecasts' if prior is None else 'forecast_revisions'
                if prior is not None:
                    sequence=con.execute('SELECT COALESCE(MAX(rowid),0)+1 FROM forecast_revisions').fetchone()[0]
                    key=hashlib.sha256(json.dumps([event['event'],rank,observed,model_id,raw,pred,current,issued_ms,sequence]).encode()).hexdigest()
                added+=con.execute('INSERT OR IGNORE INTO '+table+' VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(key,event['event'],rank,int(row['phase']),event['start_ms'],event['end_ms'],cutoff_ms,observed,issued_ms,model_id,raw,pred,current)).rowcount
        return added

    def feedback_rows(self,*,before_ms):
        with self.connect() as con:rows=con.execute('SELECT * FROM feedback WHERE end_ms<? ORDER BY end_ms,event,rank,phase',(before_ms,)).fetchall()
        return [dict(row) for row in rows]

    def seed(self,rows,*,now_ms):
        added=0
        with self.connect() as con:
            for r in rows:
                if not r['end_ms']<now_ms or r['origin']!='retrospective_v3_seed':raise ValueError('Invalid seed feedback clock/origin')
                added+=con.execute('INSERT OR IGNORE INTO feedback VALUES(?,?,?,?,?,?,?,?,?,?,?)',(r['event'],r['rank'],r['phase'],r['start_ms'],r['end_ms'],r['mean_log_error'],r['cases'],r['actual_final'],r['origin'],r.get('source'),now_ms)).rowcount
        return added

    def learn_completed(self,curves,*,now_ms,grace_ms=600000):
        finals={(c.event,c.rank):c for c in curves if c.end+grace_ms<now_ms}
        updated=0
        with self.connect() as con:
            for (event,rank),curve in finals.items():
                first=con.execute('SELECT *,0 AS revision FROM forecasts WHERE event=? AND rank=?',(event,rank)).fetchall()
                revisions=con.execute('SELECT *,1 AS revision FROM forecast_revisions WHERE event=? AND rank=? ORDER BY rowid',(event,rank)).fetchall()
                latest={}
                for row in sorted(list(first)+list(revisions),key=lambda r:(r['issued_ms'],r['revision'])):latest[row['observed_at']]=row
                rows=list(latest.values())
                if not rows:continue
                for phase in sorted({r['phase'] for r in rows}):
                    part=[r for r in rows if r['phase']==phase]
                    # Unique observed times were enforced when issued. Each event
                    # contributes one mean per rank/phase regardless of polls.
                    error=float(np.mean([np.log(r['raw_prediction'])-np.log(curve.final) for r in part]))
                    previous=con.execute('SELECT * FROM feedback WHERE event=? AND rank=? AND phase=?',(event,rank,phase)).fetchone()
                    if previous and previous['origin']=='live' and previous['actual_final']==curve.final and previous['cases']==len(part):continue
                    con.execute('INSERT OR REPLACE INTO feedback VALUES(?,?,?,?,?,?,?,?,?,?,?)',(event,rank,phase,curve.start,curve.end,error,len(part),curve.final,'live',None,now_ms));updated+=1
        return updated

    def fact(self,key,value=None):
        with self.connect() as con:
            if value is not None:con.execute('INSERT OR REPLACE INTO facts VALUES(?,?)',(key,json.dumps(value,ensure_ascii=False)))
            row=con.execute('SELECT value FROM facts WHERE key=?',(key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def summary(self):
        with self.connect() as con:
            counts={table:con.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ['pins','forecasts','forecast_revisions','feedback']}
            counts['live_feedback_groups']=con.execute("SELECT COUNT(*) FROM feedback WHERE origin='live'").fetchone()[0]
            counts['last_issued_ms']=con.execute('SELECT MAX(issued_ms) FROM (SELECT issued_ms FROM forecasts UNION ALL SELECT issued_ms FROM forecast_revisions)').fetchone()[0]
        return counts
