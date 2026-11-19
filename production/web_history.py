"""Clock-guarded, read-only views of stored activities and issued forecasts."""
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3

from analysis.data_policy import MIN_EVENT_START_MS


def _rows(root, filename, sql, parameters=()):
    database=Path(root).resolve()/filename
    if not database.is_file():
        return []
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=2)) as con:
        con.row_factory=sqlite3.Row
        return [dict(row) for row in con.execute(sql,parameters).fetchall()]


def event_catalog(root, ranks, now_ms):
    slots=','.join('?' for _ in ranks)
    rows=_rows(root,'data.sqlite3',f'''SELECT e.*,COUNT(c.rank) AS curve_ranks
        FROM events e LEFT JOIN curves c ON c.event=e.event AND c.rank IN ({slots})
        WHERE e.start_ms>=? AND e.start_ms<=? AND e.end_ms>e.start_ms
        GROUP BY e.event HAVING COUNT(c.rank)>0 OR e.end_ms>?
        ORDER BY e.start_ms DESC,e.event DESC''',(*ranks,MIN_EVENT_START_MS,now_ms,now_ms))
    counts={r['event']:r['cases'] for r in _rows(root,'state.sqlite3',
        'SELECT event,COUNT(*) AS cases FROM forecasts WHERE issued_ms<=? GROUP BY event',(now_ms,))}
    return [row|{'live_forecasts':counts.get(row['event'],0)} for row in rows]


def curve_points(row, now_ms):
    """Keep source timestamps, including the established five-minute final window."""
    upper=min(now_ms,row['end_ms']+300000)
    points={}
    for point in json.loads(row.get('points_json') or '[]'):
        timestamp,score=point.get('time'),point.get('ep')
        if (isinstance(timestamp,int) and not isinstance(timestamp,bool)
            and isinstance(score,(int,float)) and not isinstance(score,bool)
            and math.isfinite(score) and score>=0 and row['start_ms']<=timestamp<=upper):
            points[timestamp]={'time':timestamp,'ep':float(score)}
    return [points[key] for key in sorted(points)]


def terminal_score(row, points, now_ms):
    if (row.get('verified_closed') and row['end_ms']<now_ms and len(points)>=5
        and row['end_ms']-60000<=points[-1]['time']<=row['end_ms']+300000
        and points[-1]['ep']>0):
        return {'ep':points[-1]['ep'],'reported_at':points[-1]['time'],'event_end_ms':row['end_ms']}
    return None


def event_rows(root, event, ranks, now_ms):
    raw=_rows(root,'data.sqlite3','''SELECT e.event,e.start_ms,e.end_ms,c.* FROM events e
        JOIN curves c ON c.event=e.event WHERE e.event=? AND e.start_ms>=? AND e.start_ms<=?''',
        (event['event'],MIN_EVENT_START_MS,now_ms))
    by_rank={row['rank']:row for row in raw}
    issued=_rows(root,'state.sqlite3','''SELECT rank,observed_at,issued_ms,prediction,raw_prediction,model_id,0 AS revision,rowid AS seq
        FROM forecasts WHERE event=? AND issued_ms<=? AND observed_at<=? UNION ALL
        SELECT rank,observed_at,issued_ms,prediction,raw_prediction,model_id,1 AS revision,rowid AS seq
        FROM forecast_revisions WHERE event=? AND issued_ms<=? AND observed_at<=?
        ORDER BY observed_at,issued_ms,revision,seq''',
        (event['event'],now_ms,now_ms,event['event'],now_ms,now_ms))
    latest={row['rank']:row for row in issued}
    historical=event['end_ms']<=now_ms
    result=[]
    for rank in ranks:
        source=by_rank.get(rank)
        points=curve_points(source,now_ms) if source else []
        last=points[-1] if points else None
        final=terminal_score(source,points,now_ms) if source else None
        record=latest.get(rank)
        observation_available=bool(last and (historical or now_ms-last['time']<=3600000))
        prediction_available=bool(record and (historical or now_ms-record['observed_at']<=3600000))
        result.append({'event':event['event'],'rank':rank,'start_ms':event['start_ms'],'end_ms':event['end_ms'],
            'current_ep':last['ep'] if last else None,'observed_at':last['time'] if last else None,
            'actual_final_ep':final['ep'] if final else None,
            'predicted_final_ep':record['prediction'] if prediction_available else None,
            'raw_predicted_final_ep':record['raw_prediction'] if prediction_available else None,
            'forecast_issued_at':record['issued_ms'] if record else None,
            'forecast_observed_at':record['observed_at'] if record else None,
            'observation_available':observation_available,'available':prediction_available,
            'historical':historical,'progress':1. if historical else
                max(0.,min(1.,(now_ms-event['start_ms'])/(event['end_ms']-event['start_ms']))),
            'reason':('该活动没有上线预测记录' if record is None else
                '预测观测已超过60分钟' if not prediction_available else None)})
    return result
