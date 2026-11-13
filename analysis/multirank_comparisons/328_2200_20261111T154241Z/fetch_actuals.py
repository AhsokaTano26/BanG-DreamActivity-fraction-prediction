"""Fetch end observations without accessing or modifying any prediction model."""
import asyncio,gzip,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import httpx
from download_ranks import RANKS,Limiter,request,normalize_cutoffs,normalize_top
from analysis.holdout_dataset import sha256,write_json
RUN=Path(__file__).resolve().parent
PRED=ROOT/'analysis/multirank_predictions/20261111_2200_replay'
async def main():
    seal=json.loads((PRED/'prediction_manifest.json').read_text())
    for name,digest in seal['artifacts'].items():
        if sha256(PRED/name)!=digest:raise ValueError('Sealed prediction artifact changed: '+name)
    pred=json.loads((PRED/'prediction.json').read_text());end=int(datetime.fromisoformat(pred['predictions'][0]['event_end']).timestamp()*1000)
    if (RUN/'raw').exists():raise FileExistsError('Never overwrite previously fetched actuals')
    (RUN/'raw').mkdir();limiter=Limiter(2.);rows=[]
    async with httpx.AsyncClient(timeout=30,follow_redirects=True) as client:
        for rank in RANKS:
            url='https://bestdori.com/api/eventtop/data' if rank==10 else 'https://bestdori.com/api/tracker/data'
            params={'server':3,'event':328}|({'mid':0,'interval':0} if rank==10 else {'tier':rank})
            try:
                response,attempts=await request(client,url,params,limiter)
                with gzip.open(RUN/f'raw/{rank}.json.gz','wb') as f:f.write(response.content)
                points,quality=(normalize_top if rank==10 else normalize_cutoffs)(response.json())
                nearby=[p for p in points if end-60000<=p['time']<=end+300000]
                row={'rank':rank,'fetched_at':datetime.now(timezone.utc).isoformat(),'url':str(response.url),'attempts':attempts,
                    'http_date':response.headers.get('date'),'http_age':response.headers.get('age'),'point_count':len(points),
                    'end_ms':end,'near_end_point_count':len(nearby),'near_end_final':nearby[-1] if nearby else None,
                    'latest_observation':points[-1] if points else None,'raw_sha256':sha256(RUN/f'raw/{rank}.json.gz')}
                if nearby:row['changes_after_final_window']=sum(p['ep']!=nearby[-1]['ep'] for p in points if p['time']>end+300000)
                rows.append(row)
                print(json.dumps(row),flush=True)
            except (ValueError,httpx.HTTPError) as exc:
                rows.append({'rank':rank,'error':str(exc)});print(json.dumps(rows[-1]),flush=True)
    write_json(RUN/'actual_observations.json',{'event':328,'prediction_sha256':sha256(PRED/'prediction.json'),
        'prediction_manifest_sha256':sha256(PRED/'prediction_manifest.json'),'definition':'latest observation within [end-60s,end+300s]; preserve later updates separately','rows':rows})
    write_json(RUN/'fetch_manifest.json',{'artifacts':{str(p.relative_to(RUN)):sha256(p) for p in RUN.rglob('*') if p.is_file()}})
asyncio.run(main())
