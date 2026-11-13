"""Register future-cohort rules and verify whether a CN activity is available now."""
import json
from datetime import datetime,timezone
from pathlib import Path
import httpx
from analysis.holdout_dataset import sha256,write_json
from analysis.temporal_dataset import STAGES

r=Path(__file__).resolve().parent
if (r/'prospective_registration.json').exists():raise FileExistsError('Registration already frozen')
now=datetime.now(timezone.utc);t=round(now.timestamp()*1000)
url='https://bestdori.com/api/events/all.5.json'
response=httpx.get(url,timeout=30,follow_redirects=True);response.raise_for_status()
(r/'next_event_metadata.json').write_bytes(response.content)
ongoing=[];upcoming=[]
for event,info in response.json().items():
 starts,ends=info.get('startAt',[]),info.get('endAt',[])
 if len(starts)<=3 or len(ends)<=3 or starts[3] is None or ends[3] is None:continue
 start,end=int(starts[3]),int(ends[3])
 row={'event':int(event),'start_ms':start,'end_ms':end}
 if start<=t<end:ongoing.append(row)
 if start>t:upcoming.append(row)
manifest=json.loads((r/'run_manifest.json').read_text())
write_json(r/'prospective_registration.json',{'registered_at':now.isoformat(),
 'model_identities':manifest['model_identities'],'selection':manifest['selection'],'combination':manifest['combination'],
 'population':'first five CN activities with actual start after registration; no activity chosen by observed error',
 'ranks':[1000,2000],'requested_stages':STAGES,'fit_policy':'no refit or member selection on new outcomes',
 'forecast_provenance':'forecast must actually be sealed before event end; do not backdate a replay as a stage-time forecast',
 'outcome_policy':'last observation between end-60s and end+300s; missing remains pending',
 'ongoing_cn_activities':ongoing,'upcoming_announced_cn_activities':sorted(upcoming,key=lambda x:x['start_ms']),
 'automatic_collection_running':False,'status':'waiting_for_new_cn_activity' if not ongoing else 'live_activity_available',
 'source':url,'http_date':response.headers.get('date'),'response_sha256':sha256(r/'next_event_metadata.json')})
print(json.dumps({'ongoing':ongoing,'announced_upcoming':upcoming,'automatic_collection_running':False},ensure_ascii=False))
