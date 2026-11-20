from pathlib import Path
from contextlib import closing
from unittest.mock import patch
import json,shutil,sqlite3,tempfile,time,os,subprocess,sys
from collections import Counter
import argparse

parser=argparse.ArgumentParser(description='在生产状态的临时副本上检查模型复用；不启动网页、不联网、不训练')
parser.add_argument('--state',type=Path,default=Path('runtime/live'))
args=parser.parse_args()
from production import model, selected_runtime
from production.data import Store

source=args.state.resolve()
latest=json.loads((source/'latest.json').read_text())
clock=latest['as_of_ms']
report={'scope':'Offline technical performance check; no training or accuracy evaluation',
        'clock_ms':clock,'source_mutated':False,'web_server_started':False}
with tempfile.TemporaryDirectory(prefix='bangdream-inference-check-') as tmp:
 root=Path(tmp)/'state';root.mkdir()
 for name in ('data.sqlite3','state.sqlite3'):
  with closing(sqlite3.connect((source/name).as_uri()+'?mode=ro',uri=True)) as src, closing(sqlite3.connect(root/name)) as dest:
   src.backup(dest)
 shutil.copytree(source/'metadata',root/'metadata')
 shutil.copytree(source/'models',root/'models')
 cfg=json.loads((source/'config.json').read_text())|{'root':str(root),'automatic_training':False,'selected_seed_run':None}
 queries=[]
 class MeasuredStore(Store):
  def load_curves(self,*args,**kwargs):
   start=time.perf_counter(); values=super().load_curves(*args,**kwargs)
   queries.append({'complete_only':kwargs.get('complete_only',True),'events':kwargs.get('events'),
                   'curves':len(values),'seconds':round(time.perf_counter()-start,6)})
   return values
 cache=model.InferenceCache();predictions=[]
 for phase in ('cold','warm'):
  queries.clear();start=time.perf_counter()
  with patch.object(selected_runtime,'Store',MeasuredStore), patch.object(model.joblib,'load',wraps=model.joblib.load) as loads, patch.object(model,'_verify_pack',wraps=model._verify_pack) as checks, model.inference_session(cache):
   result=selected_runtime.run_cycle(cfg,now_ms=clock,sync=False)
  report[phase]={'seconds':round(time.perf_counter()-start,6),'status':result['status'],
                 'predictions':len(result['predictions']),'full_model_verifications':checks.call_count,
                 'joblib_loads':dict(Counter(Path(c.args[0]).name for c in loads.call_args_list)),
                 'curve_reads':list(queries),'errors':result['errors'],
                 'resident_model_versions':len(cache._verified),
                 'process_rss_mib':round(int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())]))/1024,2)}
  predictions.append(result['predictions'])
 report['cold_warm_predictions_identical']=predictions[0]==predictions[1]
 # Query volume of the former three broad reads, without decoding their JSON.
 events=Store(root).events(clock)
 before=min((e['start_ms'] for e in events if e['start_ms']<=clock<e['end_ms']),default=clock)
 from analysis.data_policy import MIN_EVENT_START_MS
 from download_ranks import RANKS
 with closing(sqlite3.connect(root/'data.sqlite3')) as con:
  broad=[]
  for name,end,completed in [('feedback',clock,True),('unused_training_history',before,True),('prediction',clock,False)]:
   condition=' AND e.end_ms<?' if completed else ''
   params=[MIN_EVENT_START_MS,end,*RANKS]+([end] if completed else [])
   row=con.execute('SELECT COUNT(*),SUM(LENGTH(points_json)) FROM curves c JOIN events e USING(event) WHERE e.start_ms>=? AND e.start_ms<=? AND c.rank IN ('+','.join('?' for _ in RANKS)+')'+condition,params).fetchone()
   broad.append({'purpose':name,'curve_rows':row[0],'json_bytes':row[1] or 0})
  report['previous_broad_read_volume']=broad
report['reuse_regression_log']='/tmp/bangdream-inference-regressions.log'
report['web_parent_memory']={}
for name,module in [('numerical_stack_import','production.runtime'),('lightweight_web_import','production.app')]:
 code=f"import {module},os,subprocess; print(int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())]))/1024)"
 report['web_parent_memory'][name+'_rss_mib']=round(float(subprocess.check_output([sys.executable,'-c',code],text=True)),2)
report['memory_scope']='RSS in fresh local processes; not a container memory limit or private-memory total'
print(json.dumps(report,ensure_ascii=False,indent=2))
