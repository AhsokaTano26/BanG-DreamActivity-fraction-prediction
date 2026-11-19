"""Independent scalar fixed-DLM transfer check, posthoc identity retained."""
import csv,gzip,json,math,statistics,hashlib
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];D=BASE/'workspaces/D';RUN=D/'posthoc_transfer_run_001'
def rows(p):
    p=Path(p)
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return list(csv.DictReader(f))
def j(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mm(a,b):return [[sum(a[i][k]*b[k][z] for k in range(2)) for z in range(2)] for i in range(2)]
cal=j(D/'calibration_run_v2_001/calibration_seal.json');g=cal['best'];qc,qd,r,rho,m=[float(g[k]) for k in ['qc','qd','r','rho','m']];a=1-m;b=-m;q=[[qc+a*a*qd,qc+a*b*qd],[qc+a*b*qd,qc+b*b*qd]];noise=[[r,r*rho],[r*rho,r]]
raw=rows(BASE/'inputs/train_residuals.csv.gz');qs=sorted(j(BASE/'inputs/train_queries.json'),key=lambda x:(x['start_ms'],x['event']));ids=[x['event'] for x in qs];means=defaultdict(list)
for x in raw:means[int(x['event']),int(x['rank']),int(x['phase'])].append(float(x['log_error']))
means={k:statistics.fmean(v) for k,v in means.items()};pred=rows(RUN/'issued_DLM.csv');lookup={r['case_id']:r for r in raw};out=[];maxbias=0.;maxpred=0.;bad=[]
for query in qs[25:]:
    history=[x for x in qs if x['end_ms']<query['start_ms']];state=list(cal['mu']);p=[[.1,0.],[0.,.1]];previous=-1
    for old in history:
        ordinal=ids.index(old['event']);y=[means[old['event'],1000,1],means[old['event'],2000,1]];prior=[[p[i][k]+(ordinal-previous)*q[i][k] for k in range(2)] for i in range(2)];s=[[prior[i][k]+noise[i][k] for k in range(2)] for i in range(2)];det=s[0][0]*s[1][1]-s[0][1]*s[1][0];inverse=[[s[1][1]/det,-s[0][1]/det],[-s[1][0]/det,s[0][0]/det]];gain=mm(prior,inverse);v=[y[k]-state[k] for k in range(2)];state=[state[i]+sum(gain[i][k]*v[k] for k in range(2)) for i in range(2)];gp=mm(gain,prior);p=[[prior[i][k]-gp[i][k] for k in range(2)] for i in range(2)];previous=ordinal
    group=[r for r in pred if int(r['event'])==query['event']];ape=[];loge=[]
    for r in group:
        x=lookup[r['case_id']];bias=max(-.3,min(.3,state[0 if int(r['rank'])==1000 else 1]));expected=max(float(x['current']),min(float(x['current'])*math.exp(4),float(x['base_prediction'])*math.exp(-bias)));maxbias=max(maxbias,abs(bias-float(r['bias'])));maxpred=max(maxpred,abs(expected-float(r['prediction'])));ape.append(100*abs(expected/float(x['actual'])-1));loge.append((float(x['log_error'])-bias)**2)
        if int(r['history_complete_pairs'])!=len(history) or int(x['phase'])!=1 or int(x['rank']) not in [1000,2000]:bad.append(r['case_id'])
    out.append({'event':query['event'],'cases':len(group),'mape':statistics.fmean(ape),'log_mse':statistics.fmean(loge),'history_pairs':len(history),'max_history_end_ms':max(x['end_ms'] for x in history),'query_start_ms':query['start_ms']})
summary=j(RUN/'summary.json');mape=statistics.fmean(x['mape'] for x in out);logmse=statistics.fmean(x['log_mse'] for x in out);comp=rows(RUN/'event_comparisons.csv');maxmetrics=0.
for r in comp:
    ours=next(x for x in out if x['event']==int(r['event']));maxmetrics=max(maxmetrics,abs(ours['mape']-float(r['case_mape'])),abs(ours['log_mse']-float(r['log_mse'])))
manifest=j(RUN/'run_manifest.json');seal=j(D/'posthoc_transfer_seal.json');badsha=[n for n,h in manifest['outputs_sha256'].items() if sha(RUN/n)!=h]+[n for n,h in seal['sha256'].items() if sha(D/n)!=h]
# Rebuild phase1 I/S/risk comparators from B issued forecasts and raw labels.
refs=defaultdict(list)
for i,query in enumerate(qs[25:],26):
    for rr in rows(BASE/f'workspaces/B/run_v2_001/issued/{i:02d}_event{query["event"]}.csv.gz'):
        if int(rr['phase'])==1 and int(rr['rank']) in [1000,2000] and rr['candidate'] in ['independent_fixed_recipe','fully_shared_pair','risk_recent_k8']:
            y=float(lookup[rr['case_id']]['actual']);refs[rr['candidate'],query['event']].append(100*abs(float(rr['prediction'])/y-1))
comparators={name:statistics.fmean(statistics.fmean(v) for (c,e),v in refs.items() if c==name) for name in ['independent_fixed_recipe','fully_shared_pair','risk_recent_k8']}
result={'scope':'Fixed first25 calibration, posthoc after simulation; independent scalar filter and original issued B comparators','passed':maxbias<1e-12 and maxpred<1e-6 and maxmetrics<1e-10 and not bad and not badsha and len(pred)==120 and len(out)==10,'max_bias_difference':maxbias,'max_prediction_difference':maxpred,'max_event_metric_difference':maxmetrics,'bad_cases':bad,'bad_sha':badsha,'events':out,'event_macro_mape':mape,'event_macro_log_mse':logmse,'same_phase_comparator_mape':comparators,'summary_mape_matches':abs(mape-summary['event_macro']['case_mape'])<1e-10,'identity':summary['identity'],'script_sha256':sha(__file__),'boundary':'Retrospective information-set replay; per-event forecasts collected in memory and sealed at end of run; no prospective publication timing claim.'}
(HERE/'H_D_transfer_evidence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['passed','max_bias_difference','max_prediction_difference','event_macro_mape','event_macro_log_mse','same_phase_comparator_mape']}))
