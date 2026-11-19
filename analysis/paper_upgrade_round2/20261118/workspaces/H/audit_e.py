"""Independent interval targets, TRAIN-only power basis and selected paired CI."""
from __future__ import annotations
import csv,gzip,hashlib,importlib.util,json,math,statistics,sys
from pathlib import Path
from collections import defaultdict
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2];E=BASE/'workspaces/E';REL=E/'reliability_002';STAT=E/'statistics_002'
def j(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    p=Path(p)
    with (gzip.open(p,'rt',newline='') if p.suffix=='.gz' else p.open(newline='')) as f:return list(csv.DictReader(f))
def close(a,b):return abs(float(a)-float(b))<1e-8+1e-12*max(abs(float(a)),abs(float(b)))
def quantile(v,q):
    a=sorted(v);pos=(len(a)-1)*q;i=math.floor(pos);z=math.ceil(pos);return a[i]+(pos-i)*(a[z]-a[i])
checks={}
def check(k,v,detail=None):checks[k]={'passed':bool(v),'detail':detail}
for run in [STAT,REL]:
    seal=j(run/'output_seal.json');manifest=j(run/'analysis_manifest.json');bad=[f for f,h in seal.items() if sha(run/f)!=h];badin=[p for p,h in manifest['inputs'].items() if sha(ROOT/p)!=h]
    check(run.name+'_output_and_input_seals',not bad and not badin and sha(run/'scorer_source.py')==manifest['source_sha256'],{'outputs':len(seal),'bad_outputs':bad,'bad_inputs':badin,'snapshot_source_matches':sha(run/'scorer_source.py')==manifest['source_sha256']})
old=ROOT/'analysis/paper_upgrade_records/20261118/runs/fixed_replay_v2';dataset=ROOT/'data/rolling_holdout_v3_dense_19rank_20261117';intervals=rows(old/'test_intervals.csv');labels={r['case_id']:float(r['actual']) for r in rows(dataset/'test_labels.csv')};cases={r['case_id']:r for r in rows(dataset/'test_cases.csv')}
panels=defaultdict(list)
for r in intervals:
    case=cases[r['case_id']];y=labels[r['case_id']];lo=float(r['lower']);hi=float(r['upper']);nom=float(r['nominal_coverage']);width=hi-lo;covered=lo<=y<=hi;score=width+2/(1-nom)*max(lo-y,0)+2/(1-nom)*max(y-hi,0);progress=(int(case['observed_at'])-int(case['start_ms']))/(int(case['end_ms'])-int(case['start_ms']))
    support=float(r['own_support']);supportbin='0' if support<=.5 else '1–9' if support<=9.5 else '10–29' if support<=29.5 else '30–59' if support<=59.5 else '60+'
    panels[r['method'],r['nominal_coverage']].append(dict(r,**{k:v for k,v in case.items() if k not in r},actual=y,lo=lo,hi=hi,covered=covered,width=width,relative_width=100*width/y,score=score,own_support_bin=supportbin,actual_progress_bin='[0,.25)' if progress<.25 else '[.25,.9)' if progress<.9 else '[.9,1]'))
summaries=[];bad=[];conditional_bad=[];joint_bad=[]
for s in rows(REL/'interval_summary.csv'):
    part=panels[s['method'],s['nominal_coverage']];ev=defaultdict(list);groups=defaultdict(list)
    for r in part:ev[r['event']].append(r);groups[r['event'],r['rank'],r['phase']].append(r)
    joint_by_ev=defaultdict(list)
    for (event,_,_),v in groups.items():joint_by_ev[event].append(all(x['covered'] for x in v))
    point=statistics.fmean(statistics.fmean(x['covered'] for x in v) for v in ev.values());joint=statistics.fmean(statistics.fmean(v) for v in joint_by_ev.values());success=sum(all(x['covered'] for x in v) for v in ev.values())
    values=dict(n_events=len(ev),n_cases=len(part),n_groups=len(groups),case_coverage=statistics.fmean(r['covered'] for r in part),event_equal_point_coverage=point,event_equal_group_joint_coverage=joint,whole_event_joint_coverage=success/len(ev),whole_event_successes=success,
        finite_interval_rate=statistics.fmean(math.isfinite(r['lo']) and math.isfinite(r['hi']) for r in part),unbounded_cases=sum(not math.isfinite(r['lo']) or not math.isfinite(r['hi']) for r in part),unavailable_cases=sum(r['available']=='False' for r in part),own_support_zero_cases=sum(float(r['own_support'])==0 for r in part),
        mean_width=statistics.fmean(r['width'] for r in part),p90_width=quantile([r['width'] for r in part],.9),mean_relative_width_pct=statistics.fmean(r['relative_width'] for r in part),p90_relative_width_pct=quantile([r['relative_width'] for r in part],.9),winkler_score=statistics.fmean(r['score'] for r in part),
        relative_score_pct=statistics.fmean(100*r['score']/r['actual'] for r in part),event_equal_width=statistics.fmean(statistics.fmean(x['width'] for x in v) for v in ev.values()),event_equal_relative_width_pct=statistics.fmean(statistics.fmean(x['relative_width'] for x in v) for v in ev.values()),event_mean_p90_relative_width_pct=statistics.fmean(quantile([x['relative_width'] for x in v],.9) for v in ev.values()),event_equal_winkler_score=statistics.fmean(statistics.fmean(x['score'] for x in v) for v in ev.values()))
    if any(not close(v,s[k]) for k,v in values.items()):bad.append(s['nominal_coverage'])
    summaries.append({'nominal':float(s['nominal_coverage']),**values})
check('interval_main_targets_width_and_score_independent',not bad,{'summary_rows':len(summaries),'bad_nominals':bad})
for s in rows(REL/'interval_conditions.csv'):
    dim=s['dimension'];value=s['value'];part=[r for r in panels[s['method'],s['nominal_coverage']] if str(r[dim])==value];ev=defaultdict(list)
    for r in part:ev[r['event']].append(r['covered'])
    if len(part)!=int(s['n_cases']) or len(ev)!=int(s['n_events']) or not close(statistics.fmean(r['covered'] for r in part),s['case_coverage']) or not close(statistics.fmean(statistics.fmean(v) for v in ev.values()),s['event_equal_point_coverage']):conditional_bad.append([s['nominal_coverage'],dim,value])
    if any(k.startswith('whole_event') or k.startswith('event_equal_group_joint') for k in s):conditional_bad.append(['partial_target_field_leaked'])
check('conditional_point_denominators_and_target_identity',not conditional_bad,{'conditional_rows':len(rows(REL/'interval_conditions.csv')),'bad':conditional_bad})
for s in rows(REL/'interval_full_group_conditions.csv'):
    dim=s['dimension'];value=s['value'];groups=defaultdict(list)
    for r in panels[s['method'],s['nominal_coverage']]:groups[r['event'],r['rank'],r['phase']].append(r)
    selected={k:v for k,v in groups.items() if (str(v[0][dim]) if len({str(x[dim]) for x in v})==1 else 'mixed')==value};byev=defaultdict(list)
    for (event,_,_),v in selected.items():byev[event].append(all(x['covered'] for x in v))
    if len(selected)!=int(s['n_groups']) or len(byev)!=int(s['n_events']) or sum(map(len,selected.values()))!=int(s['n_cases']) or not close(statistics.fmean(statistics.fmean(v) for v in byev.values()),s['event_equal_full_group_joint_coverage']):joint_bad.append([s['nominal_coverage'],dim,value])
check('conditional_full_group_denominators',not joint_bad,{'rows':len(rows(REL/'interval_full_group_conditions.csv')),'bad':joint_bad})
# Training variance is reconstructed from blind C1 predictions and train answers,
# independently of E's summary and of historical TEST interval observations.
answers={r['case_id']:float(r['actual']) for r in rows(BASE/'inputs/train_residuals.csv.gz')};loss=defaultdict(list)
for r in rows(BASE/'workspaces/C/runs/c1_002/predictions.csv.gz'):
    if r['method'] in ['legacy_pair','independent']:
        a=100*abs(float(r['predicted'])-answers[r['case_id']])/answers[r['case_id']];loss['global19',r['method'],r['event']].append(a)
        if int(r['rank']) in [1000,2000]:loss['T1000_T2000',r['method'],r['event']].append(a)
diff={}
for panel,method,event in loss:
    if method=='legacy_pair':diff[panel,event]=statistics.fmean(loss[panel,'legacy_pair',event])-statistics.fmean(loss[panel,'independent',event])
training_rows=rows(REL/'power_training_activity_differences.csv');check('power_TRAIN35_basis_independent',len(training_rows)==70 and all(close(diff[r['panel'],r['event']],r['difference_pp']) for r in training_rows),{'panels':2,'activities_per_panel':35})
from scipy.stats import t,nct,beta
power_bad=[];maxpd=0.
def power(n,sigma,effect):
    critical=float(t.ppf(.975,n-1));nc=abs(effect)*math.sqrt(n)/sigma
    return float(nct.sf(critical,n-1,nc)+nct.sf(critical,n-1,-nc))
for r in rows(REL/'power_grid.csv'):
    n=int(r['required_events']);sigma=float(r['sigma_pp'])*math.sqrt((1+float(r['rho']))/(1-float(r['rho'])));effect=float(r['effect_pp']);target=float(r['target_power']);p=power(n,sigma,effect);previous=power(n-1,sigma,effect) if n>2 else 0.
    pd=abs(power(20,sigma,effect)-float(r['power_at_20_events']));maxpd=max(maxpd,pd)
    if p<target-1e-12 or previous>=target+1e-12 or pd>1e-12:power_bad.append([r['panel'],r['sigma_scenario'],n])
check('power_integer_boundary_and_20event_probability',not power_bad,{'grid_rows':len(rows(REL/'power_grid.csv')),'max_20event_power_difference':maxpd,'bad':power_bad})
cp_bad=[]
for s in rows(REL/'interval_summary.csv'):
    n=int(s['n_events']);success=int(s['whole_event_successes']);lo=0. if success==0 else float(beta.ppf(.025,success,n-success+1));hi=1. if success==n else float(beta.ppf(.975,success+1,n-success))
    if not close(lo,s['whole_event_cp_low95']) or not close(hi,s['whole_event_cp_high95']):cp_bad.append(s['nominal_coverage'])
check('whole_event_binomial_reference',not cp_bad,{'interpretation':'Exact binomial reference uses iid activity assumption; fixed-loss bootstrap target is separately labeled.'})
# A few prespecified paired rows, independent implementation of circular blocks.
import numpy as np
events=rows(STAT/'B_events.csv');paired=rows(STAT/'B_paired.csv');ci_bad=[];tested=0
for s in paired:
    if s['panel']=='main_pair' and s['period']=='diagnostic' and s['left'] in ['fully_shared_pair','risk_recent_k8'] and s['metric']=='event_macro_MAPE_pct':
        left={r['event']:r for r in events if r['panel']==s['panel'] and r['period']==s['period'] and r['method']==s['left']};right={r['event']:r for r in events if r['panel']==s['panel'] and r['period']==s['period'] and r['method']==s['right']};order=sorted(left,key=lambda e:(int(left[e]['start_ms']),int(e)));d=[float(left[e]['mape'])-float(right[e]['mape']) for e in order];block=int(s['block']);rng=np.random.default_rng(42);starts=rng.integers(0,len(d),size=(5000,math.ceil(len(d)/block)));samples=[]
        for startrow in starts:
            ix=[(int(start)+k)%len(d) for start in startrow for k in range(block)][:len(d)];samples.append(statistics.fmean(d[z] for z in ix))
        lo=quantile(samples,.025);hi=quantile(samples,.975)
        if not close(lo,s['low95']) or not close(hi,s['high95']) or not close(statistics.fmean(d),s['difference']):ci_bad.append([s['left'],block])
        tested+=1
check('selected_B_paired_CIs_independent_loops',tested==8 and not ci_bad,{'rows':tested,'bad':ci_bad})
plan=j(REL/'confirmation_plan.json');check('future_plan_draft_unactivated',plan['status']=='draft_unactivated_no_future_scores' and plan['supersedes_existing_registry'] is False,{'planned_events':plan['planned_events'],'primary_comparison':plan['sole_primary_comparison']})
addendum=j(E/'confirmation_plan_addendum.json');required={}
for r in rows(REL/'power_grid.csv'):
    if r['sigma_scenario']=='normal_iid_chi2_high95' and float(r['effect_pp'])==.5 and float(r['target_power'])==.9 and float(r['rho'])==.3:required[r['panel']]=int(r['required_events'])
target=10*math.ceil(max(60,*required.values())/10)
check('future_80event_addendum_basis_and_seal',required=={'global19':4,'T1000_T2000':72} and target==addendum['target_events']==80 and addendum['amends_plan_sha256']==sha(REL/'confirmation_plan.json') and addendum['future_events_at_addendum']==0 and addendum['future_mape'] is None and addendum['status']=='pending_new_cohort_unactivated',{'model_requirements':required,'engineering_floor':60,'rounded_target':target,'addendum_sha256':sha(E/'confirmation_plan_addendum.json')})
# Test the actual future HAC function against hand-assembled chronological sums.
spec=importlib.util.spec_from_file_location('h_E_pure',E/'statistics_audit.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
x=[-.3,.2,.9,.7,-.2,-.5,.1,.6,.3,-.4,.8,.4];n=len(x);mean=statistics.fmean(x);c=[v-mean for v in x];band=math.ceil(n**(1/3));gamma0=sum(v*v for v in c)/n;lrv=gamma0+2*sum((1-h/(band+1))*sum(c[i]*c[i-h] for i in range(h,n))/n for h in range(1,band+1));actual=module.hac_interval(x);constant=module.hac_interval([.2]*12)
check('future_HAC_ordered_manual_and_degenerate',close(actual['long_run_variance'],lrv) and actual['bandwidth']==band and close(actual['se_mean'],math.sqrt(lrv/n)) and constant['status']=='degenerate_variance_no_decision' and constant['p_value'] is None,{'bandwidth':band,'manual_LRV':lrv,'source_sha256':sha(E/'statistics_audit.py'),'constant_status':constant['status']})
out={'scope':'Independent interval arithmetic from original frozen intervals, TRAIN35 power basis and numerical power boundary; no E scorer imported','checks':checks,'interval_recomputed':summaries,'script_sha256':sha(__file__),'limitations':['Historical test interval rows used for descriptive reliability only, no interval/method selection','Power uses retrospective TRAIN differences and approximate AR1 variance inflation','Eight selected paired CI rows replayed; all E bootstrap rows were not regenerated']}
(HERE/'H_E_numeric_evidence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'checks':len(checks),'failures':[k for k,v in checks.items() if not v['passed']],'planned_events':plan['planned_events']}))
