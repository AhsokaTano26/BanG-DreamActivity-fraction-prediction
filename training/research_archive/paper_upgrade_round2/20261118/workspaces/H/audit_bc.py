"""Independent read-only B/C arithmetic and seal audit (no model scorer imported)."""
from __future__ import annotations
import csv, gzip, hashlib, json, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[2]
sys.dont_write_bytecode=True

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p): return json.loads(Path(p).read_text())
def rows(p):
    p=Path(p)
    with (gzip.open(p,'rt',newline='') if p.suffix=='.gz' else p.open(newline='')) as f:
        return list(csv.DictReader(f))
def digest(v): return hashlib.sha256(json.dumps(sorted(v)).encode()).hexdigest()
def close(a,b,atol=1e-8): return abs(float(a)-float(b))<=atol+1e-12*max(abs(float(a)),abs(float(b)))
def quantile(v,q):
    x=sorted(v); pos=(len(x)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return x[lo]+(pos-lo)*(x[hi]-x[lo])
def metrics(data,predcol,answers):
    ev=defaultdict(list); er=defaultdict(list); logev=defaultdict(list); apes=[]; diffs=[]
    for r in data:
        p=float(r[predcol]); y=answers[r['case_id']]
        a=abs(p-y)/y*100; ev[r['event']].append(a); er[r['event'],r['rank']].append(a)
        logev[r['event']].append(math.log(p/y)**2); apes.append(a); diffs.append(p-y)
    er_event=defaultdict(list)
    for (e,_),v in er.items(): er_event[e].append(statistics.fmean(v))
    return dict(events=len(ev),cases=len(data),mape=statistics.fmean(map(statistics.fmean,ev.values())),
                rank_mape=statistics.fmean(map(statistics.fmean,er_event.values())),
                case_mape=statistics.fmean(apes),mae=statistics.fmean(map(abs,diffs)),
                rmse=math.sqrt(statistics.fmean(d*d for d in diffs)),p90=quantile(apes,.9),
                log_squared=statistics.fmean(map(statistics.fmean,logev.values())))

def audit():
    out={'scope':'Independent stdlib arithmetic from issued predictions plus train labels; no implementation scorer imported',
         'checks':{},'failures':[],'not_run':['B full35 replay','C2 108 refits','bootstrap'],'input_files':{}}
    checks=out['checks']; failures=out['failures']
    def check(name,condition,detail=None):
        checks[name]={'passed':bool(condition),'detail':detail}
        if not condition:failures.append(name)
    inputs=BASE/'inputs'; raw=rows(inputs/'train_residuals.csv.gz'); queries=sorted(j(inputs/'train_queries.json'),key=lambda x:(x['start_ms'],x['event']))
    for fn in ['train_residuals.csv.gz','train_queries.json']:out['input_files'][fn]=sha(inputs/fn)
    answers={r['case_id']:float(r['actual']) for r in raw}; clock={int(r['event']):(int(r['start_ms']),int(r['end_ms'])) for r in raw}
    unique_physical={(r['event'],r['rank'],r['cutoff']) for r in raw}
    check('input_unique_physical',len(raw)==len(answers)==len(unique_physical),{'rows':len(raw),'events':len(clock),'ranks':sorted({int(r['rank']) for r in raw})})
    check('canonical_case_ids',all(r['case_id']==f"train:{r['event']}:{r['rank']}:{r['cutoff']}" for r in raw))
    check('dense_training_only',all(r['split']=='train' and int(r['start_ms'])>=1706504400000 for r in raw))
    check('schedule_35_actual_time',len(queries)==35 and [int(q['event']) for q in queries]==sorted(clock,key=lambda e:(clock[e][0],e)))

    # Every hash in each run/evaluation seal is recomputed from bytes.
    targets=[('B/run_v2_001','run_manifest.json'),('C/runs/c1_002','configuration_seal.json'),('C/runs/c1_002','blind_prediction_seal.json'),
             ('C/runs/c1_002_scores','evaluation_seal.json'),('C/runs/c2_002','configuration_seal.json'),('C/runs/c2_002','blind_prediction_seal.json'),('C/runs/c2_002_scores','evaluation_seal.json')]
    for directory,filename in targets:
        p=BASE/'workspaces'/directory; m=j(p/filename); bad=[f for f,h in m['files'].items() if not (p/f).is_file() or sha(p/f)!=h]
        check('hash_'+directory+'_'+filename,not bad,{'files':len(m['files']),'bad':bad})
    b=BASE/'workspaces/B/run_v2_001'; issued=[]; bad_b=[]; seal_records=[]
    for index,q in enumerate(queries,1):
        e=int(q['event']); fn=f'{index:02d}_event{e}'; seal=j(b/'seals'/f'{fn}.json'); pred=rows(b/'issued'/f'{fn}.csv.gz'); issued+=pred
        expected=sorted(ev for ev,(start,end) in clock.items() if end<int(q['start_ms']))
        expected_rows=sum(1 for r in raw if int(r['event']) in expected)
        last=max([clock[ev][1] for ev in expected],default=None)
        good=(seal['history_events']==expected and seal['history_cases']==expected_rows and seal['latest_history_end_ms']==last
              and seal['prediction_sha256']==sha(b/'issued'/f'{fn}.csv.gz') and seal['experts_sha256']==sha(b/'experts'/f'{fn}.csv.gz')
              and len(pred)==int(q['target_cases'])*12 and (last is None or last<int(q['start_ms']))
              and all(not {'actual','log_error','target','expert__independent'}&set(r) for r in pred))
        if not good:bad_b.append(e)
        seal_records.append({'event':e,'start_ms':q['start_ms'],'history_events':len(expected),'latest_history_end_ms':last})
    check('B_35_history_and_issued_seals',not bad_b,{'bad_events':bad_b,'events':len(seal_records)})
    bsummary=rows(b/'summary.csv'); b_group=defaultdict(list)
    # External/base controls are rebuilt from input columns, never from B scored_cases.
    for r in issued:b_group[r['block'],r['candidate']].append(r)
    blockmap={str(q['event']):('warmup' if i<15 else 'selection' if i<25 else 'diagnostic') for i,q in enumerate(queries)}
    for r in raw:
        for name,col in [('external_original_independent','expert__independent'),('uncorrected_base','base_prediction')]:
            v=dict(r,prediction=r[col]);b_group[blockmap[r['event']],name].append(v)
    b_err=[]; b_values=[]; max_formula=0.; physical_bad=0
    for r in issued:
        lam=float(r['lambda']); bias=max(-.3,min(.3,(1-lam)*float(r['expert_ind'])+lam*float(r['expert_shared'])))
        cur=float(r['current']); expected=max(cur,min(cur*math.exp(4),float(r['base_prediction'])*math.exp(-bias)))
        max_formula=max(max_formula,abs(expected-float(r['prediction'])))
    for s in bsummary:
        part=[r for (block,c),v in b_group.items() if c==s['candidate'] and (s['block']=='all' or block==s['block']) for r in v]
        if s['panel']=='main_pair':part=[r for r in part if int(r['rank']) in (1000,2000)]
        if len({r['case_id'] for r in part})!=len(part):physical_bad+=1
        val=metrics(part,'prediction',answers)
        aliases={'events':'events','cases':'cases','mape':'event_macro_mape','rank_mape':'rank_equal_event_macro_mape','case_mape':'case_mape',
                 'mae':'mae','rmse':'rmse','p90':'p90_ape','log_squared':'event_macro_log_squared'}
        if any(not close(val[k],s[col],1e-7) for k,col in aliases.items()): b_err.append([s['block'],s['candidate'],s['panel']])
        b_values.append({k:s[k] for k in ['block','candidate','panel']}|val)
    check('B_independent_arithmetic_all_summary',not b_err,{'summary_rows':len(bsummary),'bad':b_err})
    check('B_prediction_mixture_recomputed',max_formula<1e-6,{'max_abs_score_difference':max_formula})
    check('B_all_panels_unique_cases',physical_bad==0)
    seal=j(b/'selection_seal.json'); menu=[c['name'] for c in j(b/'configuration.json')['candidates']]
    selection=[v for v in b_values if v['block']=='selection' and v['panel']=='main_pair' and v['candidate'] in menu]
    selected=min(selection,key=lambda x:(x['mape'],menu.index(x['candidate'])))['candidate']
    diag_start=int(queries[25]['start_ms']);check('B_selection_independent_recomputed',selected==seal['selected_candidate']=='independent_fixed_recipe',{'selected':selected})
    check('B_selection_seal_before_query26',seal['freeze_start_ms']==diag_start and seal['issued_events_before_seal']==25 and seal['latest_feedback_end_ms']<diag_start
          and seal['selection_events']==[int(q['event']) for q in queries[15:25]])
    states={}
    for r in issued:
        if r['block']=='diagnostic' and int(r['rank']) in (1000,2000):states[r['candidate'],r['event'],r['rank'],r['phase']]=r
    gated=[r for (c,*_),r in states.items() if c=='risk_gated_k8']
    check('B_gated_diagnostic_all_closed',len(gated)==60 and all(float(r['lambda'])==0 and r['gate_open']=='False' for r in gated),{'states':len(gated)})
    t1000_adj=[r for r in issued if r['candidate']=='adjacent_risk_k8' and int(r['rank'])==1000]
    check('B_missing1500_adjacent_support',all(int(r['shared_history_events'])==0 and float(r['lambda'])==0 for r in t1000_adj))
    out['B_recalculated_summary']=b_values;out['B_time_evidence']=seal_records

    # C summaries use independently opened TRAIN answers and their blind predictions.
    c_values={}; c2_queries=None
    for exp in ['c1','c2']:
        run=BASE/f'workspaces/C/runs/{exp}_002'; pred=rows(run/'predictions.csv.gz'); summaries=rows(run.with_name(run.name+'_scores')/'summary.csv')
        if exp=='c1':ans=answers;grp=['period','method']
        else:
            data=ROOT/'data/temporal_holdout_v2_dense_t1000_t2000_20261115';ans={r['case_id']:float(r['actual']) for r in rows(data/'train_labels.csv')};grp=['endpoint','method']
        index=defaultdict(list)
        for r in pred:index[tuple(r[k] for k in grp)].append(r)
        bad=[]; values=[]
        for s in summaries:
            part=index[tuple(s[k] for k in grp)]
            if s['panel']=='T1000_T2000':part=[r for r in part if int(r['rank']) in (1000,2000)]
            val=metrics(part,'predicted',ans)
            aliases={'events':'events','cases':'cases','mape':'event_macro_mape_pct','rank_mape':'event_rank_macro_mape_pct','case_mape':'case_mape_pct','mae':'mae','rmse':'rmse','p90':'p90_ape_pct'}
            if any(not close(val[k],s[col],1e-7) for k,col in aliases.items()):bad.append([s[k] for k in grp])
            values.append({k:s[k] for k in grp+['panel']}|val)
        check(exp.upper()+'_independent_arithmetic',not bad,{'summary_rows':len(summaries),'bad':bad})
        check(exp.upper()+'_case_uniqueness',all(len({r['case_id'] for r in part})==len(part) for part in index.values()))
        c_values[exp]=values
        if exp=='c1':
            bad=[];c1_seals=[]
            for i,q in enumerate(queries,1):
                p=run/'predictions'/f'query_{i:02d}_event_{q["event"]}.csv.gz'; s=j(p.with_suffix('.issuance_seal.json'))
                hist=[r for r in raw if int(r['end_ms'])<int(q['start_ms'])];target=[r for r in raw if int(r['event'])==q['event']]
                valid=(s['max_history_end_ms']==max((int(r['end_ms']) for r in hist),default=None) and s['history_cases']==len(hist)
                       and s['history_case_ids_sha256']==digest([r['case_id'] for r in hist]) and s['target_case_ids_sha256']==digest([r['case_id'] for r in target])
                       and s['history_event_ids_sha256']==digest(list({int(r['event']) for r in hist})) and s['prediction_sha256']==sha(p))
                if not valid:bad.append(q['event'])
                c1_seals.append(s)
            check('C1_35_seals_independent_history_reconstruction',not bad,{'bad_events':bad,'seals':len(c1_seals)})
            predmaps={method:{r['case_id']:float(r['predicted']) for r in v} for (_,method),v in index.items()}
            # Rebuild over all periods, whose cases are disjoint.
            predmaps=defaultdict(dict)
            for r in pred:predmaps[r['method']][r['case_id']]=float(r['predicted'])
            identity=max(abs(v-predmaps['same_panel_independent'][k]) for k,v in predmaps['same_filter_pair'].items())
            check('C1_same_filter_identity',identity<1e-6,{'max_absolute_difference':identity})
            check('C1_complete_support_is_zero_perturbation',predmaps['legacy_pair']==predmaps['complete_support'])
        else:
            query=rows(run/'query_cases.csv');c2_queries={int(r['event']) for r in query}; physical={(r['event'],r['rank'],r['cutoff']) for r in query}
            check('C2_unique_query_panel',len(query)==len(physical)==442 and len(c2_queries)==12,{'cases':len(query),'events':sorted(c2_queries)})
            byendpoint=defaultdict(set)
            for r in query:byendpoint[r['endpoint']].add(r['case_id'])
            check('C2_all_method_panels_equal',all({r['case_id'] for r in part}==byendpoint[key[0]] for key,part in index.items()))
            ensemble_bad=[]
            pmap={(r['endpoint'],r['group'],r['structure'],r['seed'],r['case_id']):float(r['predicted']) for r in pred}
            for r in pred:
                if r['seed']=='mean':
                    mean=statistics.fmean(pmap[r['endpoint'],r['group'],r['structure'],str(seed),r['case_id']] for seed in [42,123,2026])
                    if not close(mean,r['predicted'],1e-6):ensemble_bad.append(r['case_id'])
            check('C2_final_score_ensemble',not ensemble_bad,{'bad_cases':len(ensemble_bad)})
            costs=rows(run/'fit_costs.csv'); groups=rows(run/'history_groups.csv'); budget=j(run/'fit_budget.json');cfg=j(run/'configuration.json')
            traincases=rows(data/'train_cases.csv'); trainclock={int(r['event']):(int(r['start_ms']),int(r['end_ms'])) for r in traincases}
            sorted_events=sorted(trainclock,key=lambda e:(trainclock[e][0],e));bad_groups=[];bad_weights=[]
            for g in groups:
                endpoint=int(g['endpoint']);last32=sorted_events[endpoint-32:endpoint]
                expected=(last32 if g['group']=='N32full' else last32[-16:] if g['group']=='N16recent' else [last32[i] for i in [0,2,4,6,8,10,12,14,17,19,21,23,25,27,29,31]])
                qevents=sorted_events[endpoint:endpoint+4];qstart=min(trainclock[e][0] for e in qevents); span=(max(trainclock[e][0] for e in expected)-min(trainclock[e][0] for e in expected))/86400000
                if json.loads(g['event_ids'])!=expected or json.loads(g['query_events'])!=qevents or any(trainclock[e][1]>=qstart for e in expected) or not close(span,g['span_days']):bad_groups.append([endpoint,g['group']])
                w=rows(run/'models'/f'endpoint{endpoint}_{g["group"]}'/'training_weights.csv'); masses=defaultdict(float); rankm=defaultdict(float)
                for r in w:masses[r['event']]+=float(r['weight']);rankm[r['event'],r['rank']]+=float(r['weight'])
                if max(masses.values())-min(masses.values())>1e-8 or max(rankm.values())-min(rankm.values())>1e-8:bad_weights.append([endpoint,g['group']])
            check('C2_time_groups_and_spans',not bad_groups,{'bad_groups':bad_groups,'groups':len(groups)})
            check('C2_event_rank_equal_weights',not bad_weights,{'bad_groups':bad_weights})
            dim={(g['endpoint'],g['group']):int(g['feature_dimension']) for g in groups}; bad_budget=[]
            for r in costs:
                d=dim[r['endpoint'],r['group']];shared=r['structure'].startswith('shared');params=32*d+66 if shared else 16*d+33
                n=int(r['rows']);epochs=int(r['epochs']);mac=n*epochs*((32*d+64) if shared else (16*d+16))
                if int(r['parameters'])!=params or int(r['optimizer_steps'])!=epochs or int(r['parameter_updates'])!=params*epochs or int(r['forward_linear_MACs'])!=mac:bad_budget.append(r)
            total={k:sum(int(r[k]) for r in costs) for k in ['optimizer_steps','parameter_updates','forward_linear_MACs']}
            check('C2_physical_cost_formula_recomputed',not bad_budget and len(costs)==108 and all(total[k]==budget[k] for k in total),{'physical_fit_rows':len(costs),'totals':total,'bad':len(bad_budget)})
            out['C2_budget_by_structure']={name:{k:sum(float(r[k]) for r in costs if r['structure']==name) for k in ['optimizer_steps','parameter_updates','forward_linear_MACs','cpu_seconds','wall_seconds']}|{'physical_fits':sum(r['structure']==name for r in costs)} for name in ['shared60','shared30','independent60']}
            out['C2_history_groups']=groups
    out['C_recalculated_summaries']=c_values
    overlap=sorted(set(clock)&c2_queries);check('cross_study_overlap_recomputed',len(overlap)==12,{'B_events':35,'C2_events':12,'overlap_events':overlap,'union_events':len(set(clock)|c2_queries),'interpretation':'Different estimands on overlapping activities; independent activity count for the union is 35.'})
    out['script_sha256']=sha(__file__)
    (HERE/'H_numeric_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':len(checks),'failed':failures,'evidence':'H_numeric_evidence.json'},ensure_ascii=False))
    return int(bool(failures))
if __name__=='__main__':raise SystemExit(audit())
