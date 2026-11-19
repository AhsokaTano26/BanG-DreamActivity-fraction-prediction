"""Second implementation of B expert, risk and gate arithmetic using stdlib."""
import csv,gzip,json,math,statistics,hashlib
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];RUN=BASE/'workspaces/B/run_v2_001'
def rows(path):
    with gzip.open(path,'rt') as f:return list(csv.DictReader(f))
raw=rows(BASE/'inputs/train_residuals.csv.gz');queries=sorted(json.loads((BASE/'inputs/train_queries.json').read_text()),key=lambda q:(q['start_ms'],q['event']));cfg=json.loads((RUN/'configuration.json').read_text());configs={c['name']:c for c in cfg['candidates']};pair=(1000,2000);adjacent=[tuple(x) for x in cfg['adjacent_groups']]
def ewma(vals):
    state=0.
    for i,x in enumerate(vals):state=x if i==0 else .7*state+.3*x
    return max(-.3,min(.3,state*len(vals)/(len(vals)+10)))
def risk(vals,k,decay):
    if not vals:return dict(lam=0.,A=0.,C=0.,n_eff=0.,usable=False)
    ws=[decay**(len(vals)-1-i) for i in range(len(vals))];sw=sum(ws)
    a=sum(w*x[0] for w,x in zip(ws,vals))/sw;c=sum(w*x[1] for w,x in zip(ws,vals))/sw;ne=sw*sw/sum(w*w for w in ws);usable=len(vals)>=5 and a>1e-8
    return dict(lam=max(0.,min(1.,c/(a*(1+k/ne)))) if usable else 0.,A=a,C=c,n_eff=ne,usable=usable)
def gate(table):
    if len(table)<8:return dict(gate_pairs=len(table),rho=0.,gap_ratio=0.,drift_ratio=0.,gate_open=False)
    table=table[-12:];x=[v[0] for v in table];y=[v[1] for v in table];mx=statistics.fmean(x);my=statistics.fmean(y)
    vx=sum((v-mx)**2 for v in x);vy=sum((v-my)**2 for v in y);rho=sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(vx*vy) if math.sqrt(vx/len(x))>1e-12 and math.sqrt(vy/len(y))>1e-12 else 0.
    rms=lambda v:math.sqrt(statistics.fmean(a*a for a in v));gap=rms([a-b for a,b in zip(x,y)])/(rms(x)+rms(y)+1e-8)
    dx=[b-a for a,b in zip(x,x[1:])];dy=[b-a for a,b in zip(y,y[1:])];drift=rms([a-b for a,b in zip(dx,dy)])/(rms(dx)+rms(dy)+1e-8)
    return dict(gate_pairs=len(table),rho=rho,gap_ratio=gap,drift_ratio=drift,gate_open=rho>=.15 and gap<=.5 and drift<=.75)
clock={int(r['event']):(int(r['start_ms']),int(r['end_ms'])) for r in raw};means=defaultdict(list)
for r in raw:means[int(r['event']),int(r['rank']),int(r['phase'])].append(float(r['log_error']))
means={key:statistics.fmean(v) for key,v in means.items()};issued_experts={};maxima=defaultdict(float);bad=[];states=0
for i,q in enumerate(queries,1):
    event=q['event'];history=sorted([e for e,(s,end) in clock.items() if end<q['start_ms']],key=lambda e:(clock[e][1],clock[e][0],e))
    ex=rows(RUN/'experts'/f'{i:02d}_event{event}.csv.gz');target={r['case_id']:r for r in raw if int(r['event'])==event}
    state_expected={}
    for rr in ex:
        r=target[rr['case_id']];rank=int(r['rank']);phase=int(r['phase']);key=(rank,phase)
        if key not in state_expected:
            own=[means[e,rank,phase] for e in history if (e,rank,phase) in means];value=ewma(own);expected={'expert_ind':value,'ind_history_events':len(own)}
            for family,groups in [('pair',[pair]),('adjacent',adjacent)]:
                group=next((g for g in groups if rank in g),None);complete=[e for e in history if group and len(group)>1 and all((e,a,phase) in means for a in group)]
                expected['expert_'+family]=ewma([statistics.fmean(means[e,a,phase] for a in group) for e in complete]) if complete else value
                expected[family+'_history_events']=len(complete)
                if family=='pair':expected['expert_samepanel']=ewma([means[e,rank,phase] for e in complete]) if complete else value
            state_expected[key]=expected;issued_experts[event,rank,phase]=expected
        for col,value in state_expected[key].items():maxima[col]=max(maxima[col],abs(value-float(rr[col])))
    pred=rows(RUN/'issued'/f'{i:02d}_event{event}.csv.gz');unique={(r['candidate'],int(r['rank']),int(r['phase'])):r for r in pred}
    for (candidate,rank,phase),r in unique.items():
        conf=configs[candidate];pool=conf.get('pool','pair');group=(pair if pool=='pair' and rank in pair else next((g for g in adjacent if rank in g),()) if pool=='adjacent' else ())
        active=len(group)>1 and rank in conf.get('apply_ranks',cfg['ranks']);lam=0.;samples=[]
        if active and conf['kind']=='fixed':lam=conf['lambda']
        elif active and conf['kind'] in ('risk','gated'):
            for e in history:
                if (e,rank,phase) in means:
                    exp=issued_experts[e,rank,phase];d=exp['expert_'+pool]-exp['expert_ind'];samples.append((d*d,(means[e,rank,phase]-exp['expert_ind'])*d))
            value=risk(samples,conf['k'],conf['decay']);lam=value['lam']
            for col,other in [('A','A'),('C','C'),('n_eff','n_eff')]:maxima[col]=max(maxima[col],abs(value[other]-float(r[col])))
            deletions=[risk(samples[:z]+samples[z+1:],conf['k'],conf['decay'])['lam'] for z in range(len(samples))] if value['usable'] else []
            lo=min(deletions) if deletions else lam;hi=max(deletions) if deletions else lam
            maxima['lambda_delete_min']=max(maxima['lambda_delete_min'],abs(lo-float(r['lambda_delete_min'])));maxima['lambda_delete_max']=max(maxima['lambda_delete_max'],abs(hi-float(r['lambda_delete_max'])))
            if int(r['support'])!=len(samples):bad.append([event,candidate,rank,phase,'support'])
            if conf['kind']=='gated':
                table=[(means[e,1000,phase],means[e,2000,phase]) for e in history if (e,1000,phase) in means and (e,2000,phase) in means][-12:];g=gate(table)
                for col in ['rho','gap_ratio','drift_ratio','gate_pairs']:maxima[col]=max(maxima[col],abs(g[col]-float(r[col])))
                if g['gate_open']!=(r['gate_open']=='True'):bad.append([event,candidate,rank,phase,'gate'])
                if not g['gate_open']:lam=0.
        maxima['lambda']=max(maxima['lambda'],abs(lam-float(r['lambda'])));states+=1
passed=not bad and all(v<1e-12 for v in maxima.values())
out={'scope':'Independent scalar arithmetic, no B model/scorer imported','passed':passed,'event_rank_phase_candidate_states':states,'max_absolute_discrepancies':dict(maxima),'bad_states':bad,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(HERE/'H_B_scalar_evidence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
