"""H-authored small adversarial fixtures and checkpoint inventory, no fitting."""
from __future__ import annotations
import importlib.util, json, math, os, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; BASE=HERE.parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
b=load('adaptive_sharing',BASE/'workspaces/B/adaptive_sharing.py')
replay=load('h_replay',BASE/'workspaces/B/replay.py')
findings=[];checks={}
def assert_check(name,ok,detail=None):
    checks[name]={'passed':bool(ok),'detail':detail}
def rejected(fn):
    try:fn()
    except ValueError:return True
    return False
raw=pd.read_csv(BASE/'inputs/train_residuals.csv.gz',dtype={'event_type':str,'band_group':str},low_memory=False)
q=sorted(json.loads((BASE/'inputs/train_queries.json').read_text()),key=lambda x:(x['start_ms'],x['event']))
target=raw.loc[raw.event==q[1]['event'],list(b.BLIND_COLUMNS)].copy()
hist=raw.loc[raw.event==q[0]['event'],list(b.BLIND_COLUMNS)+['log_error']].copy()
experts=b.issue_experts(hist.iloc[:0],hist[list(b.BLIND_COLUMNS)])
for k,col in [('issued_ind','expert_ind'),('issued_pair','expert_pair'),('issued_adjacent','expert_adjacent'),('issued_samepanel','expert_samepanel')]:hist[k]=experts[col].to_numpy()
bad=hist.copy();bad['end_ms']=int(target.start_ms.iloc[0]);assert_check('strict_equal_end_rejected',rejected(lambda:b.issue_experts(bad,target)))
for field in ['actual','log_error','expert__independent','target','future_label']:
    badtarget=target.copy();badtarget[field]=1.;assert_check('current_'+field+'_rejected',rejected(lambda:b.issue_experts(hist,badtarget)))
duplicate=pd.concat([target,target.iloc[:1]],ignore_index=True);assert_check('duplicate_case_id_rejected',rejected(lambda:b.issue_experts(hist,duplicate)))
alias=target.iloc[:1].copy();alias['case_id']=alias.case_id+'::alias';aliases=pd.concat([target,alias],ignore_index=True)
alias_rejected=rejected(lambda:b.issue_experts(hist,aliases))
assert_check('physical_alias_duplicate_rejected',alias_rejected,{'physical_key':['event','rank','cutoff'],'frozen_input_duplicate_count':int(raw.duplicated(['event','rank','cutoff']).sum())})
if not alias_rejected:findings.append({'id':'H-BC-01','severity':'important','scope':'input robustness; frozen B/C scores unaffected','finding':'B pure validator accepts repeated physical prefix with a distinct case_id alias. Frozen input has zero such aliases.','clear_condition':'Reject duplicate (event,rank,cutoff) when cutoff is present, or enforce canonical case_id at trusted input boundary; preserve the already-sealed run and demonstrate adversarial alias fixture rejection.'})
degenerate=[]
for n in range(5):
    r=b.risk_lambda([(1.,1.)]*n,k=8,decay=.85);degenerate.append(r);assert_check('small_support_'+str(n)+'_fallback',r['lambda']==0 and math.isfinite(r['lambda']))
for n in [5,12,40]:
    r=b.risk_lambda([(0.,0.)]*n,k=8,decay=.85);assert_check('constant_experts_'+str(n)+'_fallback',r['lambda']==0 and not r['risk_usable'])
samples=[(.1,.2),(.2,-.3),(.3,.1),(.1,.01),(.2,.08),(.3,.2)]
r=b.risk_lambda(samples,k=8,decay=.85)
w=.85**np.arange(len(samples)-1,-1,-1);a=float(np.dot(w,[v[0] for v in samples])/w.sum());c=float(np.dot(w,[v[1] for v in samples])/w.sum());eff=float(w.sum()**2/np.dot(w,w));expected=max(0.,min(1.,c/(a*(1+8/eff))))
assert_check('risk_formula_independent_arithmetic',abs(r['lambda']-expected)<1e-15)
# A one-event whole-panel replication must preserve event-mean risk records.
rep=hist.copy();rep['case_id']=rep.case_id+'::replica';replicated=pd.concat([hist,rep],ignore_index=True)
one=b.risk_observations(hist,rank=1000,phase=0,pool='pair');two=b.risk_observations(replicated,rank=1000,phase=0,pool='pair')
assert_check('whole_event_replication_preserves_event_risk',np.allclose(one,two))
# Coordinator end-to-end fixture: perturb diagnostic labels by 10^100, compare
# all forecasts and frozen selection. The fixture has four events and two rules.
records=[];queries=[]
for i in range(4):
    start=b.MIN_START_MS+i*10000;end=start+8000;event=i+1
    for rank in [1000,2000]:
        for t,progress in enumerate([.1,.5,.95]):
            current=1000.+t*50; base=current*1.5;z=.02*(i+1)*(1 if rank==1000 else -1)
            records.append(dict(case_id=f'train:{event}:{rank}:{start+t+1}',event=event,rank=rank,phase=int(np.searchsorted([.25,.9],progress,side='right')),progress=progress,start_ms=start,end_ms=end,cutoff=start+t+1000,observed_at=start+t+900,current=current,base_prediction=base,split='train',log_error=z,actual=base*math.exp(-z)))
    queries.append(dict(event=event,start_ms=start,end_ms=end,split='train',target_cases=6))
frame=pd.DataFrame(records);cfg={'schedule':{'warmup':2,'selection':1,'diagnostic':1},'budgets':{'wall_seconds':30},'candidates':[{'name':'independent','kind':'fixed','lambda':0.,'pool':'pair'},{'name':'shared','kind':'fixed','lambda':1.,'pool':'pair'}]}
fixture_root=HERE/'fixtures';fixture_root.mkdir(exist_ok=True)
paths=[]
for name,changed in [('original',False),('future_extreme',True)]:
    f=frame.copy()
    if changed:f.loc[f.event==4,'actual']*=1e100;f.loc[f.event==4,'log_error']=np.log(f.loc[f.event==4,'base_prediction']/f.loc[f.event==4,'actual'])
    path=fixture_root/name;replay.run_replay(f,queries,cfg,path,allow_fixture=True);paths.append(path)
matches=[]
for file in sorted((paths[0]/'issued').glob('*.csv.gz')):
    a=pd.read_csv(file);bb=pd.read_csv(paths[1]/'issued'/file.name);matches.append(a.equals(bb))
assert_check('future_extreme_labels_all_predictions_unchanged',all(matches),{'queries':4,'extreme_factor':1e100})
assert_check('future_extreme_labels_selection_unchanged',json.loads((paths[0]/'selection_seal.json').read_text())['selected_candidate']==json.loads((paths[1]/'selection_seal.json').read_text())['selected_candidate'])

# Load checkpoint tensors and recorded optimizer counters; zero optimizer calls.
import torch
torch.set_num_threads(1)
run=BASE/'workspaces/C/runs/c2_002';inventory=[];bad=[];physical=0;steps=0;updates=0
for path in sorted((run/'models').rglob('*.pt')):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True);d=int(checkpoint['features']);params=sum(v.numel() for v in checkpoint['state_dict'].values());states=checkpoint['fit_states'];physical+=len(states)
    for state in states:
        op=state['optimizer_state_dict'];counters=[int(v['step'].item()) for v in op['state'].values()];epoch=checkpoint['epochs'];size=sum(v['exp_avg'].numel() for v in op['state'].values())
        steps+=epoch;updates+=epoch*size
        if not counters or set(counters)!={epoch} or len(state['losses'])!=epoch or not all(math.isfinite(v) for v in state['losses']):bad.append(str(path))
    if params!=32*d+66:bad.append(str(path))
    inventory.append({'path':str(path.relative_to(run)),'features':d,'parameters':params,'structure':checkpoint['structure'],'epochs':checkpoint['epochs'],'physical_fit_states':len(states),'state_shapes':{k:list(v.shape) for k,v in checkpoint['state_dict'].items()}})
assert_check('C2_checkpoint_tensor_parameter_count',len(inventory)==81 and not bad,{'checkpoint_files':len(inventory),'bad':bad,'physical_fit_states':physical})
assert_check('C2_optimizer_actual_steps_and_updates',physical==108 and steps==5670 and updates==7611300,{'physical_states':physical,'optimizer_steps':steps,'parameter_updates':updates})
out={'scope':'H-authored small fixtures and read-only checkpoint inventory; no model training','checks':checks,'findings':findings,'checkpoint_inventory':inventory,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'probe_failures':[]}
(HERE/'H_fixture_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'failed':[k for k,v in checks.items() if not v['passed']],'findings':len(findings)},ensure_ascii=False))
