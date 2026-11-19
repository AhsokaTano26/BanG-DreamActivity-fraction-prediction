"""Copy minimum train-only closure, forbid original project reads, replay B/C.

Run only in the PI-assigned single-thread numerical window. This launcher writes
auditor-owned artifacts and a temporary standalone source/data project. Installed
packages in .venv remain the declared shared dependency environment.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, shutil, subprocess, sys, tempfile, time, site
from importlib.machinery import PathFinder
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def closure(initial):
    seen=set();queue=list(initial)
    def local(name):
        return (ROOT/(name.replace('.','/')+'.py')).is_file() and (name.startswith('analysis.') or PathFinder.find_spec(name.split('.')[0],site.getsitepackages()) is None)
    while queue:
        m=queue.pop();p=ROOT/(m.replace('.','/')+'.py')
        if m in seen or not p.is_file():continue
        seen.add(m)
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n,ast.Import):queue.extend(a.name for a in n.names if local(a.name))
            if isinstance(n,ast.ImportFrom) and n.module and (n.module=='analysis' or local(n.module)):
                queue.extend('analysis.'+a.name for a in n.names) if n.module=='analysis' else queue.append(n.module)
    return sorted(seen)
def launch():
    parser=argparse.ArgumentParser();parser.add_argument('--b-full35',action='store_true');parser.add_argument('--c-single-fit',action='store_true');args=parser.parse_args()
    timestamp=str(time.time_ns());out=HERE/('portability_'+timestamp);out.mkdir();temp=Path(tempfile.mkdtemp(prefix='round2_H_standalone_'));project=temp/'project';project.mkdir()
    copied=[]
    def cp(src,dest):
        dest=project/dest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);copied.append({'path':str(dest.relative_to(project)),'sha256':sha(dest)})
    for name in ['adaptive_sharing.py','replay.py','preregistration_v2.json']:cp(BASE/'workspaces/B'/name,Path('B')/name)
    cp(BASE/'integration/sharing_input_guard.py','sharing_input_guard.py')
    for name in ['train_residuals.csv.gz','train_queries.json','manifest.json']:cp(BASE/'inputs'/name,Path('inputs')/name)
    mods=closure(['analysis.paper_upgrade_replay_models','analysis.rolling_residual_stats','analysis.multirank_model','analysis.backtest','analysis.holdout_dataset'])
    for m in mods:rel=Path(m.replace('.','/')+'.py');cp(ROOT/rel,rel)
    c_rel=Path('analysis/paper_upgrade_round2/20261118/workspaces/C')
    cp(BASE/'workspaces/C/c_fairness.py',c_rel/'c_fairness.py')
    data=Path('data/temporal_holdout_v2_dense_t1000_t2000_20261115')
    for name in ['train_inputs.jsonl.gz','train_cases.csv','train_labels.csv']:cp(ROOT/data/name,data/name)
    # Audit hook is installed before importing any copied model source. Original
    # project source/data reads are rejected; .venv supplies installed packages.
    worker=r'''import sys,os,json,hashlib,importlib.util,time
from pathlib import Path
sys.dont_write_bytecode=True
original=Path(sys.argv[1]).resolve();project=Path(sys.argv[2]).resolve();do_b=sys.argv[3]=='1';do_c=sys.argv[4]=='1';denied=[]
def hook(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(original) and not path.is_relative_to(original/'.venv'):
            denied.append(str(path));raise PermissionError('H original-project read rejected: '+str(path))
sys.addaudithook(hook)
os.chdir(project);sys.path[:]=[str(project)]+[p for p in sys.path if p and (not Path(p).resolve().is_relative_to(original) or Path(p).resolve().is_relative_to(original/'.venv'))]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
result={'denied_original_project_reads':denied,'working_project':str(project),'shared_dependency_environment':str(original/'.venv')}
started=time.monotonic()
if do_b:
    sys.path.insert(0,str(project/'B'));import replay
    from sharing_input_guard import GuardedSharingBackend
    replay.model=GuardedSharingBackend(replay.model)
    cfg=json.loads((project/'B/preregistration_v2.json').read_text());qs=json.loads((project/'inputs/train_queries.json').read_text());raw=replay.load_train_vault(project/'inputs/train_residuals.csv.gz')
    manifest=replay.run_replay(raw,qs,cfg,project/'B_replay')
    result['B']={'calls':manifest['predict_calls'],'selected':manifest['selected_candidate'],'seconds':manifest['elapsed_seconds'],'guard_adapter':'GuardedSharingBackend injected over unchanged frozen model'}
if do_c:
    import numpy as np,torch,joblib
    from analysis.multirank_model import make_preprocessor
    c=load('c_fairness',project/'analysis/paper_upgrade_round2/20261118/workspaces/C/c_fairness.py')
    cases,features=c.load_train_prefixes(c.DATA);block=c.training_groups(cases)[0];events=block['groups']['N32full'];mask=cases.event.isin(events).to_numpy();qmask=cases.event.isin(block['query_events']).to_numpy();train=cases.loc[mask];query=cases.loc[qmask]
    labels=c.allowed_train_labels(c.DATA/'train_labels.csv',train.case_id);actual=np.array([labels[k] for k in train.case_id]);y=np.log(actual/train.current.to_numpy())+np.log(train.progress.to_numpy());w=c.event_rank_weights(train.event.to_numpy(),train['rank'].to_numpy())
    prep=make_preprocessor(features.loc[mask]);x=prep.fit_transform(features.loc[mask]);qx=prep.transform(features.loc[qmask]);mean=np.average(x,axis=0,weights=w);sd=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=w)),1e-4)
    nx=np.clip((x-mean)/sd,-8,8);nqx=np.clip((qx-mean)/sd,-8,8);rank=cases['rank'].map({1000:0,2000:1}).to_numpy(int)
    pred,checkpoint,cost=c._fit_network(nx,y,w,rank[mask],nqx,rank[qmask],structure='shared',epochs=60,seed=42)
    torch.save(checkpoint,project/'C_refit.pt');score=query.current.to_numpy()*np.exp(np.clip(pred-np.log(query.progress.to_numpy()),0,4));query.assign(predicted=score).to_csv(project/'C_refit_predictions.csv',index=False)
    result['C']={'features':nx.shape[1],'train_rows':len(nx),'query_rows':len(query),'physical_fits':len(cost),'parameters':sum(v.numel() for v in checkpoint['state_dict'].values())}
# Deliberate negative-control probe proves the hook runs. It is distinct from
# dependency reads: required-source failures abort before this line.
probe=False
try:(original/'analysis/paper_upgrade_replay_models.py').read_bytes()
except PermissionError:probe=True
result['guard_negative_control_passed']=probe;result['seconds']=time.monotonic()-started
(project/'worker_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    (out/'standalone_worker.py').write_text(worker);shutil.copyfile(out/'standalone_worker.py',project/'standalone_worker.py')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONPATH='')
    start=time.monotonic();proc=subprocess.run([sys.executable,str(project/'standalone_worker.py'),str(ROOT),str(project),'1' if args.b_full35 else '0','1' if args.c_single_fit else '0'],cwd=project,env=env,capture_output=True,text=True,timeout=45 if args.b_full35 else 20)
    (out/'stdout.log').write_text(proc.stdout);(out/'stderr.log').write_text(proc.stderr)
    result={'temporary_standalone_project':str(project),'copied_closure':copied,'local_analysis_modules':mods,'returncode':proc.returncode,'elapsed_seconds':time.monotonic()-start,'launcher_sha256':sha(__file__),'worker_sha256':sha(out/'standalone_worker.py'),'comparisons':{}}
    if proc.returncode==0:
        result['worker']=json.loads((project/'worker_result.json').read_text())
        import pandas as pd,numpy as np
        if args.b_full35:
            a=pd.read_csv(BASE/'workspaces/B/run_v2_001/scored_cases.csv.gz');bb=pd.read_csv(project/'B_replay/scored_cases.csv.gz');key=['candidate','case_id'];a=a.sort_values(key);bb=bb.sort_values(key)
            result['comparisons']['B']={'same_keys':list(map(tuple,a[key].to_numpy()))==list(map(tuple,bb[key].to_numpy())),'prediction_max_abs':float(np.max(np.abs(a.prediction.to_numpy()-bb.prediction.to_numpy()))),'rows':len(a)}
        if args.c_single_fit:
            import torch
            old=torch.load(BASE/'workspaces/C/runs/c2_002/models/endpoint32_N32full/shared60_seed42.pt',map_location='cpu',weights_only=True);new=torch.load(project/'C_refit.pt',map_location='cpu',weights_only=True)
            delta=max(float(torch.max(torch.abs(v-new['state_dict'][k]))) for k,v in old['state_dict'].items());a=pd.read_csv(BASE/'workspaces/C/runs/c2_002/predictions.csv.gz');a=a[(a.endpoint==32)&(a.group=='N32full')&(a.structure=='shared60')&(a.seed.astype(str)=='42')].set_index('case_id');bb=pd.read_csv(project/'C_refit_predictions.csv').set_index('case_id')
            result['comparisons']['C']={'state_tensor_max_abs':delta,'prediction_max_abs':float(np.max(np.abs(a.predicted-bb.predicted))),'rows':len(bb)}
    shutil.copytree(project,out/'standalone_project',dirs_exist_ok=True)
    (out/'portability_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'result':str(out/'portability_result.json'),'returncode':proc.returncode,'comparisons':result['comparisons']}))
    return proc.returncode
if __name__=='__main__':raise SystemExit(launch())
