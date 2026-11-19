"""Independent scalar likelihood/aggregates and D neighborhood fixtures."""
from __future__ import annotations
import csv,gzip,hashlib,importlib.util,json,math,statistics,sys
from pathlib import Path
from collections import defaultdict
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];D=BASE/'workspaces/D';CAL=D/'calibration_run_v2_001';SIM=D/'simulation_run_v2_001'
def j(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    p=Path(p)
    with (gzip.open(p,'rt',newline='') if p.suffix=='.gz' else p.open(newline='')) as f:return list(csv.DictReader(f))
checks={}
def check(k,v,detail=None):checks[k]={'passed':bool(v),'detail':detail}
def covariance(g):
    qc,qd,r,rho,m=[float(g[k]) for k in ['qc','qd','r','rho','m']];a=1-m;b=-m
    return [[qc+a*a*qd,qc+a*b*qd],[qc+a*b*qd,qc+b*b*qd]],[[r,r*rho],[r*rho,r]]
def mm(a,b):return [[sum(a[i][k]*b[k][z] for k in range(2)) for z in range(2)] for i in range(2)]
def nll(y,ordinal,g):
    q,r=covariance(g);mean=list(y[0]);p=[row[:] for row in r];loss=0.
    for i in range(1,len(y)):
        prior=[[p[a][b]+(ordinal[i]-ordinal[i-1])*q[a][b] for b in range(2)] for a in range(2)];s=[[prior[a][b]+r[a][b] for b in range(2)] for a in range(2)];det=s[0][0]*s[1][1]-s[0][1]*s[1][0]
        inverse=[[s[1][1]/det,-s[0][1]/det],[-s[1][0]/det,s[0][0]/det]];v=[y[i][a]-mean[a] for a in range(2)];quad=sum(v[a]*inverse[a][b]*v[b] for a in range(2) for b in range(2));loss+=.5*(2*math.log(2*math.pi)+math.log(det)+quad)
        gain=mm(prior,inverse);mean=[mean[a]+sum(gain[a][b]*v[b] for b in range(2)) for a in range(2)];gp=mm(gain,prior);p=[[prior[a][b]-gp[a][b] for b in range(2)] for a in range(2)];off=(p[0][1]+p[1][0])/2;p[0][1]=p[1][0]=off
    return loss

for run in [CAL,SIM]:
    m=j(run/'run_manifest.json');bad=[fn for fn,h in m['outputs_sha256'].items() if sha(run/fn)!=h];badsrc=[fn for fn,h in m['source_seal']['sha256'].items() if sha(D/fn)!=h]
    check(run.name+'_hashes',not bad and not badsrc,{'outputs':len(m['outputs_sha256']),'bad_outputs':bad,'bad_sources':badsrc})
raw=rows(BASE/'inputs/train_residuals.csv.gz');qs=sorted(j(BASE/'inputs/train_queries.json'),key=lambda q:(q['start_ms'],q['event']));event_ids=[q['event'] for q in qs];means=defaultdict(list)
for r in raw:means[int(r['event']),int(r['rank']),int(r['phase'])].append(float(r['log_error']))
means={k:statistics.fmean(v) for k,v in means.items()};cal=j(CAL/'calibration_seal.json');y=[[means[e,1000,1],means[e,2000,1]] for e in event_ids[:25]];ordinals=list(range(25));grid=rows(CAL/'calibration_profile.csv')
losses=[nll(y,ordinals,g) for g in grid];best=min(range(len(losses)),key=losses.__getitem__);delta=max(abs(loss-float(g['nll'])) for loss,g in zip(losses,grid));supported=[i for i,loss in enumerate(losses) if 2*(loss-losses[best])<=10]
check('calibration_grid_independent_scalar_NLL',len(grid)==900 and delta<1e-9 and best==cal['best']['grid_index'],{'grid_points':len(grid),'max_abs_NLL_difference':delta,'best_grid_index':best,'support_points':len(supported)})
check('calibration_train25_only',cal['complete_pair_events']==event_ids[:25] and cal['activity_ordinals']==ordinals and set(cal['complete_pair_events']).isdisjoint(event_ids[25:]))
for key,index in [('qc',0),('qd',1),('r',2),('rho',3),('m',4)]:pass
deleted=2;yy=[v for k,v in enumerate(y) if k!=deleted];oo=[v for v in ordinals if v!=deleted];dl=[nll(yy,oo,g) for g in grid];dbest=min(range(len(dl)),key=dl.__getitem__);export=next(row for row in cal['delete_one_best'] if row['deleted_ordinal']==deleted)
check('delete_one_parameter_argmin_independent',dbest==export['grid_index'],{'deleted_ordinal':deleted,'selected_grid_index':dbest,'deletion_NLL':dl[dbest],'exported_full25_NLL':export['nll'],'deletion_complete_pairs':24,'deletion_adjacent_differences':sum(b-a==1 for a,b in zip(oo,oo[1:])),'exported_full25_complete_pairs':export['complete_pairs']})
out={'checks':checks,'delete_one_independent_metadata':checks['delete_one_parameter_argmin_independent']['detail']}
# H creates nonmonotone stitched indices, including wrap seams and original gaps.
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
M=load('mechanism',D/'mechanism.py');R=load('h_D_run',D/'run_mechanism.py')
import numpy as np
linear=np.array([[float(i),3.*i] for i in range(12)]);attached=R.pair_contributions(linear,np.arange(12));ix=np.array([8,9,10,0,1,2,8,9,3,4,5,6]);stats=R.contribution_statistics(attached,ix);valid=attached['delta'][ix];valid=valid[np.isfinite(valid).all(axis=1)]
check('block_seams_do_not_create_linear_spikes',np.max(np.var(valid,axis=0))==0 and max(abs(stats.get('Gamma_delta0_'+str(a)+str(b)) or 0.) for a in range(2) for b in range(2))==0,{'sampled_indices':ix.tolist(),'delta_variance':np.var(valid,axis=0).tolist()})
ordinal=np.array([0,1,4,5,8,9]);z=np.array([[float(i),2.*i] for i in ordinal]);attached=R.pair_contributions(z,ordinal);ix=np.array([0,2,3,4,5,1]);stats=R.contribution_statistics(attached,ix)
check('original_gaps_keep_missing_adjacency',stats['adjacent_differences']==3 and stats['difference_lag_pairs']==0,{'adjacent':stats['adjacent_differences'],'lag_pairs':stats['difference_lag_pairs']})
# Recalculate simulation summaries from published path losses. No simulated
# path is regenerated, and no Monte Carlo/bootstrap loop is executed here.
paths=rows(SIM/'per_path.csv');by=defaultdict(list)
for r in paths:by[r['regime'],r['method']].append(r)
bad=[];maxmean=0.;maxse=0.
for r in rows(SIM/'failure_regions.csv'):
    part=by[r['regime'],r['method']];metric=r['metric'];v=[float(p[metric]) for p in part];d=[float(p[metric+'_minus_I']) for p in part];mean=statistics.fmean(v);diff=statistics.fmean(d);se=statistics.stdev(d)/math.sqrt(len(d));maxmean=max(maxmean,abs(mean-float(r['mean'])),abs(diff-float(r['mean_minus_I'])));maxse=max(maxse,abs(se-float(r['paired_mc_se'])))
    if len(part)!=300 or abs(mean-float(r['mean']))>1e-7 or abs(diff-float(r['mean_minus_I']))>1e-7 or abs(se-float(r['paired_mc_se']))>1e-7:bad.append([r['regime'],r['method'],metric])
check('simulation_aggregates_independent_path_arithmetic',not bad and len(paths)==43200,{'path_method_rows':len(paths),'unique_regimes':len({r['regime'] for r in paths}),'max_mean_difference':maxmean,'max_MC_SE_difference':maxse,'bad':bad})
check('simulation_prespecified_menu_and_seeds',len({r['regime'] for r in paths})==24 and {int(r['seed']) for r in paths}=={42,123,2026} and len({r['method'] for r in paths})==6)
check('simulation_calibration_binding',j(SIM/'run_manifest.json')['calibration_seal_sha256']==sha(CAL/'calibration_seal.json'))
out.update(checks=checks,script_sha256=sha(__file__),not_run=['full bootstrap replay','all simulated trajectories regeneration'])
(HERE/'H_D_numeric_evidence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'checks':len(checks),'failures':[k for k,v in checks.items() if not v['passed']],'deletion':out['delete_one_independent_metadata']}))
