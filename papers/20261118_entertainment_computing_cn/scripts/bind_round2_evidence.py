"""Bind sealed round-two artifacts to LaTeX; no fitting, selection or scoring.

Paths are explicit. Run with --repo-root /absolute/repository --output /absolute/manuscript.
"""
from pathlib import Path
import argparse,csv,json,hashlib,math,subprocess,sys,re
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo-root',required=True,type=Path)
p.add_argument('--output','--paper-root',dest='output',required=True,type=Path)
p.add_argument('--skip-legacy',action='store_true')
a=p.parse_args(); ROOT=a.repo_root.resolve(); OUT=a.output.resolve()
BASE=ROOT/'analysis/paper_upgrade_round2/20261118/workspaces'
for n in ['data','tables','qa']: (OUT/n).mkdir(parents=True,exist_ok=True)
if not a.skip_legacy:
 subprocess.run([sys.executable,str(OUT/'scripts/bind_upgrade_evidence.py'),'--repo-root',str(ROOT),'--output',str(OUT)],check=True)
SOURCES={}; BINDINGS=[]; MACROS={}; OUTPUTS=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def source(rel):
 path=BASE/rel
 if not path.is_file():raise FileNotFoundError(path)
 SOURCES[str(path.relative_to(ROOT))]=sha(path)
 owner=Path(rel).parts[0];delivery=BASE/owner/'delivery_manifest.json'
 if delivery.is_file():
  dm=json.loads(delivery.read_text());expected=dm.get('files',{}).get(str(Path(rel).relative_to(owner)))
  SOURCES[str(delivery.relative_to(ROOT))]=sha(delivery)
  if expected and expected!=sha(path):raise ValueError('Delivery source hash mismatch: '+str(path))
 return path
def rows(rel):
 with source(rel).open(newline='') as h:return list(csv.DictReader(h))
def js(rel):return json.loads(source(rel).read_text())
def one(rs,**kw):
 z=[r for r in rs if all(str(r.get(k))==str(v) for k,v in kw.items())]
 if len(z)!=1:raise ValueError((kw,len(z)))
 return z[0]
def inferred_unit(rel,row,field,mult):
 if field in ['event_macro_mape','event_macro_mape_pct','event_macro_mape','case_mape','event_macro_mape']:return 'MAPE (%)'
 if field in ['event_macro_log_squared','event_macro_log_mse']:return 'squared log error'
 if field=='event_macro_mape':return 'MAPE (%)'
 if field in ['delta_vs_independent_pp','difference']:return 'MAPE difference (percentage points)'
 if field in ['task_states','lambda_positive_states']:return 'task states (count)'
 if field.startswith('lambda_') or field=='delete_range_max':return 'dimensionless sharing coefficient'
 if field=='optimizer_steps':return 'optimizer steps (count)'
 if field=='parameter_updates':return 'scalar parameter updates (count)'
 if field=='forward_linear_MACs':return 'training forward linear MACs (count)'
 if field=='cpu_seconds':return 'fit CPU seconds'
 if field=='wall_seconds':return 'fit wall seconds'
 if field in ['n_events','whole_event_successes','target_events','minimum_activity_floor','required_events','primary_zero_null_model_n','mechanism_secondary_zero_null_model_n','future_events_at_addendum']:return 'activities (count)'
 if field in ['n_groups']:return 'complete rank-phase groups (count)'
 if field in ['n_cases']:return 'unique physical cases (count)'
 if 'coverage' in field or field in ['low95','high95'] and mult==100:return 'coverage (%)' if mult==100 else 'coverage (fraction)'
 if field in ['low95','high95']:return 'MAPE difference (percentage points)'
 if field in ['event_equal_relative_width_pct']:return 'interval width / actual final score (%)'
 if 'calibration_parameters' in rel:return 'dimensionless model parameter' if row.get('parameter') in ['rho','m','q_r_ratio'] else 'squared log error per query activity'
 if field=='Q_minimum_eigenvalue':return 'squared log error per query activity'
 if 'pearson' in field or field.startswith('own_lag_'):return 'dimensionless correlation'
 return 'dimensionless'
def val(rel,row,field,where,digits=3,mult=1,name=None,unit=None):
 v=float(row[field])*mult
 unit=unit or inferred_unit(rel,row,field,mult)
 if not math.isfinite(v):raise ValueError((rel,field,v))
 r=f'{v:.{digits}f}'
 BINDINGS.append({'binding':name or f'{rel}:{where}:{field}','source':str((BASE/rel).relative_to(ROOT)),'source_sha256':sha(BASE/rel),'field':field,'row_condition':where,'raw_value':row[field],'value':v,'rendered':r,'multiplier':mult,'unit':unit,'rounding':f'{digits} decimals'})
 if name:
  if not name.isalpha():raise ValueError('LaTeX macro name must contain letters only: '+name)
  MACROS[name]=r
 return r
def macro(name,rel,rs,field,where,digits=3,mult=1,unit=None):
 return val(rel,one(rs,**where),field,where,digits,mult,name,unit)
def table(file,title,label,headers,body,columns=None,note='',long=False):
 note=re.sub(r'(?<!\\)%',r'\\%',note)
 col=columns or 'L'+' r'*(len(headers)-1)
 if long:
  t='\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{'+col+'}\n\\caption{'+title+'}\\label{'+label+'}\\\\\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\\endfirsthead\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\\endhead\n'
  t+='\n'.join(' & '.join(r)+' \\\\' for r in body)+'\n\\bottomrule\n\\end{longtable}\n'+note+'\\par\n\\endgroup\n'
 else:
  t='\\begin{table}[!htbp]\\centering\n\\caption{'+title+'}\\label{'+label+'}\n\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabularx}{\\linewidth}{'+col+'}\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
  t+='\n'.join(' & '.join(r)+' \\\\' for r in body)+'\n\\bottomrule\\end{tabularx}\n'
  if note:t+='\\par\\vspace{3pt}\\begin{minipage}{\\linewidth}\\footnotesize '+note+'\\end{minipage}\n'
  t+='\\end{table}\n'
 path=OUT/'tables'/file;path.write_text(t);OUTPUTS.append(path)
# Read sealed evidence and attest input files, including independent audit evidence.
for rel in ['B/run_v2_001/run_manifest.json','B/run_v2_001/selection_seal.json','C/evidence_001/evidence_seal.json','H/H_numeric_evidence.json','H/H_B_scalar_evidence.json','H/H_D_numeric_evidence.json','H/H_D_transfer_evidence.json','H/H_E_numeric_evidence.json']:
 if (BASE/rel).exists():source(rel)
# Audit all artifact seals, using only their stored hashes.
for directory,manifest,key in [('B/run_v2_001','run_manifest.json','files'),('C/evidence_001','evidence_seal.json',None),('D/calibration_run_v2_001','run_manifest.json','outputs_sha256'),('D/simulation_run_v2_001','run_manifest.json','outputs_sha256'),('E/statistics_002','output_seal.json',None),('E/reliability_002','output_seal.json',None)]:
 seal=js(directory+'/'+manifest)
 entries=seal[key] if key else seal
 if 'files' in entries and isinstance(entries['files'],dict):entries=entries['files']
 for filename,digest in entries.items():
  if not isinstance(digest,str) or len(digest)!=64:continue
  path=BASE/directory/filename
  if not path.is_file():raise FileNotFoundError('Sealed artifact missing: '+str(path))
  if sha(path)!=digest:raise ValueError('Artifact seal mismatch: '+str(path))
# B: all twelve candidates, selected on the middle ten events.
bp='B/run_v2_001/summary.csv';br=rows(bp)
bs='E/statistics_002/B_paired.csv';bci=rows(bs)
bnames={'independent_fixed_recipe':'独立EWMA（预定选择）','fixed_quarter':'固定1/4共享','fixed_half':'固定1/2共享','fully_shared_pair':'两线完全共享','risk_k4':'风险收缩 k=4','risk_k12':'风险收缩 k=12','risk_recent_k8':'近期风险 k=8','risk_gated_k8':'稳健门控 k=8','same_panel_independent':'同面板独立','risk_only_T1000':'风险规则仅改T1000','risk_only_T2000':'风险规则仅改T2000','adjacent_risk_k8':'相邻排名风险规则'}
body=[]
for method,label in bnames.items():
 w={'block':'selection','candidate':method,'panel':'pair1000_2000'}
 # Native B uses pair panel name; enforce discovery with exact stable candidate/block.
 panel=next(r['panel'] for r in br if r['candidate']==method and r['block']=='selection' and r['panel']!='all19')
 w['panel']=panel; ss=one(br,**w);tw={**w,'block':'diagnostic'}
 if not any(r.get('block')=='diagnostic' for r in br):tw['block']='temporal_diagnostic'
 tr=one(br,**tw)
 nums=[val(bp,ss,'event_macro_mape',w),val(bp,tr,'event_macro_mape',tw),val(bp,tr,'event_macro_log_squared',tw,6,unit='squared log error')]
 body.append([label,*nums])
 if method in ['independent_fixed_recipe','fully_shared_pair','risk_recent_k8','fixed_quarter','risk_gated_k8']:macro({'independent_fixed_recipe':'RTIndependent','fully_shared_pair':'RTPool','risk_recent_k8':'RTRisk','fixed_quarter':'RTQuarter','risk_gated_k8':'RTGated'}.get(method,'RTUnused'+''.join(x.title() for x in method.split('_'))),bp,br,'event_macro_mape',tw)
 if method=='independent_fixed_recipe':macro('RSIndependent',bp,br,'event_macro_mape',w)
table('round2_adaptive.tex','训练内滚动选择：全部12个固定共享候选','tab:round2adaptive',['候选','选择10期 / \\%','末10期 / \\%','末段log平方风险'],body,note='两线主面板；选择块372案例，末段352案例；各10活动等权。次10期按候选顺序处理并列，选中独立EWMA。log平方损失与MAPE分别保留。')
# Stable named paired intervals from independent E.
for method,prefix in [('fully_shared_pair','Pool'),('risk_recent_k8','Risk')]:
 cond={'panel':panel,'period':'diagnostic','left':method,'right':'independent_fixed_recipe','metric':'event_macro_MAPE_pct','block':'1'}
 if not any(r['period']=='diagnostic' for r in bci):cond['period']='temporal_diagnostic'
 for field,suffix in [('difference','Delta'),('low95','Low'),('high95','High')]:macro('RT'+prefix+suffix,bs,bci,field,cond)
lp='B/lambda_diagnostics.csv';lr=rows(lp); lbody=[]
for method in ['risk_k4','risk_k12','risk_recent_k8','risk_gated_k8','risk_only_T1000','risk_only_T2000','adjacent_risk_k8']:
 cond={'block':tw['block'],'candidate':method};r=one(lr,**cond)
 lbody.append([bnames[method],val(lp,r,'lambda_positive_states',cond,0)+'/'+val(lp,r,'task_states',cond,0),val(lp,r,'lambda_mean',cond),val(lp,r,'lambda_max',cond),val(lp,r,'delete_range_max',cond)])
table('round2_lambda.tex','末10期的共享系数与删除敏感性','tab:round2lambda',['规则','正系数/状态','均值','最大值','最大删除跨度'],lbody,note='60状态为10活动×2排名×3阶段。门控行的删除跨度属于门控前风险估计；实际系数全为零。删除跨度描述有限支持敏感性。')
# C1: all ablations on the frozen last-ten panel.
cp='C/evidence_001/C1_last10_all_ablations.csv';cr=rows(cp)
methods=[r['method'] for r in cr if r['panel']=='T1000_T2000']
clabels={'uncorrected':'未校正','independent':'独立EWMA','independent_reference':'独立EWMA','ewma_reference':'独立EWMA','legacy_pair':'原锚点差异','legacy_only_T1000':'仅替换T1000','legacy_only_T2000':'仅替换T2000慢锚点','swapped_only_T2000':'T1000作锚，仅替换T2000','pair_anchor_only':'仅锚点','pair_difference_only':'仅差异','same_filter_pair':'同滤波坐标负控','same_panel_independent':'同面板独立','equal_alpha':'统一衰减系数','equal_shrinkage':'统一收缩','complete_support':'完整配对支持','no_anchor_shrink':'锚点不收缩','no_difference_shrink':'差异不收缩','no_shrink':'全部不收缩','anchor_alpha_0p3':'锚点衰减改为.3','difference_alpha_0p3':'差异衰减改为.3','anchor_gap':'原锚点差异','original_anchor_gap':'原锚点差异'}
cb=[]
for r in cr:
 if r['panel']!='T1000_T2000':continue
 cond={'panel':r['panel'],'period':r['period'],'method':r['method']}
 cb.append([clabels.get(r['method'],r['method'].replace('_',r'\_')),*[val(cp,r,k,cond) for k in ['event_macro_mape_pct','delta_vs_independent_pp']]])
table('round2_ablations.tex','训练末10期：18项预定结构消融','tab:round2ablations',['固定规则','两线MAPE / \\%','相对独立差 / pp'],cb,note='各规则共用10期352案例。方法标识对应封存配置；仅替换某一线时，另一线保留独立校正。该表保留预定消融身份，末段未用于重选。')
# C2: endpoint ensembles, actual budget and all individual seeds.
c2p='E/statistics_002/C2_summary.csv';c2=rows(c2p)
body=[];seedbody=[]
parp='C/evidence_001/C2_actual_parameter_equality.csv';pars=rows(parp)
for endpoint in ['32','44','56']:
 for group in ['N32full','N16span','N16recent']:
  vals=[]
  for structure in ['independent60','shared60','shared30']:
   w={'endpoint':endpoint,'group':group,'structure':structure,'seed':'mean'}
   vals.append(val(c2p,one(c2,**w),'event_macro_mape_pct',w))
  pw={'endpoint':endpoint,'group':group,'seed':'42'};pr=one(pars,**pw)
  if len({pr[k] for k in ['independent60','shared60','shared30']})!=1:raise ValueError('Actual parameter mismatch')
  body.append([endpoint,group,val(parp,pr,'independent60',pw,0,unit='parameters'),*vals])
  for seed in ['42','123','2026']:
   z=[]
   for structure in ['independent60','shared60','shared30']:
    w={'endpoint':endpoint,'group':group,'structure':structure,'seed':seed};z.append(val(c2p,one(c2,**w),'event_macro_mape_pct',w))
   seedbody.append([endpoint,group,seed,*z])
table('round2_fairness.tex','训练期多终点：总参数匹配与近似前向计算匹配','tab:round2fairness',['终点','历史组','总参数','独立60轮','共享60轮','共享30轮'],body,columns='r L r r r r',note='单位为三种子最终分数均值集成的MAPE（%）；每终点4活动，合计12个查询活动、442个唯一案例。同单元总参数均为32d+66。')
table('round2_fairness_seeds.tex','训练期公平对照的全部27个种子单元','tab:round2fairseeds',['终点','历史组','种子','独立60轮','共享60轮','共享30轮'],seedbody,columns='r p{2.5cm} r r r r',long=True,note='单位为活动等权MAPE（%）。27个种子单元涉及108次物理拟合；共享结构各27次，独立结构54次。种子与重复终点维持活动重叠关系。')
bup='C/evidence_001/C2_budget_totals.csv';bur=rows(bup);bub=[]
for r in bur:
 cond={'structure':r['structure']}
 bub.append([r['structure'],*[val(bup,r,k,cond,0 if k not in ['cpu_seconds','wall_seconds'] else 3) for k in ['optimizer_steps','parameter_updates','forward_linear_MACs','cpu_seconds']]])
table('round2_budget.tex','公平训练的实际更新与计算预算','tab:round2budget',['结构','优化器步数','标量参数更新','前向线性MAC','拟合CPU秒'],bub,note='MAC只计训练前向线性层；反向传播、激活、偏置、优化器及预处理另有成本。共享30轮比独立60轮多约1.73%前向MAC，每参数更新次数减半。')
# D: original moments, restricted profile and all simulation directions.
dp='D/calibration_run_v2_001/paired_diagnostics.csv';dr=rows(dp);dw={'segment':'calibration','phase':'1'};d=one(dr,**dw)
for field,name,digits in [('level_pearson','DContemp',4),('own_lag_1000','DLagOne',4),('own_lag_2000','DLagTwo',4),('Q_minimum_eigenvalue','DQMin',7),('level_pearson_block3_low','DCorrLow',4),('level_pearson_block3_high','DCorrHigh',4)]:macro(name,dp,dr,field,dw,digits)
# Derived quantities retain both input fields and explicit aggregation.
ratio=float(d['R_raw_00'])/float(d['R_raw_11']);MACROS['DRatio']=f'{ratio:.2f}'
BINDINGS.append({'binding':'DRatio','source':str((BASE/dp).relative_to(ROOT)),'source_sha256':sha(BASE/dp),'field':['R_raw_00','R_raw_11'],'row_condition':dw,'aggregation':'R_raw_00 / R_raw_11','value':ratio,'rendered':MACROS['DRatio'],'unit':'ratio'})
pr='D/calibration_run_v2_001/calibration_parameters.csv';prs=rows(pr);pd=[]
for r in prs:
 cond={'parameter':r['parameter']};pd.append([r['parameter'].replace('_',r'\_'),*[val(pr,r,k,cond,6) for k in ['best','support_low','support_high','delete_one_low','delete_one_high']]])
table('round2_calibration.tex','受限局部水平族的参数支持范围','tab:round2calibration',['参数','最优','支持下限','支持上限','删一下限','删一上限'],pd,note='900点离散网格；描述性支持条件为2ΔNLL≤10，保留17点。参数限于预定等观测方差与独立过程创新族。删一范围仅使用删除后所选参数；原表附带似然元数据属于完整25期。')
sp='D/simulation_run_v2_001/failure_regions.csv';sr=rows(sp);sbody=[]
snames={'fully_shared':'完全共享','fully_shared_pair':'完全共享','fixed_half':'固定半共享','risk_recent_k8':'近期风险','dlm_fitted':'校准DLM','dlm_calibrated':'校准DLM','dlm_oracle':'已知基准协方差oracle','fitted_fullcov_DLM':'校准完整协方差DLM','oracle_fullcov_DLM':'已知基准协方差oracle'}

for method in dict.fromkeys(r['method'] for r in sr):
 if method=='independent_ewma':continue
 vals=[]
 for metric in ['state_mse','synthetic_unbounded_ape_pct']:
  z=[r for r in sr if r['method']==method and r['metric']==metric]
  if not z: metric=next(k for k in {r['metric'] for r in sr} if ('ape' in k) and k!='raw_state_mse');z=[r for r in sr if r['method']==method and r['metric']==metric]
  count=(sum(float(r['mean_minus_I'])<0 for r in z),sum(float(r['mean_minus_I'])>0 for r in z))
  vals.append(f'{count[0]}/{count[1]}')
  BINDINGS.append({'binding':f'D-count-{method}-{metric}','source':str((BASE/sp).relative_to(ROOT)),'source_sha256':sha(BASE/sp),'field':'mean_minus_I','row_condition':{'method':method,'metric':metric},'aggregation':'negative/positive count of 24 fixed regime means','value':list(count),'rendered':vals[-1],'unit':'regimes'})
 sbody.append([snames.get(method,method.replace('_',r'\_')),*vals])
table('round2_simulation.tex','真实量纲校准模拟中的完整方向计数','tab:round2sim',['规则','状态MSE减少/增加','模拟APE减少/增加'],sbody,note='3个参数锚点×8扰动，共24格；每格3种子各100路径，方法与格间共用底层随机数。独立Monte Carlo单位为完整路径；24格描述设计覆盖。')
tp='D/posthoc_transfer_run_001/summary.json';tr=js(tp)
# H independent scalar-filter evidence carries the same-phase comparators.
hp='H/H_D_transfer_evidence.json';hr=js(hp)
for field,name in [('event_macro_mape','DTransfer'),('event_macro_log_mse','DTransferLog')]:
 val(hp,hr,field,{},6,name=name,unit='MAPE (%)' if field=='event_macro_mape' else 'squared log error')
val(hp,hr['same_phase_comparator_mape'],'independent_fixed_recipe',{'object':'same_phase_comparator_mape'},6,name='DTransferI',unit='MAPE (%)')
# E: coverage targets and prespecified conditional subsets.
ep='E/reliability_002/interval_summary.csv';er=rows(ep);eb=[]
for r in er:
 w={'nominal_coverage':r['nominal_coverage']};nom=round(100*float(r['nominal_coverage']))
 vals=[val(ep,r,k,w,2,100 if ('coverage' in k) else 1,unit='coverage (%)') for k in ['event_equal_point_coverage','event_equal_group_joint_coverage','event_equal_group_joint_coverage_low95','event_equal_group_joint_coverage_high95']]
 vals.append(val(ep,r,'whole_event_successes',w,0)+'/'+val(ep,r,'n_events',w,0))
 vals.append(val(ep,r,'event_equal_relative_width_pct',w,2,unit='interval width / actual final score (%)'))
 eb.append([str(nom),vals[0],vals[1]+' ['+vals[2]+','+vals[3]+']',vals[4],vals[5]])
 macro('EWidth'+('Eighty' if nom==80 else 'Ninety'),ep,er,'event_equal_relative_width_pct',w,2)
 macro('EWinkler'+('Eighty' if nom==80 else 'Ninety'),ep,er,'event_equal_winkler_score',w,2,mult=1e-4,unit='ten thousand EP')
table('round2_reliability.tex','三个覆盖目标与区间宽度','tab:round2reliability',['名义 / \\%','逐案例 / \\%','完整组联合 [95\\%区间]','整活动成功','相对宽度 / \\%'],eb,note='逐案例、完整组联合及宽度均先在活动内汇总，再活动等权。各水平5220案例、878个完整排名×阶段组、16活动；均为有限区间。重抽样固定已发区间。')
ec='E/reliability_002/interval_full_group_conditions.csv';ecs=rows(ec);ecb=[]
for dim,value,label in [('rank','1000','T1000'),('event_type','challenge','challenge类型'),('source','all_event_phase_pool','跨排名回退池')]:
 for nom in ['0.8','0.9']:
  w={'dimension':dim,'value':value,'nominal_coverage':nom};r=one(ecs,**w)
  ecb.append([label,str(round(100*float(nom))),val(ec,r,'n_events',w,0)+'/'+val(ec,r,'n_groups',w,0),val(ec,r,'event_equal_full_group_joint_coverage',w,2,100), '['+val(ec,r,'low95',w,2,100)+','+val(ec,r,'high95',w,2,100)+']'])
table('round2_conditions.tex','预定条件审计的低覆盖与回退实例','tab:round2conditions',['条件','名义 / \\%','活动/组数','联合覆盖 / \\%','95\\%活动区间'],ecb,note='完整76个逐案例分层与76个完整组分层保留于原生CSV。条件间活动、排名、阶段及历史支持组成共同变化。回退池高覆盖伴随较宽区间。')
pp='E/reliability_002/power_grid.csv';pwr=rows(pp);pwbody=[]
for effect in ['0.25','0.5','1.0']:
 v=[]
 for rho in ['0.0','0.3','0.6']:
  w={'panel':'T1000_T2000','sigma_scenario':'normal_iid_chi2_high95','effect_pp':effect,'target_power':'0.9','rho':rho};r=one(pwr,**w);v.append(val(pp,r,'required_events',w,0,unit='events'))
 pwbody.append([effect,*v])
table('round2_power.tex','训练差异驱动的两线样本量敏感性','tab:round2power',['假设备择改善 / pp','ρ=0','ρ=.3','ρ=.6'],pwbody,note='双侧α=.05、功效.9、训练35期正态iid卡方区间的上限σ；非中心t基于独立正态活动差，AR(1)长期方差膨胀仅为规划近似。该表规划零差检验。')
ap='E/confirmation_plan_addendum.json';ad=js(ap)
for field,name in [('target_events','ConfirmTarget'),('minimum_activity_floor','ConfirmFloor'),('primary_zero_null_model_n','ConfirmGlobalN'),('mechanism_secondary_zero_null_model_n','ConfirmPairN'),('future_events_at_addendum','ConfirmFuture')]:val(ap,ad,field,{},0,name=name,unit='events')
# Include exact raw inputs in a portable artifact directory; manifest always names repo-relative sources.
(OUT/'data/round2_macros.tex').write_text('% Generated only from sealed evidence.\n'+''.join('\\newcommand{\\'+n+'}{'+v+'}\n' for n,v in MACROS.items()))
for binding in BINDINGS:
 binding.setdefault('raw_value',binding.get('value'))
manifest={'schema_version':2,'scope':'read-only binding of sealed round-two data; legacy scalar bindings remain under upgrade_number_bindings.json','repo_root_parameter':'explicit --repo-root','sources':SOURCES,'bindings':BINDINGS,'generated_files':{str(x.relative_to(OUT)):sha(x) for x in OUTPUTS},'macros_sha256':sha(OUT/'data/round2_macros.tex')}
(OUT/'qa/round2_number_bindings.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(f'Bound {len(MACROS)} round-two macros, {len(BINDINGS)} scalar/aggregate entries and {len(OUTPUTS)} native tables.')
