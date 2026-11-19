"""Bind sealed independent scores to native LaTeX. No fitting or scoring occurs."""
from pathlib import Path
import csv
import hashlib
import json
import argparse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo-root", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args=parser.parse_args()

PAPER = args.output.resolve()
ROOT = args.repo_root.resolve()
(PAPER / "qa").mkdir(parents=True, exist_ok=True)
RECORD = ROOT / 'analysis/paper_upgrade_records/20261118'
STAT = RECORD / 'results/statistics'
TABLES = PAPER / 'tables'
BINDINGS, SOURCES = [], {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f'Finalized evidence required: {p}')
    key = str(p.relative_to(ROOT))
    SOURCES[key] = sha(p)
    if p.is_relative_to(STAT):
        manifest = p.parent / 'manifest.json'
        m = json.loads(manifest.read_text())
        if m['artifacts'].get(p.name) != sha(p):
            raise ValueError(f'Sealed scorer artifact mismatch: {key}')
        SOURCES[str(manifest.relative_to(ROOT))] = sha(manifest)
    return p


def rows(p):
    p = source(p)
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def js(p):
    return json.loads(source(p).read_text())


def get(rs, **conditions):
    found = [r for r in rs if all(str(r[k]) == str(v) for k, v in conditions.items())]
    if len(found) != 1:
        raise ValueError(f'Expected unique row: {conditions}; found {len(found)}')
    return found[0]


def number(name, value, path, field, where, digits=3, unit='%', multiplier=1):
    value = float(value) * multiplier
    if not (-float('inf') < value < float('inf')):
        raise ValueError(f'Nonfinite manuscript value: {name}')
    out = f'{value:.{digits}f}'
    BINDINGS.append(dict(binding=name, value=value, rendered=out, source=str(Path(path).relative_to(ROOT)),
                         source_sha256=sha(Path(path)), field=field, row_condition=where,
                         unit=unit, rounding=f'round to {digits} decimal places', multiplier=multiplier))
    return out


def cell(path, r, field, where, digits=3, unit='%', name=None, multiplier=1):
    return number(name or f'table:{Path(path).stem}:{where}:{field}', r[field], path, field, where, digits, unit, multiplier)


def signed(x, digits=3):
    return f'{float(x):+.{digits}f}'


def tab(path, caption, label, columns, header, body, note='', long=False):
    if long:
        text = '\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
        text += '\\begin{longtable}{' + columns + '}\n\\caption{' + caption + '}\\label{' + label + '}\\\\\n'
        text += '\\toprule\n' + header + ' \\\\\n\\midrule\\endfirsthead\n\\toprule\n' + header + ' \\\\\n\\midrule\\endhead\n'
        text += '\n'.join(body) + '\n\\bottomrule\n\\end{longtable}\n'
        if note:
            text += '\\noindent ' + note + '\\par\n'
        text += '\\endgroup\n'
    else:
        text = '\\begin{table}[!htbp]\\centering\n\\caption{' + caption + '}\\label{' + label + '}\n'
        text += '\\small\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.15}\n'
        text += '\\begin{tabularx}{\\linewidth}{' + columns + '}\\toprule\n' + header + ' \\\\\n\\midrule\n'
        text += '\n'.join(body) + '\n\\bottomrule\\end{tabularx}\n'
        if note:
            text += '\\par\\vspace{4pt}\\begin{minipage}{\\linewidth}\\footnotesize ' + note + '\\end{minipage}\n'
        text += '\\end{table}\n'
    (TABLES / path).write_text(text)


def main():
    TABLES.mkdir(exist_ok=True)
    fp = STAT / 'fixed_replay_v2'
    mp, vp, sp = fp/'test_metrics.csv', fp/'validation_metrics.csv', fp/'summary.json'
    metrics, validation, summary = rows(mp), rows(vp), js(sp)
    if len(metrics) != 54 or len(validation) != 54:
        raise ValueError('Fixed menu must retain all 54 methods')
    if any(int(r['events']) != 16 or int(r['cases']) != 5220 for r in metrics):
        raise ValueError('Test panel changed')
    effects = summary['splits']['test']['paired_effects']
    def effect(m, ref='ewma_reference'):
        return get(effects, method=m, reference=ref, weighting='activity_case_weighting')
    def pairfield(name,m,field,ref='ewma_reference',digits=3,unit='percentage points'):
        r=effect(m,ref)
        return number(name,r[field],sp,field,dict(method=m,reference=ref,weighting='activity_case_weighting'),digits,unit)
    macros={}
    for prefix,m in [('Base','uncorrected'),('Ewma','ewma_reference'),('Pair','legacy_pair'),('Difference','pair_difference_only'),('Dlm','dlm_fullcov')]:
        r=get(metrics,method=m)
        macros[prefix+'Test']=cell(mp,r,'event_macro_mape_pct',{'method':m},name=prefix+'Test')
        if prefix in ['Base','Ewma','Pair']:
            macros[prefix+'Val']=cell(vp,get(validation,method=m),'event_macro_mape_pct',{'method':m},name=prefix+'Val')
    macros['PairDelta']=pairfield('PairDelta','legacy_pair','mean_difference_points')
    macros['PairAnchorIncrement']=pairfield('PairAnchorIncrement','legacy_pair','mean_difference_points','pair_difference_only')
    macros['DlmDelta']=pairfield('DlmDelta','dlm_fullcov','mean_difference_points')
    oldp=PAPER/'data/main_models.csv'; old=rows(oldp)
    macros['PairTwoTest']=cell(oldp,get(old,method='legacy_pair'),'original2_test_mape_pct',{'method':'legacy_pair'},4,name='PairTwoTest')
    labels={
      'uncorrected':'未校正基础预测','ewma_reference':'独立EWMA（固定主方法）','legacy_pair':'历史锚点差异组合',
      'pair_difference_only':'仅差异状态','pair_anchor_only':'仅锚点状态','pair_swapped_anchor':'交换锚点',
      'same_filter_pair':'同滤波锚点负控','same_panel_independent':'同面板独立负控','window5':'5期窗口均值',
      'kalman_level':'独立局部水平Kalman','dlm_fullcov':'完整协方差DLM','time_proportion':'时间比例增长',
      'history_growth':'历史增长中位数','residual_mean':'历史残差均值','window10':'10期窗口均值','ewma_fast':'快EWMA',
      'kalman_trend':'独立趋势Kalman'}
    chosen=['uncorrected','ewma_reference','legacy_pair','pair_difference_only','pair_anchor_only','pair_swapped_anchor','same_filter_pair','same_panel_independent','window5','kalman_level','dlm_fullcov','time_proportion']
    body=[]
    for m in chosen:
        r,v=get(metrics,method=m),get(validation,method=m)
        body.append(labels[m]+' & '+cell(vp,v,'event_macro_mape_pct',{'method':m})+' & '+cell(mp,r,'event_macro_mape_pct',{'method':m})+' & '+cell(mp,r,'p90_ape_pct',{'method':m})+r' \\')
    tab('upgrade_main_results.tex','固定19线基础预测上的主要控制（活动等权MAPE）','tab:main','L r r r','方法 & 验证 / \\% & 测试 / \\% & 案例P90 / \\%',body,
        '验证14期4361案例，测试16期5220案例。完整菜单、延迟通道及敏感性点见表~\\ref{tab:fullmenu}。各行均为预先固定控制；本表没有按测试误差重选主方法。')
    body=[]
    for r in metrics:
        m=r['method'];v=get(validation,method=m)
        body.append(r'\path{'+m+'} & '+cell(vp,v,'event_macro_mape_pct',{'method':m})+' & '+cell(mp,r,'event_macro_mape_pct',{'method':m})+' & '+cell(mp,r,'case_mape_pct',{'method':m})+' & '+cell(mp,r,'p90_ape_pct',{'method':m})+r' \\')
    tab('upgrade_full_menu.tex','完整54种固定规则及全部反馈策略','tab:fullmenu',r'p{6.8cm} r r r r','方法标识 & 验证MAPE & 测试MAPE & case-MAPE & P90',body,
        '全部误差单位为\\%。延迟/删减只作用于残差校正反馈，基础预测沿用原V3缓存。每方法测试均为16期5220唯一案例；总体评分未连接时间条件成员关系。',True)
    # Fixed issued-loss uncertainty; structural increments use their own comparator.
    body=[]
    comparisons=[('历史组合−独立参考','legacy_pair','ewma_reference'),('历史组合−仅差异','legacy_pair','pair_difference_only'),('DLM−独立参考','dlm_fullcov','ewma_reference')]
    for label,m,ref in comparisons:
        r=effect(m,ref);where=dict(method=m,reference=ref,weighting='activity_case_weighting')
        delta=cell(sp,r,'mean_difference_points',where,unit='percentage points')
        cis=[]
        for b in ['1','3','5','7']:
            a=r['bootstrap'][b]['ci95']
            lo=number(f'CI:{m}:{ref}:block{b}:low',a[0],sp,f'bootstrap.{b}.ci95[0]',where,3,'percentage points')
            hi=number(f'CI:{m}:{ref}:block{b}:high',a[1],sp,f'bootstrap.{b}.ci95[1]',where,3,'percentage points')
            cis.append('['+lo+', '+hi+']')
        body.append(label+' & '+delta+' & '+cis[0]+' & '+cis[2]+r' \\')
        # Export headline robust fields with exact provenance.
        if m=='legacy_pair' and ref=='ewma_reference':
            for nm,f,dig,mul,unit in [('PairLooLow','leave_one_out_min',3,1,'percentage points'),('PairLooHigh','leave_one_out_max',3,1,'percentage points'),('PairTopThree','top3_absolute_share',1,100,'%'),('PairImproved','improved_events',0,1,'activities')]:
                macros[nm]=cell(sp,r,f,where,dig,unit,nm,mul)
    tab('upgrade_robustness.tex','固定已发布损失的活动级配对不确定性','tab:robustness','L r C{3.1cm} C{3.1cm}','比较 & 差值 / 百分点 & 活动配对95\\%区间 & 5活动块95\\%区间',body,
        '5000次重采样，种子42。3和7活动块区间及逐活动损失保存在独立统计summary.json。所有区间为探索性，条件于固定方法和已发布损失，未重跑反馈历史、未校正搜索。')
    # Matched support correlations.
    cp=STAT/'matched_train_v3/matched_correlation_summary.csv';corr=rows(cp)
    panel_names={'log_final':'对数终分','remaining_growth_p50':'半程剩余对数增长','raw_error_phase1':'中期原始外推误差','independent_error_phase1':'中期独立校正误差'}
    body=[]
    for panel,label in panel_names.items():
        cells=[cell(cp,get(corr,panel=panel,base_family_pair=f),'median_spearman',dict(panel=panel,base_family_pair=f),3,'Spearman rho') for f in ['context_context','mixed','shared_shared']]
        body.append(label+' & '+' & '.join(cells)+r' \\')
    tab('upgrade_matched_correlations.tex','同一34期18线面板的线对相关中位数','tab:matchedcorr','L r r r','对象 & 两上下文线 & 混合路线线对 & 两共享路线线',body,
        '三列分别有1、32、120个线对。每个系数使用相同34期活动；线对共享活动，相关差区间按活动索引重采样。T1500无训练支持，活动261缺半程增长，均退出共同交集。')
    # Learning study: exactly the same validation panel, no test selector.
    lp=STAT/'learning_curve_v2/learning_metrics.csv';learn=rows(lp)
    if len(learn)!=56 or any(int(r['cases'])!=488 or int(r['events'])!=14 for r in learn):
        raise ValueError('Learning study must be 56 cells on 488/14 fixed panel')
    body=[]
    for n in [16,32,48,67]:
        cells=[]
        for fam,structure,seed in [('catboost','shared','42'),('catboost','independent','42'),('neural','shared','mean'),('neural','rank_heads','mean'),('neural','independent','mean')]:
            w=dict(training_events=n,family=fam,structure=structure,seed=seed)
            r=get(learn,**w)
            cells.append(cell(lp,r,'event_macro_mape_pct',w))
            if n==67:
                nm={'catboost_shared':'TreeSharedFull','catboost_independent':'TreeIndependentFull','neural_shared':'NnSharedFull','neural_rank_heads':'NnHeadsFull','neural_independent':'NnIndependentFull'}[fam+'_'+structure]
                macros[nm]=cell(lp,r,'event_macro_mape_pct',w,name=nm)
        body.append(str(n)+' & '+' & '.join(cells)+r' \\')
    tab('upgrade_learning_curve.tex','相同488验证案例上的训练规模诊断（MAPE / \\%）','tab:learning','r r r r r r','训练活动 & 树共享 & 树独立 & 网络共享 & 网络专属头 & 网络独立',body,
        '验证活动固定14期。网络三列为三种子最终分数平均后的评分；树固定种子42。当前局部特征和CatBoost200/神经基础拟合与19档回放的混合基础路线分别解释。')
    body=[]
    for r in learn:
        w={k:r[k] for k in ['training_events','family','structure','seed']}
        body.append(' & '.join([r['training_events'],r['family'],r['structure'].replace('_',r'\_'),r['seed'],cell(lp,r,'event_macro_mape_pct',w),cell(lp,r,'p90_ape_pct',w)])+r' \\')
    tab('upgrade_learning_seeds.tex','新拟合学习曲线全部56个评分单元','tab:learningseeds','r l l l r r','训练期数 & 家族 & 结构 & 种子/集成 & MAPE / \\% & P90 / \\%',body,
        'mean指三个种子的最终解码分数均值，区别于三个误差的平均。全部行共用14期488案例。',True)
    costs_path=RECORD/'runs/two_rank_learning_curve_v2/fit_costs.csv';costs=rows(costs_path)
    body=[]
    for n in [16,32,48,67]:
        for fam,structure in [('catboost','shared'),('catboost','independent'),('neural','shared'),('neural','rank_heads'),('neural','independent')]:
            subset=[r for r in costs if int(r['training_events'])==n and r['family']==fam and r['structure']==structure]
            par=sorted(set(float(r['parameters']) for r in subset if r['parameters']))
            params='—' if not par else '/'.join(str(int(v)) for v in par)
            cpu=sum(float(r['cpu_seconds']) for r in subset)
            wall=sum(float(r['wall_seconds']) for r in subset)
            where=dict(training_events=n,family=fam,structure=structure)
            BINDINGS.append(dict(binding='budget:'+str(where),source=str(costs_path.relative_to(ROOT)),source_sha256=sha(costs_path),row_condition=where,aggregation='sum(cpu_seconds),sum(wall_seconds);unique(parameters)',fits=len(subset),cpu_seconds=cpu,wall_seconds=wall,parameters=par))
            body.append(f'{n} & {fam} & '+structure.replace('_',r'\_')+f' & {len(subset)} & {params} & {cpu:.3f} & {wall:.3f}'+r' \\')
    tab('upgrade_fit_budget.tex','实际拟合数、参数量及成本','tab:fitbudget','r l l r r r r','训练期数 & 家族 & 结构 & 拟合数 & 每模型参数 & CPU秒 & 墙钟秒',body,
        'CatBoost各200轮，神经各60轮。网络独立每个种子拟合两个模型，参数列为单模型参数，合计预算随模型数变化。首轮初始化开销计入实际时间；这些小规模CPU成本具有环境依赖。',True)
    # Interval reliability: never hide unbounded fallback.
    ip=fp/'test_interval_reliability.csv';ir=rows(ip);body=[]
    for r in ir:
        where=dict(method=r['method'],nominal_coverage=r['nominal_coverage'])
        lev=cell(ip,r,'nominal_coverage',where,0,'%',multiplier=100)
        cov=cell(ip,r,'event_macro_coverage',where,2,'%',multiplier=100)
        avail=cell(ip,r,'available_rate',where,1,'%',multiplier=100)
        width=cell(ip,r,'supported_mean_relative_width_pct',where,2,'%')
        ep=cell(ip,r,'supported_mean_width_ep',where,1,'10^4 EP',multiplier=.0001)
        body.append(f'{lev} & {cov} & {avail} & {r["unsupported_cases"]} & {width} & {ep}'+r' \\')
        suffix={'80':'Eighty','90':'Ninety'}[lev]
        macros['Coverage'+suffix]=cov;macros['Width'+suffix]=width
    tab('upgrade_interval_reliability.tex','独立参考历史经验区间的测试可靠性','tab:intervals','r r r r r r','名义 / \\% & 活动覆盖 / \\% & 可用 / \\% & 无支持案例 & 相对宽度 / \\% & 宽度 / 万EP',body,
        '测试16期5220案例均有有限支持，无界回退为0。覆盖先在活动内取案例均值再按活动平均；宽度为有限支持案例均值，相对宽度按真实终分归一。名义水平属于预设顺序统计量参数，表中数值为回溯描述。')
    # Match interval scoring target to the max-score calibration unit.
    joint_root=STAT/'interval_joint_and_learning_contrasts_v1'
    jp=joint_root/'interval_joint_summary.csv';jr=rows(jp)
    cip=joint_root/'interval_coverage_intervals.csv';cir=rows(cip)
    body=[];suppbody=[]
    for lev,suffix in [('0.8','Eighty'),('0.9','Ninety')]:
        w=dict(split='test',method='ewma_reference',nominal_coverage=lev)
        r=get(jr,**w)
        pct=cell(jp,r,'nominal_coverage',w,0,'%',multiplier=100)
        group=cell(jp,r,'event_equal_group_joint_coverage',w,2,'%',name='Joint'+suffix,multiplier=100)
        macros['Joint'+suffix]=group
        c=get(cir,**w,target='group_joint_coverage',block=1)
        low=cell(cip,c,'low95',dict(w,target='group_joint_coverage',block=1),2,'%',multiplier=100)
        high=cell(cip,c,'high95',dict(w,target='group_joint_coverage',block=1),2,'%',multiplier=100)
        point=cell(jp,r,'event_equal_pointwise_coverage',w,2,'%',multiplier=100)
        width=macros['Width'+suffix]
        body.append(f'{pct} & {group} & [{low}, {high}] & {point} & {width}'+r' \\')
        entire=cell(jp,r,'entire_event_joint_coverage',w,2,'%',multiplier=100)
        suppbody.append(f'{pct} & {r["groups"]} & {r["events"]} & {group} & {point} & {entire}'+r' \\')
    tab('upgrade_interval_reliability.tex','区间构造目标及测试可靠性（全部单位为\\%）','tab:intervals','r r C{3.4cm} r r','名义 & 组内联合覆盖 & 活动重采样95\\%区间 & 逐案例覆盖 & 相对宽度',body,
        '组为活动×排名×阶段，878组/16期；一组内所有冻结前缀均落入各自区间才计覆盖。覆盖先活动内平均再活动等权。逐案例为更弱目标；相对宽度是有限支持案例按真实终分归一后的均值。5220案例均有有限支持，无界回退为0。区间为5000次活动配对重采样，未校正探索。')
    tab('upgrade_interval_targets.tex','三种覆盖目标的区别','tab:intervaltargets','r r r r r r','名义 / \\% & 组数 & 活动数 & 组内联合 / \\% & 逐案例 / \\% & 整活动联合 / \\%',suppbody,
        '校准目标是同活动、排名和阶段内的全部前缀。整活动联合要求该活动所有排名和阶段均覆盖，是更强的额外诊断；80\\%和90\\%各有0/16、1/16期整活动全部覆盖。各目标的块3/5/7区间另存完整CSV。')
    # All neural ensemble contrasts, on the same repeatedly reused validation events.
    lcp=joint_root/'learning_ensemble_contrasts.csv';lcr=rows(lcp);body=[]
    for n in [16,32,48,67]:
        for a,b,label in [('rank_heads','shared','专属头−共享'),('independent','shared','独立−共享'),('independent','rank_heads','独立−专属头')]:
            w=dict(method=f'neural_{a}_n{n}_mean',reference=f'neural_{b}_n{n}_mean',block=1)
            r=get(lcr,**w);d=cell(lcp,r,'mean_difference_points',w,3,'percentage points')
            lo=cell(lcp,r,'low95',w,3,'percentage points');hi=cell(lcp,r,'high95',w,3,'percentage points')
            c=get(lcr,method=w['method'],reference=w['reference'],block=5)
            cw=dict(w,block=5);clo=cell(lcp,c,'low95',cw,3,'percentage points');chi=cell(lcp,c,'high95',cw,3,'percentage points')
            body.append(f'{n} & {label} & {d} & [{lo}, {hi}] & [{clo}, {chi}]'+r' \\')
    tab('upgrade_learning_contrasts.tex','两线新拟合神经集成的活动配对比较','tab:learningcontrasts','r L r C{3cm} C{3cm}','训练活动 & 比较 & 差值 / 百分点 & 活动配对95\\%区间 & 5活动块95\\%区间',body,
        '均以同一14期验证的活动损失重采样，5000次、种子42。该验证已被多轮使用，区间未校正搜索。参数数量和独立模型数不同，预算见表~\\ref{tab:fitbudget}。')
    # Same-support correlation contrasts resample activities, not rank pairs.
    ccp=STAT/'matched_train_v3/matched_correlation_contrasts.csv';ccr=rows(ccp);body=[]
    for panel,label in [('remaining_growth_p50','剩余增长−终分'),('raw_error_phase1','原始误差−终分'),('independent_error_phase1','校正误差−终分')]:
        w=dict(panel=panel,reference='log_final',base_family_pair='all',block=1)
        r=get(ccr,**w);d=cell(ccp,r,'difference',w,3,'Spearman rho difference')
        lo=cell(ccp,r,'low95',w,3,'Spearman rho difference');hi=cell(ccp,r,'high95',w,3,'Spearman rho difference')
        body.append(f'{label} & {d} & [{lo}, {hi}]'+r' \\')
    tab('upgrade_correlation_contrasts.tex','共同34期153线对的相关中位数差','tab:corrcontrasts','L r C{4.3cm}','比较 & 中位相关差 & 活动配对95\\%区间',body,
        '重采样对所有排名使用同一活动索引，避免把153个线对当独立样本。每次先重估相关再求中位数差；5000次、种子42。块3/5/7及基础路线分组均保留在原CSV，解释为探索性。')
    # Frozen data identifiers and protocol counts.
    factp=PAPER/'data/facts.json';facts=js(factp);body=[]
    specs=[('V1两线','v1_two','不回灌'),('V2两线','v2_two','不回灌'),('V2 19线','v2_all','不回灌'),('V3两线','v3_two','已结束活动用于后续'),('V3 19线','v3_all','已结束活动用于后续')]
    # Dataset keys are preserved exactly as in the original facts manifest.
    keys=list(facts['datasets'])
    aliases={'v2_all':'v2_all19','v3_all':'v3_all19'}
    dbody=[]
    for label,key,feedback in specs:
        if key not in facts['datasets']: key=aliases.get(key,key)
        if key not in facts['datasets']:
            # Find only by the uniquely matching case tuple, never invent an ID.
            expected=(21984,4361,5220) if '19' in label else ((2390,130,154) if label.startswith('V1') else (2452,488,560))
            protocol=label[:2].lower()
            candidates=[(k,v) for k,v in facts['datasets'].items() if k.startswith(protocol) and tuple(v['cases'][s] for s in ['train','validation','test'])==expected]
            if len(candidates)!=1: raise KeyError((key,keys))
            key,d=candidates[0]
        else: d=facts['datasets'][key]
        vals=[str(d['cases'][s]) for s in ['train','validation','test']]
        body.append(label+' & '+' & '.join(vals)+' & '+feedback+r' \\')
        dbody.append(label+r' & \path{'+d['dataset_id']+r'} \\')
        BINDINGS.append(dict(binding='dataset:'+key,source=str(factp.relative_to(ROOT)),source_sha256=sha(factp),field='datasets.'+key,dataset_id=d['dataset_id'],cases=d['cases']))
    tab('upgrade_protocols.tex','冻结协议的案例数及历史信息条件','tab:protocols','L r r r L','协议范围 & 训练 & 验证 & 测试 & 终分反馈',body,
        '三组活动为67/14/16。V3训练列是父数据集前缀量；32期基础热身后35期形成逐期外推误差，二者角色分列。V3 19线为后续冻结扩展。')
    tab('upgrade_dataset_ids.tex','冻结数据完整SHA-256标识','tab:datasetids',r'p{2.2cm} p{12.8cm}','协议 & 数据指纹',dbody,long=True)
    # Historical static models remain descriptive and separate from replay.
    staticp=PAPER/'data/static_v2_models.csv';static=rows(staticp)
    body=[]
    for r in static:
        # Export source file rows as an evidence index; compact original columns only.
        label={
            'catboost_rank':'排名共享CatBoost','retrieval__retrieval_residual':'检索残差模型',
            'growth__growth_dual_clock':'双时钟增长网络','growth__growth_progress':'进度增长网络',
            'blend_new_25':'后续模型组合','foundation__foundation_tabpfn_v2_context':'TabPFNv2上下文增强',
            'foundation__foundation_tabpfn_v2':'TabPFNv2基础配方','growth__next_growth_progress_aux025':'多时距进度辅助监督',
            'foundation__foundation_chronos2':'Chronos-2（零样本）','foundation__foundation_timesfm25':'TimesFM2.5（零样本）',
            'dual_tcn_mean':'双线TCN三种子均值','phase_route':'阶段路由组合'
        }.get(r.get('method'),r.get('description',r.get('method','')).replace('_',r'\_'))
        fields=['train_mape_pct','validation_mape_pct','test_mape_pct']
        if all(k in r for k in fields):
            vals=[cell(staticp,r,k,{'method':r.get('method')}) if r[k] and not (k=='train_mape_pct' and r['method'] in ['foundation__foundation_chronos2','foundation__foundation_timesfm25']) else '—' for k in fields]
            body.append(label+' & '+' & '.join(vals)+r' \\')
    if body:
        tab('upgrade_historical_static.tex','既有V2两线静态实验的描述性汇总','tab:historicalstatic','L r r r','方法 & 训练 / \\% & 验证 / \\% & 测试 / \\%',body,
            '训练列依方法包含训练拟合或上下文重代入；零样本方法留空。输入、权重版本和预算随历史配方变化。该表与V3回放具有不同信息条件。')
    # Synthetic summaries aggregate sealed cell means; they never touch real outcomes.
    synp=RECORD/'synthetic/summary.csv';syn=rows(synp)
    pairp=RECORD/'synthetic/paired_comparisons.csv';synpairs=rows(pairp)
    auditp=RECORD/'synthetic/negative_control_audit.json';syn_audit=js(auditp)
    for name,digest in syn_audit['artifact_hashes_verified'].items():
        if sha(RECORD/'synthetic'/name)!=digest:raise ValueError('Synthetic artifact mismatch: '+name)
    body=[]
    snames={'uncorrected':'零校正','independent_ewma':'独立EWMA','fully_shared_ewma':'完全共享EWMA',
            'equal_filter_anchor_gap':'同滤波锚点负控','anchor_gap':'既有锚点差异','independent_kalman_fixed':'独立固定Kalman',
            'dlm_diagonal_R_fixed':'删除观测交叉项DLM','dlm_fullcov_fixed':'完整协方差固定DLM','dlm_oracle':'已知生成参数oracle'}
    for m,label in snames.items():
        current=[r for r in syn if r['method']==m and r['metric']=='state_mse']
        if len(current)!=64:raise ValueError('Synthetic grid coverage changed')
        avg=sum(float(r['mean']) for r in current)/64
        def counts(metric):
            c=[r for r in synpairs if r['method']==m and r['metric']==metric]
            lower=sum(float(r['mean']) < -1e-12 for r in c);higher=sum(float(r['mean']) > 1e-12 for r in c)
            return lower,higher
        sc,ac=counts('state_mse'),counts('synthetic_unbounded_ape_pct')
        if m=='independent_ewma':ds=da='参考'
        elif m=='equal_filter_anchor_gap':ds=da='64组数值等价'
        else:ds='/'.join(map(str,sc));da='/'.join(map(str,ac))
        body.append(f'{label} & {avg:.6f} & {ds} & {da}'+r' \\')
        BINDINGS.append(dict(binding='synthetic-summary:'+m,source=str(synp.relative_to(ROOT)),source_sha256=sha(synp),field='mean',row_condition=dict(method=m,metric='state_mse'),aggregation='equal mean of all 16 cells x 4 history lengths',value=avg,rendered=f'{avg:.6f}',unit='squared log state error',rounding='6 decimal places',direction_counts=dict(state=sc,ape=ac),paired_source=str(pairp.relative_to(ROOT)),paired_sha256=sha(pairp)))
    tab('upgrade_synthetic_summary.tex','预登记合成网格的完整方向计数','tab:syntheticcount','L r r r','固定方法 & 网格平均状态MSE & 状态风险下降/上升 & 模拟APE下降/上升',body,
        '16格与4历史长度等权，64组合的权重由设计给定。状态MSE使用统一偏差截断；方向相对独立EWMA，负控按10的负12次方容差判数值等价。oracle拥有真实生成参数，其角色为诊断参照。')
    # Real registration exists; do not replace waiting status with synthetic test fixtures.
    regp=RECORD/'prospective/registry_v2/registry/config.json';reg=js(regp)
    statusp=RECORD/'prospective/status_v2.json';status=js(statusp)
    from datetime import datetime, timezone
    registered=datetime.fromtimestamp(reg['registered_ms']/1000,timezone.utc).isoformat(timespec='milliseconds').replace('+00:00',' UTC')
    macros['ProspectiveRegistered']=registered.replace('T',' ',1)
    macros['ProspectiveTarget']=number('ProspectiveTarget',reg['plan']['target_events'],regp,'plan.target_events',{},0,'activities')
    macros['ProspectiveEventCount']=number('ProspectiveEventCount',status['events'],statusp,'events',{},0,'activities')
    BINDINGS.append(dict(binding='ProspectiveRegistered',source=str(regp.relative_to(ROOT)),source_sha256=sha(regp),field='registered_ms',value=reg['registered_ms'],rendered=macros['ProspectiveRegistered'],unit='UTC datetime from epoch ms',conversion='datetime.fromtimestamp(ms/1000,UTC)',status=status['status']))
    macro_text='% Generated from finalized independent evidence; run scripts/bind_upgrade_evidence.py.\n'
    for name,val in macros.items(): macro_text+='\\newcommand{\\'+name+'}{'+val+'}\n'
    (PAPER/'data/upgrade_macros.tex').write_text(macro_text)
    audit={'schema_version':1,'scope':'read-only evidence binding; no fitting, rescoring or method selection','sources':SOURCES,'bindings':BINDINGS,'generated_files':{str(p.relative_to(PAPER)):sha(p) for p in sorted(TABLES.glob('upgrade_*.tex'))},'macros_sha256':sha(PAPER/'data/upgrade_macros.tex')}
    (PAPER/'qa/upgrade_number_bindings.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(f'Bound {len(macros)} macros; {len(BINDINGS)} numerical entries; {len(audit["generated_files"])} native tables.')

if __name__=='__main__': main()
