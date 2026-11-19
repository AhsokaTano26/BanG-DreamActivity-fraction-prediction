"""Bound readout for the fixed shared/rank/peer experiment; no model selection."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.rolling_benchmark import verify_seal,seal_directory,write_new_json
from analysis.multiline_dataset import MultilineDataset
from analysis.task_learning_runner import RUN,CONTROLS,CANDIDATES
from analysis.task_learning_audit import RECORDS,read
from analysis.task_learning_models import MODES,SEEDS
from analysis.temporal_evaluate import aligned_frame,measured
from analysis.v3_coupled_report import relative_comparison

LABELS={'uncorrected':'基础预测','ewma_reference':'各线独立暖EWMA（本轮参考）',
    'legacy_pair':'历史双线校正＋其余独立校正','previous_neural':'上一轮多线门控网络',
    'previous_selected_graph':'上一轮验证选中的图校正','shared_single':'共享编码＋共享输出',
    'rank_heads':'共享编码＋每线专属头','related_static':'再加相关线信息',
    'related_phase':'再加阶段图与注意力门控'}

def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])

def fmt(x):return '—' if x is None or not np.isfinite(x) else f'{x:.4f}'

def report(run=RUN,records=RECORDS,output=None):
    run,records=Path(run),Path(records);output=Path(output or records/'report')
    if output.exists():raise FileExistsError('Preserve previous readout')
    blind=verify_seal(run/'blind_manifest.json');evalseal=verify_seal(run/'evaluation_manifest.json')
    cfg=json.loads((run/'run_manifest.json').read_text());e=json.loads((run/'evaluation.json').read_text())
    audit=json.loads((records/'pre_scoring_audit.json').read_text())
    tests=json.loads((records/'final_tests_receipt.json').read_text())
    assert audit['status']=='pass' and tests['exit_code']==0 and tests['status']=='pass'
    assert tests['log_sha256']==sha256(records/tests['log_file'])
    assert e['run_id']==cfg['run_id']==blind['run_id']==audit['run_id']
    assert e['evaluation_id']==evalseal['evaluation_id'] and evalseal['identity']['blind_manifest_sha256']==audit['blind_manifest_sha256']==sha256(run/'blind_manifest.json')
    for name,digest in cfg['source_hashes'].items():assert sha256(ROOT/name)==digest,name
    selection=json.loads((run/'selection.json').read_text());primary=selection['selected_recipe']
    assert primary==blind['selected_recipe'] and primary in CANDIDATES and sha256(run/'selection.json')==audit['selection_sha256']
    oldroot=Path(cfg['base_run']);verify_seal(oldroot/'evaluation_manifest.json')
    old=json.loads((oldroot/'evaluation.json').read_text())
    # Controls reproduce the same frozen panels, including the old historical best.
    for name,oldname in CONTROLS.items():
        for panel in ['all19','original2','cold1500','shared18']:
            for split in ['validation','test']:
                a=e['panels'][split][panel][name]['event_macro_mape_pct']
                b=old['panels'][split][panel][oldname]['event_macro_mape_pct']
                if a is None or b is None:assert a is None and b is None
                else:np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)
    rows=[]
    for method in cfg['methods']:
        c=next(x for x in e['paired_comparisons'] if x['method']==method)
        rows.append({'method':method,'description':LABELS.get(method,method),'selected':method==primary,
            'eligible':method in CANDIDATES,'validation_mape_pct':e['splits']['validation'][method]['overall']['event_macro_mape_pct'],
            'test_mape_pct':e['splits']['test'][method]['overall']['event_macro_mape_pct'],
            'original2_test_mape_pct':e['panels']['test']['original2'][method]['event_macro_mape_pct'],
            'T1500_test_mape_pct':e['panels']['test']['cold1500'][method]['event_macro_mape_pct'],
            'delta_vs_reference_points':c['method_minus_reference_mape_points'],
            'iid_low':c['paired_event_95_ci'][0],'iid_high':c['paired_event_95_ci'][1],
            'block5_low':c['circular_block_95_ci']['5'][0],'block5_high':c['circular_block_95_ci']['5'][1],
            **e['splits']['test'][method]['overall']})
    frame=pd.DataFrame(rows);main=frame[frame.selected].iloc[0];anchor=frame[frame.method=='ewma_reference'].iloc[0]
    matched={'selected_vs_anchor':relative_comparison(e,primary,'ewma_reference'),
        'shared_vs_anchor':relative_comparison(e,'shared_single','ewma_reference'),
        'private_heads_vs_shared':relative_comparison(e,'rank_heads','shared_single'),
        'static_peers_vs_private_heads':relative_comparison(e,'related_static','rank_heads'),
        'phase_attention_vs_static_peers':relative_comparison(e,'related_phase','related_static'),
        'selected_vs_previous_neural':relative_comparison(e,primary,'previous_neural'),
        'selected_vs_legacy_all19':relative_comparison(e,primary,'legacy_pair'),
        'selected_vs_legacy_original2':relative_comparison({'paired_comparisons':e['panel_comparisons']['original2']},primary,'legacy_pair')}
    data=MultilineDataset(cfg['dataset_directory']);support=read(run/'rank_support.csv').set_index('rank');rankrows=[]
    for rank in cfg['ranks']:
        a=e['splits']['test'][primary]['by_rank'][str(rank)]
        b=e['splits']['test']['ewma_reference']['by_rank'][str(rank)]
        rankrows.append({'rank':rank,'train_curves':int(support.loc[rank,'train']),
            'validation_curves':int(support.loc[rank,'validation']),'test_curves':int(support.loc[rank,'test']),
            'test_events':a['events'],'test_cases':a['cases'],'reference_mape_pct':b['event_macro_mape_pct'],
            'selected_mape_pct':a['event_macro_mape_pct'],'difference_points':a['event_macro_mape_pct']-b['event_macro_mape_pct'],
            **{m+'_mape_pct':e['splits']['test'][m]['by_rank'][str(rank)]['event_macro_mape_pct'] for m in MODES}})
    stagerows=[]
    for stage in data.manifest['protocol']['stages']:
        key=stage['stage_id'];v=e['splits']['test']['ewma_reference']['by_stage'][key]
        stagerows.append({'stage':key,'axis':stage['axis'],'cases':v['cases'],'events':v['events'],
            **{m+'_mape_pct':e['splits']['test'][m]['by_stage'][key]['event_macro_mape_pct'] for m in ['ewma_reference',*MODES]}})
    warm=read(run/'warm_predictions.csv');warm_ids=set(warm.case_id)
    cases=data.case_index('train').query('case_id in @warm_ids')
    labels=data.labels('train');labels={c:labels[c] for c in warm_ids};train=[]
    for method,p in warm.groupby('method'):
        train.append({'method':method,**measured(aligned_frame(cases,labels,p))})
    losses=[];checkpoints=[]
    for path in sorted((run/'events').glob('*/models/diagnostics.json')):
        info=json.loads(path.read_text());event=int(path.parents[1].name)
        for method,value in info.get('training',{}).items():losses.append({'event':event,'method':method,**value})
        checkpoints.extend(path.parent.glob('*/seed*.pt'))
    assert len(checkpoints)==audit['seed_checkpoints_replayed']
    latest=max((run/'events').glob('*/event_manifest.json'),key=lambda p:json.loads(p.read_text())['query_start_ms'])
    latestseal=verify_seal(latest);latestdir=latest.parent
    refcfg=json.loads((oldroot/'run_manifest.json').read_text());rawroot=Path(refcfg['base_run'])
    rawcfg=json.loads((rawroot/'run_manifest.json').read_text());contextroot=Path(rawcfg['context_run'])
    contextcfg=json.loads((contextroot/'run_manifest.json').read_text())
    base_models={'shared_rank600':rawroot/'events'/str(latestseal['event'])/'shared.joblib',
        'context1000_2000':Path(contextcfg['base_run'])/'events'/str(latestseal['event'])/'rolling.joblib'}
    originalbest=e['panels']['test']['original2']['legacy_pair']['event_macro_mape_pct']
    legacy19=e['panels']['test']['all19']['legacy_pair']['event_macro_mape_pct']
    c=matched['selected_vs_anchor'];b=c['circular_block_95_ci']['5']
    conclusion=(f'验证集事前选中 `{primary}`；验证MAPE {main.validation_mape_pct:.4f}%，19线测试 {main.test_mape_pct:.4f}%。'
        f'各线独立暖EWMA测试 {anchor.test_mape_pct:.4f}%，差值 {c["mean_difference_points"]:+.4f} 个百分点，'
        f'配对活动95%区间 [{c["paired_95_ci"][0]:+.4f}, {c["paired_95_ci"][1]:+.4f}]，5活动时间块区间 [{b[0]:+.4f}, {b[1]:+.4f}]。')
    if c['mean_difference_points']<0 and c['paired_95_ci'][1]<0 and b[1]<0:
        conclusion+=' 本队列支持误差下降；仍须新活动前瞻检验。'
    else:conclusion+=' 本轮尚不足以确认稳定改进，按测试分数重新选模型不属于本实验结论。'
    if primary=='ewma_reference':conclusion+=' 四种新结构在验证中均落后，按预登记规则未替换参考。'
    core=frame[~frame.method.str.contains('_seed')]
    sections=['# 共享编码、各线专属头与相关线学习：V3实验（2026-11-18）',conclusion,
        '**限制：测试曾暴露，验证反复使用，元数据为缺少历史时点存档的回溯快照；本轮设计前也分析过留出组的盲前缀关系。** '
        '所有新计算使用同97活动67/14/16分组和2024-01-29密集采样起点；验证4361案例（18条有标签线）、测试5220案例/16独立活动。不能将快照数视为独立活动数，区间未作多方法比较校正。',
        '## 主对照与四种新结构',
        table(['方法','可参与选择','验证MAPE%','19线测试%','原两线测试%','T1500测试%'],
            [[r.description+(' **（事前选择）**' if r.selected else ''),'是' if r.eligible else '描述性对照',
              fmt(r.validation_mape_pct),fmt(r.test_mape_pct),fmt(r.original2_test_mape_pct),fmt(r.T1500_test_mape_pct)] for r in core.itertuples()]),
        '只有独立参考和四个新网络三种子集成参与本轮验证选择。三个单独种子和旧方法均不参与选择；全部21组完整结果见comparison.csv，未删除表现较差结构。',
        '## 各层增加后是否有收益',
        table(['匹配比较（新减旧，负值较好）','差值/百分点','配对95%区间','5活动块95%区间','改善活动数'],
            [[name,f'{v["mean_difference_points"]:+.4f}',f'[{v["paired_95_ci"][0]:+.4f}, {v["paired_95_ci"][1]:+.4f}]',
              f'[{v["circular_block_95_ci"]["5"][0]:+.4f}, {v["circular_block_95_ci"]["5"][1]:+.4f}]',f'{v["improved_events"]}/{v["events"]}']
             for name,v in [('共享输出 − 独立参考',matched['shared_vs_anchor']),('各线头 − 共享输出',matched['private_heads_vs_shared']),('相关线 − 各线头',matched['static_peers_vs_private_heads']),
                ('阶段与注意力 − 静态相关线',matched['phase_attention_vs_static_peers'])]]),
        '最后一层同时加入阶段图与动态门控，是一个组合消融；不能单独归因于注意力。静态和动态相关线分支都读取当前活动的已观测前缀，因此与上一轮仅传播历史偏差的图校正不同。跨线高相关本身不保证模型有新增预测信息。',
        '## 与历史结果同口径比较',
        f'旧`legacy_pair`在本19线案例面板为 {legacy19:.4f}%，原T1000/T2000面板为 {originalbest:.4f}%；'
        f'本轮事前主模型在相同原两线为 {main.original2_test_mape_pct:.4f}%。历史约6.54%的两线成绩不能直接和19线汇总比较。'
        '五个旧控制的每条预测及各子面板成绩均已复现，原始Context400/共享600树基线没有更换。匹配区间见matched_comparisons.json。',
        '## 每条线的覆盖和主模型成绩',
        table(['排名','训练曲线','验证曲线','测试曲线','独立参考MAPE%','主模型MAPE%','差值/点'],
            [[f'T{r["rank"]}',r['train_curves'],r['validation_curves'],r['test_curves'],fmt(r['reference_mape_pct']),fmt(r['selected_mape_pct']),f'{r["difference_points"]:+.4f}'] for r in rankrows]),
        'T1500训练0、验证0、测试14活动曲线，未参与参数选择。首次预测只能使用共享部分和预登记的连续log排名先验，专属头严格关闭；前期T1500结果结束后允许后续V3查询使用，因此14期平均不是14次全程零样本实验。T10验证/测试各13曲线，缺失保持原冻结名单。逐期支持见rolling_support.csv。',
        '## 四种新结构的各线表现',
        table(['排名','独立参考','共享输出','各线头','静态相关线','阶段门控'],
            [[f'T{r["rank"]}',fmt(r['reference_mape_pct']),*[fmt(r[m+'_mape_pct']) for m in MODES]] for r in rankrows]),
        '各时间点×结构的相同案例比较见stage_comparison.csv；活动类型、乐队、排名×时点的完整明细保留在独立评分目录。排名与时点子组用于定位问题，不用于重新选择参数。',
        '## 结构、训练与信息边界',
        '共享编码器16→16，小共享头8→1；各排名仅增加一个线性专属头并按历史支持收缩。相关线图来自本期开始前已结束活动的参考预测误差，每对至少8个活动才使用相关系数，否则用固定log排名距离先验；最多3个邻居。阶段分界25%与90%。每个历史训练前缀保留其当时发布预测时的图和支持，未用后来结果改写。',
        '60轮AdamW、学习率0.002、权重衰减0.05，42/123/2026三个种子；最终分数取算术平均。训练损失是活动等权MAPE加小幅残差惩罚。只拟合历史本线节点的标准化与历史类别字典，其他线专用节点和当前节点都不能拟合预处理。历史少于12个已发布活动时沿用参考。网络在已校正的EWMA上学习剩余误差，不重复扣旧偏差。',
        '可信准备从原盲前缀构建31565个案例的缓存；每个同行输入裁剪到目标本线的实际观测时刻，迟于该时刻的信息不能进入模型，超过1小时的同行观测带缺失掩码。运行器核对固定缓存seal、7个原盲来源、排名和构建源码。模型本身只接收脱离数据集路径的当前及合法历史数组。',
        '## 训练期成绩与可复用权重',
        f'训练组后35个活动的 {len(warm_ids)} 个前缀为逐期外推，见warm_prequential_metrics.csv；最初32活动仅供基础模型热身。'
        '这些不是训练拟合误差。训练损失轨迹保留在training_losses.csv，仅供优化诊断，不能替代活动外推成绩。',
        f'全库 {tests["tests"]} 项测试通过；审计回放 {audit["seed_checkpoints_replayed"]} 个已训练种子权重，'
        f'核对 {audit["audited_events"]} 期所有预测、图、预处理、活动权重和 {audit["protected_old_files"]} 个既有文件。'
        '全部计算单任务串行、CPU1、nice19及后台优先级。',
        f'最后历史查询为活动 {latestseal["event"]}，12个种子权重、标准化、排名图、参考预测和基础模型路径均保存。'
        '这些是该历史活动开始前训练的研究模型，未用最新实况重新训练，未接入生产自动任务。生产模型与定时更新保持原状。',
        '回放审计发现原读取函数按JSON排序后的类别字典排列输入，与训练时活动类型→乐队的顺序不同。'
        '公开回放入口为analysis.task_learning_replay.replay，显式恢复训练布局；原训练源码、权重、预处理JSON和所有已发布分数均保留不变。'
        '新增不同类别数量的回归测试覆盖此问题，并对全部历史权重逐期回放验证。原task_learning_models.replay仅作为封存源码保留，不作交付读取入口。',
        '[全部21组成绩](comparison.csv) · [各线成绩](rank_comparison.csv) · [时间点成绩](stage_comparison.csv) · [匹配区间](matched_comparisons.json) · '
        '[模型文件清单](candidate_artifacts.json) · [训练期外推](warm_prequential_metrics.csv) · [来源](provenance.json)']
    output.mkdir(parents=True)
    (output/'README.md').write_text('\n\n'.join(sections)+'\n')
    frame.to_csv(output/'comparison.csv',index=False);pd.DataFrame(rankrows).to_csv(output/'rank_comparison.csv',index=False)
    pd.DataFrame(stagerows).to_csv(output/'stage_comparison.csv',index=False)
    pd.DataFrame(train).to_csv(output/'warm_prequential_metrics.csv',index=False);pd.DataFrame(losses).to_csv(output/'training_losses.csv',index=False)
    write_new_json(output/'matched_comparisons.json',matched);write_new_json(output/'pretest_selection.json',selection)
    write_new_json(output/'candidate_artifacts.json',{'query_event':latestseal['event'],'query_start_ms':latestseal['query_start_ms'],
        'selected_method':primary,'event_manifest_sha256':sha256(latest),
        'replay_reader':{'module':'analysis.task_learning_replay.replay','source_sha256':sha256(ROOT/'training/analysis/task_learning_replay.py'),
            'categorical_block_order':['event_type','band_group'],'preserves_original_artifacts':True},
        'base_models':{name:{'path':str(p.relative_to(ROOT)),'sha256':sha256(p)} for name,p in base_models.items()},
        'reference_feedback_state':{str((oldroot/'events'/str(latestseal['event'])/n).relative_to(ROOT)):sha256(oldroot/'events'/str(latestseal['event'])/n)
            for n in ['event_manifest.json','features.csv','released_residuals.csv','audit.json','predictions.csv']},
        'weights_preprocessing_and_state':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(latestdir.rglob('*')) if p.is_file()},
        'total_seed_checkpoints':len(checkpoints),'deployment':'Research only; before last historical query, not latest live refit or production promotion'})
    write_new_json(output/'provenance.json',{'run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],
        'blind_manifest_sha256':sha256(run/'blind_manifest.json'),'evaluation_manifest_sha256':sha256(run/'evaluation_manifest.json'),
        'preregistration_sha256':sha256(records/'preregistration.json'),'pre_scoring_audit_sha256':sha256(records/'pre_scoring_audit.json'),
        'full_tests_receipt_sha256':sha256(records/'final_tests_receipt.json'),'report_source_sha256':sha256(Path(__file__)),
        'replay_reader':'analysis.task_learning_replay.replay','replay_reader_sha256':sha256(ROOT/'training/analysis/task_learning_replay.py'),
        'caveats':[cfg['exposure'],cfg['metadata_caveat'],'16 independent test activities; exploratory intervals unadjusted','T1500 absent from initial train/validation']})
    (output/'source.py').write_text(Path(__file__).read_text())
    seal_directory(output,'report_manifest.json',{'run_id':cfg['run_id'],'evaluation_id':e['evaluation_id']})
    print(core[['method','selected','validation_mape_pct','test_mape_pct','original2_test_mape_pct','T1500_test_mape_pct']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--records',type=Path,default=RECORDS);p.add_argument('--output',type=Path)
    a=p.parse_args();report(a.run,a.records,a.output)
