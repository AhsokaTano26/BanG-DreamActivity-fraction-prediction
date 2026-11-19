"""Source-bound full readout of the frozen conditional V3 follow-up."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.rolling_benchmark import verify_seal,write_new_json,seal_directory
from analysis.rolling_dataset import RollingDataset
from analysis.temporal_evaluate import score_temporal
from analysis.v3_coupled_runner import RUN

LABELS={'uncorrected':'原Context无校正','ewma_reference':'原冷启动EWMA','ewma_warm':'暖EWMA','kernel_reference':'原核校正',
 'coupled_gap':'原两线差异校正','blend_gap25':'核校正＋25%两线校正','blend_gap50':'核校正＋50%两线校正',
 'blend_gap75':'核校正＋75%两线校正','gap_changepoint':'两线差异变点模型','gap_covariate5':'条件两线回归α5',
 'gap_covariate50':'条件两线回归α50','prior_expert_mix':'固定专家权重','fixed_share':'允许切换的在线专家组合','context_gate':'活动条件门控网络（三种子）'}

def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def relative_comparison(e,method,reference):
    a=next(r for r in e['paired_comparisons'] if r['method']==method)
    b=next(r for r in e['paired_comparisons'] if r['method']==reference)
    assert a['event_order']==b['event_order']
    delta=np.array(a['event_differences'])-np.array(b['event_differences']);n=len(delta)
    rng=np.random.default_rng(42);boot=delta[rng.integers(0,n,(5000,n))].mean(axis=1)
    blocks={}
    for k in [3,5]:
        rng=np.random.default_rng(42);starts=rng.integers(0,n,(5000,int(np.ceil(n/k))))
        idx=((starts[:,:,None]+np.arange(k))%n).reshape(5000,-1)[:,:n]
        blocks[str(k)]=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist()
    return {'method':method,'reference':reference,'events':n,'mean_difference_points':float(delta.mean()),
            'improved_events':int((delta<0).sum()),'paired_95_ci':np.quantile(boot,[.025,.975]).tolist(),
            'circular_block_95_ci':blocks,'event_order':a['event_order'],'event_differences':delta.tolist(),
            'multiple_comparisons_adjusted':False}

def report(run,records,output):
    run,records,output=Path(run),Path(records),Path(output)
    if output.exists():raise FileExistsError('Preserve earlier report')
    blind=verify_seal(run/'blind_manifest.json');scoreseal=verify_seal(run/'evaluation_manifest.json')
    cfg=json.loads((run/'run_manifest.json').read_text());e=json.loads((run/'evaluation.json').read_text())
    audit=json.loads((records/'pre_scoring_audit.json').read_text());selected=json.loads((run/'selection.json').read_text())
    assert audit['status']=='pass' and audit['blind_manifest_sha256']==sha256(run/'blind_manifest.json')
    assert audit['selection_sha256']==sha256(run/'selection.json') and scoreseal['identity']['blind_manifest_sha256']==audit['blind_manifest_sha256']
    assert e['run_id']==blind['run_id']==cfg['run_id'] and e['evaluation_id']==scoreseal['evaluation_id']
    for name,digest in cfg['source_hashes'].items():assert sha256(ROOT/name)==digest,name
    method=selected['selected_recipe'];assert method==blind['selected_recipe']
    rows=[]
    for key,test in e['splits']['test'].items():
        val=e['splits']['validation'][key];p=next(c for c in e['paired_comparisons'] if c['method']==key)
        rows.append({'method':key,'description':LABELS.get(key,key),'selected':key==method,
             'validation_mape_pct':val['overall']['event_macro_mape_pct'],'test_mape_pct':test['overall']['event_macro_mape_pct'],
             'T1000_mape_pct':test['by_rank']['1000']['event_macro_mape_pct'],'T2000_mape_pct':test['by_rank']['2000']['event_macro_mape_pct'],
             'delta_vs_cold_points':p['method_minus_reference_mape_points'],'iid_low':p['paired_event_95_ci'][0],'iid_high':p['paired_event_95_ci'][1],
             'block5_low':p['circular_block_95_ci']['5'][0],'block5_high':p['circular_block_95_ci']['5'][1],**test['overall']})
    frame=pd.DataFrame(rows);main=frame[frame.selected].iloc[0]
    matched=[relative_comparison(e,'context_gate','prior_expert_mix'),relative_comparison(e,'context_gate','ewma_warm'),
             relative_comparison(e,'gap_covariate5','coupled_gap'),relative_comparison(e,'gap_changepoint','coupled_gap')]
    source=Path(cfg['base_run']);old_receipt=verify_seal(source/'evaluation_manifest.json')
    old_evaluation=json.loads((source/'evaluation.json').read_text())
    assert old_evaluation['dataset_id']==cfg['dataset_id']
    nn=frame[frame.method=='context_gate'].iloc[0];prior=frame[frame.method=='prior_expert_mix'].iloc[0]
    cold=frame[frame.method=='ewma_reference'].iloc[0]
    old_nn=old_evaluation['splits']['test']['factorized_mlp']['overall']['event_macro_mape_pct']
    old_nn_val=old_evaluation['splits']['validation']['factorized_mlp']['overall']['event_macro_mape_pct']
    nn_matched=matched[0]
    data=RollingDataset(blind['dataset_directory']);warm=pd.read_csv(run/'warm_predictions.csv');ids=set(warm.case_id)
    cases=data.case_index('train').query('case_id in @ids');members=data.memberships('train').query('case_id in @ids')
    labels=data.labels('train');labels={c:labels[c] for c in ids};trainrows=[]
    for key,p in warm.groupby('method'):
        measured=score_temporal(cases,members,labels,p,stage_catalog=data.manifest['protocol']['stages'])
        trainrows.append({'method':key,**measured['overall']})
    train=pd.DataFrame(trainrows)
    relationships=json.loads((records/'training_relationships.json').read_text())
    sections=['# 两线误差的条件自适应实验（2026-11-17）',
      f'事前按验证选择 `{method}`，验证MAPE {main.validation_mape_pct:.4f}%，测试MAPE {main.test_mape_pct:.4f}%。原冷EWMA测试为{e["splits"]["test"]["ewma_reference"]["overall"]["event_macro_mape_pct"]:.4f}%。无论其他候选测试分数如何，均不改选主结果。',
      '**限制：同一测试队列此前已暴露、验证反复使用、元数据为缺少历史时点档案的回溯快照。所有结论均为探索性，不能称全新盲测。** 本轮仅在35训练活动上分析新关系；全部候选和参数固定后生成65期盲预测，验证选择在第一测试期前封存，随后完成全库测试、审计和独立评分。',
      '## 所有方法：活动等权MAPE%，越低越好',
      table(['方法','验证','测试','T1000','T2000','较原冷EWMA/点'],[[LABELS.get(r.method,r.method)+(' **（事前主结果）**' if r.selected else ''),
        f'{r.validation_mape_pct:.3f}',f'{r.test_mape_pct:.3f}',f'{r.T1000_mape_pct:.3f}',f'{r.T2000_mape_pct:.3f}',f'{r.delta_vs_cold_points:+.3f}'] for r in frame.itertuples()]),
      '## 本轮能得出什么',
      f'新门控网络验证{nn.validation_mape_pct:.3f}%、测试{nn.test_mape_pct:.3f}%；上一轮分解残差网络分别为{old_nn_val:.3f}%、{old_nn:.3f}%。两轮使用相同缓存、预序列残差、冻结案例和三种子最终分数平均，结构化专家池加活动条件门控因此构成一个更均衡的候选。相较原冷EWMA测试误差降低{(cold.test_mape_pct-nn.test_mape_pct)/cold.test_mape_pct*100:.2f}%，但这个候选不是本轮事前主结果。',
      f'固定专家先验组合已达到{prior.test_mape_pct:.3f}%。网络学习组合权重后仅额外减少{-nn_matched["mean_difference_points"]:.3f}点，匹配对照的配对95%区间[{nn_matched["paired_95_ci"][0]:+.3f}, {nn_matched["paired_95_ci"][1]:+.3f}]。不能把全部改善归因于神经网络，更不能据此认定网络学习部分已稳定有效。',
      '主结果依然由验证集最低分决定；不能因为其他方法在测试更好就换主模型。下一步应把候选在新活动上预先登记并做影子预测，比较真实终分到达后的误差，当前生产模型继续保持。',
      '## 与匹配控制比较',
      '下面的差值为方法减对照，负值表示改善；活动配对5000次bootstrap和3/5活动时间块均保留。区间未做多方法探索校正，不代替前瞻证据。',
      table(['方法','对照','差值/点','配对95%区间','5活动时间块区间'],[[LABELS[c['method']],LABELS[c['reference']],f"{c['mean_difference_points']:+.3f}",
          f"[{c['paired_95_ci'][0]:+.3f}, {c['paired_95_ci'][1]:+.3f}]",f"[{c['circular_block_95_ci']['5'][0]:+.3f}, {c['circular_block_95_ci']['5'][1]:+.3f}]"] for c in matched]),
      '## 训练期的关系与外推诊断',
      '只使用35个训练活动、1282个预序列案例检查关系。第一阶段两线误差差与观测分数比的Spearman相关约−0.50，中段约−0.39；中段与24h增速约−0.46。误差差的lag1在三个阶段接近零。这里的相关是描述性线索，不是因果结论，35个活动也不等于1282个独立样本。',
      table(['阶段','活动数','误差差lag1','与当前两线比相关','与24h增速相关'],[[r['phase'],r['events'],f"{r['gap_lag1']:.3f}",f"{r['gap_spearman_ratio']:.3f}",f"{r['gap_spearman_rate24']:.3f}"] for r in relationships['phase_relationships']]),
      '阶段0为进度低于25%，阶段1为25%至不足90%，阶段2为90%及以后。审计重建状态时会读取每次查询前已依法释放的历史误差（包含之前已结束的V3测试活动），未打开独立测试答案文件或使用本期结果。',
      '32个训练活动只作原基础模型热身；后35个活动先预测封存、再回灌结果。下表是这些训练时期活动的逐期外推误差，不是训练拟合误差，也不与16活动测试的不同队列总分直接比较。',
      table(['方法','训练期逐期外推MAPE%'],[[LABELS.get(r.method,r.method),f'{r.event_macro_mape_pct:.3f}'] for r in train.itertuples()]),
      '## 本轮机制',
      '- 基础模型及全部参考预测使用同一封存缓存，控制新旧模型的信息条件。所有新模型只收到当前盲特征、已结束活动的误差和那些活动当时已封存的专家预测。\n- 条件回归同时考虑活动类型、乐队、进度、观测两线比、增速与期初状态；稳健权重在每轮重新归一，保持每活动总权重为1。T2000维持原锚点；T1000从实际有界预测的隐含偏差向条件差异估计收缩。\n- 变点模型用Student-t预测分布跟踪历史误差差的运行长度；固定概率在观测前开始新段，新段包含该观测。它是明确该约定的实现，不声称复现其他软件的逐位结果。\n- 固定共享组合可在先前专家误差变化时重新分配权重；神经网络在同4专家上学习活动条件权重，使用相同的先验组合控制、3固定种子和最终分数平均。',
      '## 保存内容与运行范围',
      '同一冻结V3：14验证活动/488案例、16测试活动/560案例；所有17方法均使用相同名单和时间成员关系。按时点×排名、活动类型、乐队、观察滞后、MAE/RMSE/P90等结果在原运行评分文件中。',
      '继承的独立评分器evaluation.json内train_scope仍是旧67活动热身的通用文字。此轮实际是32初始基础热身＋35因果残差热身，以run_manifest、逐期审计和本报告为准，不改变任何计分数字。',
      '研究计算按用户要求单任务串行、单数值线程、nice19和后台优先级运行。生产代码、训练权重、活动模型固定指针和自动任务均保留。本轮没有部署测试期事后优胜者。',
      '[完整数字](comparison.csv) · [匹配对照](matched_comparisons.json) · [训练期外推](warm_prequential_metrics.csv) · [训练候选和状态](candidate_artifacts.json) · [来源绑定](provenance.json)',
      '## 方法参考',
      '[Adams与MacKay的在线变点检测](https://arxiv.org/abs/0710.3742)给出运行长度后验思路；[Herbster与Warmuth的专家跟踪](https://mwarmuth.bitbucket.io/pubs/J39.pdf)讨论随时间变化的专家组合。文献用于说明方法来源，不是本数据上的效果证据。']
    output.mkdir(parents=True);frame.to_csv(output/'comparison.csv',index=False);train.to_csv(output/'warm_prequential_metrics.csv',index=False)
    (output/'README.md').write_text('\n\n'.join(sections)+'\n')
    write_new_json(output/'matched_comparisons.json',matched)
    write_new_json(output/'pretest_selection.json',selected)
    write_new_json(output/'provenance.json',{'run':str(run.relative_to(ROOT)),'run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],
        'blind_manifest_sha256':sha256(run/'blind_manifest.json'),'evaluation_manifest_sha256':sha256(run/'evaluation_manifest.json'),
        'previous_neural_comparison':{'run':str(source.relative_to(ROOT)),'method':'factorized_mlp','evaluation_id':old_receipt['evaluation_id'],'evaluation_manifest_sha256':sha256(source/'evaluation_manifest.json')},
        'pre_scoring_audit_sha256':sha256(records/'pre_scoring_audit.json'),'report_source_sha256':sha256(Path(__file__))})
    last=max((run/'events').glob('*/event_manifest.json'),key=lambda p:json.loads(p.read_text())['query_start_ms'])
    event=verify_seal(last);source=Path(cfg['base_run']);source_cfg=json.loads((source/'run_manifest.json').read_text())
    base=Path(source_cfg['base_run'])/'events'/str(event['event'])/'rolling.joblib'
    write_new_json(output/'candidate_artifacts.json',{'query_event':event['event'],'query_start_ms':event['query_start_ms'],
        'selected_method':method,'event_manifest_sha256':sha256(last),'source_reference_event_manifest_sha256':sha256(source/'events'/str(event['event'])/'event_manifest.json'),
        'base_model':{'path':str(base.relative_to(ROOT)),'sha256':sha256(base)},
        'trained_models_and_state':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(last.parent.rglob('*')) if p.is_file()},
        'deployment':'not promoted; these are historical last-query models, not a latest-data production refit'})
    (output/'source.py').write_text(Path(__file__).read_text())
    seal_directory(output,'report_manifest.json',{'run_id':cfg['run_id'],'evaluation_id':e['evaluation_id']})
    print(frame[['method','selected','validation_mape_pct','test_mape_pct']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--records',type=Path,default=ROOT/'analysis/v3_coupled_records/20261117');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.run,a.records,a.output)
