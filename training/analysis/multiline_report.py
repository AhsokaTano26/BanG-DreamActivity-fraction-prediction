"""Bound19rank readout, separate two-rank/cold panels and trained artifacts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.rolling_benchmark import verify_seal,seal_directory,write_new_json
from analysis.multiline_dataset import MultilineDataset
from analysis.multiline_runner import RUN
from analysis.temporal_evaluate import score_temporal
from analysis.v3_coupled_report import relative_comparison

LABELS={'uncorrected':'基础预测无校正','ewma_reference':'各线独立暖EWMA','ewma_cold':'各线独立冷启动EWMA','legacy_pair':'原双线方法＋其余独立校正',
 'hierarchy':'全局／排名区间／单线分层校正','graph025':'相邻排名图校正λ0.25','graph1':'相邻排名图校正λ1',
 'factor2':'两因子动态校正','factor4':'四因子动态校正','share_gate':'在线专家切换','neural_gate':'多线门控网络（三种子）','prior_mix':'固定专家组合'}
RECORDS=ROOT/'analysis/multiline_records/20261117'

def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(x):return '—' if x is None or not np.isfinite(x) else f'{x:.3f}'

def training_relations(history,ranks,output):
    means=history.groupby(['event','phase','rank'],as_index=False).agg(error=('log_error','mean'));rows=[]
    def band(r):return int(np.searchsorted([50,500,5000],r,side='left'))
    for phase,p in means.groupby('phase'):
        matrix=p.pivot(index='event',columns='rank',values='error').reindex(columns=ranks)
        corr=matrix.corr(method='spearman');corr.to_csv(output/f'train_rank_correlations_phase{phase}.csv')
        within=[];between=[]
        for i,r in enumerate(ranks):
            for s in ranks[i+1:]:
                value=corr.loc[r,s]
                if np.isfinite(value):(within if band(r)==band(s) else between).append(value)
        active=matrix.notna().any();m=matrix.loc[:,active].to_numpy();center=np.nanmean(m,axis=0);scale=np.maximum(np.nanstd(m,axis=0),.02)
        z=np.where(np.isfinite(m),(m-center)/scale,0.);singular=np.linalg.svd(z,compute_uv=False);total=float((singular**2).sum())
        rows.append({'phase':int(phase),'events':matrix.shape[0],'observed_ranks':int(active.sum()),
          'within_band_median_spearman':float(np.median(within)) if within else None,'between_band_median_spearman':float(np.median(between)) if between else None,
          'two_factor_variance_fraction':float((singular[:2]**2).sum()/total) if total else None,
          'four_factor_variance_fraction':float((singular[:4]**2).sum()/total) if total else None})
    return {'scope':'35training prequential activities only, no heldout outcomes; associations not causal','relationships':rows}

def report(run=RUN,records=RECORDS,output=None):
    run,records=Path(run),Path(records);output=Path(output or records/'report')
    if output.exists():raise FileExistsError('Preserve previous readout')
    blind=verify_seal(run/'blind_manifest.json');evalseal=verify_seal(run/'evaluation_manifest.json')
    cfg=json.loads((run/'run_manifest.json').read_text());e=json.loads((run/'evaluation.json').read_text());audit=json.loads((records/'pre_scoring_audit.json').read_text())
    assert audit['status']=='pass' and audit['blind_manifest_sha256']==sha256(run/'blind_manifest.json')
    assert e['run_id']==cfg['run_id']==blind['run_id'] and e['evaluation_id']==evalseal['evaluation_id']
    assert evalseal['identity']['blind_manifest_sha256']==sha256(run/'blind_manifest.json')
    for n,h in cfg['source_hashes'].items():assert sha256(ROOT/n)==h,n
    selection=json.loads((run/'selection.json').read_text());primary=selection['selected_recipe'];reference='ewma_reference'
    assert sha256(run/'selection.json')==audit['selection_sha256'] and primary==blind['selected_recipe']
    data=MultilineDataset(cfg['dataset_directory']);rows=[]
    for method,score in e['splits']['test'].items():
        c=next(r for r in e['paired_comparisons'] if r['method']==method)
        rows.append({'method':method,'description':LABELS.get(method,method),'selected':method==primary,
            'validation_mape_pct':e['splits']['validation'][method]['overall']['event_macro_mape_pct'],
            'test_mape_pct':score['overall']['event_macro_mape_pct'],
            'original2_test_mape_pct':e['panels']['test']['original2'][method]['event_macro_mape_pct'],
            'T1500_test_mape_pct':e['panels']['test']['cold1500'][method]['event_macro_mape_pct'],
            'delta_points':c['method_minus_reference_mape_points'],'iid_low':c['paired_event_95_ci'][0],'iid_high':c['paired_event_95_ci'][1],
            'block5_low':c['circular_block_95_ci']['5'][0],'block5_high':c['circular_block_95_ci']['5'][1],**score['overall']})
    frame=pd.DataFrame(rows);main=frame[frame.selected].iloc[0];ref=frame[frame.method==reference].iloc[0]
    matched={'main_vs_reference_all19':relative_comparison(e,primary,reference),
        'neural_vs_fixed_prior_all19':relative_comparison(e,'neural_gate','prior_mix'),
        'main_vs_legacy_original2':relative_comparison({'paired_comparisons':e['panel_comparisons']['original2']},primary,'legacy_pair')}
    support=pd.read_csv(run/'rank_support.csv').set_index('rank');rankrows=[]
    for rank in cfg['ranks']:
        a=e['splits']['test'][primary]['by_rank'].get(str(rank),{});b=e['splits']['test'][reference]['by_rank'].get(str(rank),{})
        rankrows.append({'rank':rank,'train_curves':int(support.loc[rank,'train']),'validation_curves':int(support.loc[rank,'validation']),
           'test_curves':int(support.loc[rank,'test']),'reference_test_mape_pct':b.get('event_macro_mape_pct'),
           'selected_test_mape_pct':a.get('event_macro_mape_pct'),'test_cases':a.get('cases',0),'test_events':a.get('events',0),
           'difference_points':a.get('event_macro_mape_pct',0)-b.get('event_macro_mape_pct',0)})
    warm=pd.read_csv(run/'warm_predictions.csv');warm_ids=set(warm.case_id);cases=data.case_index('train').query('case_id in @warm_ids');members=data.memberships('train').query('case_id in @warm_ids')
    labels=data.labels('train');labels={c:labels[c] for c in warm_ids};train=[]
    for method,p in warm.groupby('method'):
        value=score_temporal(cases,members,labels,p,stage_catalog=data.manifest['protocol']['stages'])
        train.append({'method':method,**value['overall']})
    first_validation=min((r for r in data.event_index.values() if r['split']=='validation'),key=lambda r:(r['start_ms'],r['event']))
    train_history=pd.read_csv(run/'events'/str(first_validation['event'])/'released_residuals.csv')
    assert set(train_history.event)==set(cfg['warm_forecast_events']) and train_history.event.nunique()==35
    output.mkdir(parents=True);relations=training_relations(train_history,cfg['ranks'],output)
    conclusion=(f'事前按验证集选择 `{primary}`：19线验证MAPE {main.validation_mape_pct:.4f}%，测试 {main.test_mape_pct:.4f}%；同历史的各线独立EWMA测试 {ref.test_mape_pct:.4f}%，差值{main.delta_points:+.4f}个百分点。'
        f'配对活动95%区间[{main.iid_low:+.3f}, {main.iid_high:+.3f}]，5活动时间块区间[{main.block5_low:+.3f}, {main.block5_high:+.3f}]。')
    conclusion+=(' 本队列的两种区间均支持下降，仍需前瞻确认。' if main.delta_points<0 and main.iid_high<0 and main.block5_high<0 else ' 本轮不足以认定共享信息已经稳定优于独立校正；主结果不按测试成绩改选。')
    sections=['# 19条排名线的V3共享误差校正（2026-11-17）',conclusion,
      '**范围和限制：测试队列曾被探索使用、验证反复使用、元数据为回溯快照。** 原97活动67/14/16分组、2024-01-29密集采样下限及全部时间网格保持。19线整体与此前两线的6.54%总分属于不同范围，不直接比较；下表另列完全相同的原两线面板。',
      '## 全部15组成绩（活动等权MAPE%，越低越好）',
      table(['方法','验证*','测试19线','原两线测试','T1500测试','较独立参考/点'],[[LABELS.get(r.method,r.method)+(' **（事前主结果）**' if r.selected else ''),fmt(r.validation_mape_pct),fmt(r.test_mape_pct),fmt(r.original2_test_mape_pct),fmt(r.T1500_test_mape_pct),f'{r.delta_points:+.3f}'] for r in frame.itertuples()]),
      '*验证案例为4361，实际覆盖18条线；T1500没有训练或验证曲线，其行为没有参与参数选择。测试5220个案例、16个独立活动；不是5220个独立活动。各模型使用相同名单，没有删掉误差大的排名或时间点。',
      '## 各排名线覆盖及主模型成绩',
      table(['排名','训练组曲线','验证曲线','测试曲线','独立参考测试','主模型测试','差值/点'],[[f'T{r["rank"]}',r['train_curves'],r['validation_curves'],r['test_curves'],fmt(r['reference_test_mape_pct']),fmt(r['selected_test_mape_pct']),f'{r["difference_points"]:+.3f}'] for r in rankrows]),
      '## 两线对照与冷启动的解释',
      f'原两线面板仍为16活动/560案例，`legacy_pair`复现旧coupled_gap的{e["panels"]["test"]["original2"]["legacy_pair"]["event_macro_mape_pct"]:.4f}%。主模型在同面板为{main.original2_test_mape_pct:.4f}%；两者差值/区间保留在matched_comparisons.json。这里的基础Context400预测未更换，额外信息来自其他线此前已结束活动的误差。其余17条线使用逐期重训的共享排名模型600树，所有候选共享同一基线。',
      'T1500初始训练0、验证0、测试14曲线。首次预测依赖排名特征与跨线迁移；此前T1500活动结束后，V3允许后续预测学习这些真实结果。因此14期平均不是14期全部零样本预测。rolling_support.csv保留每期开始时的原始模型训练案例支持和已完成校正活动数。T10验证/测试各13曲线，缺线保持冻结，不补造案例。',
      '## 神经网络的匹配控制',
      f'门控网络相较相同4专家的固定组合，测试差值{matched["neural_vs_fixed_prior_all19"]["mean_difference_points"]:+.3f}点，配对95%区间[{matched["neural_vs_fixed_prior_all19"]["paired_95_ci"][0]:+.3f}, {matched["neural_vs_fixed_prior_all19"]["paired_95_ci"][1]:+.3f}]。应据此分辨专家结构的收益与网络学习权重的额外收益，不能只比较神经网络和未经校正的基线。全部区间未作多方法探索校正。',
      '## 共享的是什么',
      '- 分层方法依次追踪全局共同偏差、排名区间偏差、每条线自己的额外偏差；各活动仅贡献一次排名/阶段误差均值。\n- 图方法在相邻log排名之间平滑近期偏差变化，同时保留每条线的长期误差水平。因子方法从合法历史的活动×排名误差矩阵提取2/4个共同变化方向，缺线带掩码、未见排名作显式插值。\n- 在线专家与神经门控仅使用历史当时已经封存的专家分数、活动条件及预测前状态。当前活动的其他线晚到观测没有进入当前预测。所有预处理只拟合已结束历史。',
      '## 训练期关系与诊断',
      '以下仅分析35个训练期预序列活动的误差，供检查误差共享的依据，不作因果解释。阶段0低于25%，阶段1为25%至不足90%，阶段2为90%及以后。',
      table(['阶段','活动数','有历史排名数','区间内相关中位数','跨区间相关中位数','前2因子方差占比'],[[r['phase'],r['events'],r['observed_ranks'],fmt(r['within_band_median_spearman']),fmt(r['between_band_median_spearman']),fmt(r['two_factor_variance_fraction'])] for r in relations['relationships']]),
      f'暖训练期共35活动/{len(warm_ids)}前缀；训练期逐期外推分数见warm_prequential_metrics.csv。此前32活动只作基础模型热身，未将训练拟合误差当滚动精度。V3数据继承父V2的fit_policy/training_evaluation文字，新版本的feedback/history_boundary及运行配置明确32+35范围，按后者执行并留存审计。',
      '## 保存与复现',
      '363项全库测试通过，65个原始模型和65组神经网络逐期重放；旧数据、源码、生产权重和自动任务保留。研究计算单任务串行、CPU1、nice19和后台优先级。候选仅保存，不因测试结果自动替换生产模型。',
      '[全部成绩](comparison.csv) · [19线明细](rank_comparison.csv) · [匹配区间](matched_comparisons.json) · [模型权重及历史状态](candidate_artifacts.json) · [训练期外推](warm_prequential_metrics.csv) · [来源绑定](provenance.json)',
      '独立入口：analysis.multiline_dataset（可信准备）、analysis.multiline_base（基础盲预测）、analysis.multiline_runner（共享校正）、analysis.multiline_evaluate（独立评分）。目录不可覆盖；已完成同版本重复评分只返回封存结果。']
    (output/'README.md').write_text('\n\n'.join(sections)+'\n');frame.to_csv(output/'comparison.csv',index=False);pd.DataFrame(rankrows).to_csv(output/'rank_comparison.csv',index=False);pd.DataFrame(train).to_csv(output/'warm_prequential_metrics.csv',index=False)
    write_new_json(output/'matched_comparisons.json',matched);write_new_json(output/'training_relationships.json',relations);write_new_json(output/'pretest_selection.json',selection)
    last=max((run/'events').glob('*/event_manifest.json'),key=lambda p:json.loads(p.read_text())['query_start_ms']);event=verify_seal(last)
    raw=Path(cfg['base_run']);raw_cfg=json.loads((raw/'run_manifest.json').read_text());context=Path(raw_cfg['context_run']);context_cfg=json.loads((context/'run_manifest.json').read_text())
    source_models={'shared_rank600':raw/'events'/str(event['event'])/'shared.joblib','context1000_2000':Path(context_cfg['base_run'])/'events'/str(event['event'])/'rolling.joblib'}
    write_new_json(output/'candidate_artifacts.json',{'query_event':event['event'],'query_start_ms':event['query_start_ms'],'selected_method':primary,
       'event_manifest_sha256':sha256(last),'base_models':{name:{'path':str(p.relative_to(ROOT)),'sha256':sha256(p)} for name,p in source_models.items()},
       'calibration_weights_and_states':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(last.parent.rglob('*')) if p.is_file()},
       'deployment':'not promoted; models fit before last historical query, not latest live-data refit'})
    write_new_json(output/'provenance.json',{'run':str(run.relative_to(ROOT)),'run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],
       'blind_manifest_sha256':sha256(run/'blind_manifest.json'),'evaluation_manifest_sha256':sha256(run/'evaluation_manifest.json'),
       'pre_scoring_audit_sha256':sha256(records/'pre_scoring_audit.json'),'report_source_sha256':sha256(Path(__file__))})
    (output/'source.py').write_text(Path(__file__).read_text());seal_directory(output,'report_manifest.json',{'run_id':cfg['run_id'],'evaluation_id':e['evaluation_id']})
    print(frame[['method','selected','validation_mape_pct','test_mape_pct','original2_test_mape_pct','T1500_test_mape_pct']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--records',type=Path,default=RECORDS);p.add_argument('--output',type=Path);a=p.parse_args();report(a.run,a.records,a.output)
