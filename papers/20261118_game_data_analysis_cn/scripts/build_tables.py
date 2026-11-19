"""Bind manuscript tables and number macros to the reviewed analysis outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1];D=P/'data';OUT=P/'tables'


def escape(x):return str(x).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
def f3(x):return f'{float(x):.3f}'.replace('-','−')
def integer(x):return str(int(x))
def grouped(x):return f'{int(x):,}'.replace(',',r'\,')
def scientific(x):
    if x==0:return r'$<10^{-300}$'
    if x>=.001:return f'{x:.3f}'
    exp=int(np.floor(np.log10(x)));return rf'${x/10**exp:.2f}\times10^{{{exp}}}$'


def table(name,caption,columns,header,rows,note=''):
    content='\n'.join(' & '.join(row)+r' \\' for row in rows)
    text=rf'''\begin{{table}}[!htbp]
\centering\small
\caption{{{caption}}}\label{{tab:{name}}}
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabularx}}{{\linewidth}}{{{columns}}}
\toprule
{header} \\
\midrule
{content}
\bottomrule
\end{{tabularx}}
'''
    if note:text+=rf'''\par\vspace{{4pt}}
\begin{{minipage}}{{\linewidth}}\footnotesize\setstretch{{1.05}}
注：{note}
\end{{minipage}}
'''
    text+='\\end{table}\n';(OUT/f'{name}.tex').write_text(text)


def main():
    OUT.mkdir(exist_ok=True)
    r=json.loads((D/'results.json').read_text());coef=pd.read_csv(D/'association_coefficients.csv');bands=pd.read_csv(D/'band_summary.csv')
    profile=pd.read_csv(D/'matched_four_rank_growth.csv');year=pd.read_csv(D/'calendar_aligned_years.csv');ratio=pd.read_csv(D/'calendar_ratio_summary.csv')
    fact=pd.read_csv(D/'association_factor_tests.csv');core=pd.read_csv(D/'rank_summary.csv');coverage=pd.read_csv(D/'coverage_median_comparison.csv')
    macro={
        'NumEvents':integer(r['source_events']),'NumSourceCurves':grouped(r['source_curves']),
        'NumValidCurves':grouped(r['qualified_curves']),'NumPoints':grouped(r['points_raw']),
        'NumValidPoints':grouped(r['qualified_points']),'NumExcluded':integer(r['source_curves']-r['qualified_curves']),
        'NumCoreEvents':integer(r['correlations']['common18_events']),'NumCorrelationEvents':integer(r['correlations']['matched_growth_events']),
        'PcaFirst':f3(r['correlations']['rank_pca_first']*100),'PcaTwo':f3(r['correlations']['rank_pca_first_two']*100),
        'NumCards':grouped(r['cards_cn_released_by_last_start']),'NumFiveStar':integer(r['cards_five_star_by_last_start']),
        'NumCharacterMemberships':integer(r['character_event_memberships']),'NumTests':integer(r['factor_tests_global_family']),
        'NumNegativeAnchors':integer(r['negative_remaining_shares']),'NumAnchors':grouped(r['anchors']),
        'NumDecliningCurves':integer(r['score_decrease_curves']),'NumDeclines':integer(r['score_decrease_points']),
        'SourceFingerprint':r['source_row_fingerprint'][:12]}
    for panel,key in [('matched_final','RhoFinal'),('matched_growth_p50','RhoGrowth'),('partial_final_common18','RhoPartial')]:
        macro[key]=f3(next(x['median_pair_correlation'] for x in r['correlations']['summaries'] if x['panel']==panel))
    card=coef[(coef.model=='card_context')&(coef.term=='recent30_matched_cards')].iloc[0]
    for key,col in [('CardBeta','beta'),('CardLower','hac5_lower95'),('CardUpper','hac5_upper95')]:macro[key]=f3(card[col])
    for term,prefix in [('time_years','Time'),('log_duration','Duration'),('band_model=MyGO!!!!!','Mygo')]:
        row=coef[(coef.model=='pooled_final')&(coef.term==term)].iloc[0]
        for suffix,col in [('Beta','beta'),('Lower','hac5_lower95'),('Upper','hac5_upper95')]:macro[prefix+suffix]=f3(row[col])
    for axis,prefix in [('progress','Progress'),('remaining_h','Remaining')]:macro[prefix+'Events']=integer(profile[profile.axis==axis].events.iloc[0])
    for rowname,rank in [('Top',10),('Thousand',1000),('TwoThousand',2000),('ThirtyThousand',30000)]:
        h=profile[(profile.axis=='progress')&(profile.value==.5)&(profile['rank']==rank)].iloc[0]
        t=profile[(profile.axis=='remaining_h')&(profile.value==6)&(profile['rank']==rank)].iloc[0]
        macro['Half'+rowname]=f3(h.median_fraction*100);macro['LateSix'+rowname]=f3(t.median_remaining_share*100)
    for yr,suffix in [(2024,'First'),(2025,'Second'),(2026,'Third')]:
        row=ratio[(ratio.numerator==1000)&(ratio.denominator==2000)&(ratio.year==yr)].iloc[0];macro['RatioYear'+suffix]=f3(row.median_ratio)
    macro['MatchedRareEvents']=integer(coverage.matched_events.iloc[0])
    sensitivity=pd.read_csv(D/'measurement_sensitivity.csv')
    for value,key in [(6,'InterpolationSix'),(.25,'InterpolationQuarter')]:
        z=sensitivity[(sensitivity.axis=='remaining_h')&(sensitivity.value==value)].iloc[0]
        macro[key]=f3(z.median_step_excess_growth_pp)
    rows=[['数据采集区域','国服；服务器索引3'],['活动实际开始日期','2024年1月29日至2026年11月5日'],
        ['最后活动结束时间','2026年11月11日22:59:59（UTC+8）'],['已结束活动／排名线',rf'\NumEvents{{}}／19'],
        ['存储曲线／合格曲线',rf'\NumSourceCurves{{}}／\NumValidCurves{{}}'],['存储观测／合格曲线内观测',rf'\NumPoints{{}}／\NumValidPoints{{}}'],
        ['终点窗口未覆盖曲线',rf'\NumExcluded{{}}'],['活动类型／单乐队类别／混合乐队活动','6／8／28'],
        ['观测到的加成角色／角色与活动成员关系',rf'40／\NumCharacterMemberships{{}}'],
        ['18线共同终分／共同半程增长活动',rf'\NumCoreEvents{{}}／\NumCorrelationEvents{{}}']]
    table('cohort','数据快照与分析范围','l L','项目 & 数值或范围',rows,
        '快照于2026年11月18日从线上运行库只读提取。观测数为库中已归一化的榜线记录；每条曲线对应一个活动与一个排名。')
    rows=[]
    for row in core.itertuples():
        rows.append(['T'+str(row.rank),str(row.source_curves),str(row.valid_events),f3(row.median_final/10000),
            '['+f3(row.q25_final/10000)+'，'+f3(row.q75_final/10000)+']',f'{row.median_gap_min:.2f}'])
    table('ranks','19条排名线的终分覆盖、分布与采样间隔','l r r r L r',
        r'排名 & 源曲线 & 有效活动 & \shortstack{终分中位数\\/ 万EP} & 四分位范围 / 万EP & \shortstack{间隔中位数\\/ min}',rows,
        '每档使用其可用活动，跨档中位数的活动组成存在差异。典型间隔先在曲线内计算中位数，再在该档合格活动间取中位数。')
    rows=[]
    for yr in [2024,2025,2026]:
        a=year[(year.year==yr)&year['rank'].isin([1000,2000,30000])].set_index('rank');assert a.events.nunique()==1
        z=ratio[(ratio.numerator==1000)&(ratio.denominator==2000)&(ratio.year==yr)].iloc[0]
        rows.append([str(yr),integer(a.events.iloc[0]),*[f3(a.loc[k,'median_final']/10000) for k in [1000,2000,30000]],f3(z.median_ratio)])
    table('years','相同月份内的终分和排名比值','l r r r r L',r'年份 & 活动数 & T1000 & T2000 & T30000 & T1000/T2000',rows,
        '各年均使用实际开始日期位于2月至10月的活动。三个终分列单位为万EP；比值列先对同活动两线相除，再汇总中位数。')
    rows=[]
    for rank in [10,1000,2000,30000]:
        vals=[]
        for axis,value,col in [('progress',.5,'median_fraction'),('progress',.9,'median_fraction'),('remaining_h',6,'median_remaining_share'),('remaining_h',1,'median_remaining_share')]:
            a=profile[(profile.axis==axis)&(profile.value==value)&(profile['rank']==rank)].iloc[0];vals.append(f3(a[col]*100))
        rows.append(['T'+str(rank),*vals])
    table('growth','共同活动面板上的分数累积与后续增长',r'l *{4}{>{\centering\arraybackslash}X}',
        r'排名 & \shortstack{半程\\已观测 / \%} & \shortstack{90\%进度\\已观测 / \%} & \shortstack{结束前6h\\后续增长 / \%} & \shortstack{结束前1h\\后续增长 / \%}',rows,
        rf'进度面板为\ProgressEvents{{}}期，剩余时间面板为\RemainingEvents{{}}期；同一面板内四条线及所有时点共用活动。表中为活动中位数，后续增长相对终分归一化。')
    summaries={x['panel']:x for x in r['correlations']['summaries']};support=pd.read_csv(D/'final_pairwise_support.csv',index_col=0).to_numpy();upper=support[np.triu_indices(19,1)]
    rows=[['19线成对可用终分',f'{upper.min()}至{upper.max()}','19','171',f3(summaries['all_pairwise_final']['median_pair_correlation'])]]
    for key,label in [('common18_final','18线共同终分'),('partial_final_common18','条件调整后的终分'),('matched_final','半程匹配面板的终分'),('matched_growth_p50','同面板半程剩余增长')]:
        z=summaries[key];rows.append([label,str(z['events']),'18','153',f3(z['median_pair_correlation'])])
    table('correlations','终分与半程剩余增长的跨线关系','L r r r r',r'分析对象 & 活动数 & 排名线 & 线对 & 中位相关',rows,
        '末两行严格使用同一批活动和线对。条件调整秩相关控制时间、时长、类型与乐队，具体投影形式见正文。')
    main=coef[coef.model=='pooled_final'].set_index('term');rows=[]
    for row in bands.sort_values('events',ascending=False).itertuples():
        if row.band=='mixed':effect='参照组'
        else:
            z=main.loc['band_model='+row.band];effect=f3(z.beta)+' ['+f3(z.hac5_lower95)+'，'+f3(z.hac5_upper95)+']'
        rows.append([escape(row.band),str(row.events),str(row.t1000_events),f3(row.t1000_median/10000),effect])
    table('bands','乐队覆盖及终分的条件关联','L r r r C{4.4cm}',
        r'乐队类别 & 活动 & \shortstack{T1000\\有效活动} & \shortstack{T1000中位数\\/ 万EP} & 条件对数差 [95\%区间]',rows,
        '模型同时调整排名、时间、时长和活动类型，混合乐队为参照。区间为逐参数边际区间，采用按活动汇总得分向量的5期HAC估计。')
    names={'rank_label':'排名','kind':'活动类型','band_model':'乐队','time_years':'时间项','log_duration':'活动时长'};rows=[]
    for key in ['rank_label','kind','band_model','time_years','log_duration']:
        x=fact[(fact.model=='pooled_final')&(fact.factor==key)].iloc[0]
        rows.append([names[key],str(x.df),f3(x.partial_r2),scientific(x.p),scientific(x.q_by_global)])
    table('factors','主关联模型的删项诊断与整体检验','L r r L L',r'因素 & 自由度 & 偏$R^2$ & HAC检验$p$ & 全局BY调整$q$',rows,
        rf'偏$R^2$按删去该因素后的加权残差平方和定义。各因素的删项基准不同。BY调整覆盖五个固定模型中的全部\NumTests{{}}组因素检验。')
    (P/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in macro.items())+'\n')
    (P/'qa/manuscript_number_bindings.json').write_text(json.dumps(macro,ensure_ascii=False,indent=2)+'\n')
    print('Generated 7 native tables and',len(macro),'number bindings.')


if __name__=='__main__':main()
