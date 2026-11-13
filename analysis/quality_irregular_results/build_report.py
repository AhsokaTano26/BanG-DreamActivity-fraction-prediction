"""Render the report from audited development artifacts, never test answers."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text())
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    metrics=read('audit_v2/verified_metrics.json');older=read('audit_v1_reviewed/verified_metrics.json')
    names={'remaining_rank':'分排名剩余增速基线','additive_spline':'共享加性样条',
           'hierarchical_spline':'类型/排名分层样条','quality_spline':'质量感知分层样条',
           'quality_shared_tree':'质量感知共享树','quality_type_tree':'类型专家部分汇聚树',
           'time_gru_ensemble3':'不规则时间 GRU，三种子','neural_cde_ensemble3':'Neural CDE，三种子',
           'current_reference':'原有混合方案'}
    def score(split,name):
        value=metrics[split].get(name,{}).get('overall',{}).get('event_macro_mape_pct')
        return '—' if value is None else f'{value:.3f}%'
    comparison=table(['方法','训练拟合 MAPE','训练内部时间验证 MAPE','固定验证 MAPE'],
        [[label,score('train',name),score('oof',name),score('validation',name)] for name,label in names.items()])
    budget=pd.read_csv(ROOT/'budget_comparison.csv')
    budget_table=table(['模型','种子','80轮上限：最佳轮次 / 验证MAPE','240轮上限：最佳轮次 / 验证MAPE','实际停止轮次'],
        [[r.method,r.seed,f'{r.best_epoch_80} / {r.validation_mape_80:.3f}%',
          f'{r.best_epoch_240} / {r.validation_mape_240:.3f}%',r.epochs_240] for r in budget.itertuples()])
    folds=read('v2/folds.json')
    fold_table=table(['时间段（各25个活动）','分排名基线','分层样条','质量分层样条','质量共享树','类型专家树'],
        [[['2021-09—2022-03','2022-03—2022-09','2022-10—2023-08','2023-09—2024-05'][f['fold']]]+
         [f'{f["scores"][n]:.3f}%' for n in ['remaining_rank','hierarchical_spline','quality_spline','quality_shared_tree','quality_type_tree']] for f in folds])
    ablations=read('audit_v2/paired_ablations.json')
    ablation_table=table(['改动（配对基线）','内部时间验证变化，95%区间','固定验证变化，95%区间'],
        [[label]+[f'{ablations[split][name]["delta_mape_pp"]:+.3f} pp [{ablations[split][name]["ci95_pp"][0]:+.3f}, {ablations[split][name]["ci95_pp"][1]:+.3f}]'
                    for split in ['oof','validation']]
         for name,label in [('hierarchical_spline','加类型/排名偏差（共享样条）'),
                            ('quality_spline','加可靠性和历史先验（分层样条）'),
                            ('quality_type_tree','加类型专家（质量共享树）')]])
    blend_table=table(['新增成员（占25%，原组合占75%）','固定验证 MAPE'],
        [[names[name],score('validation','blend25_'+name)] for name in names if name!='current_reference'])
    text=f'''# 类型分层、采样质量与不规则时间神经网络：实验结果

**结论：本轮没有得到足以替换现有组合的新模型。** 原有组合的固定验证活动 MAPE 为 **3.802%**，仍是本轮候选及预设组合中的最低值。类型分层和历史质量信息在较早时段有价值，但跨时期不稳定；GRU、Neural CDE 延长训练后有所改善，仍未胜出。

本轮完成6个统计方案、2种神经结构×3种子，以及一次完整披露的训练预算敏感性检查。所有结果保留；没有请求测试输入/答案，没有产生新测试预测或测试评分，也没有改动生产默认方案。

## 评测口径

使用 `holdout_v1`，指纹 `d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb`。训练198活动/4377前缀；固定验证42活动/414前缀。统计模型内部时间验证为最早98个活动拟合，随后四段各25活动，共100活动/2190前缀；每折重拟合预处理和模型。各折历史池仅包含该折训练活动，不随评价块更新。

主指标为每个活动等权的 event-macro MAPE（%，越低越好）。所有模型使用同一批冻结案例；全局训练权重也按活动等权，跨两个排名计算，类型/排名专家不重新归一化。训练历史统计排除本期和本期开始时尚未结束的活动。

## 完整方案比较

{comparison}

神经网络列采用扩展预算下的三种子最终分数算术平均。训练拟合、内部时间验证、固定验证的活动和前缀时点不同，只能在同一列内比较，不能把列间差距全解释为过拟合。神经网络本轮未重跑四折时间验证；原组合也未重训，空白项不是零。

完整 case-MAPE、MAE、RMSE、P90 及排名/时点/类型/乐队分组见[独立核验后的指标](audit_v2/verified_metrics.json)，可直接处理的表格见[comparison.csv](audit_v2/comparison.csv)。

## 分层与质量信息：局部有效，跨时期失效

共享样条以39个普通特征加进度、log剩余小时的二次B样条为基线。分层版增加类型以及类型×排名的样条偏差；对新增偏差块施加更强的L2惩罚，分别为共享系数的4倍、16倍。这是惩罚回归形式的部分汇聚，不是完整贝叶斯后验估计。惩罚倍数只针对新增偏差块，普通特征仍保留。

质量版增加因果历史先验、形状偏差及可靠性 `n_eff/(n_eff+10)/(1+span_h/2)`。这是预先固定的启发式特征，不是经校准的置信概率，也不改变活动权重。没有有效历史形状时，形状偏差为零，仍保留可用的进度/结束时间先验。

类型专家树把专家与共享树预测按 `n/(n+20)` 汇聚，其中n是该类型不同训练活动数；未知类型退回共享树。两个树方案使用相同质量特征，可以单独判断增加类型专家的价值。与分排名基线相比，质量共享树同时改变了共享方式和特征，因此该比较不能单独归因于采样质量。

{ablation_table}

以上为5000次配对活动 bootstrap，负值表示改进；区间未作多重比较校正，属于开发集上的描述性证据。分层样条对共享样条的内部收益明确，但未迁移到固定验证；类型专家相对共享树在两个队列都变差。

{fold_table}

最后一段的退化部分抵消了前三段的收益。质量分层样条相对分排名基线的整体内部变化为 −0.347 pp，95%区间 [−0.798, +0.078]；不能确认整体稳定改进。分层或更细的类型专家并未解决跨年代变化。

## 神经网络：延长训练有效，但没有超越现有模型

两种网络接收实际观测时刻，每档最多保留32个真实节点，保留首末点；控制通道为进度、归一化分数、观测强度和实际经过时间。两个排名分别编码，再与相同39维原始表格特征（训练集拟合预处理）融合。缺失/过期的另一排名显式屏蔽。GRU读取节点及时间/分数增量；CDE学习由观测路径驱动的连续隐藏状态。

CDE仍使用节点间**线性插值**及Heun数值积分，并不意味着没有插值，更不能恢复未观测到的冲刺。密集前缀压缩为最多32个节点也会损失细节；本轮只检验这一具体原型。

首次方案预先固定80轮上限；六次拟合都触及上限，最佳轮次为72—80。随后依据已看到的验证轨迹，另登记一次240轮上限的收敛敏感性检查，其他结构、优化器、早停、输入和种子全部不变。**这项扩展是事后触发的补充，不属于最初预登记。** v1永久保留，v2另存；没有继续追加预算或搜参。

{budget_table}

三种子集成的固定验证结果：GRU由 **{older['validation']['time_gru_ensemble3']['overall']['event_macro_mape_pct']:.3f}% → {score('validation','time_gru_ensemble3')}**，CDE由 **{older['validation']['neural_cde_ensemble3']['overall']['event_macro_mape_pct']:.3f}% → {score('validation','neural_cde_ensemble3')}**。扩展训练均在114—195轮按原规则停止，没有再触及240轮上限；这说明80轮预算确实影响表现，但不证明已达到全局最优。

两个预算的所有前80轮训练损失/验证轨迹完全一致，统计模型在训练、内部验证和固定验证上的预测也完全一致，见[预算核验](budget_verification.json)。没有挑单个种子作为最终结果。

把CDE每段Heun积分从1步细化至4步，单模型预测平均相对变化约0.051%—0.135%，最大单例变化0.819%；各种子验证MAPE改变不超过0.016 pp。本次固定检查点的积分细化变化远小于与现有组合约2.8 pp的差距；它不是总数值误差的上界，也不能排除改用更细求解器重新训练后的不同表现。细化求解器的预测仅用于诊断，没有进入选择。详见 v2/models 下各种子 JSON 的 `solver_sensitivity`。

预先固定的25%新成员混合也全部未胜出：

{blend_table}

## 新诊断：采样密度可能被模型当成年代信号

训练前缀观测点数中位数为18，固定验证为263；中位采样间隔由2.775h变为约0.501h。固定验证414个案例中，102个案例的观测点数超过训练前缀最大值。另一排名的可用性也随数据年代显著变化。

对已经训练好的质量感知样条作精确线性分解：`quality_log_points` 对训练→验证的活动加权平均预测log剩余增速变化贡献约 **+0.322**，是最大的单项；排名分数比值特征约+0.096。数值来自标准化特征均值变化×该模型系数，见[分布统计](feature_distributions.csv)与[均值分解](spline_mean_shift_decomposition.csv)。

这是事后模型诊断，解释的是该模型的平均输出变化，不是对预测误差的因果分解。训练/验证还同时改变了活动类型、时点和时期，不能说“观测更密导致活动涨分”，也不能据此宣布删除某特征会提高准确率。

更值得下一步验证的是：**让同一已观测前缀在不同稀疏化版本下保持预测一致**，把采样密度更多用于表达不确定性，减少把测量方式当热度的机会。可固定同一标签与预测起点，增加原始/稀疏视图一致性正则，并保留活动等权；它比继续扩大网络更直接针对这次发现。然后用训练内部的跨时期最差折作稳定性检查。这个方向尚未实现，不能当作已取得的提升。

当前建议是保留原组合，把本轮分层/不规则时间模型作为已保存的研究对照。TabPFN等预训练表格模型本轮没有安装或训练，没有相关成绩可报告。

## 参考与实现区别

- [Kidger 等，Neural Controlled Differential Equations，NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html)：不规则采样多变量序列的连续控制状态建模。本文仅借鉴其模型形式，采用自己的小型双档网络和固定Heun求解器，非论文实验复现。
- [作者维护的 Diffrax Neural CDE 文档](https://docs.kidger.site/diffrax/examples/neural_cde/)：说明控制路径插值、矩阵向量场和有界输出。本轮用现有PyTorch直接实现，没有安装Diffrax或torchcde。
- [Stan 分层回归文档](https://mc-stan.org/docs/stan-users-guide/regression.html#hierarchical-regression)：分组信息共享与部分汇聚的依据。本轮以固定惩罚近似该思路，没有进行完整贝叶斯推断。
- [scikit-learn SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) 与 [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)：样条基函数与带样本权重的惩罚回归实现。论文或工具能力不等于本项目预测收益。

## 复现、审查与限制

- 训练源码、预处理、模型、预测、依赖、参数和数据指纹均已归档在 [v1](v1/run_manifest.json)、[v2](v2/run_manifest.json)；各有 `development_seal.json`，已封存目录拒绝覆盖。
- 69项单测通过；48个已安装包依赖兼容。独立审查未发现Critical/Important问题。审查指出的Python版本核验、实时库与冻结库身份区别已在独立audit中补充，保留训练源码指纹不变。
- [v1复核](audit_v1_reviewed/verification.json)、[v2复核](audit_v2/verification.json)重算全部分组指标，回放全部最终检查点，验证种子均值/混合配方/原基线/参考模型身份。全量日志和[审查记录](review.md)一起保存。
- 本轮没有新测试评分，不能称固定测试准确率提高。该测试队列曾参与早期探索，不是完全未见过的盲测；元数据为当前回溯快照，缺少历史时点存档。这两个限制继续保留。
- 固定验证集已经多次用于模型开发；本轮含一次明确披露的验证轨迹触发补充。验证优势也会存在选择偏差，未来通过新活动做前瞻检验更有说服力。
- Bootstrap以活动为单位，没有把前缀当独立样本；仍未建模相邻活动间的时间依赖，区间不代表对未来时期的保证。乐队按元数据中的加成角色构成分组；本轮未新增乐队神经嵌入或宣称乐队因果效应。

复核命令：

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib .venv/bin/python -m analysis.quality_irregular_benchmark --run analysis/quality_irregular_results/v2 --verify
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib .venv/bin/python -m analysis.quality_irregular_budget_audit
.venv/bin/python -m unittest discover -s tests -v
```

重新训练需选择新的目录，入口为 `analysis.quality_irregular_benchmark`（80轮）或 `analysis.quality_irregular_extension`（240轮上限），不会覆盖本轮结果。
'''
    (ROOT/'README.md').write_text(text)


if __name__=='__main__':main()
