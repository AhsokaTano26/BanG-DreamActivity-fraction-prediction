# E 二轮统计与区间可靠性审计设计

2026-11-18。本角色仅写 E 工作区，读取已封存预测和独立答案。模型、参数、候选排序和区间宽度均保持封存值。真实计算在 PI 许可后以 CPU1、nice19 串行执行；B/C 结果复核和区间/功效分为两阶段，每阶段预算120秒，重抽样5000次、种子42。

## 统计对象

主MAPE先平均活动内唯一物理案例，再活动等权。敏感性先在每活动每rank平均，再活动内rank等权、活动等权。案例MAPE、MAE、RMSE和P90另列；活动等权MAE、活动等权MSE开方和活动P90均值也另列。排名和phase分层保持原案例，不以支持筛除；逐期差和留一范围保留。

B对全部12个候选按原15/10/10期拆分，主两线及名义19线分别计算；全部候选减固定独立EWMA，保留同面板恒等与负结果。唯一选择仍为原seal选中的independent_fixed_recipe。C1固定18方法全部减independent，C2主表为三种子最终分数平均，单种子附表。C2每endpoint四期及合并12期报告，组别与结构固定对比；9个cell共享活动，活动为推断单位。数量、跨度及活动组成的残余混杂在解释中保留。

所有配对抽样共享活动索引，按实际开始时间排序。iid及循环块长1/3/5/7，每期保留已发轨迹。小于块长的样本标记insufficient_events。主差、rank等权差、活动MAE、MSE开方差和活动P90均值差给95%分位区间；尾部采用活动P90均值，另给pooled case-P90点值，避免混用两种尾部目标。区间描述固定损失的探索性重抽样变化，未重建反馈历史，也未校正候选搜索。

## 区间审计

读取第一轮test_intervals.csv的10440行、冻结test_cases和test_labels，仅作已探索测试描述。名义80%与90%分别报告。保留无界/不可用案例，显示有限率及own_support=0回退；无限宽度保留为inf，有限子集指标另列。

覆盖目标分开：案例覆盖；先期内案例均值再活动等权pointwise；先检查event×rank×phase内全部前缀同时覆盖，再期内组等权、活动等权；整个活动全部排名阶段前缀联合。整体联合另给Clopper–Pearson活动二项区间，补充全零bootstrap退化的信息。每行带n_events/n_cases/n_groups。

条件预登记：19个rank；phase实际字段；actual progress=[0,.25),[.25,.9),[.9,1]，由(observed_at-start)/(end-start)计算；event_type；own_support=0、1–9、10–29、30–59、60+；source。条件pointwise使用实际行字段。条件全组联合保持完整event×rank×phase；同组条件字段变化时归mixed，不把一个组拆成更容易覆盖的小组。actual progress跨phase边界时也照此处理。所有缺失/越界另列。

宽度为upper-lower，报告分数宽度、相对实际终分宽度、P90和活动级CI。标准Winkler interval score为width+(2/alpha)(lower-y)1[y<lower]+(2/alpha)(y-upper)1[y>upper]，alpha=1-nominal，欠覆盖与过覆盖两侧均正罚。分数尺度评分保留标准定义；除实际终分的相对分数仅作可比性补充。两个名义水平对应不同误覆盖成本，分别解释，未用评分挑宽度。

## 未来功效与固定流程

sigma和lag仅由TRAIN35 legacy_pair减independent的已发活动差估计，主19线、两线附表；效应网格固定0.25/0.5/1pp，双侧alpha=.05，power=.8/.9。独立正态活动差假设下用noncentral-t求最小n，rho=0/.3/.6采用AR(1)长期方差因子(1+rho)/(1-rho)作依赖敏感性，解释为近似。报告sigma的删一范围及独立正态chi-square95%区间，并把这些sigma分别带入功效表。既有20期仍为pilot。

confirmation_plan.json为新草案，不改registry_v2和三方法承诺；正式新队列需晚于新计划封存。独立EWMA保持主方法身份，唯一正式比较为原legacy_pair−EWMA；B新候选仅影子机制。最小有用差值事前设0.5pp，功效目标.9、规划rho=.3、sigma上限95%；优势检验零差，实用意义另看相对−.5pp的CI。未来人口清单、全部失败与缺失、准时提交、实际标签释放日志进入固定评分，未到齐的结果不填补；0活动时无MAPE。

参考：区间评分定义见[Gneiting & Raftery, 2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)；noncentral-t实现见[SciPy官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html)。本计划保留旧探索测试、验证重复使用和元数据回溯快照三项限制。
