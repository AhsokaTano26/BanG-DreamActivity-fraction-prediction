# 二轮图目录

输入路径/SHA和输出SHA以qa/upgrade_figure_manifest.json为准。

## upgrade_scene_framework — fig:upgrade-scene

单位：Mechanism only; equal prefix bars indicate a shared observation cutoff, not score magnitude.

建议图注：多排名终分预测的信息接口。排名叠层的等长横条表示相同实际观测截止，不表示分数大小；公开信息按国服发布时间过滤，历史账本只含期初合法释放的已结束活动。本期全时点共享同一期初校正状态；未知终分在预测阶段隔离。轻侧面仅表达对象层次，不编码数据。

## upgrade_information_timeline — fig:upgrade-timeline

单位：Activity count; chronology is schematic, not to physical time scale.

建议图注：固定97期活动的时间归属与预测反馈顺序。前67期分为32期基础热身和35期可用因果残差热身。当前活动全部预测共享期初历史；其终分在预测封存后释放，供后续严格满足时间守卫的活动使用。验证被重复使用，测试队列已有探索暴露。

## upgrade_activity_forecasts — fig:upgrade-curves

单位：Score in million EP; elapsed time in days.

建议图注：预定的两个真实活动示例：按实际开始时间排序的第1和第8测试活动，各展示T1000与T2000。当前分数和预测仅在冻结时点绘制，连接线辅助阅读；完整采样轨迹未在此复原。水平点划线为活动结束后独立评分所用终分。测试来自既有探索队列。

## upgrade_anchor_difference — fig:upgrade-anchor

单位：Dimensionless log-error bias; update rates and shrinkage constants dimensionless; support in activities.

建议图注：原锚点差异校正。T2000慢锚点使用该线全部合法历史，T1000快差异使用同活动同阶段的两线误差差；更新率为0.2/0.6，收缩常数为10/5。图中重组偏差在解码时扣除，T1000缺配对支持时保留本线独立回退。相同面板、线性滤波及收缩下满足底部恒等式。该图不代表另行评估的自适应凸组合。

## upgrade_sampling_diagnosis — fig:upgrade-density

单位：Observations per activity; observation interval in minutes; observations per day.

建议图注：历史归档的局部采样密度诊断。边界前186期306条合格训练曲线，边界后12期24条；图示活动内观测数中位数、逐曲线间隔中位数再取中位数、每天观测数。归档覆盖至2024年5月，用户指定的政策日期与技术采集切换时刻分别解释。数据从既有summary读取。

## upgrade_matched_correlations — fig:upgrade-correlation

单位：Spearman correlation, dimensionless; support in activities.

建议图注：同一34期18排名完整面板的Spearman相关。终分、半程剩余增长、中期原始外推误差和独立校正后误差共用相同活动支持；叉号标记负相关。T1500因训练缺失不进入共同面板。T1000/T2000与其余排名基础路线不同，相关结构仅作描述。

## upgrade_eventwise_improvement — fig:upgrade-eventwise

单位：MAPE difference in percentage points; cumulative absolute contribution in percent.

建议图注：双线校正相对独立滚动校正的逐活动MAPE差，以及按绝对差排序后的累计贡献。负差表示双线较低；全部16期均保留。集中程度使用绝对差，改善期数单独标出。活动序列按国服实际开始时间排序。

## upgrade_sensitivity — fig:upgrade-sensitivity

单位：Activity-macro MAPE in percent; rates/shrinkage dimensionless; minimum support in activities.

建议图注：完整预登记单因素敏感性。五个面板采用相同MAPE纵轴尺度；每次只改变一个参数，其他设置固定，竖虚线为原配方。点线连接实际运行值，重复使用验证与已探索测试分别标示。测试只作描述，没有按该图选择更新规则。

## upgrade_structural_ablations — fig:upgrade-ablation

单位：Activity-macro MAPE difference in percentage points; exploratory95% intervals.

建议图注：固定基础预测下的结构消融与传统滤波对照。相同滤波分解与同面板独立控制具有相同历史支持和更新设置；原锚点差异法具有不同更新率、收缩与支持。点为16期测试的相对MAPE差，粗蓝线与细灰线分别为活动配对和5活动循环块的95%探索性重采样区间。仅差异分量在该测试的点估计较低，完整法与其配对差区间跨零。完整54方法清单见补充结果。

## upgrade_training_size — fig:upgrade-learning

单位：Training activities; validation activity-macro MAPE in percent.

建议图注：真实基础模型训练规模实验。按实际开始时间取最早16、32、48、67期训练活动，在相同两线488个验证前缀上独立评分；树模型和神经网络使用相同本线输入、目标与有界解码。神经细虚线为42/123/2026逐种子，主线为三种子最终分数均值集成的独立成绩。共享输出、分线输出头及完全独立网络的参数量与CPU预算分别报告。该协议与19线混合基础模型分开解释。

## upgrade_rank_stage — fig:upgrade-rank-stage

单位：MAPE difference in percentage points; progress in percent; support in activities.

建议图注：排名与进度条件下的活动等权MAPE差和共同支持。负差表示锚点差异法较低；点号为零差，横线为缺失，T1000/T2000之外两方法输出相同。右侧逐格报告相同案例的支持活动数。时间分面使用网格成员关系，总体按物理案例去重。T1500冷启动与缺失条件完整保留。

## upgrade_neural_comparison — fig:upgrade-neural

单位：MAPE difference in percentage points; exploratory95% intervals.

建议图注：既有神经结构在相同5220测试案例上的匹配比较。点为相对独立滚动校正的活动等权MAPE差，粗蓝线与细灰线分别为活动配对和5活动循环块的探索性95%重采样区间。训练预算与输入范围见正文；机制解释覆盖这些具体实现，长期方法搜索的不确定性另列。

## upgrade_synthetic_sharing — fig:upgrade-synthetic

单位：Dimensionless log-state MSE difference multiplied by1000; variance units0.0001.

建议图注：合成机制实验的16格共享效果图，历史长度32。两个方法均相对独立EWMA比较已截断偏差状态MSE，负差表示状态风险较低。每格300条独立模拟序列，星号表示配对Monte Carlo均值差的95%区间位于零的一侧；其余历史长度、方法及逐种子结果完整保留。同滤波代数负控最大偏差差为机器精度量级。该图的生成机制与真实活动评分分别解释。

## upgrade_tail_interval — fig:upgrade-interval

单位：Relative point metrics, coverage and relative interval width in percent; support in activities.

建议图注：固定封存预测的点/尾部取舍与经验区间可靠性。（a）基础预测归一为100。（b）单前缀、排名×阶段全部前缀联合、整活动联合为不同目标；整活动仅0/16与1/16期覆盖。（c）完整组阶段条件覆盖，阶段按实际进度25%和90%划分。（d）本排名阶段来源与跨排名回退的活动等权相对宽度；回退组联合覆盖均100%，但仅涉及4/10期且较宽。点和组误差棒为活动配对探索性95%重采样区间，整活动为iid二项参考Clopper–Pearson区间，活动时间依赖另作敏感性分析。未来覆盖另待新活动验证。

## upgrade_prospective_loop — fig:upgrade-prospective

单位：Activity counts and strict actual-time inequalities; no prospective accuracy values.

建议图注：前瞻记录与反馈闭环。活动须实际开始晚于配置登记，预测按物理案例去重后准时封存；终分保留可信来源及实际到达时刻。历史只有在活动结束和终分实际到达均严格早于下一期开始时释放。原20期可行性登记保持，独立80期采集方案尚待激活；当前新队列0期，无未来精度数值。

## upgrade_adaptive_diagnostics — fig:round2-adaptive

单位：Activity-macro MAPE in percent; mixture weight dimensionless; support in task states or activities as labelled.

建议图注：训练期自适应共享诊断。35期按实际时间分为15期历史、10期选择、10期末段诊断；两线选择和诊断分别372/352唯一案例。全部12候选按预登记顺序呈现，选择块选中独立固定配方，末段未重选。λ面板按活动×排名×阶段去重为60任务状态；横线至最大λ，点为均值，右侧为正λ状态数。门控末段按活动×阶段0/30打开，与独立相同源于全部关闭。删一跨度是敏感性而非置信区间。新增结果属于训练期回溯时间外推诊断。

## upgrade_fairness_diagnostics — fig:round2-fairness

单位：MAPE difference in percentage points; compute/update ratios dimensionless; actual-start span in days.

建议图注：容量、预算与时间支持的训练期诊断。（a）同一截止和历史组内总参数精确匹配，格值为共享60轮减独立60轮MAPE。（b）三块合并12查询活动的配对与5活动循环块探索性95%区间，均跨零；源表的独立减共享方向在绘图时取负。（c）训练前向线性MAC、标量参数更新和optimizer.step为不同计数；共享30轮近似匹配MAC，但每参数更新数减半。（d）32完整与16同跨度共享首末训练日期，16近期具有较短跨度；组成和类别支持仍变化。三种子42/123/2026最终分数均值，各单元4查询活动。

## upgrade_calibrated_transfer — fig:round2-transfer

单位：Dimensionless trace ratio and NLL; log-error noise variance times1000; simulation relative risk changes in percent; real MAPE difference in percentage points.

建议图注：真实残差约束的机制诊断与固定迁移。（a）900点受限高斯状态模型网格，描绘trace(Q)/trace(R)下的剖面最小值及2ΔNLL≤10的17个描述性支持点；比值按小数12位等价键稳定分组，支持散点保留原值，阈值的置信覆盖尚待校准。（b）前25完整活动对中期的PSD矩估计噪声对角不相同，与剖面采用的等方差限制形成解释边界。（c）四方法各完整24条件，每条件300路径；双轴采用零附近线性、远端对数的对称尺度，左下表示两目标均改善，两种损失按各自口径解释。（d）使用前25期固定参数的一次事后DLM真实迁移，在后10期phase1的120案例上高于独立法；全部活动保留，图中平均差不用于再选择。

