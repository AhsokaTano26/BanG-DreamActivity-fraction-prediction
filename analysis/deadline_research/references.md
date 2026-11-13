# 参考材料与项目适用性

核查日期：2026-11-11。以下采用原论文、作者/发布方资料或官方文档。文献中的胜率和性能不是本项目的实验结果。

| 材料 | 与本项目的关系 | 当前状态与限制 |
|---|---|---|
| Jiao、Aue、Ombao，**Functional time series prediction under partial observation of the future curve**，[论文](https://arxiv.org/abs/1906.00281)，2019 预印本、2022 修订 | 历史完整曲线与当前已观测片段共同预测后续曲线，直接对应“活动前缀→终分”的任务形式。 | 本轮借鉴任务建模思路，测试了时间对齐的历史先验与残差；没有复现论文的完整 PFP/函数主成分方法。验证未超过现有方案。 |
| Kidger 等，**Neural Controlled Differential Equations for Irregular Time Series**，NeurIPS 2020，[论文](https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html) | 用观测路径驱动连续时间隐藏状态，适合不规则、多变量和部分缺失的序列。 | 候选方向，尚未训练。仍需对观测路径做插值，无法恢复未被观测的真实冲刺；不能宣传为“消除了插值误差”。 |
| Hollmann 等，**Accurate predictions on small data with a tabular foundation model**，Nature 2025，[论文](https://www.nature.com/articles/s41586-024-08328-6) | 在合成任务上预训练表格模型，再对具体小样本任务做上下文学习，提供了不同于从零训练网络的路线。 | 本项目只有 198 个独立训练活动，值得比较；原论文外部基准表现不能直接推算活动预测收益。 |
| Prior Labs，**TabPFN-3.5: Technical Report**，2026-09-15，[官方报告](https://priorlabs.ai/technical-reports/tabpfn-3-5) | 本次核查时官方已有 3.5 系列，并讨论分组、时间和复杂表格数据基准；不应把旧 v2 当作最新版本。 | 发布方技术报告，未在本项目运行。需先核实活动等权、上下文隔离、依赖和本地资源；不得把 4377 个相关前缀当成 4377 个独立活动。官方报告列有本地基础模型与 Fast 检查点，本轮未安装或下载。 |
| Grinsztajn、Oyallon、Varoquaux，**Why do tree-based models still outperform deep learning on tabular data?**，2022，[论文](https://arxiv.org/abs/2207.08815) | 比较树模型与神经网络的归纳偏好，讨论无信息特征、特征方向和不规则响应函数。 | 支持保留强树模型对照，不意味着新一代表格基础模型必然弱于树，也不能据此断言神经网络不适合本任务。 |
| scikit-learn，**StackingRegressor**，[官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html) | 二层模型应使用基模型的折外预测；官方明确指出同数据预拟合后再做堆叠会有较大过拟合风险。 | 本轮采用按活动时间推进的折外残差，非默认 KFold；已实测简单全局 Ridge 校正在时段变化时会失效。 |
| Stan User's Guide，**Hierarchical regression**，[官方文档](https://mc-stan.org/docs/stan-users-guide/regression.html#hierarchical-regression) | 分组参数在总体共享与每组独立之间进行部分汇合，适合样本很少的活动类型或乐队组。 | 后续方案依据；本轮没有拟合贝叶斯层级曲线模型。把少量乐队单独拟合为自由专家会不稳定，应先检验共享及收缩的程度。 |

## 文献带来的具体实现约束

1. **函数曲线先验必须因果构造。** 训练样本的历史库也必须剔除本期及它开始时尚未结束的活动。验证和测试期间不能逐期把答案加入历史。
2. **残差校正必须使用真实的折外误差。** 不能拿上一轮训练集约 2% 的样本内误差训练校正器，再声称它能修正未来活动约 8% 的误差。
3. **不规则采样模型要保留真实时间和可靠性。** 观测缺失与真实零增速不同；历史 shape 无法比较时，应回退为类型/时长先验，而不是认为完全匹配。
4. **层级建模与预训练都要保留活动为独立单位。** 多个快照、两个排名、三个随机种子不能增加独立活动数量。
5. **文献支持的是研究方向，不是本项目收益保证。** 本轮所有新增原型仅在训练内部和固定验证集比较；没有生成新测试成绩。
