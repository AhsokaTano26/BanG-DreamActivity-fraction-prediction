# 双档结构化神经网络实验

固定 `holdout_v1` 上完成 9 个配置 × 3 个种子，共 27 次训练；每个方法评价相同的 412 个测试案例、43 个活动。

验证集选定 **`blend_neural_0.25`**：验证活动 MAPE **3.802%**，测试 **8.068%**。对照是上一轮验证选定的组合，测试 **8.140%**。

神经网络中验证最优的三种子集成为 `dual_tcn_metric_ensemble3`：验证 4.440%，测试 8.944%。单个种子只作透明报告，不进入最终方案选择。

## 设计与预先固定的比较

网络学习后续平均增速相对已有平均增速的对数倍率，终分 = 当前分数 × [1 + exp(倍率对数) × (1−进度)/进度]。倍率对数裁剪到 [-6,4]；预测不低于当前分数，剩余时长趋零时回到当前分数。

- 共享 MLP 与分排名头 MLP：检验排名专属输出头的作用。
- 双档 GRU、因果 TCN、全程加最近 48 小时 TCN：共享编码器，融合本档、另一档、差和乘积。
- 跨档 patch attention：每 8 个时间位置作为 patch，联合两档 token，并用排名 query 汇总。
- 门控 TCN：三个软门控专家，各输出两档预测头。
- TCN 目标损失对照：0.8 平滑相对终分误差 + 0.2 Huber 对数倍率误差。
- TCN 稀疏化对照：训练时随机将一半前缀观测减少至约四分之一，保留首末观测、当前分数和预测时刻。

每档输入为 64 个共享时间位置 × 8 通道：相对累计分数、相对增速、可用标记、观测年龄、插值跨度、活动进度和昼夜正余弦。另一档只取当前已观测时刻以内数据；缺失或超时用 mask 表达。数值汇总包含 39 个紧凑原始特征，去掉绝对分数和乐队特征，以检验前一轮推荐结构；活动类型保留。

固定宽度 48、dropout 0.2、AdamW 学习率 0.001/权重衰减 0.005、batch 256、上限 100 轮、至少 30 轮/耐心 20、梯度裁剪 1。普通损失为 beta=0.2 的 Huber。所有样本使用全局活动等权，同一活动两档合计权重不变；预处理只拟合训练数据。每个配置固定种子 42、123、2026，先逐种子选验证最佳检查点，再对最终分数取均值。

验证集决定三个神经集成成员及与旧方案 25%/50%/75% 的混合，旧方案本身也列为候选。测试预测全部封存后，原独立评分器才读取答案；没有根据测试结果换模型或重训。

## 三种子集成与组合结果

| 方法 | 验证活动 MAPE % | 测试活动 MAPE % |
|---|---:|---:|
| `blend_neural_0.25` | 3.802 | 8.068 |
| `blend_neural_0.5` | 3.871 | 8.148 |
| `reference` | 3.885 | 8.140 |
| `blend_neural_0.75` | 4.062 | 8.343 |
| `neural_top3` | 4.374 | 8.615 |
| `dual_tcn_metric_ensemble3` | 4.440 | 8.944 |
| `dual_gru_ensemble3` | 4.491 | 8.522 |
| `dual_tcn_thin_ensemble3` | 4.536 | 8.670 |
| `dual_tcn_ensemble3` | 4.577 | 8.783 |
| `multiscale_tcn_ensemble3` | 4.620 | 8.873 |
| `rank_mlp_ensemble3` | 4.680 | 8.759 |
| `cross_patch_ensemble3` | 4.877 | 8.724 |
| `gated_tcn_ensemble3` | 5.004 | 8.845 |
| `shared_mlp_ensemble3` | 5.644 | 8.975 |

验证选出的三个神经成员：`dual_tcn_metric_ensemble3`, `dual_gru_ensemble3`, `dual_tcn_thin_ensemble3`。

## 逐种子结果

| 配置 | 种子 | 最佳轮数 | 验证 MAPE % | 测试 MAPE % |
|---|---:|---:|---:|---:|
| `shared_mlp` | 42 | 83 | 5.062 | 8.770 |
| `shared_mlp` | 123 | 45 | 6.356 | 9.208 |
| `shared_mlp` | 2026 | 89 | 6.563 | 9.626 |
| `rank_mlp` | 42 | 59 | 4.984 | 8.918 |
| `rank_mlp` | 123 | 77 | 4.758 | 8.990 |
| `rank_mlp` | 2026 | 84 | 4.889 | 8.782 |
| `dual_gru` | 42 | 49 | 4.554 | 8.627 |
| `dual_gru` | 123 | 64 | 4.623 | 8.348 |
| `dual_gru` | 2026 | 78 | 4.749 | 8.946 |
| `dual_tcn` | 42 | 40 | 4.683 | 8.964 |
| `dual_tcn` | 123 | 66 | 4.959 | 9.103 |
| `dual_tcn` | 2026 | 96 | 4.718 | 8.885 |
| `multiscale_tcn` | 42 | 57 | 4.778 | 8.963 |
| `multiscale_tcn` | 123 | 64 | 4.704 | 9.011 |
| `multiscale_tcn` | 2026 | 48 | 4.713 | 8.939 |
| `cross_patch` | 42 | 58 | 4.955 | 8.913 |
| `cross_patch` | 123 | 94 | 5.098 | 8.754 |
| `cross_patch` | 2026 | 89 | 5.145 | 8.874 |
| `gated_tcn` | 42 | 45 | 4.793 | 8.767 |
| `gated_tcn` | 123 | 65 | 5.381 | 9.100 |
| `gated_tcn` | 2026 | 80 | 5.643 | 9.301 |
| `dual_tcn_metric` | 42 | 59 | 4.524 | 8.971 |
| `dual_tcn_metric` | 123 | 65 | 4.636 | 9.140 |
| `dual_tcn_metric` | 2026 | 78 | 4.699 | 8.960 |
| `dual_tcn_thin` | 42 | 50 | 4.814 | 8.706 |
| `dual_tcn_thin` | 123 | 64 | 4.746 | 8.674 |
| `dual_tcn_thin` | 2026 | 85 | 4.670 | 9.084 |

## 分排名与预测时点

下表同时给出旧方案、事先选定方案和验证最优神经网络 `dual_tcn_metric_ensemble3`。

| 分组 | 旧方案 MAPE % | 选定方案 MAPE % | 验证最优神经 MAPE % |
|---|---:|---:|---:|
| 1000 | 12.544 | 12.322 | 12.799 |
| 2000 | 3.737 | 3.813 | 5.089 |
| 24h remaining | 6.107 | 5.753 | 6.137 |
| 25% | 12.771 | 13.038 | 15.156 |
| 50% | 9.882 | 9.897 | 11.168 |
| 6h remaining | 3.861 | 3.380 | 3.281 |
| 75% | 7.646 | 7.773 | 8.370 |

选定方案相对旧方案的活动 MAPE 差值为 -0.073 个百分点，配对活动 bootstrap 95% 区间 [-0.201, 0.054]；25/43 个活动改善。若选定方案就是 reference，差值恒为零，不构成新的改善证据。

case-MAPE、MAE、RMSE、P90、活动类型与乐队明细见 [evaluation.json](evaluation.json)；全部方法汇总见 [all_method_comparison.csv](all_method_comparison.csv)。乐队分组按加成角色构成，不等同于活动故事主角。

## 本轮决策

保留验证事先选定的 `blend_neural_0.25` 作为本轮版本。它在本队列的测试差值为 -0.073 个百分点；区间和分组结果一起判断，不能只看单个种子的最佳值。

后续应优先做训练内部按活动、按时间的验证，检查神经网络能否学习已有结构模型的剩余误差，并保留共享编码与排名专属输出头的对照。对新活动建立前瞻记录，核对 1000 线和早期阶段。增加网络层数本身并不构成预期收益；本轮实验也不能把某种架构的一次胜负推广为普遍规律。

## 审计与限制

- 数据指纹：`d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb`；训练/验证/测试仍为 198/42/43 个活动，4377/414/412 个前缀。
- 运行身份：`744df452c51e14d366213b274766ea1f2355d3da2b149c27f09b6002bc82c569`；评分身份：`343caf39798d61fb35433b56b37691a8168bcf8320cf94c5ba6c969e66cf95f9`。
- 独立逐活动重算 41 组主指标，最大差异 1.8e-15；数据、开发、盲预测及评分封存均通过校验。
- [paired_comparison_and_verification.json](paired_comparison_and_verification.json) 保存每个方法相对 reference 的配对活动区间。区间为队列内探索比较，没有多重模型比较校正，也没有建模活动间长期时间相关；不能解释成未来预测区间。
- 原数据库未用于模型输入；数据读取使用 HoldoutDataset 校验固定文件。模型、预处理、源代码、依赖版本和预测已归档。
- **测试队列曾被早期探索使用，不能称为完全未见过的盲测；元数据是回溯快照，缺少历史时点存档。** 训练活动只有 198 个，重复前缀不是独立活动；采样密度和跨档可用性存在年代变化。
- 验证集已在多轮研究中复用，存在选择乐观；三种子不能增加独立活动数。稀疏网格仍使用插值近似，倍率裁剪不能精确表达零后续增长。

方法借鉴 [PatchTST 的分块思路](https://arxiv.org/abs/2211.14730) 与 [GRU-D 对缺失和观测间隔的处理动机](https://arxiv.org/abs/1606.01865)。这里的跨档 attention、GRU+间隔通道是任务适配实现，未复现这两篇论文的完整架构。

```bash
# 新实验另存目录，保留本轮封存结果
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib .venv/bin/python -m analysis.structured_neural_benchmark develop --run analysis/holdout_results/new_neural_run --device mps
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib .venv/bin/python -m analysis.structured_neural_benchmark predict --run analysis/holdout_results/new_neural_run
.venv/bin/python -m analysis.holdout_evaluate --run analysis/holdout_results/new_neural_run
.venv/bin/python -m analysis.structured_neural_report --run analysis/holdout_results/new_neural_run
```
