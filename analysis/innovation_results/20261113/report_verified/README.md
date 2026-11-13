# 固定数据集创新方法实验

结论：本轮完成了正式对照，但两套数据均未超过对应的既有已选模型，不替换现有主模型。新的组合优于本轮曲线控制组；双时间轴解码在验证和测试汇总上优于同编码器的直接输出，但首个内部时间折退步。上述局部信号不等于已获得稳定的整体最优方法。

本轮实施历史检索特征/先验残差、类型与乐队分层收缩、正增量双时间轴神经网络，使用冻结 V1 和 V2 两档数据。所有分数为活动等权 MAPE，单位 %。训练拟合误差不代表泛化。

每协议预先固定9个配方及2个最终分数组合。神经网络每配方三个种子42/123/2026，各80轮；仅三种子最终分数均值可参与选择。训练内部3折用于稳定性诊断，随后完整训练；验证选择后不重训，独立评分器最后读取测试答案。

## V1

验证选择 **family_mean**：验证 6.047%，测试 8.473%。
实际案例覆盖：训练 198 活动/4377 前缀，验证 42 活动/414 前缀，测试 43 活动/412 前缀。
所选组合按最终分数等权平均：retrieval__retrieval_residual, hierarchy__hierarchy_phase_shrinkage, growth__growth_dual_clock。

| 方法 | 训练拟合 | 验证 | 测试 | 内部折1 | 内部折2 | 内部折3 |
|---|---:|---:|---:|---:|---:|---:|
| family_mean | 4.553 | 6.047 | 8.473 | — | — | — |
| blend_control_50 | 5.719 | 6.519 | 9.074 | — | — | — |
| growth__growth_dual_clock | 6.077 | 6.585 | 9.120 | 11.058 | 10.982 | 8.204 |
| retrieval__retrieval_residual | 4.619 | 7.007 | 8.893 | 7.648 | 6.977 | 9.241 |
| retrieval__retrieval_features | 4.558 | 7.054 | 10.220 | 11.052 | 7.162 | 7.508 |
| growth__growth_progress | 6.223 | 7.102 | 9.510 | 10.968 | 11.876 | 8.579 |
| retrieval__retrieval_control | 5.765 | 7.167 | 9.698 | 11.224 | 9.641 | 7.289 |
| hierarchy__hierarchy_phase_shrinkage | 5.170 | 7.323 | 9.407 | 8.504 | 5.864 | 11.461 |
| hierarchy__hierarchy_additive_shrinkage | 5.199 | 7.325 | 9.432 | 8.512 | 5.884 | 11.465 |
| growth__growth_direct | 6.680 | 7.941 | 10.425 | 10.625 | 14.095 | 8.941 |
| hierarchy__hierarchy_static_context | 2.443 | 8.360 | 10.762 | 4.463 | 7.563 | 7.766 |
| linear | 13.002 | 16.025 | 17.378 | 11.061 | 12.634 | 12.798 |
| hold_current | 46.172 | 43.095 | 44.182 | 44.503 | 42.905 | 46.751 |

组合成员由固定验证集选定，因此不把选择后组合回填到内部折并称为未见表现。内部折采用固定的较早训练历史；这不同于上一轮按查询逐步扩充训练历史的检索 pilot。

### 同案例历史对照

| 方法 | 测试 MAPE | T1000 | T2000 |
|---|---:|---:|---:|
| family_mean | 8.473 | 12.521 | 4.424 |
| old_catboost_peer | 9.290 | 13.810 | 4.770 |
| old_selected | 8.068 | 12.322 | 3.813 |

| 对照 | 新减旧/百分点 | 配对活动95%区间 | 改善/变差 |
|---|---:|---|---|
| old_catboost_peer | -0.817 | [-1.926, 0.185] | 21/22 |
| old_selected | 0.405 | [-0.523, 1.269] | 13/30 |
| retrieval__retrieval_control | -1.225 | [-1.674, -0.789] | 36/7 |

区间按冻结案例归属对活动配对重采样5000次，未作多重比较校正；只作描述，不用于测试后重新挑选模型。

### 神经网络逐种子

| 配方/种子 | 验证 MAPE | 测试 MAPE |
|---|---:|---:|
| growth__growth_direct | 7.941 | 10.425 |
| growth__growth_direct__seed_42 | 7.813 | 10.390 |
| growth__growth_direct__seed_123 | 8.420 | 10.601 |
| growth__growth_direct__seed_2026 | 8.621 | 11.251 |
| growth__growth_progress | 7.102 | 9.510 |
| growth__growth_progress__seed_42 | 7.573 | 10.154 |
| growth__growth_progress__seed_123 | 7.507 | 9.710 |
| growth__growth_progress__seed_2026 | 7.025 | 9.439 |
| growth__growth_dual_clock | 6.585 | 9.120 |
| growth__growth_dual_clock__seed_42 | 7.308 | 9.705 |
| growth__growth_dual_clock__seed_123 | 6.569 | 9.280 |
| growth__growth_dual_clock__seed_2026 | 6.782 | 9.215 |

数据指纹：`d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb`；运行：`analysis/holdout_results/innovation_20261113`。

## V2

验证选择 **family_mean**：验证 6.895%，测试 9.233%。
实际案例覆盖：训练 196 活动/2229 前缀，验证 42 活动/1548 前缀，测试 43 活动/1514 前缀。
所选组合按最终分数等权平均：retrieval__retrieval_residual, hierarchy__hierarchy_static_context, growth__growth_dual_clock。

| 方法 | 训练拟合 | 验证 | 测试 | 内部折1 | 内部折2 | 内部折3 |
|---|---:|---:|---:|---:|---:|---:|
| family_mean | 2.491 | 6.895 | 9.233 | — | — | — |
| blend_control_50 | 2.915 | 7.360 | 9.983 | — | — | — |
| retrieval__retrieval_residual | 3.194 | 7.374 | 10.212 | 4.336 | 4.065 | 6.798 |
| hierarchy__hierarchy_static_context | 1.972 | 8.230 | 10.014 | 3.726 | 3.724 | 5.180 |
| retrieval__retrieval_features | 3.040 | 8.636 | 11.387 | 7.910 | 4.399 | 6.236 |
| growth__growth_dual_clock | 3.963 | 8.755 | 11.764 | 8.469 | 4.297 | 6.563 |
| retrieval__retrieval_control | 3.385 | 8.765 | 11.351 | 8.589 | 4.717 | 5.532 |
| growth__growth_progress | 4.266 | 8.852 | 11.786 | 8.651 | 4.696 | 7.534 |
| hierarchy__hierarchy_phase_shrinkage | 3.285 | 8.927 | 11.266 | 6.783 | 3.210 | 7.080 |
| hierarchy__hierarchy_additive_shrinkage | 3.435 | 9.018 | 11.362 | 6.820 | 3.237 | 7.120 |
| growth__growth_direct | 4.584 | 10.863 | 12.996 | 6.320 | 7.437 | 11.448 |
| linear | 11.443 | 18.389 | 19.074 | 12.647 | 8.058 | 11.092 |
| hold_current | 26.797 | 43.642 | 45.170 | 29.074 | 19.292 | 27.540 |

组合成员由固定验证集选定，因此不把选择后组合回填到内部折并称为未见表现。内部折采用固定的较早训练历史；这不同于上一轮按查询逐步扩充训练历史的检索 pilot。

### 同案例历史对照

| 方法 | 测试 MAPE | T1000 | T2000 |
|---|---:|---:|---:|
| family_mean | 9.233 | 13.200 | 5.263 |
| old_single | 9.753 | 14.526 | 4.970 |
| old_paired | 9.441 | 14.323 | 4.550 |
| old_selected | 9.118 | 13.512 | 4.721 |

| 对照 | 新减旧/百分点 | 配对活动95%区间 | 改善/变差 |
|---|---:|---|---|
| old_single | -0.519 | [-0.978, -0.021] | 29/14 |
| old_paired | -0.208 | [-1.603, 1.113] | 12/31 |
| old_selected | 0.115 | [-1.212, 1.390] | 11/32 |
| retrieval__retrieval_control | -2.118 | [-2.970, -1.291] | 34/9 |

区间按冻结案例归属对活动配对重采样5000次，未作多重比较校正；只作描述，不用于测试后重新挑选模型。

### 神经网络逐种子

| 配方/种子 | 验证 MAPE | 测试 MAPE |
|---|---:|---:|
| growth__growth_direct | 10.863 | 12.996 |
| growth__growth_direct__seed_42 | 10.906 | 13.662 |
| growth__growth_direct__seed_123 | 11.382 | 13.384 |
| growth__growth_direct__seed_2026 | 11.194 | 12.888 |
| growth__growth_progress | 8.852 | 11.786 |
| growth__growth_progress__seed_42 | 9.099 | 12.110 |
| growth__growth_progress__seed_123 | 9.470 | 12.141 |
| growth__growth_progress__seed_2026 | 8.855 | 12.035 |
| growth__growth_dual_clock | 8.755 | 11.764 |
| growth__growth_dual_clock__seed_42 | 9.615 | 12.170 |
| growth__growth_dual_clock__seed_123 | 8.772 | 11.918 |
| growth__growth_dual_clock__seed_2026 | 8.590 | 12.121 |

### V2 全部时间点

| 时点 | 本轮选择 | 上轮选择 | 活动/案例 |
|---|---:|---:|---|
| elapsed_1h | — | — | 0/0 |
| elapsed_3h | — | — | 0/0 |
| elapsed_6h | 18.440 | 15.542 | 31/62 |
| elapsed_12h | 19.295 | 16.684 | 39/77 |
| elapsed_24h | 17.668 | 16.041 | 40/80 |
| elapsed_48h | 13.398 | 13.073 | 41/82 |
| elapsed_72h | 12.630 | 11.934 | 42/84 |
| progress_10pct | 19.638 | 17.931 | 41/81 |
| progress_25pct | 14.064 | 13.435 | 42/84 |
| progress_50pct | 10.497 | 10.459 | 42/84 |
| progress_75pct | 8.792 | 8.560 | 41/82 |
| progress_90pct | 5.557 | 6.309 | 40/80 |
| progress_95pct | 4.558 | 4.846 | 40/80 |
| remaining_48h | 8.233 | 8.716 | 42/84 |
| remaining_24h | 5.963 | 7.274 | 41/82 |
| remaining_12h | 4.808 | 4.898 | 39/78 |
| remaining_6h | 3.418 | 4.271 | 40/80 |
| remaining_3h | 2.699 | 3.398 | 39/78 |
| remaining_1h | 1.975 | 2.745 | 40/80 |
| remaining_0.5h | 1.792 | 2.886 | 39/78 |
| remaining_0.25h | 1.792 | 2.886 | 39/78 |

缺失早期时点继续保持缺失，时间轴别名不重复计入总体。逐案例实际观测时刻见冻结数据目录的 test_cases.csv（observed_at）；≤15min 新鲜子集、共同活动×排名面板、观测滞后汇总及完整类型/乐队指标见本运行 evaluation.json。


数据指纹：`1f76f39af8151bba185496411bbcee064b16a4c5aca5c515c1d957a5056fe854`；运行：`analysis/temporal_results/innovation_20261113`。

## 设计与解释边界

- 新控制组只使用51个自身曲线/日历/排名特征，不含活动类型；类型与乐队由命名上下文配方引入。历史参考保留原有输入，不能把战胜新控制组自动解释为战胜旧模型。
- 检索只用该活动开始前已结束且属于当前拟合池的同排名训练曲线，排除本期；保存邻居覆盖与滞后，推理忽略外部传入历史。
- 分层校正使用训练内部前向OOF残差；主效应和阶段交互有固定收缩，不在验证上拟合校正。
- 神经网络只比较解码器：同一个小型MLP编码器，直接输出、进度基函数积分、进度加末24h/6h基函数积分。本轮只用终分监督，不能把分解系数解释为真实玩家行为；尚未验证区间辅助监督、自监督预训练、Neural CDE或19档曲面。
- V1和V2分别拟合/选择，跨协议汇总数字不能直接比较。V2 57、68无可用前缀但保留原分组；198组中实际拟合196活动。
- 所有正式案例保留，训练按活动等权。验证参与多候选选择，存在选择乐观偏差。

**限制：测试队列曾参与早期探索，不能称完全未见盲测；元数据为缺少历史时点存档的回溯快照。**
