# 全库固定留出评测 holdout_v1

验证集事先选定的组合为 **LightGBM peer + PatchTST 三种子集成 + CatBoost peer**，三个成员的最终分数等权平均。测试活动等权 MAPE 为 **8.920%**；简单线性外推为 **17.378%**。1000 线误差仍明显高于 2000 线，应分别看待。

本轮完成 14 个树/线性模型配置、10 个神经网络配置各 3 个种子，保存 44 个训练检查点；加上 5 个外推/模板基线、逐种子结果和组合，共 60 组测试预测、24720 条结果。每组都是同样 43 个活动、412 个时点，不按方法删除样本。

## 固定数据与边界

- 原库 328 个活动、656 条国服排名曲线；500 条曲线、283 个活动合格。排除 152 条最终观测不够靠近结束、2 条观测不足 5 点、2 条尚未结束的曲线。保留稀疏旧曲线，不限定 EventID>=226。
- 训练 198 活动 / 330 曲线 / 4377 前缀：2019-06-11 至 2024-05-22；验证 42 / 84 / 414：2024-05-24 至 2025-07-25；测试 43 / 86 / 412：2025-08-08 至 2026-10-24。按真实开始时间排序，以活动 ID 成组，跨排名和全部快照不拆分。
- 最终观测要求结束前最多 60 秒或结束后最多 300 秒；前缀至少 4 点、滞后最多 60 分钟。固定阶段缺失 24 个时点（验证 6，测试 18），原因和名单已冻结。没有猜测缺失最终分数。
- 每个活动训练总权重相同。插补、标准化、类别编码、完整历史曲线和历史最终分数仅来自训练；训练行另排除自身和未来活动。peer 只见另一档截至本次已观测时刻的前缀。卡牌按国服发布时间过滤。
- 神经网络种子 42/123/2026，最多 80 轮，至少 25 轮、耐心 18，以验证活动 MAPE 选检查点；三种子集成平均最终分数。验证集选模型和组合，测试不重训、不补入验证/测试活动的最终分数。
- 主指标先对每个活动的全部有效时点/排名求 APE 均值，再对活动平均；不是准确率，也不是直接把所有快照等权。主指标 8.920%，case-MAPE 8.871%，APE 中位数 5.832%，P90 19.960%，MAE 593231 EP，RMSE 1052449 EP。

数据指纹：`d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb`。训练运行：`aef5892fd06e218af95c2a93610eae2ec22438cc31833cde189346be6885e2e8`。实际评分版本：`9493fbe3a4a89c1f3da13b117e3306287e2aca03dcaadf36440f868ab25a1aed`。

## 全部候选配置

以下按验证分数排序，选型依据是验证集；测试分数不反过来选种子或修改模型。完整 60 组逐种子和集成结果在 [test_summary.csv](test_summary.csv)。

| 方法（网络为三种子集成） | 验证活动 MAPE % | 测试活动 MAPE % |
|---|---:|---:|
| `validation_top3_ensemble` | 4.321 | 8.920 |
| `lightgbm_peer` | 4.993 | 9.431 |
| `patchtst_ensemble3` | 5.255 | 9.160 |
| `catboost_peer` | 5.277 | 9.290 |
| `itransformer_ensemble3` | 5.390 | 9.379 |
| `lstm_ensemble3` | 5.401 | 9.127 |
| `tcn_ensemble3` | 5.465 | 9.373 |
| `gru_ensemble3` | 5.503 | 9.457 |
| `mlp_ensemble3` | 5.577 | 9.382 |
| `tsmixer_ensemble3` | 5.668 | 9.415 |
| `transformer_ensemble3` | 5.732 | 9.626 |
| `dlinear_ensemble3` | 6.084 | 9.956 |
| `catboost_sequence` | 6.482 | 9.560 |
| `lightgbm_static` | 7.312 | 10.064 |
| `catboost_static` | 7.544 | 9.983 |
| `extra_trees_sequence` | 7.666 | 10.588 |
| `tcn_peer_ensemble3` | 7.900 | 10.858 |
| `catboost_history` | 7.983 | 10.713 |
| `lightgbm_sequence` | 8.005 | 10.553 |
| `xgb_sequence` | 8.133 | 10.801 |
| `lightgbm_band` | 8.318 | 10.761 |
| `lightgbm_history` | 9.047 | 12.675 |
| `shape_analog` | 9.055 | 11.104 |
| `catboost_band` | 9.106 | 10.760 |
| `same_type_template` | 9.880 | 12.245 |
| `xgb_tab_growth` | 10.313 | 12.371 |
| `template` | 11.287 | 13.551 |
| `linear` | 16.025 | 17.378 |
| `ridge_growth` | 16.055 | 16.827 |
| `recent_24h` | 16.695 | 17.657 |

`sequence` 表示基础表格特征加 64×8 前缀序列。`band` 加入乐队构成；`static` 再加入角色、已发布卡牌和日历；`history` 再加入训练集历史热度/增长统计；`peer` 再加入另一排名档的已观测走势。神经网络为针对前缀预测最终分数的适配实现；DLinear 使用分解加非线性回归头。配置、代码、特征列、检查点、依赖版本均已保存。

单独加入乐队标签没有改善这次比较：CatBoost 基础 9.560% → band 10.760%，LightGBM 基础 10.553% → band 10.761%。完整背景和 peer 更有效，CatBoost peer 为 9.290%、LightGBM peer 为 9.431%。这些是本数据集内的特征对照，不证明某个乐队造成分数变化。背景 TCN 也未优于基础 TCN，不能假设更多特征或更复杂网络必然更准。

LSTM 三种子集成在单个网络家族的测试结果中最低（9.127%），但不据此替换验证集选定的 PatchTST 组合成员。验证选择的最终组合为 8.920%，优于本轮其他配置。

## 分排名与阶段

| 排名档 | 测试活动数 | MAPE % | P90 绝对百分比误差 % |
|---|---:|---:|---:|
| 1000 | 43 | 13.495 | 24.296 |
| 2000 | 43 | 4.345 | 9.329 |

| 预测时点 | 测试活动数 | MAPE % |
|---|---:|---:|
| 25% | 42 | 13.171 |
| 50% | 42 | 10.451 |
| 75% | 41 | 9.109 |
| 24h remaining | 41 | 6.638 |
| 6h remaining | 40 | 4.740 |

## 分活动类型与乐队

| 活动类型 | 测试活动数 | MAPE % |
|---|---:|---:|
| challenge | 8 | 8.398 |
| festival | 2 | 8.853 |
| live_try | 11 | 8.944 |
| medley | 4 | 7.005 |
| mission_live | 11 | 9.939 |
| versus | 7 | 8.993 |

| 加成角色的乐队构成 | 测试活动数 | MAPE % |
|---|---:|---:|
| Poppin'Party | 4 | 12.043 |
| RAISE A SUILEN | 4 | 10.135 |
| Afterglow | 4 | 9.993 |
| Morfonica | 4 | 7.428 |
| Hello, Happy World！ | 5 | 6.972 |
| Pastel＊Palettes | 3 | 5.876 |
| MyGO!!!!! | 3 | 8.189 |
| Roselia | 5 | 8.639 |
| 混团 | 11 | 9.538 |

乐队由加成角色组成推断，不代表唯一故事主角。Festival 只有 2 个测试活动，多数单乐队也只有 3–5 个活动，这些分组只用于检查误差，不能据此作可靠的乐队排名。

## 不确定性和适用范围

| 对照 | 组合减对照的 MAPE 差值（百分点） | 活动配对 bootstrap 95% 区间 |
|---|---:|---|
| `linear` | -8.458 | [-10.946, -5.931] |
| `lightgbm_sequence` | -1.633 | [-2.896, -0.256] |
| `catboost_peer` | -0.370 | [-0.694, -0.020] |

区间以 43 个活动为单位配对重采样 5000 次，种子 42；负数表示组合误差更低。它衡量本队列的平均误差差异，不是未来活动的预测区间。重采样没有建模相邻活动之间的长期相关性，区间可能偏乐观。

验证误差 4.321% 上升为测试 8.920%，跨时期泛化仍有限；1000 线及较早预测阶段是主要薄弱处。当前训练冻结于 2024 年，无法等同于滚动更新模型。此前 6.58% 等探索结果的训练历史、过滤和平均口径均不同，不应直接对比。

**这批测试活动曾出现在早期探索中，并非完全未见过的新队列。** 元数据也是回溯快照，缺少历史时点存档。后续参数开发仍只用训练/验证，继续沿用本固定测试名单；新增活动另做前瞻验证，不覆盖 v1。

## 验证和复现

45 项自动化测试通过，48 个安装包兼容性检查通过。独立审查核对了分组、时间边界、训练历史池与权重；发现并修复了评分版本归档和 bootstrap 活动归属的一致性问题，没有改动本轮模型预测。

[dataset_audit.json](dataset_audit.json) 核对全部前缀/peer、名单及时间边界；[independent_verification.json](independent_verification.json) 使用独立的标准库计算重新验证全部 60 组 event/case MAPE、MAE、RMSE，最大舍入差异 2.33e-10。模型、数据和评分产物均有指纹，源数据库保持只读。

- [selection.json](selection.json)：测试评分前封存的选择。
- [evaluation.json](evaluation.json)：完整总体及分组指标。
- [test_summary.csv](test_summary.csv)：全部 60 组汇总。
- [scored_test_predictions.csv](scored_test_predictions.csv)：逐时点预测、真实值和误差。
- [run_manifest.json](run_manifest.json)、[blind_test_manifest.json](blind_test_manifest.json)、[evaluation_manifest.json](evaluation_manifest.json)：配置和指纹。

```bash
# 本轮已封存；同版本评分再次执行只校验并返回原结果
.venv/bin/python -m analysis.holdout_evaluate --run analysis/holdout_results/v1

# 后续模型使用新的运行目录，固定数据集默认不变
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib .venv/bin/python -m analysis.holdout_benchmark \
  --run analysis/holdout_results/new_method --device auto
.venv/bin/python -m analysis.holdout_evaluate --run analysis/holdout_results/new_method
```
