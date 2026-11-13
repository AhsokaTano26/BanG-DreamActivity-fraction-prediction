# V2 多 agent 全方向对照

验证集选中 **family_mean**。验证 MAPE **4.76%**，测试 MAPE **9.12%**。所有数字为活动等权 event-macro MAPE。

三个 agent 各自开发6个固定配方，统一重新拟合后比较全部18个新配方、7个验证选择组合、5个历史参考。各 agent 仅访问训练/验证组；主入口在选择及组合参数封存后才打开测试盲前缀，独立评分器最后读取答案。测试成绩未用于改模型或改主方法。

## 各方向验证最优及历史参考

| 方法 | 训练 MAPE | 验证 MAPE | 测试 MAPE | T1000 测试 | T2000 测试 |
|---|---:|---:|---:|---:|---:|
| reference_single | 2.22% | 7.35% | 9.75% | 14.53% | 4.97% |
| reference_paired | 2.05% | 5.50% | 9.44% | 14.32% | 4.55% |
| reference_tcn | 2.13% | 6.18% | 10.59% | 13.92% | 7.25% |
| remaining__remaining_baseline_log_rmse | 1.99% | 5.51% | 9.44% | 14.31% | 4.55% |
| fusion__fusion_cat_rank_progress | 1.86% | 5.21% | 9.12% | 13.50% | 4.73% |
| shift__shift_calendar_rank | 2.08% | 5.34% | 9.74% | 13.87% | 5.62% |
| family_mean **（验证选中）** | 1.89% | 4.76% | 9.12% | 13.51% | 4.72% |

训练组误差仅为拟合表现。旧TCN为上一轮三个固定种子的最终分数均值；本轮没有择优挑选旧种子，也没有新增神经训练。

## 全部候选（按验证误差排序）

| 方法 | 训练 MAPE | 验证 MAPE | 测试 MAPE |
|---|---:|---:|---:|
| family_mean | 1.89% | 4.76% | 9.12% |
| family_median | 1.91% | 4.85% | 9.22% |
| blend_50 | 1.90% | 4.88% | 9.17% |
| blend_75 | 1.87% | 4.92% | 9.11% |
| phase_route | 1.77% | 4.95% | 9.05% |
| rank_route | 1.91% | 5.06% | 9.33% |
| blend_25 | 1.96% | 5.10% | 9.27% |
| fusion__fusion_cat_rank_progress | 1.86% | 5.21% | 9.12% |
| shift__shift_calendar_rank | 2.08% | 5.34% | 9.74% |
| shift__shift_rank_calibrated | 2.19% | 5.49% | 9.93% |
| reference_paired | 2.05% | 5.50% | 9.44% |
| remaining__remaining_baseline_log_rmse | 1.99% | 5.51% | 9.44% |
| fusion__fusion_cat_rmse_constant | 1.90% | 5.58% | 9.08% |
| fusion__fusion_cat_rmse_progress | 1.90% | 5.59% | 9.10% |
| fusion__fusion_cat_huber_progress | 1.90% | 5.71% | 9.26% |
| shift__shift_calendar_pool | 1.95% | 6.02% | 9.77% |
| reference_tcn | 2.13% | 6.18% | 10.59% |
| remaining__remaining_rate_log_mae | 1.72% | 6.18% | 8.90% |
| shift__shift_pool_calibrated | 2.06% | 6.33% | 10.21% |
| remaining__remaining_rate_log_rmse_by_rank | 3.02% | 6.86% | 9.01% |
| fusion__fusion_extra_progress | 1.19% | 7.01% | 9.53% |
| fusion__fusion_cat_mae_progress | 1.85% | 7.09% | 9.01% |
| reference_single | 2.22% | 7.35% | 9.75% |
| shift__shift_calendar_phase | 1.44% | 7.81% | 10.00% |
| remaining__remaining_rate_log_rmse | 3.09% | 8.57% | 10.24% |
| shift__shift_era_ridge | 4.41% | 10.36% | 13.20% |
| remaining__remaining_ratio_mape | 4.68% | 14.70% | 17.45% |
| remaining__remaining_ratio_mape_by_rank | 4.72% | 15.84% | 18.78% |
| linear | 11.44% | 18.39% | 19.07% |
| hold_current | 26.80% | 43.64% | 45.17% |

主方法严格来自验证排序首位，不用测试排序替换。全部配方、权重和源码位于 ../run_manifest.json、../models/、../sources/。全量机器可读汇总见 all_candidates.csv。

## 验证阶段选定的组合规则

```json
{
  "family_mean": {
    "kind": "mean",
    "members": [
      "remaining__remaining_baseline_log_rmse",
      "fusion__fusion_cat_rank_progress",
      "shift__shift_calendar_rank"
    ]
  },
  "family_median": {
    "kind": "median",
    "members": [
      "remaining__remaining_baseline_log_rmse",
      "fusion__fusion_cat_rank_progress",
      "shift__shift_calendar_rank"
    ]
  },
  "blend_25": {
    "kind": "weighted",
    "members": [
      "fusion__fusion_cat_rank_progress",
      "reference_paired"
    ],
    "fraction": 0.25
  },
  "blend_50": {
    "kind": "weighted",
    "members": [
      "fusion__fusion_cat_rank_progress",
      "reference_paired"
    ],
    "fraction": 0.5
  },
  "blend_75": {
    "kind": "weighted",
    "members": [
      "fusion__fusion_cat_rank_progress",
      "reference_paired"
    ],
    "fraction": 0.75
  },
  "rank_route": {
    "kind": "rank",
    "routes": {
      "0": "reference_tcn",
      "1": "fusion__fusion_cat_rank_progress"
    }
  },
  "phase_route": {
    "kind": "phase",
    "routes": {
      "0": "shift__shift_calendar_rank",
      "1": "shift__shift_calendar_rank",
      "2": "remaining__remaining_rate_log_mae"
    },
    "edges": [
      0.25,
      0.9
    ]
  }
}
```

其中阶段组合边界预先固定为25%/90%；它在三个方向开发发现验证上的阶段互补后、正式统一训练和任何新测试推理前加入，实验计划记录了这一补充。验证集参与多候选选择可能导致乐观偏差，因此同时保留完整队列的独立测试及活动级不确定性。

## 配对活动比较

| 方法 | 参考 | 新减旧/百分点 | 95%下界 | 95%上界 | 改善/变差活动数 |
|---|---|---:|---:|---:|---|
| family_mean | reference_single | -0.635 | -1.927 | 0.801 | 33/10 |
| family_mean | reference_paired | -0.323 | -0.650 | 0.010 | 27/16 |
| remaining__remaining_baseline_log_rmse | reference_single | -0.314 | -1.692 | 1.207 | 32/11 |
| remaining__remaining_baseline_log_rmse | reference_paired | -0.002 | -0.030 | 0.027 | 26/17 |
| fusion__fusion_cat_rank_progress | reference_single | -0.635 | -1.536 | 0.363 | 34/9 |
| fusion__fusion_cat_rank_progress | reference_paired | -0.324 | -0.999 | 0.316 | 26/17 |
| shift__shift_calendar_rank | reference_single | -0.008 | -1.682 | 1.790 | 29/14 |
| shift__shift_calendar_rank | reference_paired | 0.303 | -0.365 | 0.976 | 20/23 |

按冻结case活动归属先计算活动误差，再对43个活动配对重采样5000次；负差值表示新方法误差更低。区间是未做多重比较校正的描述性边际区间，不能用于测试后择优或直接推断因果。

## 全部测试时间点

| 请求时点 | 原单线 | 上轮双线 | 本轮验证选中 | 活动数/案例数 |
|---|---:|---:|---:|---|
| 开始后1h | — | — | — | 0/0 |
| 开始后3h | — | — | — | 0/0 |
| 开始后6h | 17.61% | 17.51% | 15.54% | 31/62 |
| 开始后12h | 16.61% | 17.34% | 16.68% | 39/77 |
| 开始后24h | 17.87% | 17.50% | 16.04% | 40/80 |
| 开始后48h | 14.59% | 13.76% | 13.07% | 41/82 |
| 开始后72h | 12.57% | 12.66% | 11.93% | 42/84 |
| 进度10% | 21.70% | 18.77% | 17.93% | 41/81 |
| 进度25% | 14.17% | 14.08% | 13.43% | 42/84 |
| 进度50% | 11.54% | 10.77% | 10.46% | 42/84 |
| 进度75% | 10.09% | 8.78% | 8.56% | 41/82 |
| 进度90% | 7.15% | 6.78% | 6.31% | 40/80 |
| 进度95% | 5.34% | 5.14% | 4.85% | 40/80 |
| 结束前48h | 10.19% | 8.79% | 8.72% | 42/84 |
| 结束前24h | 8.04% | 7.58% | 7.27% | 41/82 |
| 结束前12h | 5.48% | 5.08% | 4.90% | 39/78 |
| 结束前6h | 4.14% | 4.52% | 4.27% | 40/80 |
| 结束前3h | 2.74% | 3.54% | 3.40% | 39/78 |
| 结束前1h | 1.72% | 2.36% | 2.75% | 40/80 |
| 结束前0.5h | 1.60% | 2.10% | 2.89% | 39/78 |
| 结束前0.25h | 1.60% | 2.10% | 2.89% | 39/78 |

1h/3h早期没有冻结合格案例，继续报告缺失；请求时点和实际观测时点分别保存。不同方法不能各自删案例；时间轴重叠的物理案例总体只计一次。共同活动×排名面板、滞后≤15min子集、活动类型、乐队、MAE/RMSE/P90等完整诊断保存于 ../evaluation.json；全部时点×排名见 ../stage_rank_metrics.csv。

## 数据、运行与限制

数据指纹 `1f76f39af8151bba185496411bbcee064b16a4c5aca5c515c1d957a5056fe854`。主评分标识 `d05fbed194639d2c335a013dab74f49f5e55dd7baef9370adea931a2ae26d7d9`。比较评分标识 `fad963221c4da427fca9c587d28c80a07172023e1487c04bec8cb8a730cdc307`。

训练/验证/测试2229/1548/1514个前缀，原分组198/42/43个活动；训练实际参与196个活动，57/68没有可用前缀，仍保留分组。所有训练活动总权重相同，排名分支切片全局权重；OOF校准只使用训练活动并严格按已结束活动时间组织。

双线输入采用此前安全配对规则，训练/验证/测试可用配对为950/1548/1512个；覆盖差异可能影响泛化。缺失保留，掩码与门控显式处理。

旧V1/V2、数据库与旧封存结果保留。本轮是用户指定V2范围的比较，不构成V1主标准改进声明。

**限制：该测试队列曾参与早期探索，不能称为完全未见盲测；元数据是缺少历史时点存档的回溯快照。**

```bash
.venv/bin/python -m analysis.research_search --run analysis/temporal_results/<new-run>
.venv/bin/python -m analysis.temporal_evaluate --run analysis/temporal_results/<new-run>
.venv/bin/python -m analysis.research_compare --run analysis/temporal_results/<new-run>
.venv/bin/python -m analysis.research_report --run analysis/temporal_results/<new-run>
```
