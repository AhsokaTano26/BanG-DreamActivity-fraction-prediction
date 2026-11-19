# 密集采样时期 V1 扩展模型重训

固定队列：`data/holdout_v1_dense_20261115`，指纹
`80e354c3a5340df6724a40ef1df490f92c08007311d55849342de509eaf5ee50`。
国服实际开始时间不早于 `1706504400000`，67/14/16 个训练、验证、测试活动，
2390/130/154 个前缀。旧数据、模型和封存结果不变。

## 预登记执行范围

入口 `analysis.dense_v1_extensions` 重用旧实现中的固定拟合配方，新增数据政策守卫、
同队列引用校验、测试答案拒绝器、训练拟合预测和独立盲预测封存。所有 CPU 工作限制为两线程。
Structured 在专用 `analysis.dense_v1_mps` 入口使用 MPS；其他家族仍在 CPU 执行。

| 家族 | 固定配置 | 新运行目录 |
|---|---|---|
| Relationship | 23 个 CatBoost、LightGBM、GAM 配方，原有消融与组合菜单 | `dense_relationship_20261115` |
| Structured neural | 9 架构/损失/采样配置 × 42/123/2026，最多 100 轮，原早停规则 | `dense_structured_mps_20261115` |
| Deadline | 3 个模型，3 个训练 OOF 校正收缩量 | `dense_deadline_20261115` |
| Quality / irregular | 6 个层次/质量统计模型，Time-GRU 和 Neural-CDE × 三种子，最多 80 轮 | `dense_quality_20261115` |
| Quality 延长敏感性 | 相同固定菜单，神经网络最多 240 轮，原早停规则 | `dense_quality_extended_20261115` |
| Innovation | retrieval、hierarchy、growth 全部原配方与组合 | `dense_innovation_20261115` |

Relationship、Deadline 和 Quality 的原 198 活动 OOF 边界不适用于新队列。
本轮按实际时间排序，前 35 个活动作为初始训练，随后四个各 8 活动的连续留出块，
每折仅使用在该块开始前已经结束的训练活动；对应训练活动数为 35、43、51、59。
Innovation 保留其原有比例规则：前半训练预热、后三个连续留出块。
全部 OOF 都仅使用 67 个训练活动；任何验证或测试终分不会进入历史或校正拟合。

Relationship 引用本轮 `dense_core_20261115`；Structured 引用本轮 Relationship；
Deadline 和 Quality 引用本轮 MPS Structured。引用仅使用已经封存的同队列预测。
Quality 240 轮敏感性独立运行，不覆盖 80 轮结果。历史统计关系、采样稀疏化、
分布漂移脚本是诊断报告，未定义新的预测配方，不作为待重训模型遗漏处理。

CPU Structured 首个 dual-TCN 种子耗时 148.73 秒。MPS 空闲后，保留未完成的
`dense_structured_20261115` 与中断说明，在新目录完整重跑所有 27 个神经拟合。
设备调整在测试评分前确定，只为执行耗时；CPU 未完成结果不参与模型或设备选择。
原适配器没有修改，新入口单独归档自身及原入口，训练预测重放也显式使用 MPS 张量。

每个家族沿用原验证选择菜单，单个神经种子仅作诊断，不参与最终模型选择。
所有家族盲预测封存后才运行 `analysis.holdout_evaluate` 独立读取测试答案。
汇总比较各家族验证选中的方法；不按测试分数重新选择方法或组合。
训练重代入误差是拟合度，OOF 与测试是独立信息条件下的评估，分开列示。

新队列仍来自曾参与探索的活动，验证也被反复使用；元数据是缺少历史时点存档的
回溯快照。不能称为完全未见过的盲测，也不能以新旧队列总分差声称模型改进。

## 结果

六组全部盲预测与训练重放已核对，合计 144 个方法输出；每个方法均覆盖完整
2390 个训练、130 个验证、154 个测试案例，分别为 67、14、16 个活动。
训练完整覆盖来自独立重放侧车，包含原封存配方的全部引用及组合，没有再次拟合。
覆盖审计位于 `analysis/dense_retrain_records/20261115/v1_coverage_before_scoring.csv`
及同名 JSON；审计执行时六组均尚未生成独立评分结果。

| 家族 | 配方数 | 神经种子拟合数 | 报告方法数 | 完整训练重放方法数 | 运行秒数 |
|---|---:|---:|---:|---:|---:|
| Relationship | 23 | 0 | 28 | 28 | 19.02 |
| Structured MPS | 9 | 27 | 41 | 41 | 193.56 |
| Deadline | 3 | 0 | 7 | 7 | 22.07 |
| Quality 80 | 8 | 6 | 23 | 23 | 80.47 |
| Quality 240 | 8 | 6 | 23 | 23 | 80.49 |
| Innovation | 9 | 9 | 22 | 22 | 88.74 |

方法数包括三种子成员、均值、固定组合和引用，不能当作独立模型或独立样本数。
秒数包含各组训练/OOF/预测工作，不包含等待前置引用及已保留的 CPU 中断尝试。
Quality 的“80/240”是最大轮数，仍执行原早停规则。

Core 另有 3 个纯历史基线（template、same_type_template、shape_analog）在第一期
密集训练活动没有历史可用，因此其全训练拟合 MAPE 记为不可用，未删案例或编造回退。
Core 验证及测试仍完整覆盖 60 方法；本表六个扩展组的训练重放没有缺失方法。

所有方法盲预测先完成封存，再统一由 `analysis.holdout_evaluate` 独立评分。
下表每行方法均在测试评分前按验证集选定，单位为活动等权 MAPE（%）。

| 家族 | 验证所选方法 | 训练拟合 | 验证 | 测试 |
|---|---|---:|---:|---:|
| Relationship | blend_new_0.25 | 2.3678 | 3.9530 | 14.1270 |
| Structured MPS | reference（沿用上行） | 2.3678 | 3.9530 | 14.1270 |
| Deadline | deadline_features | 3.1747 | 3.8255 | 14.3908 |
| Quality 80 | blend25_quality_shared_tree | 2.6079 | 3.8362 | 13.4445 |
| Quality 240 | blend25_quality_shared_tree | 2.6079 | 3.8362 | 13.4445 |
| Innovation | family_mean | 3.0879 | 4.3355 | 14.7774 |

本轮 V1 总体方法在测试前按验证最低误差选为 Deadline 的 `deadline_features`，
因此主结果仍为测试 **14.3908%**，不能看到测试后改选 Quality。该方法测试
case-MAPE 为 14.3681%，案例 APE 的 P90 为 34.7193%。验证误差与测试误差差距
明显；训练拟合误差不能代表泛化精度，现有结果也不足以声称复杂神经网络优于引用模型。

Quality 六个神经拟合分别在 42、44、57、53、47、71 轮早停，均未触及 80 轮上限。
因此将上限延长至 240 轮没有改变任何拟合；两组全部 23 方法测试预测的最大绝对差为 0。

完整 144 行候选表（包括逐种子与三种子平均）和六组验证所选结果分别见
`analysis/dense_retrain_records/20261115/v1_extension_all_candidates.csv`、
`v1_extension_selected_summary.csv`；对应评分及文件指纹见
`v1_extension_results_manifest.json`。每个运行的 `evaluation.json` 仍保留按排名、
时间点、活动类型和乐队的完整明细。独立样本量为 16 个测试活动，不能将
154 个前缀当作独立活动。新旧队列结果不可直接比较来宣称改进。
