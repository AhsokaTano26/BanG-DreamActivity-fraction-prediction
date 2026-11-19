# 当前默认：历史最佳组合 legacy_pair

2026-11-19运行方式更新：系统定时任务已按用户要求暂停。日常使用 `forecast.py --rank 1000` 做一次预测，或 `forecast.py serve` 手动启动持续预测与网页；参见[运行指南](running.md)。下文 `selected_cli` 是底层运维入口，`install` 仅在主动需要系统后台调度时执行。

2026-11-18用户选择目前同口径测试成绩最好的组合。T1000/T2000使用400树Context基础模型，再用T2000共同偏差＋两线误差差值进行校正；其余排名使用600树共享基础模型＋各线独立收缩EWMA。固定配方来自已封存的coupled_gap/legacy_pair，后续不按实时测试成绩自动换算法。

该组合在既有19线V3测试面板的活动等权MAPE为6.540281%，原两线面板为6.539744%。测试队列曾被探索使用、验证反复使用、元数据为回溯快照；这些成绩属于历史探索性评测，不是当前生产重新拟合版本的精度保证。旧实验、分组、评分、源码和权重保留。

## 当前活动与新活动

当前活动沿用期初已固定的基础模型和校正快照，输出标记`pinned_independent_ewma`。新活动开始后默认建立`legacy_pair`快照。状态中的`selected_forecast_method`是默认选择；每条预测的`forecast_method`和`active_event_models`是实际使用方法。两者可以在当前活动结束前不同。

全部监督训练与校正均使用国服实际开始时间不早于2024-01-29 13:00、结束严格早于本期开始的历史。每期历史池固定。本期终分在结束并通过质量检查后才能回灌；新数据继续触发原有固定基础模型配方训练，失败保留旧版本。

校正初始状态使用已封存65期因果RAW预测和本地核实的已结束真实曲线。每个活动/排名/阶段一个误差均值；真正上线后的反馈优先于回溯种子。覆盖旧回溯种子前存入`selected_feedback_history`，不把种子称作新的前瞻验证。保留原始预测和修订历史。

种子来源完整性检查或加载失败时，已有活动继续使用固定状态；新活动暂不建立状态，等待下一轮重试。正常缺失曲线会明确记录为`partial`，按已核实历史和既定回退规则工作。

种子只读取固定外层清单列出的活动包，额外放入目录的文件不会加入历史。后台任务的安装和卸载须取得运行锁，并核对实际已加载任务的路径、参数和工作目录，避免中断正在运行的预测或操作其他安装。

## 当前操作入口

```bash
.venv/bin/python -m production.selected_cli status
.venv/bin/python -m production.selected_cli cycle
.venv/bin/python -m production.selected_cli cycle --offline
.venv/bin/python -m production.selected_cli install
.venv/bin/python -m production.selected_cli uninstall
```

后台任务仍为`com.tano.bangdream-forecast`，每900秒运行一次；入口为`production.selected_cli`，后台优先级及nice19，数值库线程环境为1。Mac须保持登录、清醒和联网。`runtime/live/latest.json/latest.csv`为实际输出，`selected_seed.json`保存回溯种子来源。

旧`production.cli`及旧说明保留供旧版本复现；日常运行请使用上述新入口。基础权重仍保存为不可覆盖版本，不因更换校正层重训神经网络。部署备份和验收证据位于`production_records/20261118_best/`。
