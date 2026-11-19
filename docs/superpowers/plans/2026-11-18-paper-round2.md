# Paper Round2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement the tasks.

**Goal:** 在训练期证据、机制可检验性、公平对照和可靠性上升级预测论文。

**Architecture:** A–H隔离工作区交付，PI单点整合；先冻结计划与输入，再串行数值、独立审计、文字图表。以现有32页工件和不可变实验为基准。

**Tech Stack:** project .venv Python, NumPy/pandas/SciPy/CatBoost/PyTorch, Matplotlib SVG/PDF, XeLaTeX/BibTeX.

**Spec:** docs/superpowers/specs/2026-11-18-paper-round2.md

## Global Constraints

实际CNstart>=1706504400000，97期67/14/16不移动。历史测试探索性、重复验证、回溯元数据限制保留。数值CPU1串行nice19。旧结果保留。无未来虚构。所有agents只写自己隔离目录，PI集成。

## Review Focus

- 时间边界相等的标签不能释放；未来目标扰动不得影响早期输出。
- 支持缺失、常量残差和极小样本时lambda必须有限并回退独立。
- 容量/预算/时间跨度的混杂必须以清单与实际参数计数验收。
- 条件覆盖分母缺失和小活动数必须标注，不能以案例数替代活动数。
- 图表、表格和文本必须对应已完成封存结果，未来状态始终真实。

### Task 1: 基准归档、只读输入及责任清单（PI）
- [x] 归档完整32页稿、补充、图、QA并SHA；输出baseline/manifest.json。
- [x] 导出35期train-only误差和查询顺序，输入只读，metadata记录来源和caseID。
- [x] 完成A–H任务接口、依赖与预算列表。

### Task 2: A独立审稿与H审计设计
- [x] A贡献矩阵、拒稿问题和数学/引用核验，H检查冻结输入和选择协议。
- [x] 验收：真实主文/来源逐项定位，问题有严重度和修复条件。

### Task 3: B自适应共享、C消融与公平训练
- [x] B先preregistration.json，建立lambda纯函数与未来扰动/退化情形测试，保存RED/GREEN。
- [x] C先公平性manifest，证明参数与预算计数、固定时间分组不使用val/test答案。
- [x] PI审核封存后串行运行，全候选/负结果保留；生成逐案例、逐活动预测与预算。
- [x] H独立重算得分、案例集合、支持信息和实际调用。

### Task 4: D真实机制与E可靠性/功效
- [x] D训练残差估计、假设检验、参数可辨识说明、固定非高斯/突变/延迟模拟并配对结果。
- [x] E活动/排名等权、条件覆盖与宽度、活动级不确定性、训练差驱动功效和未来固定评分。
- [x] H核验数值和限制，禁止测试择优。

### Task 5: F图表与G中文LaTeX整合
- [x] 保留15图科研问题，根据审计和新证据改图，不按用户序号猜文件身份。
- [x] 重写贡献为协议/机制/候选算法的实际证据层次，补充详细菜单和预算。
- [x] 一键绑定数值、矢量出图、编译正文及补充、逐页PDF检查。

### Task 6: 最终独立复现与交付
- [x] H核查新代码/数据/配置/结果/稿件；A模拟终审，所有Important问题实际闭合或明确保留限制。
- [x] 运行相关测试和仓库要求unittest整套，检查ZIP独立构建/数值重放。
- [x] 更新质量变化报告、证据索引、剩余风险和投稿建议；交付PDF与源码实验包。
