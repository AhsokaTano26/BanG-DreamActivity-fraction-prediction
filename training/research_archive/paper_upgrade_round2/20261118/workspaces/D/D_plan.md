# D 校准与机制实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Native execution is assigned by PI; no further agents.

**Goal:** 在训练期隔离下，交付真实残差条件诊断和24格配对机制模拟。

**Architecture:** 纯数组统计/滤波/模拟核与可信读取协调器分开。协调器独占真实标签，按时钟释放；预测核接收过去历史。B专家只读复用，并以小fixture核对向量化等价。

**Tech Stack:** 项目 `.venv/bin/python`，标准库、NumPy、pandas；无新依赖。

**Spec:** `D_design.md`、仓库 `docs/superpowers/specs/2026-11-18-paper-round2.md`。

## Global Constraints

- 所有新增文件在 `workspaces/D/`；旧论文、源码、结果只读。
- 前25期校准，末10期回溯时间外推诊断；只读35期训练输入。
- 数值作业待PI许可；CPU1、nice19、120秒硬预算、串行。
- 阶段边界 .25/.9 归右；严格 `end_ms<query.start_ms`；先发布后反馈。

## Review Focus

- 差分缺口：矩估计只使用相邻活动，滤波按缺口推进Q。
- 负Q/R：原矩保留，PSD投影另报，支持带属于描述性范围。
- 延迟：活动e发布前只看e-1-d，旧专家保留原发布值。
- 噪声：t5/污染方差规范，shift/异方差强度事前固定。
- 缺线与当前标签：完整配对明确支持，当前帧严格无actual/log_error。

### Task 1: 纯核与fixture

**Files:** `mechanism.py`、`test_mechanism.py`。
**Interfaces:** `moment_decomposition(diff0,diff1)`；`noise_unit_variance(kind,draws)`；`replay_arrays(y,Q,R,configuration,delay)`；B接口只读。

- [ ] 写失败测试：手算 `Gamma0=[[6,1],[1,10]],Gamma1=[[-1,-.5],[-.5,-2]]` 得 `R=[[1,.5],[.5,2]],Q=[[4,0],[0,6]]`，负特征值原值保留。
- [ ] 运行观察RED；最小实现后观察GREEN。
- [ ] 写失败测试：延迟0/1/3，预测前无本期标签；改变未来标签保持过去预测相同；历史I/S发布值完整保留。
- [ ] 写失败测试：同滤波负控、B I/S/risk固定fixture一致、完整协方差坐标等价、缺线更新、噪声规范常数和有限值。
- [ ] 最小实现并观察GREEN，记录命令/输出；数值许可前到此。

### Task 2: 真实协调器与条件校准

**Files:** `run_mechanism.py`。
**Interfaces:** 验证3个输入SHA，限定train；产出`calibration_seal.json`供模拟只读。

- [ ] fixture验证排序实际时间、阶段、严格释放、输出新目录、少于12停止、负矩保留。
- [ ] 实现固定900点剖面、删一范围、19×3诊断和末10收益信号；对称池只复用B。
- [ ] PI许可后执行单线程真实协调器，封存校准参数及SHA。

### Task 3: 固定模拟与报告

**Files:** `mechanism.py`、`run_mechanism.py`、`D_report.md`。
**Interfaces:** 只读校准seal；固定24格、6方法、三种子和100路径。

- [ ] fixture验证3×8格身份、配对路径、延迟释放、shift时刻、MC单位与结果完整性。
- [ ] 实现向量化路径和完整表，超时保留中断状态。
- [ ] PI许可后同一串行配额运行；输出失败区和simulation-real链接；摘要保留所有边界。
