# 网页与训练分离 Implementation Plan

**Goal:** 根项目提供已有模型网页服务，`training/` 提供完整训练和评测入口，统一 uv。

**Architecture:** 单一 uv 项目，通过可编辑包保留旧导入名；生产推理与显式训练分别启动，固定旧模型兼容指纹。

**Tech Stack:** Python 3.14.8、uv、CatBoost、现有网页服务、Docker Compose。

**Spec:** `docs/superpowers/specs/2026-11-19-uv-training-layout-design.md`

## 约束

- 本地 main 工作，不提交、不推送、不发布。
- 不修改冻结数据、封存预测及模型包；历史测试仍为探索性证据。
- 网页依赖不默认安装神经网络库；训练和测试计算串行。

## 工作项

- [x] 记录实际旧模型源码指纹、依赖与技术预测；保留源码迁移备份。
- [x] 添加入口、只推理边界与兼容指纹的失败测试。
- [x] 将研究代码与旧脚本移入 `training/`，修正源码位置查找，保留历史数据路径。
- [x] 将生产拟合函数移入 `training/model_fit.py`，增加显式训练入口。
- [x] 增加根 `app.py`、`pyproject.toml`、`.python-version`、`uv.lock`，实际同步并验证导入。
- [x] 更新 Docker、Actions、运行说明、训练说明、README 与忽略规则。
- [x] 核查旧权重技术重放，运行完整测试和网页验证，整理本地待提交内容。
