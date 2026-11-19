# Web Deployment and Model Releases Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement the approved scope inline, one task at a time.

**Goal:** 发布单一网页框架镜像，独立管理固定算法的新版训练权重。

**Architecture:** 复用现有网页、数据守卫和活动 pin；新增部署入口和模型包管理入口。Docker 持久化 state 与 models，GitHub Actions 分别构建镜像和发布模型资产。

**Tech Stack:** Python 3.14.8、现有固定依赖、Docker Compose、GitHub Actions、Docker Hub。

**Spec:** `docs/superpowers/specs/2026-11-19-web-deployment-and-model-releases.md`

## Global Constraints

- 使用实际国服开始时间 `>=1706504400000`；保留冻结数据、实验与现有服务。
- 单镜像、网页服务、固定算法；不发布训练数据库到镜像。
- 当前活动保留模型和校正快照，新活动只用期初之前结束的历史。
- Docker Hub Secrets 为 `DOCKERHUB_USERNAME`、`DOCKERHUB_TOKEN`。

## Review Focus

- 压缩包路径/链接、文件集合或 SHA 变化必须拒绝。
- 模型安装与自动周期争锁时不能部分激活或覆盖已有 pin。
- 部署迁移不能保留开发电脑路径或遗失预测账本。
- 自动训练不能因历史缺失替换为缩小训练集的权重。
- 模型发布不能触发镜像构建；PR 构建不使用发布凭据。

## Tasks

1. [x] 新增部署与模型包行为测试；核实缺少入口的失败。
2. [x] 实现 `production/model_release.py` 导出、校验、安装与反馈导入，保留原模型验证。
3. [x] 实现 `production/deployment.py` 配置、状态导出、网页和周期入口；扩展现有 worker 注入与可选训练控制。
4. [x] 实现 Dockerfile、依赖清单、Compose、构建上下文排除和独立 Actions 工作流。
5. [x] 完成部署与发布说明，运行506项完整测试、容器 smoke 和 Compose 校验。
6. [ ] 在独立发布分支提交所需运行源码，推送并检查 Actions；模型资产另行封存及发布。
7. [x] 整理Git提交边界，保留源码与文稿，忽略本地/大型工件，停止跟踪IDE配置。
8. [x] 原始论文添加Tano署名，重新编译、136项验证、像素对照和独立源码编译；四份PDF准备独立发布资产。
