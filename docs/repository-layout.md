# 仓库与发布工件的边界

## Git 保存的内容

- `production/`：网页、数据同步、固定配方预测、模型包校验和部署入口。
- `deploy/`、`.github/workflows/`、`Dockerfile`、`compose.yaml`：框架构建、部署及模型/论文的独立发布流程。
- `analysis/*.py`、`tests/`：研究实现与测试；二轮研究的实现、预登记、审稿报告及整合入口也保留。
- `docs/`、`AGENTS.md`：运行说明、评测约束和设计。
- `papers/`：三篇论文的 LaTeX、BibTeX、表格、绘图脚本、必要的 PDF/SVG 矢量图及轻量数据说明。
- 新冻结队列的 `manifest.json`、`partitions.csv`、覆盖/排除清单：保留分组与指纹，原始案例载荷另存。

## 留在本地或外部存储的内容

`.gitignore` 排除运行库、预测账本、数据库、训练权重、预训练模型下载、实验运行输出、源码冻结副本、回放沙箱、压缩交付包、论文编译目录与视觉检查截图。文件保留原位置，不移动、不删除，以免破坏冻结证据和复现脚本。

已经跟踪的历史冻结数据和结果保留原有版本。新增忽略规则不会将它们从 Git 历史中删除。IDE 配置停止跟踪，个人文件仍留在电脑上。机密使用环境变量或 GitHub Actions Secrets，`.env.example` 可以提交。

只有源码、轻量清单和论文编译素材进入 Git，并不表示完整研究数据已公开。完整数值复现仍需对应的冻结数据与实验归档；来源指纹与缺失的外部工件需按研究说明核对。

## 三类独立发布

| 对象 | 存放位置 | 不触发的流程 |
|---|---|---|
| 网页框架镜像 | Docker Hub `tano26/bang-dream-activity-fraction-prediction` | 不打包数据库、权重和论文 |
| 新训练模型 | GitHub Release `model-v*` | 不重建框架镜像 |
| 三篇论文及补充 PDF | GitHub Release `papers-v*` | 不重建镜像、不发布模型 |

最终 PDF 在 `papers/*/output/` 生成，校验后作为 Release 资产发布。正文引用的矢量图位于 `papers/*/figures/`，属于编译输入，因此仍可跟踪。
