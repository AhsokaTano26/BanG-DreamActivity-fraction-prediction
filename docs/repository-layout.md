# 仓库与发布工件的边界

## Git 保存的内容

- `app.py`、`production/`：网页、数据同步、固定配方推理、模型包校验和部署入口。实际拟合位于 `training/model_fit.py`。
- `deploy/`、`.github/workflows/`、`Dockerfile`、`compose.yaml`：框架构建、部署及模型/论文的独立发布流程。
- `training/`、`tests/`：完整训练实现、独立训练入口、研究方法与测试；二轮研究的预登记和审计记录保存在 `training/research_archive/`。
- `pyproject.toml`、`uv.lock`、`.python-version`：统一依赖与 Python 版本；`docs/`、`AGENTS.md` 保存运行说明、评测约束和设计。
- `papers/published/`：仅包含按标题命名的三篇正文 PDF 和一份补充 PDF。
- `docs/research/`：README 展示所需的实验结论、结果摘录和预测案例预览。
- 新冻结队列的 `manifest.json`、`partitions.csv`、覆盖/排除清单：保留分组与指纹，原始案例载荷另存。

## 留在本地或外部存储的内容

`.gitignore` 排除运行库、预测账本、数据库、训练权重、预训练模型下载、实验运行输出、源码冻结副本、回放沙箱和压缩交付包。论文的 LaTeX、BibTeX、表格、绘图源文件、编译目录与视觉检查文件也全部留在本地；这些研究原稿保留原位置。

已经跟踪的历史冻结数据和结果保留原有版本。新增忽略规则不会将它们从 Git 历史中删除。IDE 配置停止跟踪，个人文件仍留在电脑上。机密使用环境变量或 GitHub Actions Secrets，`.env.example` 可以提交。

Git 保存项目源码、轻量清单、结果说明和论文 PDF。完整数值复现需要本地冻结数据与实验归档；来源指纹与外部工件需按研究说明核对。

## 三类独立发布

| 对象 | 存放位置 | 不触发的流程 |
|---|---|---|
| 网页框架镜像 | Docker Hub `tano26/bang-dream-activity-fraction-prediction` | 不打包数据库、权重和论文 |
| 新训练模型 | GitHub Release `model-v*` | 不重建框架镜像 |
| 三篇论文及补充 PDF | GitHub Release `papers-v*` | 不重建镜像、不发布模型 |

在保有论文原稿的本地工作区运行 `python deploy/paper_assets.py sync`，将已有编译结果复制到 `papers/published/`，文件名与标题一致。论文 Release 仅接受这四份 PDF，并逐字节对照仓库中的阅读版；校验结果写入工作流日志。LaTeX、源码包、图表源文件与校验报告均不作为论文下载附件。
