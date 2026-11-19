# 网页部署与独立模型发布

部署只提供网页服务，使用固定的 CatBoost、特征处理及 legacy_pair 校正规则。Docker 镜像为 `tano26/bang-dream-activity-fraction-prediction`。权重、数据库和账本均在镜像外；后续增加训练数据发布新权重，无须构建镜像。

## 三类独立发布

| 对象 | 发布位置 | 触发方式 | 内容 |
|---|---|---|---|
| 网页框架 | Docker Hub | 框架源码/依赖变化，`framework-v*` 标签或手动执行 | 服务、网页、固定算法、依赖 |
| 训练模型 | GitHub Release | 准备 `model-v*` 草稿后手动执行模型工作流 | 权重、预处理、训练元数据、清单、反馈种子和 SHA-256 |
| 三篇论文 | GitHub Release | 准备 `papers-v*` 草稿后手动执行论文工作流 | 三份正文 PDF、独立补充材料、页数/作者信息和 SHA-256 |

`.github/workflows/framework-image.yml` 构建、容器测试并推送 amd64/arm64 镜像，标签包括 `latest`、`sha-<完整提交>`，框架版本标签另生成版本号。PR 只构建测试，不登录或推送。`model-v*` 标签不触发镜像构建。

`.github/workflows/model-release.yml` 下载已准备的草稿资产，检查 SHA-256，并在指定框架镜像中执行清单校验和真实权重重放；通过后发布模型 Release，不构建镜像。GitHub 的手动工作流需要先存在于默认分支。模型文件不进入 Git/LFS，也不进入 Docker 构建上下文。

GitHub 仓库 Settings → Secrets and variables → Actions 配置：

- `DOCKERHUB_USERNAME`：`tano26`。
- `DOCKERHUB_TOKEN`：具有推送权限的 Docker Hub Personal Access Token。

模型工作流使用 GitHub 自动提供的 `GITHUB_TOKEN`，不需要另外提供 GitHub Token。

## 论文 PDF 发布

三篇正文分别是原始预测论文、二轮 Entertainment Computing 预测论文、游戏数据分析论文，另附二轮独立补充材料。四份 PDF 的首页和作者元数据均为 Tano。

在已有编译结果的完整工作区运行：

```bash
.venv/bin/python deploy/paper_assets.py build --output /tmp/papers-v20261119.1
.venv/bin/python deploy/paper_assets.py verify /tmp/papers-v20261119.1
```

脚本逐字节复制当前 PDF，并检查可解析性、页数、作者、SHA-256；不重新运行实验，不重新编译或修改其他稿件。输出三份正文、补充 PDF、`papers-manifest.json` 和 `SHA256SUMS`。

将这些文件上传为 `papers-v*` 草稿 Release，默认分支已有工作流后执行 Actions → Publish three paper PDFs。工作流只校验并发布这组资产，不构建 Docker、不发布权重。PDF 成稿和压缩源码包不纳入 Git；论文 LaTeX、参考文献、表格及必要矢量图继续版本管理。

## 初次部署

服务器安装 Docker 与 Compose，放置仓库的 `compose.yaml`。准备可写目录：

```bash
mkdir -p state models bootstrap
```

在 `.env` 设置宿主用户 ID、路径和镜像版本，例如普通 Linux 用户为1000：

```dotenv
PUID=1000
PGID=1000
STATE_PATH=./state
MODEL_PATH=./models
BOOTSTRAP_PATH=./bootstrap
FRAMEWORK_TAG=latest
BIND_HOST=127.0.0.1
WEB_PORT=8765
AUTO_TRAIN=false
UPDATE_INTERVAL=900
CPU_LIMIT=1.0
MEMORY_LIMIT=2g
```

实际 UID/GID 用 `id -u`、`id -g` 获取。目录归该用户所有。默认端口仅绑定服务器本机；需要直接访问时显式设置 `BIND_HOST=0.0.0.0`，或在服务器的反向代理上转发该端口。

推荐迁移现有生产历史、预测记录和活动固定状态。在原电脑执行：

```bash
.venv/bin/python -m production.deployment export-state \
  --root runtime/live --output /tmp/bangdream-deployment-export
```

把导出目录中的 `state/` 和 `models/` 复制到服务器对应目录。命令使用 SQLite backup 并取得周期锁，不修改源数据库；目标必须为新目录。启动时根据模型 ID 重建容器内路径，保留每期校正快照与原始预测记录。

也可从空目录启动：先在 `bootstrap/` 放置发布的 `model.tar.gz` 和校验文件，再安装：

```bash
docker compose pull
docker compose run --rm web model verify /bootstrap/model.tar.gz
docker compose run --rm web model install /bootstrap/model.tar.gz --root /state
docker compose up -d
```

空部署可通过在线同步取得当前活动并使用发布权重；完整历史浏览需要迁移现有状态，或在 `bootstrap/` 提供原始 `db.sqlite3` 与 `metadata/`，由时间守卫导入密集采样历史。数据库挂载只读，不复制进镜像。已有在线缓存不能替代完整训练历史。

```bash
docker compose logs -f web
curl http://127.0.0.1:8765/healthz
curl http://127.0.0.1:8765/readyz
```

`/healthz` 检查网页服务存活；`/readyz` 检查最近模型周期是否就绪。尚未安装权重时网页可以打开，就绪接口返回503并等待模型。网页每15秒读取结果，后台周期默认900秒，始终串行。

## 发布新的训练模型

训练继续使用现有固定配方及密集采样政策。部署默认 `AUTO_TRAIN=false`，只加载明确安装的权重。若希望服务器在新增完整历史后自行重训，设为 `true`；历史缺少上一模型任意活动/排名时保留旧权重，不用缩小的历史训练替换它。模型拟合不等于新的独立精度评测。

训练完成后，在训练环境封装当前准备好的权重与已核实反馈：

```bash
mkdir -p /tmp/model-v20261119.1
.venv/bin/python -m production.model_release build \
  --root runtime/live --output /tmp/model-v20261119.1/model.tar.gz
.venv/bin/python -m production.model_release verify /tmp/model-v20261119.1/model.tar.gz
```

生成 `model.tar.gz` 与 `model.tar.gz.sha256`，不会覆盖已有资产。压缩包保留原始模型清单、软件和源码指纹。生产 fixture 权重、错误配方、旧政策、未来反馈、链接或路径逃逸、重复成员和损坏文件都会拒绝。只安装可信来源的自有模型资产。

准备草稿 Release（填写实际分支或提交及说明）：

```bash
gh release create model-v20261119.1 \
  /tmp/model-v20261119.1/model.tar.gz \
  /tmp/model-v20261119.1/model.tar.gz.sha256 \
  --draft --target deployment/web-model-releases \
  --title 'Model model-v20261119.1' --notes-file /tmp/model-release-notes.md
```

默认分支已有工作流后，在 Actions → Publish trained model → Run workflow 填写 `release_tag`，`framework_tag` 推荐填写与训练代码对应的 `sha-*` 镜像标签。工作流只将通过镜像环境验证的草稿改为正式 Release。模型 Release 不设为整个仓库的 latest Release。

## 更新框架与更新模型

更新框架：

```bash
docker compose pull
docker compose up -d
```

更新模型：下载指定 Release 的两个资产到 `bootstrap/`，在该目录执行 `sha256sum -c model.tar.gz.sha256`，然后：

```bash
docker compose run --rm web model install /bootstrap/model.tar.gz --root /state
```

安装在周期锁内完成，忙时明确退出，稍后重试。当前活动继续使用原权重和校正快照；新活动使用新版本。旧权重与活动 pin 保留，恢复旧版本可重新安装已保存的旧 Release。不要手工覆盖 `versions/` 内文件或删除仍被活动引用的权重。

模型包与镜像的固定算法及依赖必须匹配。更换服务器或 CPU 架构时通过模型工作流重新执行技术重放；不跳过原有源码、软件、数据政策与期初约束检查。镜像保留历史来源文件以匹配 Mac 既有指纹，通过安全启动入口先加载第三方 XGBoost，避免旧同名脚本遮蔽库或创建数据库。

## 本地验收

```bash
docker compose config --quiet
docker build -t bangdream-web:test .
docker run --rm --cpus=1 --memory=2g --network=none bangdream-web:test test
.venv/bin/python -m unittest discover -s tests -v
```

历史测试队列曾参与探索，验证被重复使用，元数据为回溯快照；模型包中的拟合活动数和技术重放均不是新的准确率证据。部署不改变冻结研究队列和既有结果。
