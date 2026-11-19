# 训练与评测

所有命令在仓库根目录执行。`uv sync --locked` 安装固定生产配方需要的依赖；完整研究和神经网络使用 `uv sync --locked --group research`。默认网页环境不安装 PyTorch、LightGBM 和基础模型包。

## 生产模型训练

```bash
uv run --locked training/train.py --help
uv run --locked training/train.py fit \
  --database data/db.sqlite3 --metadata data/metadata \
  --before '2026-11-13T13:00:00+08:00' \
  --output training/models/versions
```

`--before` 是严格反馈截止时间，建议使用将要预测的活动开始时刻。日期统一按实际国服开始时间筛选，最早为2024-01-29 13:00（北京时间），仅使用截止前已结束且通过质量校验的历史。默认拟合19线；重复使用 `--rank 1000 --rank 2000` 可限定范围。原始数据库只读打开，终分校正造成当前分数高于终分的前缀保留原有训练质量记录。

也可以使用网页已积累的历史：

```bash
uv run --locked training/train.py fit --state runtime/live \
  --before '2026-11-13T13:00:00+08:00' --output training/models/versions
```

网页缓存可能只覆盖最近活动。训练前应确认历史覆盖，优先使用完整原始历史库。输出 JSON 包含模型路径、身份、活动数及支持排名。已有内容身份相同的包直接校验并复用，其他运行输出到新目录。拟合过程不切换网页当前权重，当前活动的期初绑定仍有效。

`model_fit.py` 包含实际 CatBoost 拟合、保存、支持度与技术重放流程。`production.model.train_pack` 为旧代码保留延迟调用接口；网页入口关闭自动拟合。隐藏的 fixture 轮数参数仅供测试，生成的包明确标记为非生产包，发布和激活均拒绝它。

训练结束后可直接打包新模型，校验后再显式安装到网页状态目录：

```bash
uv run --locked training/train.py package \
  --model 'training/models/versions/<model_id>' --output /tmp/new-model-release/model.tar.gz \
  --state runtime/live
uv run --locked python -m production.model_release install \
  /tmp/new-model-release/model.tar.gz --root runtime/live
```

打包时 `--state` 用于附带已有合法误差反馈，可以省略。发布到 GitHub 仍使用独立模型工作流；这些本地命令不执行远端发布。

## 固定实验与独立评分

```bash
uv run --locked --group research training/train.py benchmark \
  --protocol v3 --run training/runs/new_v3

# 预测封存后单独评分
uv run --locked --group research training/train.py evaluate \
  --protocol v3 --run training/runs/new_v3
```

协议取 `v1`、`v2`、`v3`，均进入当前密集采样队列入口，97活动的67/14/16归属保持冻结。V1/V2固定训练历史；V3每期只使用开始前已结束的活动，预测先于反馈。测试队列已被历史探索，验证重复使用及回溯元数据限制持续适用。评测目录禁止覆盖，训练成绩与测试成绩分别解释。

保留的完整研究模块位于 `analysis/` 子目录，安装后沿用 `analysis.*` 名称以兼容已保存对象。其他入口可通过：

```bash
uv run --locked --group research training/train.py module analysis.multiline_runner --help
uv run --locked --group research python -m analysis.current_benchmark --help
```

数据集构建与读取完整答案的准备命令仍属于可信准备阶段。模型入口、独立评分器和前瞻账本之间的信息隔离遵循根目录 [AGENTS.md](../AGENTS.md)。

## 可选基础模型研究环境

原先的基础模型依赖清单位于 `legacy/requirements-foundation.txt`，作为旧环境记录保留。需要复现该方向时，由 uv 添加明确版本到隔离环境，并记录实际软件清单：

```bash
uv run --locked --group research --with tabpfn==7.1.1 \
  --with chronos-forecasting==2.3.2 --with timesfm==2.0.2 \
  python -m analysis.v2_next_foundation --help
```

该命令使用独立临时环境，不改变生产锁文件。权重下载与运行要求仍按对应方法的封存配置处理。

## 记录与测试

- `legacy/` 保存原根脚本，依靠安装映射保留旧导入名；日常服务使用根 `app.py`。
- `research_archive/` 保存轻量设计、预登记、脚本和审计记录；完整原始运行工件仍在本地旧 `analysis/` 路径，字节不变。
- `models/`、`runs/` 为本地输出，已忽略。
- 历史实验及负面结果见 [RESEARCH_HISTORY.md](RESEARCH_HISTORY.md)，模型能力摘要见根 [README](../README.md)。

```bash
uv run --locked --group research python -m unittest discover -s tests -v
```

macOS 如需 OpenMP 动态库，设置 `DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib`。测试及生产计算按任务串行执行；保持 PyTorch 集成测试的子进程隔离。
