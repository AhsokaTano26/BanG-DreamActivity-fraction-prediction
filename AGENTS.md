# 固定评测规范

本项目从 2026-11-11 起，主要模型比较统一使用 `data/holdout_v1/`，不得用其他切分替换后声称改进。本文件适用于整个仓库。

- 数据集指纹：`d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb`。读取时使用 `analysis.holdout_dataset.HoldoutDataset` 校验全部文件。
- 全库符合质量条件的 283 个活动按国服实际开始时间，分为训练 198、验证 42、测试 43；两个排名档和全部快照以活动 ID 成组。具体名单见 `manifest.json`、`partitions.csv`，不要用活动 ID 数值范围重新划分。
- 训练输入 4377 个前缀；验证固定 414 个、测试固定 412 个。评测时点为 25%、50%、75%、结束前 24h/6h。质量过滤和缺失时点已经冻结，见 `excluded_curves.csv`、`unavailable_cases.csv`；所有方法使用相同名单，不能各自删样本。
- 训练、插补、标准化、类别编码、历史活动最终分数统计只能使用训练集。训练样本的历史统计也必须排除其本期及开始时尚未结束的活动。验证集只用于选参数/训练轮数/组合成员；测试时不重训，不把验证/测试活动的最终分数追加到热度历史。
- 模型入口只能读取盲前缀，不能读取 SQLite 快照或验证/测试完整曲线。另一排名档只能使用截至当前已观测时刻的前缀。卡牌必须按国服发布时间过滤。
- `train_curves.jsonl` 是唯一可用于完整历史曲线统计的文件，`*_inputs.jsonl` 是无目标值的前缀，`*_labels.csv` 是独立答案。测试标签只能由独立最终评分器读取，不得进入早停、特征工程或模型选择。
- 主指标为 **event-macro MAPE（每个活动等权，单位 %，越低越好）**；同时保留 case-MAPE、MAE、RMSE、P90，以及按排名、时点、活动类型、乐队的明细。独立样本量按活动计，不能把快照当独立活动。训练也按每个活动相同总权重。
- 神经网络固定随机种子 42、123、2026，报告逐种子和三种子平均最终分数集成。模型、预处理、代码、依赖版本和预测都需要保存指纹。新增模型沿用此协议，在新运行目录保存结果，不覆盖旧实验。
- 主入口：`.venv/bin/python -m analysis.holdout_benchmark --run analysis/holdout_results/<new-run>`；完成盲预测后用 `.venv/bin/python -m analysis.holdout_evaluate --run analysis/holdout_results/<new-run>` 独立评分。可用 `--resume` 恢复完全相同代码/配置；已封存预测不得重算覆盖。
- 评分另有 `evaluation_id`，记录实际执行的评分源码和依赖，`evaluation_manifest.json` 校验全部结果。相同版本重复评分只返回已封存结果，不重读答案；评分器变化必须用 `--output` 另存审计目录，不能覆盖旧结果。配对活动 bootstrap 也必须使用冻结 case 的活动归属。
- `holdout_v1` 一经生成不得覆盖、重建或随数据库更新移动分组。更新 `data/db.sqlite3` 和 `data/metadata` 不影响 v1。新增活动用于额外前瞻验证，或经明确说明创建新版本，仍保留 v1 作为对照。
- 该测试队列曾出现在早期探索实验中，不能称为完全未见过的盲测。元数据也是当前回溯快照，缺少历史时点存档。每次报告都保留这两个限制；测试分数不得用于反复试参数后择优汇报。
- `analysis/backtest.py`、`advanced_benchmark.py`、`context_benchmark.py`、`context_ablation.py`、`transfer_benchmark.py` 是早期探索入口，其滚动重训及旧曲线过滤口径不同，不能替代标准结果。生产 CLI 可以学习当前时刻以前的历史，但不能把生产拟合误差当固定测试结果。

使用项目内 `.venv/bin/python`。执行 `.venv/bin/python -m unittest discover -s tests -v`。macOS 混用 PyTorch CPU 和树模型时可设置 `DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib`；不要设置 `KMP_DUPLICATE_LIB_OK`，不要修改已安装的轮子。PyTorch 相关集成测试保持子进程隔离。

## V2 时间评测（2026-11-12 用户指定）

以上既有协议保留为 V1；V1 数据、分组、评分和结果不得覆盖。本次新增时间
评测协议位于 `docs/superpowers/specs/2026-11-12-temporal-holdout-v2.md`，数据
目录为 `data/temporal_holdout_v2/`。旧 `data/multirank_v2` 是19档数据处理的
修订号，不等同于本次评测协议 V2。

V2 沿用固定活动归属及合格曲线队列，在训练/验证/测试三个活动组使用相同的
开始后时长、活动进度、距结束时长网格，按时间点×排名报告误差与活动数。
相同活动/排名/预测时刻只存一个案例，多轴重复时点通过成员关系引用；总体
指标不重复计数。记录实际观测时刻、滞后、缺失覆盖与同活动同排名共同面板。
主评测仍禁止加入验证/测试最终分数重训；生产滚动历史须另设协议，不能混报。

新入口 `analysis.temporal_dataset` / `analysis.temporal_benchmark` /
`analysis.temporal_evaluate` 使用独立目录、盲前缀和独立封存评分。首轮固定
原模型，只改变评测时间网格；不把V1/V2不同时间点的汇总分数差称为模型改进。

### V2 两档完整训练（2026-11-12 用户指定）

`data/temporal_holdout_v2_t1000_t2000/` 是父 V2 按排名提取的冻结子集，
不重新划分活动。入口 `analysis.temporal_rank_subset` 仅用于可信数据准备，
`analysis.temporal_train` 用全部可用训练前缀拟合固定配方、仅由验证集选择；
之后调用 `analysis.temporal_evaluate` 独立评分。报告使用
`analysis.temporal_training_report`，不能沿用首轮“没有重训”的报告措辞。
首个运行目录为 `analysis/temporal_results/v2_t1000_t2000_full_20261112/`。
旧 19 档模型仅作为同案例描述性对照，不混入本轮候选选择。
两档子集指纹为 `1f76f39af8151bba185496411bbcee064b16a4c5aca5c515c1d957a5056fe854`，
训练/验证/测试可用前缀分别为 2229/1548/1514；训练组活动 57、68 在该网格没有
合格前缀，实际拟合 196 个活动，不能移动其分组或为不同方法另删案例。
