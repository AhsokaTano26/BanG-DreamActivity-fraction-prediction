# V2 本地基础模型可行性核查（2026-11-14）

三类模型均已实际下载官方公开权重并完成本地离线推理，没有使用云推理、付费服务或替代算法。本文记录实现、资源预算与合成试跑；正式验证/测试结果由独立运行器和评分器生成。本文的试跑没有读取验证或测试答案，也没有以真实训练答案挑选配方。

## 固定的软件与权重

项目 `.venv` 保留 Python 3.14.8、torch 2.15.0、NumPy 2.5.4、pandas 3.1.1、scikit-learn 1.9.1。新增依赖由主任务统一安装：`tabpfn==7.1.1`、`chronos-forecasting==2.3.2`、`timesfm==2.0.2`，解析得到 transformers 5.23.0、huggingface-hub 2.4.0、safetensors 0.8.0。完整实际依赖版本由 `provenance()` 输出并由运行器保存。

选择 TabPFN 7.1.1 是为了兼容现有 scikit-learn/pandas；旧 2.2.1 包要求 scikit-learn <1.7、pandas <3。包版本与模型版本分开记录：本实验显式传入 **TabPFN v2** checkpoint，绝不使用包默认的新版模型。[TabPFN 官方代码](https://github.com/PriorLabs/TabPFN)、[7.1.1 包元数据](https://pypi.org/pypi/tabpfn/7.1.1/json)、[2.2.1 包元数据](https://pypi.org/pypi/tabpfn/2.2.1/json)。

| 模型 | 官方仓库 / 固定 revision | 本地权重文件 | 文件字节数 |
|---|---|---|---:|
| TabPFN v2 回归 | `Prior-Labs/TabPFN-v2-reg` / `02a44705ff7790b50cded404bc59adb94da456a1` | `tabpfn_v2/tabpfn-v2-regressor.ckpt` | 44,390,977 |
| Chronos-2 | `amazon/chronos-2` / `29ec3766d36d6f73f0696f85560a422f50e8498c` | `chronos2/model.safetensors` | 477,930,472 |
| TimesFM 2.5 200M PyTorch | `google/timesfm-2.5-200m-pytorch` / `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` | `timesfm25/model.safetensors` | 925,181,104 |

以上路径均相对于 `analysis/foundation_assets/20261114/`。权重、`config.json`、README 和可用许可证已下载；`manifest.json` 与各目录 `asset_manifest.json` 保存来源 URL、字节数及 SHA256。三个官方模型 API 均返回 `gated=false`，实际下载 URL 返回 HTTP 200。权重 SHA256 另与官方 Hugging Face tree API 的 LFS oid 逐一比对：

- TabPFN：`2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`。
- Chronos-2：`ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`。
- TimesFM：`2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8`。

来源：[TabPFN v2 官方权重](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/tree/02a44705ff7790b50cded404bc59adb94da456a1)、[Chronos-2 官方权重](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c)、[TimesFM 2.5 官方权重](https://huggingface.co/google/timesfm-2.5-200m-pytorch/tree/1d952420fba87f3c6dee4f240de0f1a0fbc790e3)。

## 预登记配方与权重含义

共四个配方：`foundation_tabpfn_v2`、`foundation_tabpfn_v2_context`、`foundation_chronos2`、`foundation_timesfm25`。前两个比较基础 51 维前缀特征与同一基础特征加训练内拟合的活动类型/乐队份额编码；后两个使用相同累计分数时间序列接口。没有验证后扩展模型菜单。

TabPFN 的 `fit(X, y)` 不接受 `sample_weight`。本实验将每个活动的上下文行数补齐到训练池内最大活动前缀数。所有原始前缀至少出现一次；完整循环重复后，余数由相应种子不放回抽取，最后打乱。这样每个活动上下文总质量完全相同，每个活动内的前缀重复数最多相差一次，且不声称重复行增加了独立样本量。

在已校验的 V2 两档训练集上，2,229 个前缀来自 196 个实际拟合活动，单活动最多 38 个前缀；完整上下文为 7,448 行，其中重复 5,219 行。内部时间折重新按其合法训练池计算等量上下文。三个种子固定为 42、123、2026，每种子 `n_estimators=1`、`fit_mode='fit_with_cache'`、MPS 设备。任务目标为 `log(final/current)`，预测解码沿用固定 `[0,4]` 对数增长范围。最终集成计算三个**最终分数**的算术均值。

这些是三个训练上下文/随机预处理成员，预训练神经网络本身没有更新权重。训练集预测查询包含自身的答案行，因此是上下文内重代入诊断，不能作为 OOF 校准残差，也不能与独立时间折的误差作等价解释。类型与乐队词表只在训练池拟合；查询中新类别或缺失类别的组效果为零，沿用既有 `ContextDesign` 的定义。

Chronos-2 与 TimesFM 为零样本对照，不读取传入训练标签，不训练任务参数，不使用历史最终分数。三个成员是同一确定性预测的重复记录，报告中标为推理种子，不能称为三次独立训练。按照预登记方案省略它们的训练内 CV；完整训练/验证/测试前缀仍各自保留全案例预测。[Chronos 官方接口](https://github.com/amazon-science/chronos-forecasting)、[TimesFM 官方接口](https://github.com/google-research/timesfm)。

## 前缀时间适配与输出

通过 `TemporalDataset` 校验冻结指纹 `1f76f39af8151bba185496411bbcee064b16a4c5aca5c515c1d957a5056fe854` 后，仅检查训练输入确定范围：活动时长 129.98–226 小时，实际观测后剩余 0.25–178.10 小时。固定基础步长为 2 小时，上下文和预测长度均最多 256。

网格锚定实际最后观测 `now`，仅在已给出的前缀观测之间插值，以当前分数归一化。不会把滞后期间补成新观测，不以请求 cutoff 取代实际观测时刻。逐案例保存观测年龄、最大原始间隔、落入超过 6 小时间隔的网格点数、有效步长、上下文长度和预测步数。缺失较多、平台期、零增长和短剩余时长案例均保留。若出现超长活动，按预登记规则把步长增加为 2 小时的整数倍，使上下文跨度和整个预测终点都能在长度上限内覆盖，而非丢弃案例。

Chronos 使用 `predict_quantiles(..., quantile_levels=[0.5], cross_learning=False)`，不允许不同查询案例相互学习。TimesFM 2.0.2 的 PyTorch 实现明确返回 `full_forecast[..., 5]` 为 point forecast，对应其 0.1–0.9 分位序列的中位数；因此两者输出统计量均记录为中位累计预测。TimesFM 设置 `torch_compile=False`、固定 CPU、batch 8、归一化输入、不额外做 flip ensemble。两者未来累计预测经过不低于当前分数的单调投影，再线性插值到活动已知的准确结束时刻；若输出非有限数或案例数不一致，显式失败，不悄悄删除案例或换算法。

## 本地实测与重放

首次 smoke 使用合成序列、CPU 两线程：Chronos-2 加载约 0.11 秒，两个序列预测 96 步约 0.915 秒；TimesFM 加载约 1.38 秒，同等查询约 0.085 秒。该结果仅表明可运行，不是全队列耗时保证。

TabPFN 使用 7,448×32 合成上下文：CPU 首次 32 条预测约 72.1 秒；支持的 MPS 缓存模式上下文拟合约 8.59 秒、32 条预测约 0.50 秒、256 条预测约 0.42 秒。实际 2,229×51 训练盲前缀形状加合成目标的补充 smoke 中，三个等量上下文拟合合计约 36.09 秒，三个成员的 32 条预测（含上下文重建）约 35.01 秒。没有读取真实训练终分。该 smoke 的上下文审计、时间和依赖记录见 `analysis/foundation_assets/20261114/smoke_training_shape.json`。

只保留一个进程内模型缓存，减少多个 MPS 上下文同时占用的内存。joblib 保存预处理、训练上下文、种子抽样索引、检查点身份和运行诊断，引用上述本地权重，不重复嵌入大权重。完整基础特征上下文 bundle 约 1.18 MB；移动仓库时需一起保留 `analysis/foundation_assets/20261114`。重载后改变查询 batch 长度的最大相对差约 `1.38e-6`，来源于 MPS 浮点计算，因此正式重放容差固定为 `rtol=atol=1e-5`，而非宣称按位相同。

推理前显式设置官方 `TABPFN_DISABLE_TELEMETRY=1`、`HF_HUB_OFFLINE=1`、`HF_HUB_DISABLE_TELEMETRY=1`、`TRANSFORMERS_OFFLINE=1`。真实集成测试在子进程内禁止 `socket.connect`，执行全部四种配方、三成员预测及 joblib 重放，且在导入后先开启 `torch.use_deterministic_algorithms(True)`，模拟此前神经模型留下的进程状态。该测试通过，不需要修改轮子、设置 `KMP_DUPLICATE_LIB_OK` 或关闭既有确定性设置。

本模块的 11 项聚焦测试通过。随后执行 `.venv/bin/python -m unittest discover -s tests -v`，209 项通过，耗时 101.850 秒；临时完整日志为 `/tmp/v2_foundation_full_suite.log`。旧 SQLite fixture 测试出现未关闭连接的 `ResourceWarning`，没有失败用例。

## 解释边界

这些可行性证据不能说明基础模型优于现有模型。固定验证只用于选择已登记配方与组合；正式测试预测必须先封存，再由独立评分器读取测试答案。重复上下文没有增加独立活动数，零样本重复种子也没有增加独立模型数。

旧测试队列曾用于探索，不能称为完全未见过的盲测；元数据是缺少历史时点存档的当前回溯快照；固定验证集也已经多轮使用。这些限制必须在最终结果报告中继续保留。
