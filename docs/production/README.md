# 自动预测与增量训练

日常运行方式已于2026-11-19更新：`forecast.py --rank 1000`执行一次预测，`forecast.py serve`手动启动持续预测与网页。原系统定时任务已暂停。请使用[运行指南](running.md)；本文下方保留早期独立EWMA服务的实现记录，当前选择配方见[selected-model.md](selected-model.md)。

这是本地可运行的生产模型与自动更新服务。默认覆盖19档排名，T1000/T2000使用400树Context模型＋收缩EWMA，其余排名使用600树共享排名模型。运行依赖项目现有`.venv`，不需要另装神经网络或调用收费预测API。

## 交付位置

- `runtime/live/models/current.json`：当前已准备的模型版本与完整目录。
- `runtime/live/models/versions/<model_id>/context.joblib`：T1000/T2000已训练模型。
- 同目录`shared.joblib`：19档共享模型；`manifest.json`、`metadata.joblib`、`training_weights.json`及`training_quality.json`保存来源、预处理、权重和过滤记录。
- `runtime/live/latest.json` / `latest.csv`：最近一轮当前活动预测。
- `runtime/live/status.json`：运行阶段、错误、所用模型和更新状态。
- `runtime/live/state.sqlite3`：实际发出的预测、不可覆盖的修订历史、每期固定模型与已结束活动的误差反馈。
- `runtime/live/data.sqlite3`、`metadata/`、`snapshots/`：生产专用数据、元数据和原始响应存档。

`current.json`是已经准备好的模型；某活动实际使用的版本以`latest.json.active_event_models`及每条预测的`model_id`为准。同一期固定版本和校正快照，下一期才使用更新后的模型。模型损坏时可使用技术检查通过且时间合法的旧版本，旧pin与替换原因保存在`pin_history`。

## 自动流程

每轮刷新国服活动目录，下载当前活动和近期尚未确认终分的曲线；角色与乐队字典每日更新。接口错误保留上次可用数据并记录，下一轮重试。已确认结束的排名曲线不重复抓取。

新前缀直接用于当前活动预测；仅在活动已经结束、真实终分通过质量检查后，才用于监督训练。模型训练仍从2024-01-29 13:00国服实际开始时间起，只用本期开始前已结束的历史。相同训练数据和配置复用已有模型；有新的合格历史时生成不可覆盖的新版本。训练/验证失败保留旧版本。

在线校正使用真正发出的未校正预测与真实终分之比。每期、每个排名、每个阶段只贡献一个平均误差；同一观测重复轮询不增加权重。源站修正同一时间的分数时会保存预测修订，反馈只使用该时间点最后实际发出的修订。已结束曲线的补录可重建按结束时间排序的EWMA，不提前使用本期结果。

初始校正状态来自此前已封存V3因果预测与本地已结束真实曲线，明确标为`retrospective_v3_seed`。`live_feedback_groups`代表部署后真正完成的新活动反馈；部署之初为0是正常的。当前活动结束后才会增加。不能把回溯种子称为新的前瞻验证。

## 使用命令

在仓库目录执行：

```bash
# 初始化；已交付的runtime/live已有配置和模型，通常无需再初始化
.venv/bin/python -m production.cli init

# 立即更新数据、检查是否需要训练并预测一轮
.venv/bin/python -m production.cli cycle

# 只用当前本地缓存运行一轮
.venv/bin/python -m production.cli cycle --offline

# 查看模型、最后运行及后台定时状态
.venv/bin/python -m production.cli status

# macOS安装并启动每15分钟运行一次的用户后台任务
.venv/bin/python -m production.cli install

# 停止自动任务，保留全部模型和数据
.venv/bin/python -m production.cli uninstall

# 非macOS或前台连续运行；Ctrl+C停止
.venv/bin/python -m production.cli run
```

配置在`runtime/live/config.json`；默认15分钟。初次创建自定义目录/排名可用：

```bash
.venv/bin/python -m production.cli init --root /absolute/path/live --ranks 1000,2000 --interval 900
.venv/bin/python -m production.cli cycle --config /absolute/path/live/config.json
.venv/bin/python -m production.cli install --config /absolute/path/live/config.json
```

修改已有`ranks`需要明确编辑配置；现有活动pin保留，新增排名若未在当前版本训练会标记不可用。间隔修改后重新执行`install`更新定时配置。

## 版本回退

```bash
.venv/bin/python -m production.cli rollback --model-id <已保存的model_id>
.venv/bin/python -m production.cli cycle

# 恢复自动训练和模型切换
.venv/bin/python -m production.cli resume-auto
```

手动回退会暂停自动训练/切换，`automatic_model_updates=false`会显示在状态中；数据获取和预测继续。恢复后重新检查新数据，当前已固定活动继续保持其pin，下期使用新的合格模型。

## 运行边界

macOS任务标签为`com.tano.bangdream-forecast`，配置在`~/Library/LaunchAgents/com.tano.bangdream-forecast.plist`。它在这台Mac登录且清醒、联网时运行；关机/休眠期间不会进行预测，并不等同于云端24小时服务。登录时自动启动，新活动目录会在后续同步时发现；曲线自动补抓目前限于活动结束后的14天，长期离线后更早的缺失曲线需另外补抓。日志在`runtime/live/scheduler.stdout.log`、`scheduler.stderr.log`及`cycles.jsonl`。[Apple关于用户launchd任务的说明](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)。

当前模型的T1500历史活动数较少，会标记`provisional=true`并给出覆盖原因。观测不足或滞后超过60分钟时，输出不可用，避免把旧数据冒充当前预测。

生产训练对晚开始采集的曲线跳过没有观测的早期时点；对封号/排名修正等造成的“历史当前分数高于真实终分”的前缀，跳过模型无法拟合的监督样本并保留原曲线/真实终分，过滤详情有审计记录。原固定V1/V2/V3名单与评分均未改变。

此前实验队列被探索过，元数据为回溯快照。新生产模型使用更多已结束历史，不能拿其训练拟合误差冒充冻结测试准确率。以后持续发出的预测及终分反馈会形成真正的前瞻记录。
