# E 执行记录

- `statistics_001/`：首次prediction阶段，B/C表生成后，读取相对路径的C2汇总触发路径校验异常，退出1。该目录留存，未封存，正式报告引用002。
- 修复范围：CSV/JSON来源路径先resolve，再相对仓库根记录SHA。相对路径夹具先红后绿；指标公式、候选和抽样参数保持。
- `statistics_002/`：正式prediction阶段，21.361366584秒，2026-11-18T15:40:47.506279+00:00结束，退出0。真实源码为目录内`scorer_source.py`及`archive/statistics_audit_stage1_v2.py`，SHA `83cf92039704f33c7132a720cbbb176b14093d4a374f0e50410f975f015c559b`。
- 运行调度消息发生交叉：002在PI后发暂停消息前已按前次两阶段许可恢复，并已在暂停被本角色读取前完成。获知D/H时段后立即释放CPU，第二阶段未接续。是否存在实际时间重叠以各角色进程时间为准，消息顺序留作说明。
- 第二阶段实现改成显式`--passed-manifest`，先校验指定成功bundle全文件SHA，再取TRAIN35活动差。新增篡改夹具先红后绿；9个fixture均通过0.018秒。当前第二阶段源码SHA `0d0f4db135a50b611159eaf3888394a22d6c89c055127c856d6c72ea618860d6`。
- 第二阶段首运行`reliability_001/`：1.135777秒，运行完成后发现SciPy直接lower-CDF在极尾返回NaN，部分功效值及最小n搜索受影响。保留该封存目录，并明确以002修订。NCT对称生存函数公式修复，极尾fixture先红后绿；数值NaN没有补为0。
- 正式第二阶段`reliability_002/`：1.140430秒，2026-11-18T15:48:21.549852+00:00结束，退出0。源码SHA `a7400e0abc38d3cb74f1b0cfb9fabc7e30519829ab1c1d1453a08698810e750c`，快照也另存`archive/statistics_audit_stage2_v2.py`。12个fixture通过0.019秒。包含极尾公式修复和HAC/无界宽度明确处理；两阶段已释放数值CPU。
- PI在未激活新队列、无未来成绩时补充80期采集预算，单独存`confirmation_plan_addendum.json`，引用002计划SHA；002源码、分数与manifest保持封存。原registry_v2的20期继续为pilot。
- 第一阶段不再重跑，原实验、数据与生产保持原样。仓库全量unittest由PI串行安排；本角色仅声明12个E夹具结果。
