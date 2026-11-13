# 独立审查与最终证据

用户要求验证已封存新组合。audit_new_activity_data agent负责只读primary API档案与终分质量；audit_validation_scope agent负责模型/预测时间来源、队列外资格和新增代码审查。二者未编辑旧资产或参与训练。

审查发现的三个问题在正式运行前修复并有RED→GREEN证据：

1. 三个模型必须与新数据的parent_dataset_id一致，并保存各自路径、run_id与seal SHA。
2. 数据构建保存实际执行的规范化/快照/前缀/时间网格等传递源码与版本；历史零分允许，当前前缀仍须正分。
3. 旧实时快照使用旧封存起止时间和类型，并核对原预测的观测时刻及当前分数，避免新版元数据改变历史输入。

主/独立评分输出可另存--output审计目录。新模型及V2两个基线均晚于327/328结束，明确是额外活动回放；只有Nov11 19:21旧structured_neural记录属于提前封存，未混用。

新模块7项边界测试；全套133项通过，77.530s，日志verification-tests.log。旧夹具ResourceWarning仍保留，无测试失败。正式predict在outcomes、sqlite3.connect、CatBoost.fit禁止守卫下成功封存（prediction.log），无训练/重选。

最终verification.json证据：原始响应重算终分一致；327不被冒充已评分活动；缓存不读答案；74个网格前缀、2个旧快照在三个方法间完全一致；复制保存模型重放一致；另一线不晚于当前已观测时刻；父数据/模型身份/源码/旧数据库/旧封存结果全部校验。未运行后台定时采集。
