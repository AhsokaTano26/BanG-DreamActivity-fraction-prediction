# 多 agent 分工、交叉审查与验证

用户明确要求多agent并行测试。三个agent均仅使用训练/验证：

- `/root/search_remaining_targets`：6个目标/损失配方；新增 research_remaining.py、5项专项测试及开发封存。
- `/root/search_fusion`：6个双线稳健融合配方；新增 research_fusion.py、5项专项测试、开发封存及说明。
- `/root/search_shift_calibration`：6个时期/排名/OOF校准配方；新增 research_shift.py、5项专项测试、开发封存及说明。

各agent不共享编辑文件。主agent负责research_search统一入口、组合规则、盲预测、独立评分与交付。

交叉只读审查：remaining审查fusion及search的训练边界、活动权重和验证选择；fusion审查shift的OOF已结束活动时序及search封存顺序。阶段路由在统一正式训练/测试前补充，并由remaining额外审查；固定25%/90%边界，不根据测试改头。未发现未关闭实质问题。

统一入口专项测试4项通过，完整 `.venv/bin/python -m unittest discover -s tests -v` 126项通过，78.074s。并行开发早期其他模块尚未创建时有临时测试失败，最终集成全部通过。完整日志见verification-tests.log。未选中的Ridge曾发出近常量方差sqrt警告，但全部预测有限且专项持久化测试通过；既有夹具保留SQLite ResourceWarning。

正式运行前冻结全部18配方、7组合、5参考；选择由验证集最低活动宏MAPE决定，三方向最终分数均值被选中。其后才打开test前缀，独立评分器最后读取测试标签。未将测试最低候选替换为主方法。

运行后独立核查：30方法全部案例与分组相同，5参考预测逐案例不变；18模型与7组合在完整test盲前缀上重放一致；每配方196个活动总权重相等；OOF校准全部折的训练最晚结束严格早于留出最早开始；主评分及比较评分缓存不再次读取答案；V1、V2、数据库与旧结果摘要保持不变。证据见verification.json及verification.log。

没有声称本轮稳定显著改善：对上轮双线的95%活动配对区间仍略跨0，末段时间点有退步；完整结果公开保留。
