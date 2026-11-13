# 独立审查及修复证据

只读审查员 `/root/review_temporal_paired_tcn` 审查新的 features/network/benchmark/tests/plan，未读取真实测试答案或成绩。

两个重要问题均在正式训练前修复：

1. 三层卷积末状态仅能看到最后15点。新增 `test_network_can_use_the_early_part_of_both_prefixes` 先失败，后将末状态与全前缀均值一起汇总，测试通过，配置同步记录。审查员确认此项关闭。
2. 同请求时点的对方前缀截到当前实际观测时刻可能为空。新增 `test_empty_or_short_peer_after_truncation_remains_missing` 用空和1点两个合法原始前缀复现 `snapshot(None)`，随后增加空值分支，将其按缺失保留，测试通过。

审查其余核对通过：同split/活动/请求时点配对、未来后缀截断、案例保留、活动等权及阶段控制、训练集拟合、验证集选checkpoint/方法、三种子最终分数均值、单种子排除候选、独立测试评分边界、不可覆盖封存及代码/版本重放。

正式运行前全仓库命令 `.venv/bin/python -m unittest discover -s tests -v` 107项通过（73.350s）。日志 `verification-tests.log` 保留旧测试夹具的 SQLite ResourceWarning；没有测试失败。

运行后验证：所有6个可重放输出（两种新树、三个神经种子、种子预测均值）在完整测试盲前缀上重放一致；全部9方法使用相同冻结案例；活动及活动内阶段权重符合规定；旧数据库、V1/V2及旧封存结果不变。详见 verification.json、verification-source.py、verification.log。

补充 comparison/ 独立评分只对已固定模型进行总体配对活动bootstrap，另有评分ID和封存，未改变模型、选择或主评分产物。
