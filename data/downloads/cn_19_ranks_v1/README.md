# 国服19档活动分数线：本地下载清单

已逐一检查 **6,232 个活动×排名组合**，范围为元数据中截至下载开始已在国服开放的 328 个活动。名单依据国服实际开始时间，不以活动ID数值范围替代。

本次取得 3,085 条非空曲线；3,145 组接口有效返回空记录；2 组响应未通过结构/值域校验；剩余网络失败 0。下载开始时刻（UTC）为 `2026-11-11T12:47:22.066909+00:00`，各组实际抓取时刻保存在 `records.jsonl`，并非同一秒的同步快照。

导入后，`data/db.sqlite3` 共含 3,085 行活动排名记录，其中19档的非空活动曲线 3,085 条、标准化分数观测点 **3,646,891 个**。一次T10完整榜单快照在数据库中计作一个分数线观测点，不把前十玩家的记录计作十个独立分数线样本。

| 排名档 | 本次非空活动曲线 | 接口空记录 | 数据库非空活动曲线 | 数据库观测点 |
|---|---:|---:|---:|---:|
| T10 | 264 | 64 | 264 | 2,912,022 |
| T20 | 103 | 225 | 103 | 42,064 |
| T30 | 103 | 225 | 103 | 42,062 |
| T40 | 103 | 225 | 103 | 42,060 |
| T50 | 219 | 109 | 219 | 48,215 |
| T100 | 327 | 0 | 327 | 50,735 |
| T200 | 103 | 225 | 103 | 42,056 |
| T300 | 219 | 109 | 219 | 46,352 |
| T400 | 103 | 225 | 103 | 42,053 |
| T500 | 218 | 109 | 218 | 46,054 |
| T1000 | 328 | 0 | 328 | 50,006 |
| T1500 | 17 | 311 | 17 | 6,465 |
| T2000 | 328 | 0 | 328 | 45,172 |
| T3000 | 103 | 225 | 103 | 38,881 |
| T4000 | 103 | 225 | 103 | 38,867 |
| T5000 | 135 | 193 | 135 | 39,114 |
| T10000 | 103 | 225 | 103 | 38,735 |
| T20000 | 103 | 225 | 103 | 38,364 |
| T30000 | 103 | 225 | 103 | 37,614 |

## 保存位置

- 主库：[项目数据库](../../db.sqlite3)。按 `Country='中国大陆' AND Rank=<排名>` 查询。
- 原始响应：`raw/` 下的 gzip JSON，保留源接口返回，带逐文件指纹。
- 标准化曲线：`curves/<活动ID>_<排名>.json`，仅含时间戳与分数等曲线字段。
- 元数据：[events.json](events.json)；完整状态：[records.jsonl](records.jsonl)。
- 每档覆盖：[audit/coverage.csv](audit/coverage.csv)；每条曲线质量：[audit/curve_quality.csv](audit/curve_quality.csv)。
- SQLite导入前备份：`/Users/tano/Documents/GitHub/personal/BanG-Dream-Activity-fraction-prediction/data/backups/db_before_cn_19_ranks_v1.sqlite3`。原库已有的时间点保留，重叠时间点使用此次接口返回的修订值。

新增数据库曲线 2,429 条，合并已有曲线 656 条，新增时间点 3,551,841 个，修订重叠时间点 0 个；详见 [database_changes.csv](database_changes.csv)。空响应和无效响应不会删除或清空已有曲线，因此本次非空下载数量与数据库非空数量可能不同。

## T10口径与质量检查

普通分数线接口不接受T10。本次用 `eventtop/data?server=3&event=<ID>&mid=0&interval=0` 取得榜单历史，在同一时间戳至少有10个不同玩家时取第10高分；不沿用最终第十名玩家的个人曲线，不为缺失快照补齐玩家。独立审计重新从原始榜单计算并核对了 2,912,022 个T10完整快照。

其余18档使用 `tracker/data?server=3&event=<ID>&tier=<rank>`。原始响应有效返回空列表只代表该源在抓取时未提供该组历史，不等于分数为零。网络503有有限重试，并在后续续传中补齐；观察到错误增多后降低了请求速率，运行调整记录在 [fetch_sessions.json](fetch_sessions.json)。

分数下降没有被静默改成累计最大值：本次审计发现 54 条已导入曲线有下降段，保留来源数据并标记，后续建模需检查。本次导入曲线中的实际零分记录 0 个，同样保留并与缺失区分。结束附近观测、起始前/结束后观测及当前进行中活动均有单独质量字段；不能把所有曲线直接当作合格训练样本或完整最终标签。

所有已导入点均独立核对“本次原始数据＋导入前备份”的合并结果；逻辑键唯一、SQLite完整性检查通过。下载可断点续传；导入使用一致性内容指纹、备份和写事务，已验证WAL变更拒绝及提交后收据写入异常恢复。未声称经过断电耐久性测试。

## 评测与下一步

冻结 `data/holdout_v1/` 指纹仍为 `d0ccf9f70aee19ad1b2152925e7afe8692d67f4a6e539d558f87e2cd33329bbb`，没有覆盖、移动分组或重新评分。本次是数据采集与入库；未训练19档模型，现有两档模型的准确率不能直接推广到其他排名。

为防止扩库隐式改变旧入口，`predict.py` 与 `predict_context.py` 继续只使用T1000/T2000历史；背景入口离线查询严格选择对应的另一档，不会误把T10当作T1000伙伴。旧模型核心与封存实验源码未改动。早期探索脚本直接读取数据库，其样本范围会受扩库影响，不能作为固定v1比较。

下一步建立19档的可用案例清单和质量过滤，再按活动成组训练/验证。原有v1仍保留为T1000/T2000对照；扩展排名需要明确登记新的评测版本。旧测试队列曾参与早期探索，元数据是回溯快照且缺少历史时点存档，这两个限制仍适用。

## 来源与复现

数据源为Bestdori公开API。接口形式已对照安装的bestdori-api源码和[维护者的分数线文档](https://github.com/windowssov8forus/bestdori-api/blob/main/docs/api/eventtracker.md)、[T10文档](https://raw.githubusercontent.com/WindowsSov8forUs/bestdori-api/main/docs/api/eventtop.md)核实。数据是否存在以实际归档响应为准。

下载代码保存在 `download_source.py`；导入使用复查后的 `apply_source.py`，二者版本差异及来源指纹均保留。读取 [download_manifest.json](download_manifest.json)、[database_receipt.json](database_receipt.json)、[audit/verification.json](audit/verification.json) 可核验记录。

新一次下载可使用项目根目录入口，另选新目录；不要覆盖本次归档：

```bash
.venv/bin/python download_ranks.py fetch --run data/downloads/<new-run> --rate 2
.venv/bin/python download_ranks.py verify --run data/downloads/<new-run>
.venv/bin/python download_ranks.py apply --run data/downloads/<new-run>
```

80项单元测试通过。新测试覆盖T10边界变更、重复玩家、空响应、同刻冲突、重试/续传、WAL并发、冻结路径/symlink保护、导入恢复及旧两档入口兼容性；独立代码复查无剩余Critical/Important问题。
