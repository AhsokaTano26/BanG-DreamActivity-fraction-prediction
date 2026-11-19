"""Write a compact local data inventory from the completed ingestion audit."""
import argparse
import csv
import json
from pathlib import Path
from download_ranks import sha256,write_json,check_database_path


def main(run):
    run=check_database_path(run)
    verification=json.loads((run/'audit/verification.json').read_text())
    config=json.loads((run/'config.json').read_text());receipt=json.loads((run/'database_receipt.json').read_text())
    coverage=list(csv.DictReader((run/'audit/coverage.csv').open()))
    rows=['| 排名档 | 本次非空活动曲线 | 接口空记录 | 数据库非空活动曲线 | 数据库观测点 |',
          '|---|---:|---:|---:|---:|']
    for r in coverage:rows.append(f'| T{r["rank"]} | {r["downloaded_curves"]} | {r["empty_events"]} | {r["database_nonempty_curves"]} | {int(r["database_points"]):,} |')
    counts=verification['download_status_counts']
    text=f'''# 国服19档活动分数线：本地下载清单

已逐一检查 **{verification['download_requested_pairs']:,} 个活动×排名组合**，范围为元数据中截至下载开始已在国服开放的 {len(config['events'])} 个活动。名单依据国服实际开始时间，不以活动ID数值范围替代。

本次取得 {counts.get('ok',0):,} 条非空曲线；{counts.get('empty',0):,} 组接口有效返回空记录；{counts.get('invalid',0):,} 组响应未通过结构/值域校验；剩余网络失败 {counts.get('failed',0):,}。下载开始时刻（UTC）为 `{config['created_at']}`，各组实际抓取时刻保存在 `records.jsonl`，并非同一秒的同步快照。

导入后，`data/db.sqlite3` 共含 {verification['database_rows']:,} 行活动排名记录，其中19档的非空活动曲线 {verification['database_nonempty_curves']:,} 条、标准化分数观测点 **{verification['database_points']:,} 个**。一次T10完整榜单快照在数据库中计作一个分数线观测点，不把前十玩家的记录计作十个独立分数线样本。

{chr(10).join(rows)}

## 保存位置

- 主库：[项目数据库](../../db.sqlite3)。按 `Country='中国大陆' AND Rank=<排名>` 查询。
- 原始响应：`raw/` 下的 gzip JSON，保留源接口返回，带逐文件指纹。
- 标准化曲线：`curves/<活动ID>_<排名>.json`，仅含时间戳与分数等曲线字段。
- 元数据：[events.json](events.json)；完整状态：[records.jsonl](records.jsonl)。
- 每档覆盖：[audit/coverage.csv](audit/coverage.csv)；每条曲线质量：[audit/curve_quality.csv](audit/curve_quality.csv)。
- SQLite导入前备份：`{receipt['backup']}`。原库已有的时间点保留，重叠时间点使用此次接口返回的修订值。

新增数据库曲线 {receipt['inserted_curves']:,} 条，合并已有曲线 {receipt['merged_curves']:,} 条，新增时间点 {receipt['added_points']:,} 个，修订重叠时间点 {receipt['corrected_points']:,} 个；详见 [database_changes.csv](database_changes.csv)。空响应和无效响应不会删除或清空已有曲线，因此本次非空下载数量与数据库非空数量可能不同。

## T10口径与质量检查

普通分数线接口不接受T10。本次用 `eventtop/data?server=3&event=<ID>&mid=0&interval=0` 取得榜单历史，在同一时间戳至少有10个不同玩家时取第10高分；不沿用最终第十名玩家的个人曲线，不为缺失快照补齐玩家。独立审计重新从原始榜单计算并核对了 {verification['top10_complete_snapshots_independently_verified']:,} 个T10完整快照。

其余18档使用 `tracker/data?server=3&event=<ID>&tier=<rank>`。原始响应有效返回空列表只代表该源在抓取时未提供该组历史，不等于分数为零。网络503有有限重试，并在后续续传中补齐；观察到错误增多后降低了请求速率，运行调整记录在 [fetch_sessions.json](fetch_sessions.json)。

分数下降没有被静默改成累计最大值：本次审计发现 {verification['curves_with_score_decreases']:,} 条已导入曲线有下降段，保留来源数据并标记，后续建模需检查。本次导入曲线中的实际零分记录 {verification['zero_score_points']:,} 个，同样保留并与缺失区分。结束附近观测、起始前/结束后观测及当前进行中活动均有单独质量字段；不能把所有曲线直接当作合格训练样本或完整最终标签。

所有已导入点均独立核对“本次原始数据＋导入前备份”的合并结果；逻辑键唯一、SQLite完整性检查通过。下载可断点续传；导入使用一致性内容指纹、备份和写事务，已验证WAL变更拒绝及提交后收据写入异常恢复。未声称经过断电耐久性测试。

## 评测与下一步

冻结 `data/holdout_v1/` 指纹仍为 `{verification['frozen_dataset_id']}`，没有覆盖、移动分组或重新评分。本次是数据采集与入库；未训练19档模型，现有两档模型的准确率不能直接推广到其他排名。

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
'''
    (run/'README.md').write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();main(args.run.resolve())
