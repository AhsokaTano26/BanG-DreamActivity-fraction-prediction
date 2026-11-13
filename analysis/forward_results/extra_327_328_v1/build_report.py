"""Render already sealed new-activity results without reopening outcomes."""
import json
from pathlib import Path
import pandas as pd
from analysis.holdout_dataset import sha256,write_json
from analysis.temporal_dataset import STAGES
from analysis.temporal_evaluate import verify_predictions

r=Path(__file__).resolve().parent
verify_predictions(r)
receipt=json.loads((r/'evaluation_manifest.json').read_text())
for n,digest in receipt['artifacts'].items():
 if sha256(r/n)!=digest:raise ValueError('Scored output changed')
if (r/'README.md').exists():raise FileExistsError('Report already exists')
e=json.loads((r/'evaluation.json').read_text());run=json.loads((r/'run_manifest.json').read_text())
registration=json.loads((r/'prospective_registration.json').read_text())
names={'reference_single':'原单线CatBoost','reference_paired':'上轮双线CatBoost','selected_combo':'新三模型组合'}
def pct(v):return '—' if v is None else f'{v:.2f}%'
lines=['# 新活动验证：328上新组合未优于原模型','',
'本次验证不支持新组合已经稳定更好。固定队列之外的活动328现已补齐有效终分；在两条线共同的18个可用V2时点、36个前缀上，新组合MAPE为10.13%，原单线为5.09%，上轮双线为7.16%。模型、成员、参数和预测没有根据本次结果修改。','',
'## 验证范围和终分','',
'327与328均不属于V1/V2固定队列，且开始时间晚于固定测试活动最后结束时刻。新组合和两个V2参考模型均在这两期结束后才生成，因此以下是额外活动回溯验证，不是新模型的事前前瞻成绩。','',
'| 活动 | T1000终分 | T2000终分 | 状态 |','|---|---:|---:|---|',
'| 327 | — | — | 最后观测早于结束895秒，待答案 |',
'| 328 | 3,349,819 | 2,173,258 | 结束后1秒记录符合冻结窗口 |','',
'原始响应、HTTP头、抓取时间及SHA在 data/forward_validation/20261112_api_audit/。终分窗口为结束前60秒到结束后300秒；没有把327最后已知分数冒充终分。','',
'独立新活动共2个，可评分1个。完整两活动总体MAPE保持不可用；下表仅评分328，不能当作两活动总体或一般泛化准确率，也不做单活动bootstrap。','',
'## 活动328的共同案例结果','',
'| 模型 | T1000 MAPE | T2000 MAPE | 两档MAPE | P90 APE |','|---|---:|---:|---:|---:|']
for m in ['reference_single','reference_paired','selected_combo']:
 v=e['grid'][m]
 lines.append('| '+names[m]+' | '+' | '.join(pct(x) for x in [v['by_rank']['1000']['event_macro_mape_pct'],v['by_rank']['2000']['event_macro_mape_pct'],v['overall']['event_macro_mape_pct'],v['overall']['p90_ape_pct']])+' |')
lines += ['', '所有方法共用冻结案例。327额外38个前缀也已封存预测，但未纳入终分误差；328有36个前缀。1h/3h及328结束前48h没有合格前缀，保留缺失清单，不插值制造数据。训练历史及标准化、校准均未更新。','',
'## 活动328全部预设时点','',
'| 时点 | 原单线 | 上轮双线 | 新组合 | 活动数/案例数 |','|---|---:|---:|---:|---|']
for stage in STAGES:
 key,axis,value=stage['stage_id'],stage['axis'],stage['value']
 label=f'开始后{value:g}h' if axis=='elapsed' else f'进度{value:.0%}' if axis=='progress' else f'结束前{value:g}h'
 metric=e['grid']['selected_combo']['by_stage'][key]
 lines.append('| '+label+' | '+' | '.join(pct(e['grid'][m]['by_stage'][key]['event_macro_mape_pct']) for m in ['reference_single','reference_paired','selected_combo'])+f" | {metric['events']}/{metric['cases']} |")
lines += ['', '请求时点与实际观测时点分别保存；前缀最多允许60分钟滞后，结果同时保留滞后≤15分钟子集及同活动同排名共同面板，详见evaluation.json。','',
'## 核对真正提前封存的旧实时预测','',
'2026-11-11 19:21:41（北京时间）的旧模型预测早于328结束，归档和源输入指纹完整。它是旧 structured_neural_v1 的 blend_neural_0.25，不是本次三模型组合。','',
'| 排名 | 当时已封存预测 | 实际终分 | APE |','|---|---:|---:|---:|']
for row in e['legacy_actual_forward_forecast']:
 lines.append(f"| T{row['rank']} | {row['predicted']:,.0f} | {row['actual']:,.0f} | {row['ape_pct']:.3f}% |")
lines += ['', '两档平均APE约4.13%，只代表这一活动、这一时点。不能与上面18时点的总体误差直接比较，也不能移作新组合的前瞻证据。','',
'同一份当时已封存输入上的事后回放对照如下；重建使用旧归档的活动起止时间与类型，并验证观测时刻/当前分数与旧预测记录相同。','',
'| 方法（均为事后回放） | 排名 | 预测终分 | APE |','|---|---:|---:|---:|']
matched=[]
for m in ['reference_single','reference_paired','selected_combo']:
 for row in e['legacy'][m]['rows']:
  lines.append(f"| {names[m]} | T{row['rank']} | {row['predicted']:,.0f} | {row['ape_pct']:.3f}% |")
  matched.append({'method':m,**row})
pd.DataFrame(matched).to_csv(r/'matched_snapshot_scores.csv',index=False)
lines += ['', '## 真正的新活动前瞻验证状态','',
f"本次再次查询国服活动元数据的登记时刻为 `{registration['registered_at']}`，进行中活动数 {len(registration['ongoing_cn_activities'])}，已公告未开始活动数 {len(registration['upcoming_announced_cn_activities'])}。",'',
'下一阶段的模型和规则已登记在 prospective_registration.json：使用相同冻结组合与参考，未来活动必须在结束前真正封存预测，不回填过去的预测时刻；缺少终分继续待答案。当前没有后台定时采集进程运行，不能声称已经完成尚未发生的新活动前瞻验证。','',
'## 审计和限制','',
f"新数据指纹 `{run['dataset_id']}`；评分标识 `{e['evaluation_id']}`。",'',
'前缀生成与答案独立保存，改变最终答案不改变任何盲前缀的回归测试已通过。正式预测时禁用了答案读取、SQLite访问和CatBoost.fit；只从封存模型做推理。父数据指纹、三个模型的运行ID/封存摘要及执行源码/版本均校验并保留。','',
'此前固定测试队列曾参与早期探索，元数据仍是缺少历史时点存档的回溯快照。这些限制不因新增回放而消失。单个新活动的失败足以提醒不要将新组合视为已验证的稳定升级，但不足以估计长期胜率。','']
(r/'README.md').write_text('\n'.join(lines))
write_json(r/'report_manifest.json',{'evaluation_manifest_sha256':sha256(r/'evaluation_manifest.json'),
 'artifacts':{n:sha256(r/n) for n in ['README.md','build_report.py','matched_snapshot_scores.csv','prospective_registration.json','next_event_metadata.json','availability_check.py']}})
print(r/'README.md')
