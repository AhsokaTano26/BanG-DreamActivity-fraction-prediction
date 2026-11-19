# F 二轮图设计与输出约定

本工作区只绘图，不拟合、不评分，不改原稿及其他角色目录。15张原图按 LaTeX label 和文件语义逐张保留；新增3张图各对应一个科学问题。正式数值仅来自已封存结果，使用当前密集时期队列政策。新旧协议汇总分数不作模型改进比较。

## 视觉合同

- 中文 SimSun，英文与普通数字 Times New Roman，数学 Computer Modern；沿用已验证字体文件。常规标签8–9 pt，机制主节点9–10 pt。原图中不需重做的密集面板保留原有字号并披露。
- 论文宽度16 cm。实验图白底、二维、弱网格、直接标签；不以拟物效果编码数值。方法固定使用蓝/橙/灰，并配合线型、marker或文字，检查灰度与红绿色觉模拟。
- 三张机制图使用原生矢量卡片、排名叠层、时间守卫和封存图标。2.5D仅限约1 mm侧面与轻阴影，不放实验数据装饰，不把所有对象画成同一立方盒。
- SVG保留普通文字与原生几何；数学字形允许矢量路径。PDF及统计色格保持矢量，PNG为400 dpi。每个输出记录输入路径/SHA、图注、单位、目标label和限制。
- 所有解释围绕读者理解方法，图中不显示工作区路径、线程或程序内部字段。

## 优先改造

1. `scene_framework / fig:upgrade-scene`：排名前缀、元数据、期初历史三个对象，流向基础预测、校正与封存；本期终分隔离；观测和评分单位分别说明。
2. `anchor_difference / fig:upgrade-anchor`：T2000全合法支持→慢锚点，T1000成对支持→快差异，α=.2/.6、k=10/5；扣除方向、独立回退及恒等负控准确。自适应I/S/λ另图，避免混成原算法。
3. `prospective_loop / fig:upgrade-prospective`：资格、先封存、后到达、独立评分；突出结束时间与实际到达时间两道守卫及物理案例去重。当前新队列0期，20期pilot与80期采集目标区分。
4. `matched_correlations / fig:upgrade-correlation`：四矩阵同一34活动18线面板；数值语义不变，轴标签分组、字号/色条改善；不以相关性作因果解释。
5. `sensitivity / fig:upgrade-sensitivity`：完整五个离散单因素扫描；验证/已探索测试清楚分开，不由测试择参。
6. `rank_stage / fig:upgrade-rank-stage`：共享冻结案例，误差差与活动支持同图，缺失保留。零差区域以符号标识，避免把两线校正之外的零值伪装成精度收益。
7. `tail_interval / fig:upgrade-interval`：保留点/尾部取舍，更新E正式三覆盖目标；增加phase完整组条件覆盖和回退宽度代价。区间为固定已发预测探索性重采样，不保证前瞻覆盖。

## 仅新增三张图

| 文件 / label | 科学问题与面板 | 封存来源 | 样本单位 |
|---|---|---|---|
| `adaptive_diagnostics / fig:round2-adaptive` | I/S/λ机制独立小面板；12候选选择块与末段MAPE；门控/非门控λ及支持敏感性 | B/run_v2_001、B的lambda_diagnostics、E/statistics_002（若需CI） | 35训练活动；15历史/10选择/10诊断，任务状态非独立样本 |
| `fairness_diagnostics / fig:round2-fairness` | 三终点×三历史组共享−独立差；12活动分组聚合CI；参数/forward-MAC/更新次数区别 | C/c2_002_scores、C/evidence_001、E/statistics_002 | 共12查询活动，与B重叠；同cell参数32d+66 |
| `calibrated_transfer / fig:round2-transfer` | 900点受限profile与17支持点；24条件状态与APE方向；固定DLM真实phase1末段迁移 | D/calibration_run_v2_001、simulation_run_v2_001、posthoc_transfer_run_001 | 校准25pair，模拟每格300path；真实10活动120case、事后追加 |

不把B35活动与C12查询活动加成47，二者活动union为35。不用旧验证/测试为新增候选选择。模拟支持集为描述性profile阈值，不能称参数置信域。真实迁移是一次固定配方的事后检验，失败也完整呈现。

## 复现与验收

独立入口 `scripts/make_round2_figures.py` 显式接受 `--repo-root` 与 `--out-root`；旧图实现作为只读参考副本位于 `scripts/legacy_reference_figures.py`。输入读取绑定SHA并在manifest中记录，临时派生表只能写本目录。

在PI给定单线程时段，以项目`.venv/bin/python`执行，设置BLAS/OMP线程1，不启动任何拟合。实际PDF经Poppler渲染，逐图查看彩色、灰度和色觉模拟；记录SVG live text/无image、PDF字体嵌入、PNG尺寸和400 dpi、图源数学和样本数。原稿由G/PI集成，F不改main.tex。
