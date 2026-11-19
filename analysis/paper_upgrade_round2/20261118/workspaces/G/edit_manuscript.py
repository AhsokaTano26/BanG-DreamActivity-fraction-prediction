from pathlib import Path
B=Path(__file__).resolve().parent;P=B/'manuscript'
s=(B/'original_sources/main.tex').read_text()
s=s.replace('音乐游戏活动终分预测：延迟反馈下的时间评测与跨排名共享','音乐游戏活动终分预测：延迟反馈下的跨排名共享与失效诊断').replace('音乐游戏活动终分预测：\\\\延迟反馈下的时间评测与跨排名共享','音乐游戏活动终分预测：\\\\延迟反馈下的跨排名共享与失效诊断')
s=s.replace('\\input{data/upgrade_macros.tex}','\\input{data/upgrade_macros.tex}\n\\input{data/round2_macros.tex}')
s=s.replace('\\begin{document}',r'''\newcommand{\roundfigure}[3]{\IfFileExists{figures/#1.pdf}{\upfigure{#1}{#2}{#3}{round2}}{}}
\begin{document}''',1)
start=s.index('\\begin{abstract}');end=s.index('\\end{abstract}',start)+len('\\end{abstract}')
s=s[:start]+r'''\begin{abstract}
排名门槛在同一音乐游戏活动中同步增长，终分标签却要等到活动结束后才可用于下一期。本文研究一个具体问题：期初可见的历史误差信号，何时支持跨排名共享，何时应保留独立校正。研究冻结BanG Dream国服2024年1月29日起97期密集采样活动及67/14/16归属，采用预测先封存、结果后释放的活动级时间契约。19档5220个历史测试前缀的活动等权MAPE为\BaseTest\%（未校正）和\EwmaTest\%（独立EWMA）；该测试曾用于探索。进一步只用训练组35期外推活动，预定15/10/10嵌套滚动比较12个候选，选择块保留独立法；末10期独立、完全共享和近期风险共享分别为\RTIndependent\%、\RTPool\%和\RTRisk\%。18项结构消融揭示两条线贡献方向不同。108次重新拟合在总参数匹配和近似前向计算匹配下检验共享结构，优势随终点和历史组成变化。训练残差校准显示同期相关与跨期持续性存在差别；24格条件模拟中的状态风险收益，在固定DLM的真实事后迁移中消失。经验区间的点、完整排名阶段组及整活动覆盖分别审计。本文贡献为可审计时间评测、共享失效的匹配对照及真实量纲机制诊断；基础组合与平方风险公式沿用既有研究。验证反复使用、历史测试探索及回溯元数据限制持续保留，80期确认性采集计划尚未激活，未来成绩为空。
\end{abstract}'''+s[end:]
start=s.index('\\section*{Abstract}');end=s.index('\\section{引言}',start)
s=s[:start]+r'''\section*{Abstract}
{\raggedright\noindent\textbf{Final score prediction in music game events: cross-rank sharing and failure diagnosis under delayed feedback}\par}
We ask when information available at the start of an event supports sharing forecast errors across ranking cutoffs. A frozen cohort contains 97 densely sampled BanG Dream events on the Chinese server, split by event into 67 training, 14 validation and 16 test events. A chronological coordinator seals predictions before releasing completed outcomes. Event macro MAPE over 5220 historical test prefixes is \BaseTest\% without correction and \EwmaTest\% with independent EWMA correction; this test cohort had been explored previously. New designs use only 35 training-period forecast events. A prespecified 15/10/10 nested replay selects independent correction among 12 candidates. On the last ten events, independent, fully pooled and recent-risk corrections score \RTIndependent\%, \RTPool\% and \RTRisk\%, respectively. Eighteen structural ablations separate effects on the two cutoffs. A further 108 neural fits compare sharing at matched total parameter counts and approximately matched forward computation, with outcomes varying by temporal endpoint and history composition. Training residuals link contemporaneous correlation to weak and uncertain temporal persistence. A calibrated 24-regime simulation yields state-risk gains that do not transfer to a fixed DLM in a post-hoc real-data diagnostic. Interval audits distinguish point, complete rank--phase group and whole-event coverage. The contribution is an auditable temporal protocol and matched failure diagnosis grounded in real residual scales; forecast-combination and quadratic-risk formulas build on prior work. Reused validation, prior test exploration and retrospective metadata remain limitations. A separate 80-event confirmation plan is unactivated, with no future accuracy reported.

'''+s[end:]
old='本文围绕三个问题组织研究：';a=s.index(old);z=s.index('\\upfigure{upgrade_scene_framework}',a)
s=s[:a]+r'''本文把研究问题落实为五项可观测检查：过去已发出的独立与共享误差是否支持下一期组合；原锚点差异配方的收益来自哪条线和哪一项；容量及训练日历控制后共享结构是否仍有优势；真实误差与状态空间模拟相容到什么范围；经验区间和未来活动需要怎样的证据。它们共同检验“同活动相关”向“下一活动可迁移收益”的跨越。

本文的贡献分为三个层次。第一，按活动固定案例、信息时钟和结果释放顺序，保留全部方法、缺失与失败记录。第二，用同支持恒等负控、预定单线消融及容量和计算对照，定位共享效果的来源与取舍。第三，建立训练残差、条件参数集合、模拟与真实迁移之间的可复核连接，并保留新候选选择失败及模型迁移失败。凸组合、EWMA和Kalman滤波采用既有数学；两项命题提供初等机制分析。当前主方法保持独立EWMA，研究候选的额外收益接受固定规则检验。

97期队列已有探索记录，14期验证多次用于选择。新增方法只在训练组内设计和诊断；其末10期仍是回溯训练期外推。本文保留历史暴露、验证复用及回溯元数据三项限制。独立确认性证据由另行激活的新活动队列收集。
'''+s[z:]
needle='本文固定每活动、排名及阶段的对数偏差状态，重点分析两线锚点与差异的时间尺度和协方差。'
s=s.replace(needle,needle+'基于历史预测误差确定组合权重的研究可追溯至Bates与Granger\\citep{bates1969combination}；多任务平均已系统研究任务均值的凸混合与风险收缩\\citep{feldman2014averaging}。本轮风险公式与ridge使用这些既有思想，新增设计集中在期初信息集、已发出误差、支持守卫和有限样本诊断。')
s=s.replace('本文比较共享输出、专属头、相关线分支及参数完全独立的网络。','任务关系的正则学习还可显式处理负相关与异常任务\\citep{zhang2010relationships}。本文比较共享输出、专属头、相关线分支及参数完全独立的网络。')
s=s.replace('采用可识别的二维模型','采用给定载荷与创新假设的二维模型')
s=s.replace('该条件说明任务差异支付平方偏差，共享减少部分估计噪声。','该式复述两任务凸平均的基本偏差方差权衡\\citep{bates1969combination,feldman2014averaging}：任务差异支付平方偏差，共享减少部分估计噪声。')
# Compact previous exploratory fitting, neural structures and hand-selected simulation in the main text; preserve complete text in SI.
a=s.index('\\subsection{训练规模与共享结构的新拟合实验}');z=s.index('\\section{稳健性、尾部误差与区间可靠性}',a)
old_experiments=s[a:z]
(P/'legacy_experiments.tex').write_text(old_experiments)
s=s[:a]+r'''\subsection{既有结构实验的证据位置}
先前的训练规模实验用最早16、32、48、67期重新拟合，并反复评价同一14期488案例验证面板；它同时改变了样本量、日历覆盖和活动组成。既有19线残差网络与手工16格模拟分别检验神经结构和指定生成条件。完整方法、三种子、预算及正负区域保留于补充材料。下节采用训练组内部多终点和匹配预算补充这些控制；各实验的输入与反馈条件分别报告。

\input{round2_studies.tex}

'''+s[z:]
# Interval section retains legacy tail metrics and figure, replace coverage paragraph with independently audited targets.
a=s.index('经验80\\%和90\\%区间围绕独立EWMA中心构造。');z=s.index('\\section{前瞻记录与可执行复现}',a)
s=s[:a]+r'''经验80\%和90\%区间以独立EWMA为中心，历史校准分数取同活动、排名与阶段内最大绝对中心对数误差，并使用\(\lceil(n+1)\ell\rceil\)顺序统计量。支持不足先回退合法历史活动阶段池，再保留显式无界或不可用标记。分布变化下的在线共形推断已有专门理论\citep{gibbs2024online}；本文固定的历史分位数配方以回溯可靠性审计为证据范围。

表~\ref{tab:round2reliability}同时报告三个目标。每水平5220个案例对应878个完整排名阶段组，全部来自16期活动。活动等权逐案例覆盖为\CoverageEighty\%/\CoverageNinety\%，完整组联合覆盖为\JointEighty\%/\JointNinety\%；整活动全部组覆盖仅为0/16和1/16。组联合目标与历史最大误差构造对应；整活动目标要求更多事件同时成立。当前案例全部有有限区间，原始案例均值相对宽度为\WidthEighty\%/\WidthNinety\%，活动等权均值为\EWidthEighty\%/\EWidthNinety\%。两种权重各按自身定义报告。
\input{tables/round2_reliability.tex}

条件审计保留76个逐案例及76个完整组分层。T1000和challenge类型的组覆盖低于总体，类型组仅3期；跨排名回退池的联合覆盖为100\%，相对宽度也明显增大。分母与活动级区间见补充表，单期类型和全成功组的固定损失bootstrap会退化。标准区间评分同时惩罚宽度与两侧越界\citep{gneiting2007proper}，80\%/90\%的活动等权Winkler分数为\EWinklerEighty 万/\EWinklerNinety 万EP；不同名义水平采用不同越界成本。这里未据条件结果调整宽度或方法。
\upfigure{upgrade_tail_interval}{fig:upgrade-interval}{点预测尾部与经验区间的三个目标。百分比误差、绝对RMSE和P90分别评价；逐案例、完整排名阶段组与整活动联合覆盖采用各自分母。覆盖、有限可用率和宽度共同报告，活动数为16。}{interval}

'''+s[z:]
# Replace outdated prospective paragraph.
a=s.index('图~\\ref{fig:upgrade-prospective}给出闭环。');z=s.index('复现包保存配置',a)
s=s[:a]+r'''图~\ref{fig:upgrade-prospective}给出闭环。原登记于\ProspectiveRegistered\ 生效，独立参考为主方法，历史组合和完整协方差DLM为影子对照；最初\ProspectiveTarget\ 期目标保留为流程可行性pilot。新增确认性计划与独立补充保持原登记不变，等待新的激活封存和独立起始队列。唯一主比较为历史组合减独立EWMA的19线活动等权MAPE差，负差支持历史组合、正差支持独立校正；两线机制面板为次要估计目标。

规划方差只取训练35期已发活动差。给定改善0.5个百分点、双侧\(\alpha=.05\)、功效.9、\(\rho=.3\)与上95\%的规划\(\sigma\)，19线模型给出\ConfirmGlobalN\ 期，两线给出\ConfirmPairN\ 期。两线规则只改变少数排名，19线平均稀释了差异方差；该条件化模型输出保留于补充表。工程与稳定性采集下限为\ConfirmFloor\ 期，结合两线次面板并按10向上取整，计划采集\ConfirmTarget\ 期。该下限是工程预算，有限样本推断精度仍取决于活动依赖与分布条件。

主区间预定采用按实际开始时间排序的Bartlett/Newey--West HAC\citep{newey1987hac}，带宽\(L=\min\{n-1,\max(1,\lceil n^{1/3}\rceil)\}\)，并报告iid配对t及循环块敏感性。渐近近似要求弱平稳、有限所需矩与弱依赖；AR(1)功效膨胀用于规划，未来HAC的有限样本功效另待实际数据。零差检验和超过0.5个百分点的实用门槛分别评价。新计划尚未激活，已纳入未来活动\ConfirmFuture\ 期，MAPE为空。

未来按新封存后的实际国服开始时间纳入最初80期，人口清单保留无预测与缺标签活动，缺失原因逐项记录，无替换活动。沿用21时间条件、19排名、至少4点前缀、60分钟观测滞后与15分钟提交滞后上限。可信采集者核实终分来源和实际释放时刻；同一期全部时点共用期初池。流程失败使用事前准时发布的固定当前分数回退，晚补预测标记失效；全部固定人口完整后按预定时点评分。

'''+s[z:]
# Add current guarded path and artifact root.
s=s.replace('图表脚本读取统计产物，不训练或选择方法。','图表脚本读取统计产物，不训练或选择方法。二轮绑定器显式接收仓库根与稿件输出路径；原封存B预测保留，新重放入口使用物理案例别名守卫。独立审计在全部35期验证守卫路径与原预测逐例零差。')
s=s.replace('本轮工件根目录为\\path{analysis/paper_upgrade_records/20261118}', '首轮工件根目录为\\path{analysis/paper_upgrade_records/20261118}，二轮工件位于\\path{analysis/paper_upgrade_round2/20261118}')
# Discussion remains compact and incorporates the new evidence without causal overstatement.
a=s.index('\\subsection{对游戏预测应用的含义}');z=s.index('\\subsection{证据、元数据与统计限制}',a)
s=s[:a]+r'''\subsection{对游戏预测应用的含义}
期初更新低维偏差状态，可与基础预测分开封存和审计。独立EWMA在历史测试及训练期选择中保持实用参照地位。自适应共享的平方风险系数只依赖已发误差，但有限历史带来的删除敏感性较大；保守门控关闭后与独立法同轨。这项回退避免了本面板的额外共享误差，主动共享的稳定收益仍缺证据。

相关性、持久性和预测损失承担不同作用。同期两线共变支持共享假设的研究动机；训练期滞后信号较弱、原始过程矩含负特征值，以及等噪声约束的失配，共同限制了状态模型解释。受约束模拟保留了负迁移和损失排序变化，真实DLM追加迁移又出现损失增加。应用中应同时显示预测时刻、观测滞后、历史支持与区间宽度，按实际目标审视百分比误差和绝对尾部。

容量匹配消除了一个可量化差异，函数类、宽度与监督共享仍有变化。时间覆盖控制也保留活动组成和类别支持差别。结果适合指导后续结构选择与失败诊断，玩家投入、竞争偏好等行为解释需要玩家层资料或额外设计。

'''+s[z:]
s=s.replace('训练规模实验采用相同案例、目标和局部特征，参数数量及模型数量仍不同，推断范围限于其已记录预算。','首轮训练规模实验的参数与模型数量随结构变化；二轮同单元实现总参数精确匹配，并另给近似前向计算控制，仍保留函数类、更新次数、时期组成和输入维度差异。B、C1与C2的真实活动并集为35期，C2的12期全部来自该集合。')
s=s.replace('合成模型的高斯创新、线性状态与已知载荷提供受控机制，真实游戏机制的解释需额外证据。','首轮手工网格及二轮真实量纲校准模拟均使用受控状态族；二轮保留重尾、突变、延迟和异方差扰动。真实潜在状态缺少直接观测，参数范围条件于离散网格与受限噪声假设；描述性支持带的阈值未经有限样本覆盖校准。')
a=s.index('\\section{结论}');z=s.index('\\FloatBarrier',a)
s=s[:a]+r'''\section{结论}
本文建立音乐游戏活动终分的活动级时间契约，并将“同期相关是否支持下一期共享”分解为可复核的预测、消融、容量与机制检查。已探索19线测试中，独立EWMA的MAPE为\EwmaTest\%；只使用训练35期的预定嵌套选择仍选独立。共享候选、单线效应、多终点训练及条件模拟完整保留，结果显示支持规模、时间位置、噪声失配和损失目标会改变收益方向。

同滤波恒等式与平方风险展开提供已有数学下的初等解释。真实校准的固定DLM在事后迁移中增加误差，使模拟与应用之间的证据缺口具体可见。经验区间的完整组覆盖接近预设水平，整活动与部分条件覆盖仍明显较低，宽度代价需同时报告。历史探索、验证复用及回溯元数据继续限制推断；80期确认性计划等待独立激活和真实新活动。

'''+s[z:]
(P/'main.tex').write_text(s)
