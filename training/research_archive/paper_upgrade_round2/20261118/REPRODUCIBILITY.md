# 第二轮复现入口与包的边界

所有命令从解压后的项目根目录运行，使用安装了requirements-upgrade.txt依赖的Python。当前实际环境为Python3.14.8。数值库单线程，作业串行nice19；全仓测试仍按AGENTS.md执行。

## 图表、绑定和论文：一条命令

~~~sh
python -B analysis/paper_upgrade_round2/20261118/integration/reproduce_paper.py --repo-root "$PWD"
~~~

入口依次核验封存来源并生成表/宏、重画18图、编译主稿与独立补充、执行当前round2验证。XeLaTeX/BibTeX/Poppler须已安装；图表使用已记录的Times New Roman及SimSun字体，文件路径见F/qa/font_check.json。字体未安装的环境可先使用随包矢量图，命令加--skip-figures；最终PDF自身嵌入字体。

只有LaTeX源码包时，从papers/20261118_entertainment_computing_cn执行make all即可用既有矢量图编译；完整数值和图源再生成需要research bundle。

## 新增模型与实验

B固定规则（原封存源码不动，新入口有物理去重守卫）：

~~~sh
python -B analysis/paper_upgrade_round2/20261118/integration/replay_guarded.py --inputs analysis/paper_upgrade_round2/20261118/inputs --config analysis/paper_upgrade_round2/20261118/workspaces/B/preregistration_v2.json --output analysis/paper_upgrade_round2/reproduction_B_new
~~~

输出目录必须新建。原12候选、15/10/10顺序和唯一选择保持配置值，原始候选、专家、逐期封存和最终评分均保存。不要根据末段结果重新调参。

C的c_fairness.py提供prepare-c1/prepare-c2、execute-c1/execute-c2、score-c1/score-c2。先prepare到新目录，再按configuration.json的实际SHA创建资源授权receipt；execute要求authorized_by=PI、相应experiment、绝对run_directory、configuration_sha256与maximum_physical_fits。receipt是本研究协调器的串行预算记录，无外部账户权限作用。独立score另选新输出目录，盲预测封存后才打开train答案。原C/PI_C1_authorization.json及PI_C2_authorization.json提供结构样例，不能直接复用到不同目录。C1零拟合、C2恰108拟合；原始模型和预处理均随包保存。

D的run_mechanism.py提供--stage calibration与--stage simulation。采用preregistration_v2及code_seal，两个阶段新建输出，参数先封存后模拟。额外posthoc_transfer.py是已看模拟后追加的固定参数诊断，其计划和身份单列。E的statistics_audit.py提供--stage predictions及--stage reliability，后者必须显式指定已通过的--passed-manifest；正式来源为statistics_002，原中断001和极尾修订前reliability_001留存。

详细实际命令、资源上限、全部SHA及运行标识见各角色报告与原日志。H已在仓库外实际重放B35期全部候选并复拟C一个固定单元，预测和模型张量与封存结果逐值一致；相应依赖闭包和实际失败修复记录一并随包。

## 数据范围

research bundle提供本轮全部训练期误差、35查询时钟、C重新拟合所需完整两线冻结数据、D全部参数/路径结果、E全部独立评分、图表数据和原始checkpoint。19线目录中提供原manifest和case/label评分子集，供区间审计；这份子集不满足完整RollingDataset加载器的全工件校验。若需从原始19线前缀重训所有历史基础树，应回到原完整工作区，恢复完整数据及各原始run的基础缓存，不能用评分子集替代。

原32页基准330文件ZIP、旧研究数值及当前未发布的前瞻登记均保留。历史54规则的便携胶囊只复现归档基础预测之后的校正/区间，其独立性范围见第一轮报告；本轮没有重新生成全部历史基础模型。

## 证据与失败

DELIVERY_MANIFEST.json逐项记录包内文件SHA，外部receipt记录ZIP整体SHA与CRC。版本化ZIP不会覆盖旧版。verification_summary.json为当前539个唯一测试的canonical索引；初次E错误cwd调用的失败及其修正运行均保留。

旧test已探索、validation反复使用、元数据为回溯快照。新35期仍属于既有训练活动，C12期是其子集。未来80期计划尚未激活，现有0期未来结果。所有重建只复现当前证据，不产生新的确认性成绩。
