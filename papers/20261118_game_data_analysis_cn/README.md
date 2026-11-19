# BanG Dream国服活动排名分数的时序结构与活动条件关联

作者：Tano。中文研究稿，附英文摘要。采用Elsevier elsarticle的单栏预印本格式，沿用Entertainment Computing稿件的常见组成。

本篇为独立的第二篇论文，研究游戏活动数据的统计结构。正文包括数据质量、19档终分分布、时间变化、排名比值、阶段增长、跨线关联、活动类型与乐队的条件关联，以及角色和卡牌记录。现有预测论文位于相邻的20261118_entertainment_computing_cn目录。

## 文件

- main.tex：完整正文与公式，论文编辑入口。
- numbers.tex、tables/：从统计输出生成的50个数值绑定和7张原生LaTeX表格。
- references.bib：11项已核对的参考文献；来源页面记录在data/reference_sources.json。
- figures/：7幅矢量PDF及400 dpi PNG。
- output/中文论文.pdf：编译后的论文。
- data/：不可变来源快照、103期名单、排除原因、完整矩阵和敏感性分析。
- scripts/：分析、绘图、表格生成、独立数值复核与交付打包脚本。
- qa/：数值、来源、字体和PDF排版检查记录。

## 数据范围与来源版本

实际国服开始时间下限为 **2024-01-29 13:00:00（UTC+8），1706504400000毫秒**。快照只选择提取时已结束的活动，按实际开始时间排序。活动ID仅用于连接。

主来源为runtime/live/data.sqlite3于2026-11-18的只读事务快照，共103期、1871条存储曲线、1796085个库内归一化观测。终分定义采用结束前1分钟至结束后5分钟窗口，得到1780条合格曲线、1718910个观测。完整来源记录指纹：

c472560390f27a5e60b4b0c9e4e508f2e7ce8595cfb4c6e443e739cdfcc063ba

source_manifest.json保存SQL、实际时间参数、源文件哈希与快照哈希。来源数据在快照后继续更新，本文统计固定于随稿版本。

元数据为回溯缓存。活动、角色、乐队缓存保存本地文件时间及运行库的刷新记录；卡牌缓存的本地文件时间为2026-11-16，并按国服发布时间过滤。source_metadata.json.gz中早期导出字段sources.fetched_at的值来自本地文件修改时间上界，解读时使用source_manifest.json的文件时间和刷新记录。该字段的说明见data/source_provenance_notes.json；导出脚本已将同类字段命名为latest_local_file_mtime。

初次核查另使用过旧data/db.sqlite3。旧缓存存在结束后持续轮询及尚未补全的终点，主分析改用当前运行库。图2的旧缓存对照取自legacy_terminal_scores.csv，与主快照分开标记。旧缓存的完整核查副本保留在工作区qa/legacy_source_check/，交付包提供其终分摘录和来源审计记录。

## 统计口径

独立样本按活动计。回归中每期总权重为1，使用活动聚类协方差及活动得分向量的5期Bartlett HAC协方差；全部五个探索模型的26组因素检验统一进行BY调整。置信区间为边际区间。18线共同终分有94期；终分与半程增长的直接相关比较共用92期。四条代表性排名的全部进度时点共用74期，全部剩余时间时点共用86期。

年度比较固定为各年2月至10月；同活动排名比先相除再汇总。缺失覆盖、实际观测滞后、下降观测、负增长和分数修订均有单独输出。卡牌统计描述发布记录；玩家持有、活跃人数、游玩时长、付费和个人投入需要额外数据。

本篇为描述性关联研究，使用全部合格已结束活动。原97期预测队列的67训练、14验证、16测试归属保留，另6期标为descriptive_extension（238、280、282、310、327、328）。本研究的全样本回归结果用于解释当前观察范围。原有预测协议、评分结果和生产模型各自保存。

部分活动此前参与模型探索，验证数据曾多次使用；元数据缺少完整历史时点存档。乐队样本为6至11期单乐队活动，排程、规则、时间和其他未观测条件共同变化，系数按探索性条件关联解释。

## 编译

安装TeX Live及XeLaTeX、BibTeX、elsarticle、ctex、unicode-math、needspace等组件后，在本目录执行：

~~~sh
make pdf
~~~

原环境的可执行文件位于/Library/TeX/texbin/，也可显式传入：

~~~sh
make pdf LATEX=/Library/TeX/texbin/xelatex BIBTEX=/Library/TeX/texbin/bibtex
~~~

正文优先使用Times New Roman和SimSun；缺少这些字体的环境使用TeX Live提供的Termes与Fandol。图中字体已嵌入PDF。重新绘图时，依据qa/font_check.json设置本机可用字体路径并核对字形。

## 从随稿快照复算

项目内使用.venv/bin/python。独立解压环境可安装requirements-analysis.txt所列依赖。各命令串行运行，并将数值库线程数设为1：

~~~sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -B scripts/analyze_game.py
python -B scripts/measurement_checks.py
python -B scripts/make_figures.py
python -B scripts/build_tables.py
python -B scripts/validate_paper.py
make pdf
python -B scripts/verify_artifacts.py
~~~

analyze_game.py会从随稿来源快照生成清理曲线及统计输出，measurement_checks.py依赖这些曲线。交付包省去可重建的清理曲线副本。复算输出应写入解压后的副本，以保留交付版审计记录。原始来源文件受哈希校验。

snapshot_sources.py记录来源制作过程，仅用于在完整项目中建立新的独立快照，现有快照有防覆盖检查。本文复核使用随稿快照，按上述分析入口执行。

## 稿件状态

该版本为中文研究稿。正式投稿时需形成英文全文，由作者补充实际所属单位、通信信息、基金与利益冲突等声明，并按投稿时的期刊指南核对。AI辅助使用范围已在文末披露。公开研究分支是项目入口；本次材料位于本地工作区，精确版本以交付包清单和哈希确定。
