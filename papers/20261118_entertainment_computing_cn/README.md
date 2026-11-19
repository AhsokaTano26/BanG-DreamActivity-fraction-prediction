# 音乐游戏活动终分预测：二轮研究稿

作者：Tano。`output/main.pdf`为完整稿（正文、补充附录、参考文献），`output/supplement.pdf`为独立补充材料。当前数值来自冻结历史工件；80期确认性计划尚未激活，未来成绩为空。首轮32页稿和10页独立补充材料保留在二轮baseline归档。

## 编译与绑定

本目录仅使用已封存的预测与评分，绑定及绘图不执行训练、重新评分或候选选择。路径全部显式传参：

```sh
/absolute/repository/.venv/bin/python scripts/bind_round2_evidence.py \
  --repo-root /absolute/repository --paper-root /absolute/manuscript
make all
make verify REPO_ROOT=/absolute/repository
make render REPO_ROOT=/absolute/repository
```

TeX默认使用`/Library/TeX/texbin/xelatex`与`/Library/TeX/texbin/bibtex`，可通过`LATEX`和`BIBTEX`覆盖；`PYTHON`也可显式覆盖。F提供的`scripts/make_round2_figures.py --repo-root ROOT --out-root PAPER`重绘18图，应先完成绘图再执行最终PDF检查。项目级一键driver由PI提供并负责原始研究工件与代码包的完整路径映射。

`bind_upgrade_evidence.py`保留首轮606个数值/聚合绑定、30宏和15张原生表；所有旧表与原稿逐字相同。`bind_round2_evidence.py`追加新宏、原生表和来源SHA，不访问验证/测试答案进行选择。`--output`与`--paper-root`是等价参数。两份绑定清单分别记录既有研究和新增研究。

## 当前验收文件

- `qa/round2_number_bindings.json`：本轮source/field/row/SHA与舍入。
- `qa/upgrade_number_bindings.json`：既有研究数值来源，保留其范围。
- `qa/round2_document_checks.json`：当前PDF页数、字体嵌入、引用和LaTeX溢出。
- `qa/round2_render_manifest.json`及`qa/round2_render/`：当前两PDF逐页渲染和接触表。
- F的当前figure manifest：18图数据、代码和导出SHA。

历史`upgrade_*`视觉QA属于首轮，在canonical整合时归档，当前验收只读`round2_*`和本轮F清单。代码单元测试核查实现边界，预测效果由封存预测的独立活动级评分评价。

## 证据范围

新增B/C/D的真实活动并集为35期，C2查询12期为其子集；训练末10期为回溯外推。原测试已探索、验证反复使用、元数据缺少历史时点档案。风险组合、ridge与平方风险展开采用既有数学；当前主方法保持独立EWMA，所有共享负结果完整保留。受限机制参数支持带与条件模拟保留模型假设，固定DLM真实迁移单列为模拟后的事后诊断。

## 依赖范围

原生LaTeX编译只需TeX发行版、文中字体和已生成图表；图重绘使用固定Python绘图库及F字体清单。C2外部复拟沿用`requirements-upgrade.txt`中的完整ML依赖，包括模块顶层导入需要的XGBoost与httpx；该实验没有拟合XGBoost。数值依赖使用本项目虚拟环境，不修改已安装轮子。

## PI一键交付入口

在完整research bundle的项目根目录执行：

~~~sh
python -B analysis/paper_upgrade_round2/20261118/integration/reproduce_paper.py --repo-root "$PWD"
~~~

该命令串行绑定966条既有及新增记录、重绘18图、编译45页整合稿与19页独立补充、核验并逐页渲染。输出PDF的SHA与64张页面PNG在同一渲染清单关联。若仅使用随包矢量图，加--skip-figures。当前一键构建记录和539项测试索引见研究目录。

旧QA和脚本归档于qa_round1_archive、scripts_round1_archive，旧源码ZIP归档于output/round1_archive。当前18图清单仍沿用qa/upgrade_figure_manifest.json文件名，schema_version=2；其数据范围由本轮F明确标记。当前两份PDF验收以qa/round2_document_checks.json为准。

完整复现边界与各实验入口见../../analysis/paper_upgrade_round2/20261118/REPRODUCIBILITY.md。19线目录随包为评分子集，全部原基础树再拟合还需原完整工作区的基础缓存；C本轮108拟合所需两线冻结数据和依赖闭包完整提供。
