# F 二轮科研图交付

18张正式图已完成：原15张全部保留其语义身份，3张机制图重绘，4张重点统计图改善；新增3张分别回答自适应选择、公平容量与预算、校准机制向真实任务迁移的问题。每张提供SVG、矢量PDF和400dpi PNG。只写本工作区，未拟合、评分或新增模拟路径，原稿由G/PI集成。

## 阅读入口

- `qa/upgrade_figure_manifest.json`：18张图的label、完整建议图注、单位、每个输入路径/SHA、输出SHA、字体、限制和派生量说明。
- `qa/upgrade_figure_contact_color.png` / `_gray.png`：来自实际PDF渲染的彩色/灰度联系表。
- `qa/upgrade_figure_contact_protanopia.png` / `_deuteranopia.png`：两种色觉压力预览，曲线同时以线型、marker和直接标签区分。
- `figure_catalog.md`：按语义列出的图注与单位。
- `output_mapping.csv`：原15图及新增3图的文件名与LaTeX label。

## 主要变化与科学边界

`scene_framework`使用排名前缀、日历卡、误差账本、锁与不同处理卡片；等长前缀横条表示共同观测截止，避免暗示T2000分数更高。`anchor_difference`保留0.2/0.6、10/5、全部锚点支持/成对差异支持、扣除方向、本线回退及恒等负控。自适应I/S/权重另图展示。`prospective_loop`突出严格结束和实际到达两道守卫；0期新队列、保持的20期pilot和待独立激活的80期采集方案分别呈现。

相关图使用完全相同34活动18排名面板；敏感性图保留五个完整离散网格并使用共同MAPE尺度；排名阶段图保留全部排名/缺失/支持，点号明确为相同输出的零差。区间图使用E正式`reliability_002`，分开单前缀、排名×阶段联合及整活动联合，整活动使用iid二项参考CP区间；阶段覆盖及回退宽度代价完整呈现，活动依赖另作敏感性分析，未来覆盖待新活动验证。

新增自适应图保留全部12候选以及15/10/10的历史、选择、末段身份。独立参考被选中，门控末段0/30打开；60是任务状态数。公平图区分同cell精确参数匹配、每参数更新、优化器step和前向MAC，并给12活动聚合CI；源表的“独立−共享”在图中取负成为“共享−独立”。B/C的活动union为35，C的12期是B子集。

校准迁移图保留900网格、17描述支持点、噪声等方差限制、24条件×300路径及一次事后真实迁移。PI复审发现浮点近似相同比值分组尖峰后，已按`round(q_r_ratio,12)`稳定数值等价键再取剖面min，派生表为`data/figure_data/round2_profile_ratio_curve.csv`；17支持点保留原始坐标。图中保留固定DLM真实末段6.953%相对独立5.341%的失败结果。

既有学习曲线、神经结构负结果与原16条件合成图仍可放SI，语义未被二轮数据覆盖。历史队列已有探索、验证反复使用、元数据为缺少历史时点档案的回溯快照等限制继续保留。

## 实际验收

实际PDF经Poppler渲染并查看重点单图及全部18图彩/灰联系表，同时查看protanopia与deuteranopia预览。SimSun与Times New Roman均为指定原字体；普通文字491个不同字符覆盖完整，无fallback，无未解决缺字日志；数学为Computer Modern。全部PDF字体嵌入并带Unicode映射。18 SVG和18 PDF均为0图像对象，统计网格保持原生矢量；普通SVG文字仍可编辑，数学字形允许矢量路径。

`qa/round2_programmatic_audit.json`核对全部源与输出SHA、400dpi、字体字形、实际PDF字框及无图像嵌入，最终18图0问题。锚点底部公式曾有字框下越界，已对齐普通文本基线后重查，实际PDF所有字框均在页面内。

外部figure-qa检查结果在`qa/programmatic/`，18份检查均成功运行，字体均通过。原始9条启发式警告如实保留：8条把90度轴/色条文字按水平字串估宽；实际Poppler PDF字框及渲染均无裁切；1条是账本纸张/浅阴影的有意叠层。`qa/figure_qa_review.json`保留人工判定及可核对证据。未对连续色图套用类别色allow-list；色阶明度、负相关叉号、正负数值、线型/marker和实际灰度/色觉预览共同验收。

## 复现

使用项目`.venv/bin/python`，单线程、nice19，无新数值实验：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 19 .venv/bin/python analysis/paper_upgrade_round2/20261118/workspaces/F/scripts/make_round2_figures.py --repo-root "$PWD" --out-root "$PWD/analysis/paper_upgrade_round2/20261118/workspaces/F" --qa
.venv/bin/python analysis/paper_upgrade_round2/20261118/workspaces/F/scripts/qa_round2_figures.py --repo-root "$PWD" --out-root "$PWD/analysis/paper_upgrade_round2/20261118/workspaces/F"
```

`--out-root`内部生成`figures`、`data/figure_data`、`qa`。`--baseline-root`默认脚本所在工作区，为空输出提供原数据与字体元数据；完整研究包须同时保留F源码/基线数据、B/C/D/E正式工件、原records与论文数据。每次实际使用的绘图源码写入不可变`out-root/scripts/generations/<SHA>.py`，局部排版修订后仍能核对原已出图的源版本。字体依赖为当前合法安装的Times/SimSun；跨机器需提供相同可用字体或显式披露替代。

本工作区验收覆盖独立图资产，最终主文/SI版面与文内引用由G/PI编译后验收。
