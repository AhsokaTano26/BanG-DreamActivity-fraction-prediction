# 中文 LaTeX 研究论文

作者：Tano。2026-11-19已同步更新正文首页署名和PDF作者元数据。

正文为 `main.tex`，直接在该文件修改。采用 SCI 研究论文的内容结构，尚未指定某个期刊的排版模板。包含中英文摘要、7节正文、8组原生编号公式、7张三线表、5幅矢量图和19条自动引用文献。本机最终编译为16页。

## 阅读与编译

成稿位于 `output/中文论文.pdf`。原生源码不依赖 Word、Markdown 或文档转换器。

`output/paper_latex_source.zip` 包含可独立编译的正文、参考文献、图表、补充结果表和核验记录。解压后在包根目录执行下述命令；上传 Overleaf 时可直接上传该 ZIP。

```sh
cd papers/20261118_sci_cn
make pdf
```

需要 XeLaTeX、BibTeX 与常规 TeX Live 宏包。也可执行：

```sh
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output main.tex
bibtex output/main
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output main.tex
```

上传到 Overleaf 时选择 XeLaTeX，并以 `main.tex` 为主文件。图文件使用相对路径。优先使用本机已有的 SimSun 和 Times New Roman；缺失时，正文自动采用 TeX Live 的 Fandol Song 与 TeX Gyre Termes，具体分页可能变化。已提供的矢量图嵌入 SimSun 和 Times New Roman，源码包不包含字体文件。数学字体为 Latin Modern Math。

## 证据与补充表

| 文件 | 内容 |
|---|---|
| `data/supplement_all_329_results.csv` | 密集时期14个正式运行的329组方法、种子与组合结果；不是329种独立架构 |
| `data/main_models.csv` | 同一19线V3面板的9个主要方法和控制 |
| `data/static_v2_models.csv` | 静态V2两线代表性配方及训练指标性质 |
| `data/rank_results.csv` | 全部19条线的覆盖及未校正、独立校正、双线组合成绩 |
| `data/stage_results.csv` | 三个时间轴、21个条件的覆盖和成绩；缺失保留为空 |
| `data/type_results.csv` 与 `data/band_results.csv` | 活动类型和乐队分组；不作为因果效应估计 |
| `data/correlation_summary.csv` 与相关矩阵 | 训练期水平、增长及误差的统计关系 |
| `data/facts.json` | 队列完整指纹、匹配区间、历史记录和生产验收来源 |
| `data/source_manifest.json` | 44个原始证据文件的SHA-256 |
| `references.bib` 与 `data/reference_sources.json` | 19条参考文献和原始出版物／作者来源 |
| `qa/numerical_validation.json` | 136项数值、来源、结构和打印范围检查 |
| `qa/visual_validation.json` 与 `qa/font_embedding.json` | 最终16页和5幅图的视觉检查，以及PDF字体嵌入记录 |
| `delivery_manifest.json` | 交付文件的逐项SHA-256；源码包哈希另记于本地 `delivery_receipt.json` |

所有表图均从已封存结果提取。本次没有训练模型、修改分组或重新选择测试最佳参数，也没有修改线上代码、配置与权重。

在完整研究仓库根目录，可运行以下命令重新提取证据、生成图表并校验正文：

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B papers/20261118_sci_cn/scripts/extract_evidence.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B papers/20261118_sci_cn/scripts/make_figures.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B papers/20261118_sci_cn/scripts/validate_paper.py
```

独立源码包已有正文编译所需图表与数据，不需要原始数据库即可编译。来源哈希的全量复核需要完整研究仓库。重新绘图需要 NumPy、pandas、Matplotlib、Pillow，以及 `qa/font_check.json` 指向的合法字体；异机应先更新字体检查记录。本机生成图为400 dpi PNG和嵌入字体的矢量PDF，正文编译使用PDF。

## 科学结论与投稿准备

论文明确区分每轮验证事前选择、历史已知测试最优和最终生产采用。最新神经网络实验事前选择仍为独立暖校正；当前双线组合的6.540%是已探索测试上的描述性结果。16期测试、验证重复使用、回溯元数据、多重探索未校正和T1500冷启动均在正文保留。MAPE下降而RMSE未下降也已写入。

本稿可作为中文研究稿继续讨论。正式投稿前仍需由署名作者确认作者顺序与单位、贡献、资助及利益冲突等实际声明，并按目标期刊模板调整。当前证据的关键补充是未参与探索的新活动前瞻验证，以及精确数据和代码版本的公开归档；本地未推送的工件不因引用公共分支链接而变为已公开版本。

早期排版草稿保存在本地 `qa/retired_word_draft/`，不是交付来源，也不纳入可编译源码包。后续正文以 `main.tex` 为准。
