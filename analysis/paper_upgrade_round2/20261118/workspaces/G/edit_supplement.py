from pathlib import Path
B=Path(__file__).resolve().parent;P=B/'manuscript'
s=(B/'original_sources/supplement.tex').read_text().replace('\\input{data/upgrade_macros.tex}','\\input{data/upgrade_macros.tex}\n\\input{data/round2_macros.tex}')
s=s.replace('本补充材料随正文共同编译，所有数值表均为原生LaTeX文本。统计源为封存后的独立评分工件，表格生成器只读CSV/JSON并核对评分清单哈希。\\path{qa/upgrade_number_bindings.json}逐项记录数值、来源文件、行条件、单位和舍入；\\path{qa/upgrade_figure_manifest.json}记录15幅新图的输入、标签、格式及导出。旧五图及其历史验收记录继续保留，新图另行渲染检查。','本材料同时提供正文附录与独立版本，完整保留既有实验并补充训练期诊断。数值表使用原生LaTeX文本。只读绑定器核对封存哈希，并在\\path{qa/upgrade_number_bindings.json}与\\path{qa/round2_number_bindings.json}记录来源、字段、行条件、单位及舍入。图表的输入、代码与导出指纹由独立manifest记录。主文围绕研究问题解释，完整候选、预算及失败记录在此展开。')
s=s.replace('前瞻登记 & 配置、上游提交、预测封存、延迟释放、历史导出及评分接口 & 等待合格真实未来活动，真实未来精度留空',r'''二轮B/C1 & 35期，15/10/10；12候选及18消融 & 训练期回溯外推，共用已发基础预测\\
二轮C2 & 3终点、3历史组、3种子；108拟合 & 12期查询全为35期子集，总参数和近似MAC分别控制\\
二轮D & 前25期校准、24格模拟、末10期事后迁移 & 受限机制诊断，整体运行末封存的迁移身份单列\\
二轮E & 独立重算、76+76条件、功效模型 & 固定损失与探索面板，0个未来活动\\
前瞻登记 & 原20期pilot及新80期独立确认计划 & 新计划待激活，未来精度留空''')
# Preserve the original figures and full historical records in SI, without repeating them in the main narrative.
s=s.replace('\\subsection{合成网格、Monte Carlo单位及失败区域}',r'''\subsection{合成网格、Monte Carlo单位及失败区域}
\upfigure{upgrade_synthetic_sharing}{fig:upgrade-synthetic}{首轮固定合成机制实验。16个生成条件、4种历史长度与9种方法完整保留；每格每长度300条完整Monte Carlo路径，跨格复用底层随机数。状态MSE相对独立EWMA的正差表示负迁移，oracle拥有额外信息。}{synthetic}''')
s=s.replace('\\subsection{全部56单元与参数预算}',r'''\subsection{首轮规模曲线与全部56单元}
每种规模按实际开始时间取最早16、32、48、67期，分别有576、1170、1766、2452个训练前缀，预测同一14期488个两线验证案例。规模、日历覆盖及活动组成共同变化。局部特征、增长目标和边界固定；CatBoost为200轮，共享拟合1个模型、独立拟合2个模型。神经训练60轮，三个种子最终分数均值为主。共享输出、专属头和完全独立的参数及优化预算随结构变化，表格保持原身份。
\input{tables/upgrade_learning_curve.tex}
\upfigure{upgrade_training_size}{fig:upgrade-learning}{首轮重新拟合的规模诊断。所有点反复使用同一14期488案例验证面板；训练规模与日历覆盖共同变化，网络参数与模型数随结构变化。二轮公平对照另在训练内部多终点执行。}{learning}''')
s=s.replace('\\subsection{历史静态基线与预训练模型}',r'''\subsection{既有19线神经残差结构}
既有19线网络在同一基础预测与独立校正上学习残差增量。编码器两层宽度16，输出头宽度8和1；专属头按支持收缩，未见排名用共享输出。相关图使用已结束活动的参考误差，每对至少8期、每线至多3邻居，支持不足回退对数排名距离先验。AdamW学习率.002、权重衰减.05、丢弃率.1、60轮及种子42/123/2026固定，历史不足12期回退。

共享输出、专属头、静态相关线与阶段图注意力的历史测试MAPE为6.716\%、7.240\%、7.213\%、7.218\%。专属头相对共享输出差0.524个百分点，既有活动iid区间[0.261,0.844]。相关线增量差−0.027个百分点，区间跨零。其负结果限定于该轮样本、输入与预算，梯度冲突未被实测。
\upfigure{upgrade_neural_comparison}{fig:upgrade-neural}{既有神经残差结构的匹配比较。活动配对与5活动循环块区间沿用该轮独立统计，历史测试已被探索。参数规模与优化仍有变化，比较范围限于已记录结构。}{neural}

\subsection{历史静态基线与预训练模型}''')
a=s.index('\\subsection{版本绑定、失败记录与复现命令}');z=s.index('\\ifdefined\\PaperMainDocument\\else',a)
s=s[:a]+r'''\input{round2_supplement.tex}

\section{版本绑定、失败记录与独立重建}
输入、源码、配置、依赖与预测分别封存。H独立用标量算术重建B全部22680个候选活动排名阶段状态，\(\lambda\)最大差为\(3.22\times10^{-15}\)；C实际张量参数和5670次优化器更新逐项核对。H的独立临时项目完整重放B35期，并重拟C2一个预定单元，与原预测逐例一致。原C源码快照的导入闭包缺口在可搬运包中补全；全新依赖安装环境仍保留自己的验收范围。

原B backend的物理案例别名守卫由独立适配器补充，原源码和封存预测保留；新入口为\path{integration/replay_guarded.py}，配合\path{sharing_input_guard.py}。H在冻结35期验证新守卫与原输出零差。D的原块接缝修订在真实运行前完成；删一参数附带的full25似然元数据用独立审计文件澄清，原表保留。E的statistics001为中断，reliability001因非中心t极尾NaN另存修订；本文数字统一来自statistics002/reliability002，失败目录随档案保留。

稿件内的一键数值重建显式传入仓库根，避免目录层级推断：
\begin{flushleft}\small
\path{.venv/bin/python <manuscript>/scripts/bind_round2_evidence.py}\\
\path{--repo-root <absolute-repository> --output <absolute-manuscript>}
\end{flushleft}
该命令先重建原15张表与宏，再绑定本轮11张表与宏，逐项记录数值来源和SHA。它只读已封存分数，不拟合、重新评分或改变选择。图表由封存数值绘制，随后在稿件目录运行\texttt{make all}，使用XeLaTeX、BibTeX及交叉引用刷新生成完整稿与独立补充材料。\texttt{make verify}核对引用、字体及LaTeX警告；逐页渲染和接触表另存QA目录。

项目要求的全套\path{.venv/bin/python -m unittest discover -s tests -v}已由PI按单线程串行执行，469项通过。二轮B15、C19、D19、E12及独立守卫5项夹具另计，共70项通过。E相对路径夹具首次从工作目录误调用的失败日志保留，按仓库根契约重跑通过，代码保持原版本。软件测试检验执行边界与实现，预测精度来自独立评分。

\subsection{尚待完成的外部证据}
当前真实未来活动为0，80期新队列等待新的激活封存。完整人口清单、准时预测、原始来源及实际释放日志仍依赖上游采集和独立审计。跨服务器和跨游戏验证、历史元数据时点快照与玩家层行为资料，是解释和推广当前结论所需的新增证据。所有历史分数保留测试探索、验证复用与回溯元数据限制；本轮35期研究亦保留回溯身份。

'''+s[z:]
(P/'supplement.tex').write_text(s)
# Bibliography edits: identities, URLs, punctuation and six relevant original sources.
b=(P/'references.bib').read_text()
b=b.replace('doi={10.1115/1.3662552}, url={https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf}','doi={10.1115/1.3662552}, url={https://doi.org/10.1115/1.3662552}')
b=b.replace('https://arxiv.org/abs/0811.0405}', 'https://arxiv.org/abs/0811.0405v1}').replace('note={Author preprint, version 1}',r'note={\space Author preprint, version 1}')
b=b.replace('https://arxiv.org/abs/2604.13253}', 'https://arxiv.org/abs/2604.13253v1}').replace('https://arxiv.org/abs/2605.06541}', 'https://arxiv.org/abs/2605.06541v1}').replace('https://arxiv.org/abs/2510.15821}', 'https://arxiv.org/abs/2510.15821v1}')
b+=r'''
@article{bates1969combination,
  author={Bates, J. M. and Granger, C. W. J.}, title={The Combination of Forecasts},
  journal={Journal of the Operational Research Society}, year={1969}, volume={20}, number={4}, pages={451--468}, doi={10.1057/jors.1969.103}}
@article{feldman2014averaging,
  author={Feldman, Sergey and Gupta, Maya R. and Frigyik, Bela A.}, title={Revisiting {Stein}'s Paradox: Multi-Task Averaging},
  journal={Journal of Machine Learning Research}, year={2014}, volume={15}, number={106}, pages={3621--3662}, url={https://www.jmlr.org/beta/papers/v15/feldman14a.html}}
@inproceedings{zhang2010relationships,
  author={Zhang, Yu and Yeung, Dit-Yan}, title={A Convex Formulation for Learning Task Relationships in Multi-Task Learning},
  booktitle={Proceedings of the 26th Conference on Uncertainty in Artificial Intelligence}, series={Proceedings of Machine Learning Research}, year={2010}, volume={R8}, pages={700--709}, url={https://proceedings.mlr.press/r8/zhang10a.html}}
@article{gibbs2024online,
  author={Gibbs, Isaac and Candès, Emmanuel J.}, title={Conformal Inference for Online Prediction with Arbitrary Distribution Shifts},
  journal={Journal of Machine Learning Research}, year={2024}, volume={25}, number={162}, pages={1--36}, url={https://jmlr.org/papers/v25/22-1218.html}}
@article{gneiting2007proper,
  author={Gneiting, Tilmann and Raftery, Adrian E.}, title={Strictly Proper Scoring Rules, Prediction, and Estimation},
  journal={Journal of the American Statistical Association}, year={2007}, volume={102}, number={477}, pages={359--378}, doi={10.1198/016214506000001437}, url={https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf}}
@article{newey1987hac,
  author={Newey, Whitney K. and West, Kenneth D.}, title={A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix},
  journal={Econometrica}, year={1987}, volume={55}, number={3}, pages={703--708}, doi={10.2307/1913610}, url={https://www.nber.org/papers/t0055}}
'''
(P/'references.bib').write_text(b)
