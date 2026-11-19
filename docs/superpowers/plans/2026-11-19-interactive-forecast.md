# Interactive Forecast Implementation Plan

> 执行方式：在当前目录串行完成；生产代码尚未纳入 HEAD，另建 worktree 将缺少依赖，因此采用新增模块、保留原文件的方式隔离本轮变更。

**Goal:** 提供单参数排名预测与手动启动的持续网页服务。

**Architecture:** `forecast.py` 负责低负载环境与入口；`production/interactive.py` 协调一次预测；`production/web.py` 提供只读 API 和一个受控计算子进程；`production/web_assets/` 保存静态界面。

**Tech Stack:** 现有 Python 虚拟环境、标准库 HTTP/subprocess/SQLite、现有 CatBoost 模型、原生 HTML/CSS/JS。

**Spec:** `docs/superpowers/specs/2026-11-19-interactive-forecast-design.md`

## Global Constraints

- 保留密集期、期初时间约束、模型校验、不可覆盖预测账本与原始实验资产。
- 不重新安装后台任务；单次模式不训练；持续模式单个计算子进程。
- 不改动现有生产算法源码及其指纹；不改动用户其他未提交文件。

## Review Focus

- 停服不能遗留计算子进程；端口冲突时不能启动计算。
- 网络错误和过期缓存不得显示为当前成功预测。
- 指定排名不可改变基础模型训练范围或覆盖19线结果。
- 页面陈旧、活动切换、缺失排名和无活动应明确展示。
- HTTP 不读取任意路径，不写入预测状态；错误不暴露本地路径。

## Tasks

- [x] 1. 编写单线协调器和 CLI 行为测试，确认失败；实现指定排名、模型复用、期初状态、账本记录与中文输出。
- [x] 2. 编写 HTTP/进程生命周期测试，确认失败；实现只读数据与预测记录 API，串行子进程更新与退出清理。
- [x] 3. 根据界面参考实现静态网页，验证排名选择、记录曲线、表格、刷新与陈旧提示。
- [x] 4. 更新运行文档，执行完整 unittest、真实单线请求与桌面/手机浏览器检查；确认所有服务保持停止。

## Execution ledger

- 2026-11-19：通过原 selected_cli 卸载 launchd，返回 `removed`、`models_and_data_preserved=true`。
- Ruling：按用户已经明确的两种入口直接实施；默认当前国服活动与本机网页，复用现有 Python 依赖。当前目录生产代码未跟踪，采用新增独立文件，不创建缺少生产代码的 worktree。
- 单线与HTTP测试：新增16项，分别先观察失败，再通过。真实联网命令 `forecast.py --rank 1000` 成功，当前活动329输出当前分数2,735,383、预测终分6,206,489，实际观测为00:45:05；该实测用于功能验收，不是精度评测。
- 浏览器：Browser插件不可用，使用项目外临时安装的Playwright与本机Chrome。1440×1100和390×844验证19行、下拉切换、表格选线、真实预测记录、刷新与连接错误恢复；无非预期控制台错误、手机页面无横向溢出。参考图与桌面/手机截图已逐一查看，修正手机图表文字缩小问题。
- Ctrl+C实测后8765端口关闭；launchctl确认系统任务未加载且plist已移除。独立只读审查未发现可证实的问题。
- 全量验证：`DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 19 .venv/bin/python -m unittest discover -s tests -v`，485项全部通过，耗时187.989秒。结束后进程检查显示没有 `forecast.py serve` 或 `production.selected_cli cycle` 遗留进程。
