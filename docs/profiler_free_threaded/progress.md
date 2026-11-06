# 进展记录

代码修改、测试和文档已完成。候选 1 的崩溃路径已在 NPU 上复现，它的修改已在 NPU 上重测。候选 1 是否为使用方崩溃的原因，结论待定。含全部代码修改的 `9f34a2a` 已用 FT 版环境编译，并在 NPU 上运行。候选 2、3 的确认、GIL 版编译、问题 1、2、3 的修改在 NPU 上的专门测试是待办。

- 对象：分支 `fix/profiler-thread-safety`，基线 `113xiaoji/torch_npu` 的 `codex/r22-real-four-worker-20261027`（`d3216c2`，TorchNPU 2.13.0）。
- 提交时间来自 `git log`，时区 `-0800`。
- NPU 结果的日期按测试方的本地日期记录。
- 日志目录名里的时间戳有的是 UTC（以 `Z` 结尾），有的没有标时区。本文按原样引用，不做换算。

## 如何更新本文件

1. 在“进展”开头加一条，最新的在最上面。写日期、做了什么、证据（提交号、日志或命令）。
2. 状态表只保留当前状态。状态改变时，同时在“进展”中加一条。
3. 状态只用三个值：已完成、进行中、待办。
4. 只写有出处的日期和数字。

## 当前状态

| 事项 | 状态 | 下一步 |
|---|---|---|
| 定位三处线程安全问题 | 已完成 | 无 |
| 问题 1、2 的修改 | 已完成 | 手工移植到上游 master（上游做过 clang-format） |
| 问题 3 的修改 | 已完成 | 见“问题 3 的合入方式” |
| stub 压力测试（原代码 42/42 失败，修复后 0/42 失败） | 已完成 | 按 [learning.md](learning.md) B2 第 5 条加强暂停点，再运行变异测试 |
| 候选方案对比和测量 | 已完成 | 无 |
| 与上游 `ff5ad88d8a` 的对照，用本仓库测试运行上游版本 | 已完成 | 无 |
| 文档目录 | 已完成 | 发现错误时按本文记录 |
| 向使用方要配置和崩溃时机 | 待办 | [analysis.md](analysis.md) 第 7.1 节第 1 步 |
| 在 NPU 上复现候选 1 的崩溃路径（单线程、`--stack`）；在本分支补上 `b2e659d62f` 的 `PyMapping_GetItemString` 改法，另加判空（`1ece90d`；`b2e659d62f` 没有判空，见 [analysis.md](analysis.md) 第 7.1 节“第 4 步的修改做法”后的警告），在 NPU 上重测 | 已完成 | 无 |
| 候选 1 是否为使用方崩溃的原因 | 待办 | 结论待定。先做 [analysis.md](analysis.md) 第 7.1 节第 1 步 |
| 确认候选 2、3 | 待办 | [analysis.md](analysis.md) 第 7.1 节第 3–4 步 |
| 4 线程运行是否采集到 worker 线程 | 待办 | 结论待定。开始采集后启动的线程可能不被采集（推断） |
| 崩溃路径 `PythonTraceback::gather` | 待办 | 出处已丢失（测试方找不到 `WORK_REPORT.md`），原因未查明 |
| FT 版编译：用真实 torch、CPython 3.14t、CANN 头文件编译 `9f34a2a` | 已完成 | 无 |
| GIL 版编译：用真实 torch、CPython 3.14、CANN 头文件编译 | 待办 | 与 FT 版分别编译 |
| 在 NPU 上运行 `npu_check.py`（`9f34a2a`，三次：单线程开 `--stack`；4 线程 × 20 次开 `--stack`；单线程不开 `--stack`） | 已完成 | 无 |
| `npu_check.py --export db` 一轮 | 待办 | 检查 `CANNTimelineParser` 是否不再报 run failed |
| 在 NPU 上专门测试问题 1、2、3 的修改 | 待办 | 多线程推理开 profiler，覆盖 `with_stack=True`、`profile_memory=True`、`schedule` 带 warmup；运行 `test/profiler/` 确认 GIL 版行为不变 |
| 问题 3 的合入方式 | 待办 | 由接手的工程师或上游维护者决定。建议以上游为基础（[analysis.md](analysis.md) 第 7.3 节） |
| host regions 的 worker 线程算子可能采集不到 | 待办 | 核实 worker 的数量和创建时机 |
| `with_stack` 在 FT 构建下的锁争用 | 待办 | 改为每线程缓存。上游已是这种结构 |
| `ProfilerMgr` 剩余的竞态 | 待办 | 本次没有改，`ProfilerMgr` 也没有编译进测试 |
| GIL 构建中 `record_params_` 沿用到下一次采集 | 待办 | 上游原有的问题，修复会改变 GIL 构建的行为 |
| fork 出的子进程中停止采集可能无限等待 | 待办 | 已写进代码注释 |
| 两轮 stop/start 期间漏采一个线程 | 待办 | 已写进代码注释。改用上游方案后不存在 |

## 进展

### 2026-11-06：在 NPU 上重测候选 1 的修改

- 结果由 NPU 测试方提供。
- 环境：8 × Ascend950PR，驱动 25.6.rc3，CANN toolkit 9.1.0，Python 3.14.3 free-threaded（`gil_enabled=False`），torch 2.13.0+cpu，aarch64。
- 本分支构建：torch_npu `2.13.0+git9f34a2a`，含 `1ece90d`。
  - `9f34a2a` 含本分支全部代码修改（`663e8ba`、`f36d74d`、`1ece90d`）。问题 1、2、3 的修改随它编译和运行，但下面的运行没有专门触发它们。
  - `python npu_check.py --stack`：退出码 0。
  - `python npu_check.py --threads 4 --repeat 20 --stack`：退出码 0，20 次全部正常。
  - `python npu_check.py`（不加 `--stack`）：退出码 0。
- 对照构建：同一环境，修改前的 torch_npu wheel，版本号 `2.13.0+gitunknown`，10-28 构建，早于 `663e8ba`，不含本分支任何修改。基线提交未知（版本号为 `gitunknown`）。
  - `python npu_check.py --stack`：退出码 139（SIGSEGV）。
  - C 栈顶：`libtorch_npu.so` 调用 `PyObject_GetAttrString`。`libtorch_npu.so` 没有调试符号，只有偏移。
  - Python 栈：`torch/nn/modules/module.py` 的 `_wrapped_call_impl` → `npu_check.py` 的 `work`。
  - C 栈顶函数（`PyObject_GetAttrString`）与 11-03 相同。没有调试符号，`ModuleInfo::ModuleInfo` 这一层无法核对。
- 结论：在这个环境中，含 `1ece90d` 的构建不再出现候选 1 的崩溃路径。是否为使用方崩溃的原因，结论待定。
- 归因：`9f34a2a` 比对照构建多出三个代码提交（`663e8ba`、`f36d74d`、`1ece90d`）。归因到 `1ece90d` 还依据 11-03 的运行：含 `663e8ba`、`f36d74d` 但不含 `1ece90d` 的 `2148353` 同样崩溃在 `PyObject_GetAttrString`。
- 注：对照构建是 10-28 的修改前构建，与 11-03 使用的 `2148353` 不完全相同。
- 未解决：
  - 4 线程运行不能说明 worker 线程被采集。开始采集后启动的线程可能不被采集（推断）。
  - 崩溃路径 `PythonTraceback::gather`：出处已丢失（测试方找不到 `WORK_REPORT.md`），原因未查明。
- 与本分支修改无关的现象：本分支构建的每次运行都输出 9 条解析任务 "run failed"（`CANNTimelineParser` 和依赖它的 8 个任务）。msprof 日志为 "Current scene of data is based on single_op" 和 "There is no summary data to export"，没有导出 `mindstudio_profiler_output/*.csv`。`CANNTimelineParser` 等待这些 csv；只剩 NON_BLOCKING 任务时，任务管理器按设计停止。修改后的构建上开或不开 `--stack` 都出现。修改前的构建只运行了开 `--stack` 的一次，在解析前崩溃；不开 `--stack` 没有运行，所以修改前的构建上是否出现，无法判断。这是 CANN 一侧的行为，不在本分支范围内。db 导出的测试待办。

### 2026-11-03：在 NPU 上运行 `2148353`

- 结果由 NPU 测试方提供。
- 检查脚本 `test/profiler_free_threaded/npu_check.py` 由 `d2a3f84`（03:10）加入。
- 构建：`2148353`。它含 `663e8ba`、`f36d74d`，不含 `1ece90d`。它不是 [analysis.md](analysis.md) 第 7.1 节定义的原构建。
  - 单线程开 `--stack`：SIGSEGV。栈：`pyProfileFn → recordPyCall → genPyCallHashId → ModuleInfo::ModuleInfo → PyObject_GetAttrString`。
  - 单线程不开 `--stack`：退出码 0。
  - 4 线程 × 20 次，不开 `--stack`：退出码 0。
  - 4 线程 × 20 次，开 `--stack`：跑到第 14 次时磁盘写满，没有崩溃。
- 运行之后的提交：`1ece90d`（04:35）修改候选 1，`npu_check.py` 加 `--keep-output`。它的提交信息引用了上面的运行。

### 2026-11-02：按 ASD-STE100 改写本目录

- 五个文件按 ASD-STE100 的写法规则改写：短句、编号步骤、术语统一、警告单独成块，加 6 张 Mermaid 图。
- 审阅过程和本机路径移出正文，本机日志的来源列在 [analysis.md](analysis.md) 附录。
- 改写已提交为 `ee31dbb`。
- 改写后经过两轮独立审阅。按第二轮意见做的修订（`2148353`）和之后的提交（`d2a3f84` 至 `4f5bff0`）未经审阅。代码的三轮审阅只覆盖 `663e8ba`、`f36d74d`。

### 2026-11-02：整理后的五轮文档审阅

- 五轮审阅由 AI 模型子代理完成，意见未公开。每轮的修改已并入当前文本。
- 第二轮提出 21 条意见。主要修改：接手步骤改为四步；补 `PYTHONFAULTHANDLER=1`、`PYTHONMALLOC=debug`、gdb 三种诊断手段；改正第 5.4 节端点比值。同轮新增 faulthandler 和 `PYTHONMALLOC=debug` 的本机核对，结果见 [analysis.md](analysis.md) 第 7.1 节。
- 第三轮：五个文件的接手步骤统一为同一顺序；删去以调用速率作为选择依据的结论。修改提交为 `d1bbbad`。
- 第四轮：接手步骤改为“准备，再分四步”，第 3 步分单线程和多线程，第 4 步在本分支上确认修复；写明 02:33 的测量是例外。
- 第五轮：第 3.1 步按栈顶区分候选 2 和候选 3 的问题 1；第 3.2 步先关闭 `with_stack`、`with_modules`；第 4 步写明“只有第 3.2 步复现了崩溃、这里不再崩溃，才算确认修复”。修改后未再经独立审阅。

### 2026-11-02：整理文档目录

- 新建本目录。把 `docs/experimental_free_threaded/profiler_thread_safety.md` 用 `git mv` 移为 `fix_details.md`，更新 `test/cpp/profiler_thread_safety/README.md` 的链接。新增其余四个文件。
- `fix_details.md` 的改正：审阅轮数按记录重数；删去“上游不存在全局锁争用”；新增两个崩溃候选；补审计回调的前提；回调吞吐从“420 万到 470 万、110 万到 140 万”改为日志中的 418 万到 419 万、124 万到 141 万。
- 整理后第一轮文档审阅的意见没有单独记录。

### 2026-11-02 03:48 和 03:51：交接文档

- `675f2e9` 写出交接文档。`da752e6` 更正对上游方案的概括和审阅条目统计。
- 两轮文档审阅：第一轮在 `675f2e9` 之前，16 条意见；第二轮在两次提交之间，指出对上游方案的概括不准确。
- 写文档时才查到上游 `ff5ad88d8a`。

### 2026-11-02 02:47 和 03:07：修复提交

- `663e8ba`：三个问题的修改加压力测试。此前做了两轮代码审阅和修复，每轮两位审阅者。
- 前两轮共 21 条意见：16 条改了代码或测试，2 条改正措辞，1 条写进代码注释，2 条只做了性能测量（`writers_` 开销、`with_stack` 锁争用）。其中几个问题由前一轮修复引入（[learning.md](learning.md) B1）。
- 第二轮修复者用加强后的测试自己发现了两处问题（`session_` 的第一版、`recordReturn`）。
- `f36d74d`：第三轮（最终）审阅的两位审阅者都判定没有严重或重要问题。确认的一处引用泄漏在本提交修复，其余轻微问题写进代码注释或 [fix_details.md](fix_details.md) 第 6 节。
- 早期文档写的“四轮独立审阅”“第三轮发现第二轮修复中的两处问题”与记录不符，已改正。
- 提交信息中的“49/49 runs pass”包含 7 次编译检查。场景运行是 42 次。

### 2026-11-02 00:13 之前：收到任务

- 本机最早的可核对记录是 00:13。
- 任务分两步：确认这个分支做了哪些修改；修复线程安全问题并完成 profiler 适配。
- 使用方只提供一句描述：FT 下开启 profiler 出现 SIGSEGV，没有调用栈。

## 交付状态

- 问题 1、2、3 的修改（`663e8ba`、`f36d74d`）只在 stub 的 C++ 单元测试中专门测试过。11-06 在 NPU 上编译并运行了含全部代码修改的 `9f34a2a`，`npu_check.py` 的三次运行（单线程开 `--stack`；4 线程 × 20 次开 `--stack`；单线程不开 `--stack`）都正常退出，但没有专门触发这三处问题。修改前这三处问题在 NPU 上是否导致崩溃，结论待定。
- 11-03 NPU 结果：`npu_check.py --stack` 单线程 SIGSEGV（退出码 139），栈顶 `ModuleInfo::ModuleInfo → PyObject_GetAttrString`，复现了候选 1 的崩溃路径。不开 `with_stack` 的单线程和 4 线程 20 次都正常退出（出处：11-03 测试方结果，单线程和 4 线程 × 20 次都不开 `--stack`，退出码 0）。4 线程开 `with_stack` 跑到第 14/20 次时磁盘满，没有崩溃（`1ece90d` 提交信息写的是第 13 次）。
- 候选 1 的修改在 `1ece90d`。11-06 NPU 重测：`9f34a2a` 上单线程开 `--stack` 正常退出；修改前的构建在同一环境中 SIGSEGV。4 线程运行可能没有采集 worker 线程（推断），不作为候选 1 修复的证据。候选 1 是否为使用方崩溃的原因，结论待定。候选 2、3 是否也会导致崩溃，结论待定。
- 审阅：代码三轮，每轮两位审阅者，只覆盖 `663e8ba`、`f36d74d`；原交接文档两轮；整理后五个文件五轮；改写（`ee31dbb`）后两轮。按第二轮意见做的修订（`2148353`）和之后的提交（`d2a3f84` 至 `4f5bff0`）未经审阅。所有审阅和修复都由 AI 模型子代理完成，没有人工审阅。
