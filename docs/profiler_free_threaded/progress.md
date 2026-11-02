# 进展记录

- 对象：`yinwu01/torch_npu` 分支 `fix/profiler-thread-safety`（基线 `113xiaoji/torch_npu` 的 `codex/r22-real-four-worker-20261027`，`d3216c2`，TorchNPU 2.13.0）。
- 本文记录做过什么、现在停在哪里、接手的人要做什么。技术内容在 [fix_details.md](fix_details.md)、[analysis.md](analysis.md)、[learning.md](learning.md) 里，本文不重复。
- 时间说明：提交时间来自 `git log`，时区为 `-0800`；日志目录名里的时间戳有的是 UTC（以 `Z` 结尾），有的没有标时区，本文按原样引用，不做换算。
- 文中 `torch_npu-work/…` 路径都在作者本机（`/SSD1/wuyin/project/torch_npu-work/`），没有上传，接手的工程师无法打开。

## 如何更新本文件

- 在"进展"一节开头插入新的日期条目，最新的在最上面。每条写：日期、做了什么、证据（提交号、日志路径或命令）。
- 状态表只保留当前状态，不保留历史；状态改变时同时在"进展"里加一条。
- 状态列只用三个值：已完成、进行中、待办。
- 只写有出处的日期和数字：提交时间用 `git log`，测试结果用日志文件。没有出处的时间不写。

## 当前状态

| 事项 | 状态 | 证据 | 下一步 |
|---|---|---|---|
| 定位三处线程安全问题 | 已完成 | [fix_details.md](fix_details.md) 第 3 节；定位方法见 [analysis.md](analysis.md) 第 1 节 | 无 |
| 问题 1、2 的修改（`ring_buffer.h`、`data_dumper.h/.cpp`） | 已完成 | `663e8ba`、`f36d74d` | 移植到上游 master（手工合入，上游做过 clang-format） |
| 问题 3 的修改（`profiler_python.cpp`） | 已完成 | `663e8ba`、`f36d74d` | 先决定是继续用这套写法还是改用上游 `ff5ad88d8a`，见下一行 |
| stub 环境的压力测试 | 已完成 | `test/cpp/profiler_thread_safety/`；原代码 42 次场景运行全部失败，修复后 42 次全部通过（[fix_details.md](fix_details.md) 第 4.2 节） | 按 [learning.md](learning.md) B2 第 5 条加强暂停点后重新运行变异测试 |
| 候选方案对比和测量 | 已完成 | [analysis.md](analysis.md) 第 2、3、6 节；原始记录 `torch_npu-work/alt-bench/results/20261102T120205Z/` | 无 |
| 与上游 `ff5ad88d8a` 的对照、用本仓库测试运行上游版本 | 已完成 | [analysis.md](analysis.md) 第 4、5 节；日志 `torch_npu-work/upstream-harness/logs/` | 无 |
| 文档目录 `docs/profiler_free_threaded/` | 进行中 | 本目录五个文件，快照已提交为 `87dcd0e`；第三轮审阅意见的修改未提交，修改后未再经独立审阅 | 独立审阅后提交 |
| 向使用方要 profiler 配置（`with_stack`、`with_modules`、`activities` 是否含 CPU、`profile_memory`、`record_shapes`）、是否多线程和崩溃发生的时间点 | 待办 | [analysis.md](analysis.md) 第 1.5 节（按配置和时机排除候选） | 接手后的第一步，见 [analysis.md](analysis.md) 第 7.1 节第 1 步 |
| 基线缺少上游 `b2e659d62f`，3.13+ 读 `self` 得到空指针 | 待办 | [analysis.md](analysis.md) 第 4.2 节；CPython C API 行为已验证（`torch_npu-work/py314-checks/`），torch_npu 上未验证 | 按 [analysis.md](analysis.md) 第 7.1 节第 2 至 4 步：原构建单线程运行最小脚本（第一次前向时崩溃对应候选 1）；原构建只补 `b2e659d62f` 的 `PyMapping_GetItemString` 和判空部分后重新运行并包含停止采集（候选 2）；本分支补同一处修改（独立提交，不取决于前两步的结果），编译后运行多线程用例（候选 3 和修复） |
| 用真实 torch、CPython 3.14t、CANN 头文件编译 | 待办 | [fix_details.md](fix_details.md) 第 1 节"未完成的验证" | 在 NPU 机器上分别编译 GIL 版和 FT 版 |
| 在 NPU 上验证 | 待办 | 同上 | 多线程推理开 profiler，覆盖 `with_stack=True`、`profile_memory=True`、`schedule` 带 warmup；再跑 `test/profiler/` 确认 GIL 版行为不变 |
| 问题 3 的合入方式：继续用我们的写法，还是改用上游 `ff5ad88d8a` | 待办 | 两套方案的逐项对照、风险和建议见 [analysis.md](analysis.md) 第 4.8、4.11、7.3 节 | 由接手的工程师或上游维护者决定。本文的建议是以上游为基础 |
| host regions 的 worker 线程上的算子可能采集不到 | 待办 | [fix_details.md](fix_details.md) 第 6 节；`npu_profiler.cpp:260` 的 `addThreadLocalCallback` | 核实 worker 的数量和创建时机；影响是结果缺少这些线程的算子数据，没有观察到崩溃 |
| `with_stack` 在 FT 下的锁争用 | 待办 | [fix_details.md](fix_details.md) 第 5.2、6 节；吞吐见 [analysis.md](analysis.md) 第 5.4 节 | 改为每线程缓存才能彻底解决，上游已是这种结构 |
| `ProfilerMgr` 剩余的竞态：`mstx_domain_include_/exclude_` 两个 `std::vector`，以及 `path_`、`profConfig_` 等非原子成员 | 待办 | [fix_details.md](fix_details.md) 第 6 节 | 本次没有改，`ProfilerMgr` 也没有编译进测试 |
| 有 GIL 时 `record_params_` 会沿用到下一次采集 | 待办 | [fix_details.md](fix_details.md) 第 6 节 | 上游原有的问题，修复会改变 GIL 构建的行为，所以没有修改 |
| fork 出的子进程里停止采集可能无限等待 | 待办 | [fix_details.md](fix_details.md) 第 6 节；[learning.md](learning.md) B7（4/40 个子进程挂起，`torch_npu-work/final-review-upload-r1/fork/`） | 已写进代码注释；失败形态从崩溃变为挂起 |
| 一次 `startOne` 期间经历两轮完整 stop/start 时漏采该线程 | 待办 | [fix_details.md](fix_details.md) 第 6 节 | `startOne` 只重试一次，代码注释已写明。改用上游方案后这条限制不存在 |

## 进展

### 2026-11-02（本次）：按整理后第三轮文档审阅的意见修改本目录

- 第三轮审阅意见（AI 模型子代理，意见未公开）已修改，未提交，修改后未再经独立审阅。主要修改：接手步骤在五个文件中统一为同一顺序（问配置，含 `activities`；原构建单线程检查候选 1；原构建只补 `b2e659d62f` 判空部分后检查候选 2；本分支补同一处修改、编译后运行多线程用例）；`analysis.md` 第 2 节删去以调用速率作为选择依据的结论（各变体丢弃比例不同，调用速率不能比较单次调用开销）；第 5.4 节改正端点比值和 2 线程的说法；第 4.8、4.11 节改正 `c_profileobj` 的读取方；`PYTHONMALLOC=debug` 的适用范围改为不含候选 3。
- 时间顺序：方案选择依据推理；`analysis.md` 第 2、3、5 节的对比测量在 2026-11-02 03:58 至 04:23 `-0800` 进行，晚于修复提交和最初的文档，属于事后核对。
- 没有改 `torch_npu/` 下的源码，没有提交，没有推送。

### 2026-11-02：按整理后第二轮文档审阅的意见修改本目录

- 整理后的第二轮文档审阅提出 21 条意见（AI 模型子代理，意见未公开），逐条核对后修改，未提交。主要修改：
  - `analysis.md`：第 2.2、2.5、2.7、7.2 节补上接受速率的结论（8 线程时读写锁的接受速率 2.24 到 2.49，高于采用方案的 2.02 到 2.16，范围不重叠；依据 `torch_npu-work/alt-bench/results/20261102T120205Z/summary.txt`）；第 1.5 节改写漏掉的问题，并加入按配置和崩溃时机排除候选的依据；第 7.1 节改为四步（问配置、在原构建上运行最小脚本、补 `b2e659d62f`、在本分支上复测），列出 `PYTHONFAULTHANDLER=1`、`PYTHONMALLOC=debug`、gdb 三种手段和堆损坏时的读法；第 4.8、4.11 节写明两边都直接读 `PyThreadState` 的 `c_profilefunc`、`c_profileobj`；第 5.4 节端点比值改为 1.57–3.51（按 `upstream-harness/logs/bench-20261102-042250.jsonl` 逐条计算）；第 5.5 节写明 u2 同时去掉了 `enabled_` 检查；术语表补 UBSan；改正两处数字或行号（4.73、`npu_profiler.cpp:375`）。
  - `README.md`、`fix_details.md`、`progress.md`：同步上面的候选排序、诊断手段和"最小脚本在使用方原来的构建上运行"；README 术语补 UBSan、LSan、`seq_cst`；`fix_details.md` 第 4.1 节补 UBSan，第 3 节"结果"改正 `stop()` 之后不记录事件的依据（`active_`，`tracer_` 只在 `clear()` 时置空）。
  - `learning.md`：改正 A1.6 的行号（541–543）和同一处的 `tracer_` 说法；B4 的任务日期改为"没有可核对的出处"；补齐 stub 前后的空格。
  - 五个文件的审阅状态统一改为"整理后经过两轮文档审阅"。
- 新增核对（2026-11-02，本机）：`torch_npu-work/py314-checks/crash_diag_check.sh`，输出 `crash_diag_check.out`。结果：3.14.3t 上 `-X faulthandler` 只打印崩溃线程的 Python 调用栈（`<Cannot show all threads while the GIL is disabled>`），3.14.3 打印所有线程；`PYTHONMALLOC=debug` 下在线程状态未绑定解释器时调用 `PyObject_Free`，两种构建都立即以 `Fatal Python error: _PyMem_DebugFree` 终止。
- 没有改 `torch_npu/` 下的源码，没有提交，没有推送。

### 2026-11-02：整理文档目录

- 新建 `docs/profiler_free_threaded/`，把原 `docs/experimental_free_threaded/profiler_thread_safety.md` 用 `git mv` 移为本目录的 `fix_details.md`，并更新 `test/cpp/profiler_thread_safety/README.md` 顶部的链接。
- 新增 `README.md`（入口）、`analysis.md`（方案分析）、`learning.md`（背景知识和经验）、`progress.md`（本文）。
- 修改了 `fix_details.md`（相对 `HEAD` 的完整差异用 `git diff -M HEAD -- docs/` 查看）：
  - 改正三处说法：审阅轮数（第 4.4 节，按审阅记录中的标签重数，代码审阅为三轮）；"上游不存在全局锁争用"（第 6、7 节，上游 FT 构建每个事件仍取一次 `results_mutex_`）；"没有逐行对比两套方案、没有用本仓库的测试运行过上游方案"（第 1、7 节，这两件事后来都做了）。
  - 新增崩溃候选：开头加入基线缺少 `b2e659d62f` 和 issue #5658 两条；第 6 节新增 `b2e659d62f` 一行，"锁争用"一行补上上游 `results_mutex_` 的说明。
  - 第 3 节"结果"补上 `sys.setprofile` 审计回调抛异常时的前提。
  - 第 1 节上游提交标题补上 `fix:` 前缀，去掉"两者没有逐行比较"。
  - 用词统一："普通构建"改为 plain 构建，"cache line"改为缓存行。
  - 第 5.2 节回调吞吐数字从"420 万到 470 万、110 万到 140 万"改为日志中的 418 万到 419 万、124 万到 141 万，并写明日志出处。
  - 第 7 节重排步骤：补 `b2e659d62f` 放在第一步，编译步骤链接到 `docs/experimental_free_threaded/` 的具体章节，"普通版"改为 GIL 版，第 6 步补上调用栈的读法。
- 整理后第一轮文档审阅的意见和对应修改没有单独记录。
- 没有改 `torch_npu/` 下的源码，没有提交，没有推送。

### 2026-11-02 03:48 和 03:51 `-0800`：交接文档

- `675f2e9` 写出交接文档，`da752e6` 更正其中对上游方案的概括和审阅条目统计。
- 文档审阅共两轮：第一轮在 `675f2e9` 之前，提出 16 条意见；第二轮在 `675f2e9` 和 `da752e6` 之间，指出对上游方案的概括不准确（[learning.md](learning.md) B4、B6）。
- 写文档期间查上游时才发现 `ff5ad88d8a`（2026-10-29 已重构 `PythonTracer`），这件事发生在设计和修复之后。

### 2026-11-02 02:47 和 03:07 `-0800`：修复提交

- `663e8ba`：问题 1、2、3 的修改加一套压力测试。这个提交之前做了两轮审阅和修复，每轮两位审阅者（都是 AI 模型子代理），前两轮共 21 条意见（[fix_details.md](fix_details.md) 第 4.4 节、[learning.md](learning.md) B1）。
- `f36d74d`：最终审阅发现的一处引用泄漏（`startOne` 持有的 ctx 引用在 `recordStartStack` 抛 Python 异常时泄漏，LSan 报 32 字节）。
- 提交信息里的"49/49 runs pass"包含 7 次编译检查，场景运行是 42 次（[learning.md](learning.md) B5 第 3 条）。

### 2026-11-02 00:13 `-0800` 之前：收到任务

- 本机可核对的最早记录是 `torch_npu-work/clone.log`，修改时间 2026-11-02 00:13 `-0800`；`torch_npu-work/` 下其余条目都晚于它。更早的时间只在未公开的会话记录里，本文不引用。
- 任务分两步：先确认这个分支做了哪些修改，然后修复线程安全问题并完成 profiler 适配。
- 使用方提供的现场只有一句描述：FT 下开启 `torch_npu.profiler` 出现 SIGSEGV，没有调用栈。使用方判断是"上报模块初始化竞态导致空指针访问"，这个判断没有核实。
- 静态阅读上报链路后读出三处线程安全问题（[analysis.md](analysis.md) 第 1 节）。

## 这次工作的交付状态

- 代码修改只在 stub 环境中验证过，没有用真实 torch、CPython 3.14t、CANN 头文件编译，没有在 NPU 上运行。
- 使用方崩溃的根因没有确定。三个候选：(1) 基线缺少 `b2e659d62f`，3.13+ 读 `self` 得到空指针后对空指针取属性；(2) 上游 issue #5658 所述路径，原 `clear()` 在没有持有 GIL 的状态下 `Py_DECREF` `TraceContext`（本分支 `clear()` 已改为持有 GIL 再释放）；(3) 三个竞态中的某一处。候选 (1)(2) 在单线程、有 GIL 时同样成立。在不知道使用方配置和崩溃时机的情况下三者无法排序，知道之后可以排除一部分（[analysis.md](analysis.md) 第 1.5 节）。确定根因的步骤和手段见 [analysis.md](analysis.md) 第 7.1 节。
- 代码修改经过三轮审阅（每轮两位审阅者），原交接文档（即 `fix_details.md`）在 `675f2e9`、`da752e6` 期间经过两轮文档审阅；整理成本目录之后，五个文件一起经过三轮文档审阅，第三轮审阅意见已修改，修改后未再经独立审阅。这些审阅和对应的修复都由 AI 模型子代理完成，没有人工审阅。
