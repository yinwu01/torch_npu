# torch_npu profiler 在 free-threaded Python 下的线程安全

本分支修复了 profiler 上报链路中的三处线程安全问题。这三处修改（`663e8ba`、`f36d74d`）只在 stub 的 C++ 单元测试中专门测试过。11-03 在 NPU 上复现了候选 1 的崩溃路径：`npu_check.py --stack` 在 `2148353` 上 SIGSEGV，栈顶 `ModuleInfo::ModuleInfo → PyObject_GetAttrString`。是否为使用方崩溃的原因，结论待定。本分支 `1ece90d` 已修改。11-06 用真实环境编译了含本分支全部代码修改的 `9f34a2a`，并在 NPU 上运行：`npu_check.py --stack` 正常退出；修改前的构建在同一环境中仍 SIGSEGV。这些运行中没有专门触发三处线程安全问题的测试。

## 问题

使用方在 FT 构建（CPython 3.14t）下开启 `torch_npu.profiler`，进程 SIGSEGV。使用方没有提供调用栈。

已完成的工作：

- 静态阅读上报链路，找到三处线程安全问题，在本分支修复（[fix_details.md](fix_details.md)）。
- stub 环境的压力测试：原代码 42 次场景运行全部失败，修复后 42 次全部通过。
- 上游对照时发现两处在单线程、有 GIL 时同样成立的问题。本分支已修改这两处（`663e8ba`、`1ece90d`）。

## 崩溃根因的三个候选

| 候选 | 内容 | 成立条件 | 本分支 |
|---|---|---|---|
| 1 | 基线缺少上游 `b2e659d62f`。3.13 及以上读 `nn.Module` 帧的 `self` 得到空指针，随后对空指针取属性。 | 单线程、有 GIL 时也成立 | 崩溃路径已在 NPU 上复现（单线程、`--stack`）。是否为使用方崩溃的原因，结论待定。`1ece90d` 已修改，11-06 重测不再崩溃 |
| 2 | 上游 issue #5658：原 `clear()` 在不持有 GIL 时 `Py_DECREF` `TraceContext`。 | 单线程、有 GIL 时也成立 | 已完成（`663e8ba`）：`clear()` 先取 GIL 再释放 |
| 3 | 三处线程安全问题之一 | 只在并发时成立 | 已完成（`663e8ba`、`f36d74d`） |

按配置排除（推断，[analysis.md](analysis.md) 第 1.5 节）：

- 候选 1、2 要求开启 `with_stack` 或 `with_modules`，并采集 CPU activity。否则 `PythonTracer` 不启动，只剩问题 1、2。
- 3.13 及以上，采集期间只要调用过 `nn.Module`，候选 1 就在第一次前向时发生，早于停止采集。

## 接手步骤

完整步骤、流程图和诊断手段见 [analysis.md](analysis.md) 第 7.1 节。

1. 向使用方要 profiler 配置、是否多线程和崩溃时机。
2. 在原构建上运行单线程最小脚本（`npu_check.py --stack`，见“文件”一节），用 gdb 栈顶检查候选 1。
3. 在原构建上只补 `b2e659d62f` 的 `PyMapping_GetItemString` 改法，另加判空（`b2e659d62f` 没有判空，见 [analysis.md](analysis.md) 第 7.1 节“第 3 步的修改做法（与 `1ece90d` 相同）”后的警告）。单线程含停止采集，检查候选 1、2；多线程重复运行，检查候选 3。
4. 本分支已补同一处修改（`1ece90d`）。用第 3.2 步的用例运行。只有第 3.2 步复现了崩溃、这里不再崩溃，才算确认修复。

> [!WARNING]
> 按 gdb 栈顶或 ASan 报告的位置判断原因，不按是否崩溃判断。多线程用例要重复运行。

## 当前状态

| 事项 | 状态 |
|---|---|
| 三处问题的修改（`663e8ba`、`f36d74d`）和压力测试 | 已完成 |
| 候选方案对比，与上游 `ff5ad88d8a` 的对照 | 已完成 |
| 候选 1 的修改（`1ece90d`）在 NPU 上重测（[progress.md](progress.md) 2026-11-06） | 已完成 |
| 接手步骤第 2 步（候选 1）。注：11-03 在 `2148353` 上复现了崩溃路径，没有在原构建上复现 | 待办 |
| 接手步骤第 1、3、4 步 | 待办 |
| 真实 torch、CPython 3.14t、CANN 编译 `9f34a2a`，在 NPU 上运行 `npu_check.py`（[progress.md](progress.md) 2026-11-06） | 已完成 |
| 问题 1、2、3 的修改在 NPU 上的专门测试 | 待办 |
| 问题 3 的合入方式（建议以上游为基础，[analysis.md](analysis.md) 第 7.3 节） | 待办 |
| [fix_details.md](fix_details.md) 第 6 节的其他问题 | 待办 |

逐项的下一步见 [progress.md](progress.md)。

## 文件

| 文件 | 内容 |
|---|---|
| [fix_details.md](fix_details.md) | 三个问题、修改、验证、行为变化、未处理的问题 |
| [analysis.md](analysis.md) | 定位方法、候选方案和测量、与上游的对照、接手步骤 |
| [learning.md](learning.md) | 背景知识（附官方出处）和 8 条过程经验 |
| [progress.md](progress.md) | 状态表和进展记录 |

测试代码在 [`test/cpp/profiler_thread_safety/`](../../test/cpp/profiler_thread_safety)。运行方法见 [fix_details.md](fix_details.md) 第 4.3 节。依赖：Linux，g++（C++17），Python 3.8 以上。不需要 torch、CANN、NPU、网络。

NPU 检查脚本：[`test/profiler_free_threaded/npu_check.py`](../../test/profiler_free_threaded/npu_check.py)。依赖：Python 3.14t，torch 2.13.0，待测的 torch_npu，一张 NPU，不需要网络。`--stack` 同时开启 `with_stack` 和 `with_modules`。

## 术语

| 术语 | 含义 |
|---|---|
| GIL | 全局解释器锁。同一时刻只允许一个线程执行 Python 字节码。 |
| FT 构建 / GIL 构建 | 去掉 GIL 的 CPython 构建（编译宏 `Py_GIL_DISABLED`）/ 默认构建。 |
| 写入方 | 调用 `Report` 写入记录的线程，包括业务线程和 C++ 线程（算子下发、NPU 任务队列出队）。 |
| 读取方 | `DataDumper` 或 `TraceDataDumper` 的后台线程。它从 `RingBuffer` 取出记录并写文件。 |
| 停止方 | 执行 `Stop` 和 `UnInit` 的线程。 |
| 上报器 | `ProfilerMgr` 的 `dataReceiver_`、`dataReceiverWithLock_`、`traceDataReceiver_`。 |
| 开始采集 / 停止采集 | `prof.start()` 或进入 `with` 块 / `prof.stop()` 或退出 `with` 块。 |
| 问题 1、2、3 | 本分支修复的三处问题（[fix_details.md](fix_details.md) 第 3 节）。 |
| 候选 1、2、3 | 崩溃根因的三个候选（上表）。 |
| 原代码 | 基线 `d3216c2` 的源码。 |
| 原构建 | 不含本分支修改、也不含 `b2e659d62f` 的 torch_npu 构建。 |
| 本分支 / 上游 | `fix/profiler-thread-safety` / `Ascend/pytorch` master。 |
| stub | 只保留接口、实现最简的替代头文件，用于在没有 torch、CANN、CPython 3.14 的机器上编译被测源码。 |
| 场景运行 | 一个测试场景在一种编译方式下运行一次。不含编译检查。 |
| TSan、ASan、UBSan、LSan | 分别检测数据竞争、内存错误（越界、释放后访问、重复释放）、未定义行为、内存泄漏。 |
| TSan 构建 | 用 `-fsanitize=thread` 编译。 |
| ASan 构建 | 用 `-fsanitize=address,undefined` 编译，并开启 LSan。它的失败可能来自 ASan、UBSan 或 LSan。 |
| plain 构建 | 用 `-O2` 编译，不带 sanitizer。 |
| `seq_cst` | `std::memory_order_seq_cst`。所有 `seq_cst` 操作处于一个所有线程都认同的全局顺序中。 |
| 原子读-改-写 | 一次完成读出、修改、写回的原子操作，如 `fetch_add`、CAS。 |
| CAS | 比较并交换。变量等于期望值时才写入新值。 |
| 缓存行 | CPU 缓存与内存交换数据的最小单位，本机 64 字节。多个线程频繁写同一缓存行时产生争用。 |
| 热路径 | 采集期间每条记录或每个 Python 调用都执行的代码：`Report`、`Push`、profile 回调。 |
| SPSC | 单写入方、单读取方。 |
| 线性一致 | 每个操作等价于在调用到返回之间的某一时刻完成。对队列，不同写入方的元素按这个时刻的先后出队。 |
| 互斥锁 / 读写锁 | `std::mutex` / `std::shared_mutex`。 |
| tstate | `PyThreadState`，每个线程的线程状态。profile 回调存在其中。 |
| attach / detach | 线程把 tstate 绑定到解释器 / 解除绑定。GIL 构建中对应获取 / 释放 GIL。 |
| STW | stop-the-world。CPython 暂停调用线程以外的所有线程。线程在 attach 状态下阻塞时，STW 无法完成。 |
| `PyMutex` | CPython 3.13 起的互斥锁。等锁时 detach，不可重入。 |
| ctx | `TraceContext`，tracer 传给 `PyEval_SetProfile` 的 Python 对象。回调通过它找到 tracer。 |
| 重入 | 同一线程持有一把锁时，又调用需要同一把锁的函数。 |
| 变异测试 | 故意把某处修复改回错误写法，检查测试能否发现。 |
| 变体 / 参照变体 | 按候选方案改写的源码副本 / 去掉部分同步、不是有效修复的副本，只用于给出开销的参照值。 |
| host regions、r22 分支 | 基线所带的实验特性和它所在的分支，见 [`host_regions_r22.md`](../experimental_free_threaded/host_regions_r22.md)。 |
| 核实状态 | 用于事实。已验证：读过原文、运行过命令或测试。推断：只根据阅读代码得出。未验证：没有核实。 |
| 事项状态 | 用于工作事项。已完成、进行中、待办。 |

## 分支上的提交

分支 `fix/profiler-thread-safety`，远端 <https://github.com/yinwu01/torch_npu>。基线是 `113xiaoji/torch_npu` 的 `codex/r22-real-four-worker-20261027`（`d3216c2`，TorchNPU 2.13.0，基线 `46936d1`）。本分支修改的四个源文件与 2.13.0 逐字节相同，所以问题不是 r22 分支引入的。

| 提交 | 内容 |
|---|---|
| [`663e8ba`](https://github.com/yinwu01/torch_npu/commit/663e8ba3da4b6db43ca04971b82bdb9bb1e96ba8) | 三个问题的修改加压力测试 |
| [`f36d74d`](https://github.com/yinwu01/torch_npu/commit/f36d74df1e4daf3af7e81387f1d367abc1fbaee0) | `startOne` 的 ctx 引用在异常路径上泄漏；在 `data_dumper.cpp` 注释中写明 `seq_cst` 要求和 fork 限制 |
| [`675f2e9`](https://github.com/yinwu01/torch_npu/commit/675f2e9c5aa096da70cd649ca4dc00dd28b68a18) | 交接文档（现在的 `fix_details.md`） |
| [`da752e6`](https://github.com/yinwu01/torch_npu/commit/da752e6a12e87e79095da4f23050cb1f81833533) | 更正交接文档中对上游方案的概括和审阅条目统计 |
| [`d2a3f84`](https://github.com/yinwu01/torch_npu/commit/d2a3f849b66302c851842c3c86a54552524148e8) | 新增 `test/profiler_free_threaded/npu_check.py` |
| [`1ece90d`](https://github.com/yinwu01/torch_npu/commit/1ece90db455b51dc82ac3b46cb1ce53e370809ce) | 3.13 及以上用 `PyMapping_GetItemString` 读 `self`，取不到时跳过；`npu_check.py` 加 `--keep-output` |
| [`090c9ef`](https://github.com/yinwu01/torch_npu/commit/090c9ef6d05cdbc936a5de3d526ea087f11656ae) | `npu_check.py` 加 `--export {text,db}` 和每次运行的产物清单 |
| 其余提交 | 本目录的文档 |

## 交付状态

代码经过三轮审阅，文档经过多轮审阅，全部由 AI 模型子代理完成，没有人工审阅。改写（`ee31dbb`）后经过两轮独立审阅，修改在 `757b3f7`、`2148353`。之后 11-05 一轮审阅覆盖到 `4f5bff0`，26 条意见，修改在 `99d5500`、`f1eb4d3`；再一轮覆盖到 `f1eb4d3`，11 条意见，修改在 `f1eb4d3` 之后的提交（提交信息 “docs(profiler): apply review round 2; record the --export db round”）。这些修改和之后的提交未经审阅。代码的三轮审阅只覆盖 `663e8ba`、`f36d74d`。经过见 [progress.md](progress.md)。
