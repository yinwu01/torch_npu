# 学习笔记：背景知识和过程经验

A 部分列出维护这次修改需要的背景知识，每条附官方出处。B 部分列出这次修复中有记录可查的 8 条经验。

- 先读 [fix_details.md](fix_details.md) 和 [analysis.md](analysis.md)。术语见 [README.md](README.md)。
- 出处的访问日期都是 2026-11-02。CPython 源码以当天 `python/cpython` 的 `3.14` 和 `3.13` 分支为准。
- B 部分的原始记录没有公开，也不在仓库中。
- B 部分的审阅者、实现者、修复者都是 AI 模型子代理。这次没有人工代码审阅。

## A. 背景知识

### A1. free-threaded CPython

#### A1.1 PEP 703、PEP 779 和版本状态

- PEP 703 让 CPython 可以不带 GIL 编译和运行。状态 Final，Python-Version 3.13，2023-10-24 决议。用 `--disable-gil` 编译时定义 `Py_GIL_DISABLED`。GIL 构建仍是默认构建。
- PEP 779 规定 FT 构建进入第二阶段（官方支持，仍是可选构建）的条件。状态 Final，Python-Version 3.14，2025-06-16 决议。
- 3.14 What's New 写明 "Free-threaded Python is officially supported"。单线程代码的性能损失约 5–10%（官方自报）。
- FT 构建在运行时可能重新开启 GIL。触发方式：`PYTHON_GIL`、`-X gil`，或导入没有声明支持的 C 扩展（此时打印警告）。
- 与本次的关系：使用方的 SIGSEGV 出现在 3.14 的 FT 构建下（[README.md](README.md)“问题”一节）。有 GIL 时竞态很少暴露，见 [analysis.md](analysis.md) 第 1.3 节。
- 出处：<https://peps.python.org/pep-0703/>，<https://peps.python.org/pep-0779/>，<https://docs.python.org/3.14/whatsnew/3.14.html>，<https://docs.python.org/3.14/howto/free-threading-python.html>

#### A1.2 C 扩展如何识别和声明 FT 支持

- FT 构建定义 `Py_GIL_DISABLED` 为 1，GIL 构建不定义。
- 多阶段初始化的模块加 `{Py_mod_gil, Py_MOD_GIL_NOT_USED}`。单阶段初始化的模块调用 `PyUnstable_Module_SetGIL()`。不声明时，导入打印警告并开启 GIL。
- 与本次的关系：torch_npu 声明支持后才在无 GIL 状态下运行。这是问题暴露的前提。
- 位置：`torch_npu/csrc/InitNpuBindings.cpp:247` 调用 `PyUnstable_Module_SetGIL`，只在同时定义 `Py_GIL_DISABLED` 和 `TORCH_NPU_EXPERIMENTAL_FREE_THREADED` 时编译。这是 r22 分支原有的代码。
- 位置：`torch_npu/csrc/profiler/profiler_python.cpp` 的 `TracerMutex`、`TracerControlMutex`、`ctxKey()`、`start()`、`stop()` 有条件编译。`ring_buffer.h`、`data_dumper.h/.cpp` 没有。
- 出处：<https://docs.python.org/3.14/howto/free-threading-extensions.html>

#### A1.3 attach/detach 和 STW

- 只有 attach 的线程可以访问 Python 对象。GIL 构建中 attach 相当于持有 GIL。
- `PyEval_SaveThread()` 和 `Py_BEGIN_ALLOW_THREADS` 使线程 detach。官方文档写明 FT 构建仍要 detach，因为解释器要周期性地暂停所有线程。
- STW 暂停所有 attach 的线程，垃圾回收会用它。3.14 的 `_PyEval_StopTheWorld()` 只在 FT 构建中有实现。发起方要等其他线程全部 detach 或暂停。
- 与本次的关系：线程在 attach 状态下阻塞在 `std::mutex` 上时，STW 无法完成。持有该锁的线程如果也在等 STW，就死锁。所以 `profiler_python.cpp` 用 `PyMutex`。
- 位置：`TracerMutex` 上方的注释；`test/cpp/profiler_thread_safety/python_tracer/stubs/stub_all.h` 用读写锁模拟 attach/detach 和 STW。
- 出处：<https://docs.python.org/3.14/c-api/threads.html>，<https://github.com/python/cpython/blob/3.14/Python/pystate.c>（`_PyEval_StopTheWorld`、`stop_the_world`、`tstate_wait_attach`）

#### A1.4 PyMutex

- 3.13 新增，目前占 1 字节（大小不保证）。`PyMutex_Lock` 在锁已有持有者时阻塞，阻塞期间临时 detach。`PyMutex` 不可重入，同一线程第二次加锁会一直阻塞。
- 与本次的关系：等锁时 detach，不阻止 STW。代价是：重新 attach 后，其他线程可能已释放之前读到的对象（B1 中 `recordReturn` 的问题）。
- 位置：`TracerMutex`（`registration_mutex_`、`data_mutex_`），`TracerControlMutex`（`control_mutex_`，同线程重入时抛异常）。
- 出处：<https://docs.python.org/3.14/c-api/synchronization.html>

#### A1.5 critical section

- `Py_BEGIN_CRITICAL_SECTION` 于 3.13 新增。它在 FT 构建中获取某个 Python 对象的锁，在 GIL 构建中只展开为 `{` 和 `}`。
- 线程在 critical section 内阻塞时，运行时挂起它并释放锁。所以它不提供 `PyMutex` 那样的独占保证。
- 与本次的关系：要保护的是 C++ 容器，等锁时要保持独占，所以没有用它。改用它时要考虑挂起。
- 出处：<https://docs.python.org/3.14/c-api/synchronization.html>

#### A1.6 `PyEval_SetProfile` 和 `PyEval_SetProfileAllThreads`

- `PyEval_SetProfile(func, obj)` 给当前线程设置 profile 回调，调用方必须 attach。`PyEval_SetProfileAllThreads` 给所有线程设置，3.12 新增。
- 源码行为（`Python/legacy_tracing.c`，3.14 分支最近修改 `f36da66`，2026-04-10）：
  - 两者先在当前线程触发审计事件 `sys.setprofile`。
  - 然后 STW，在暂停期间替换回调，再恢复。
  - 旧的 `obj` 在恢复后才 `Py_XDECREF`。
  - 3.13 分支同样 STW。
  - 公开函数（`Python/ceval.c`）忽略审计回调的异常，只打印 unraisable 信息。
- 与本次的关系：FT 构建的 `start()` 用 `PyEval_SetProfileAllThreads` 安装共享 ctx。`stop()` 传 `nullptr` 清除。替换在暂停期间完成，所以 `stop()` 返回后不会再有线程以旧回调进入。
- 代价：`stop()` 会清掉其他工具（如 cProfile）设置的回调（[fix_details.md](fix_details.md) 第 5.2 节）。
- 前提：审计回调没有抛异常。按 3.14.3 第 541–543、566–568 行，审计回调返回 -1 时函数直接返回，不 STW，不替换。旧回调仍在时，`stop()` 之后由 `active_ == false` 保证不记录。`clear()` 之后由已置空的 `ctx->tracer_` 保证。下一轮 `startOne` 会因该线程的 `c_profilefunc` 非空而跳过它。这些没有测试。
- `startOne` 的 `PyEval_SetProfile` 也 STW。调用线程在这期间可能等待，其他线程的 `stop()`、`clear()` 可能执行完（B1）。stub 没有模拟审计回调（B3）。
- 位置：`start()`（约第 474、497 行），`startOne()`（约第 549、561 行），`stop()`（约第 589 行）。
- 出处：<https://docs.python.org/3.14/c-api/profiling.html>，<https://github.com/python/cpython/blob/3.14/Python/legacy_tracing.c>，<https://github.com/python/cpython/blob/3.13/Python/legacy_tracing.c>，<https://github.com/python/cpython/blob/3.14/Python/ceval.c>

### A2. C++ 内存模型

#### A2.1 数据竞争是未定义行为

- 数据竞争的条件：两个求值访问同一内存位置，至少一个是写，至少一个非原子操作，两者之间没有 happens-before 关系。
- 数据竞争的行为未定义。
- 与本次的关系：原代码的 `RingBuffer::is_inited_` 是普通 `bool`。多线程读写即为未定义行为，与是否崩溃无关。TSan 报告的就是这类竞争。
- 位置：`is_inited_` 改为 `std::atomic<bool>`；`TraceContext::tracer_` 为 `std::atomic<PythonTracer*>`（`profiler_python.cpp` 约第 140 行）。
- 出处：<https://en.cppreference.com/w/cpp/atomic/memory_order>

#### A2.2 relaxed、acquire/release、seq_cst

- relaxed 只保证操作本身是原子的，不建立线程间的先后关系。
- 线程 B 的 acquire 读取读到线程 A 的 release 存储时，两者 synchronizes-with。A 在存储前的写入对 B 可见。
- seq_cst 在此基础上让所有 seq_cst 操作处于一个全局顺序中。cppreference 写明，多核系统上这需要完整内存屏障。
- 与本次的关系：`ring_buffer.h` 中，写入方对槽位序号做 release 存储，读取方做 acquire 读取。把这两处改成 relaxed（A4.3 的 m3、m4），各有 8/25 次运行失败。`Report()` 开头的 relaxed 读取只用于提前返回。
- 出处：<https://en.cppreference.com/w/cpp/atomic/memory_order>

#### A2.3 先写后读协议为什么需要 seq_cst

- 写入方先 `writers_.fetch_add(1)` 再读 `start_`。停止方先 `start_.exchange(false)` 再读 `writers_`。目标是两个读取至少一个看到对方的写入。
- 只用 acquire/release 时，两次读取都可能读到旧值。没有 acquire 读到对应的 release 写，标准不禁止这一结果。
- 四个操作都是 seq_cst 时，它们处于同一全局顺序。无论两个写入在全局顺序中谁先，后写的一方随后的读取一定看到先写的一方的写入（或在它之后的修改）。所以两个读取至少一个看到对方的写入。
- 硬件上这是 x86-TSO 论文中的 SB（store buffering）例子。写入先进入本处理器的存储缓冲区。读取可以先于写入对其他处理器可见。
- 位置：`data_dumper.cpp` 的 `DataDumper::Report`（第 157 行起）、`TraceDataDumper::Report/ReportHash/ReportParam`、`StopLocked`、`WaitWriters`（第 94 行，第 96–99 行的注释写明两边都必须保持 seq_cst）。
- 出处：<https://en.cppreference.com/w/cpp/atomic/memory_order>（Sequentially-consistent ordering）；Sewell 等，"x86-TSO"，CACM 2010：<https://www.cl.cam.ac.uk/~pes20/weakmemory/cacm.pdf>

#### A2.4 release sequence

- 设 A 是对原子变量 M 的 release 操作。M 修改顺序中紧随 A 的连续读-改-写操作，由任何线程执行，与 A 构成 A 的 release sequence。读到其中任一值的 acquire 读取都与 A synchronizes-with。C++20 起只算读-改-写操作。
- 与本次的关系：每个写入方执行 `writers_.fetch_sub(1, release)`。停止方在 `WaitWriters` 中用 seq_cst 读到 0。中间的操作都是读-改-写，所以每个写入方的 release 都延续到最后的 0。因此写入方在 `Push` 中的写入，在 `Flush`、`UnInit` 之前可见。第二轮并发审阅确认了这一推理。
- 出处：<https://en.cppreference.com/w/cpp/atomic/memory_order>（Release sequence）

#### A2.5 为什么 x86 上测不出改弱

- cppreference：x86 等强内存序系统上，多数 release/acquire 不需要额外指令。ARM、PowerPC 需要。
- Sewell 的映射表：
  - x86 上 seq_cst 读取是 `MOV`，seq_cst 存储是带锁的 `XCHG`。
  - AArch64 上 acquire/seq_cst 读取是 `LDAR`，release/seq_cst 存储是 `STLR`，读-改-写是 `ldaxr`/`stlxr` 循环。
- x86-TSO：带 `LOCK` 前缀的指令结束时清空存储缓冲区。
- 与本次的关系：x86 上 `fetch_add` 不论内存顺序参数都编译为带 `LOCK` 的指令。所以写入方改成 `fetch_add(relaxed)` 加 `load(acquire)` 后（m11），x86 上 25 次全部通过。AArch64 上 relaxed 的读-改-写没有 acquire/release 语义。按 C++ 模型，“至少一个成立”不再有保证。aarch64（如鲲鹏）上是否实际出错，没有实测。
- 出处：<https://en.cppreference.com/w/cpp/atomic/memory_order>，<https://www.cl.cam.ac.uk/~pes20/cpp/cpp0xmappings.html>，<https://www.cl.cam.ac.uk/~pes20/weakmemory/cacm.pdf>

### A3. Vyukov 有界队列

- 容量为 2 的幂。每个槽位带原子序号，初值为下标。
- 写入方读到 `seq == pos`（acquire）时用 CAS 占用。写完后置 `seq = pos + 1`（release）。
- 读取方读到 `seq == pos + 1`（acquire）才取数据。取完置 `seq = pos + 容量`（release）。
- 一个写入方占用后迟迟不写完，读取方会停在这个槽位。
- 副作用：`Size()` 统计已占用的槽位，包括未写完的。读取方的 `Run` 原来按 `Size()` 决定是否读取。第一版修改后，`Run` 在有写入方未写完时会忙等（B1）。当前代码已改为取不到记录时休眠（[fix_details.md](fix_details.md) 第 3 节问题 2）。
- 每个槽位从 8 字节变为 16 字节。
- 位置：`torch_npu/csrc/toolkit/profiler/common/ring_buffer.h` 的 `Push`（第 89 行起）、`Pop`（第 127 行起）、`Size`、`struct Slot`。
- 出处：原始页面 <http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue> 当天返回 301，跳转到无关域名，没有打开。web.archive.org 在本机无法访问。
- 代码和作者 Simplified BSD 许可证的副本在 spdlog 早期版本中。文件头写明 "A modified version of Bounded MPMC queue by Dmitry Vyukov"：<https://github.com/gabime/spdlog/blob/1bd10957cb47824c0e24859db2bd637e472ab97d/include/spdlog/details/mpmc_bounded_q.h>。上面的规则与这份代码一致。

### A4. 工具

#### A4.1 ThreadSanitizer

- 编译期插桩加运行时检测。Clang 文档：典型减速 5–15 倍，内存开销 5–10 倍（官方自报）。GCC 文档：不能与 `-fsanitize=address` 或 `-fsanitize=leak` 同时用。
- 只检查实际执行到的交错。有锁保护的数据没有数据竞争，所以锁内的逻辑错误不报告。
- Lidbury 和 Donaldson（POPL 2017）写道，当时的 tsan "considers only sequentially consistent program executions"。所以它不产生 SB 这类结果。当前版本是否仍是这样，没有核实。
- 与本次的关系：TSan 构建能检出原代码的竞争，例如 `ring_buffer.h:105 in Pop`、`data_dumper.cpp:72 in GatherAndDumpData`。
- TSan 通过不代表代码正确。m11 在 TSan 下 25 次全部通过。B2 中“`ReportHash` 去掉写入方登记”的变异在 TSan 下 0/3 检出。这个变异只能由后加的语义检查 `encoded_after_stop` 发现。
- 位置：`test/cpp/profiler_thread_safety/run.py` 的 `tsan` 构建。
- 出处：<https://clang.llvm.org/docs/ThreadSanitizer.html>，<https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html>，<https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2017/POPL.pdf>

#### A4.2 AddressSanitizer 和 LeakSanitizer

- ASan 检测堆、栈、全局变量越界，释放后访问，重复释放，无效释放。Clang 文档：典型减速 2 倍（官方自报）。Linux 上 ASan 构建默认开启 LSan。
- 与本次的关系：原代码在 ASan 构建下报 heap-use-after-free 和 double-free。这是问题 1 的直接证据。加宽时序窗口后，ASan 复现了 `startOne` 中 `Py_XINCREF` 的释放后访问（3/3）。LSan 复现了异常路径上 32 字节的泄漏，`f36d74d` 修复后不再报告。
- 出处：<https://clang.llvm.org/docs/AddressSanitizer.html>，<https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html>

#### A4.3 变异测试

- 并发测试在修复后的代码上通过，只能说明测试没有发现问题。测试能否发现问题，要另外检查。
- 变异测试有意在被测代码中注入小改动（变异），例如删掉一行同步、把内存顺序改弱。然后检查测试能否发现。
- 下表的编号 m1–m11 由最终审阅者给出，只用于本分支的代码。每项 25 次运行。
- 另有两套编号。第二轮修复时另有一个编号 m6 的变异（B2 第 2 条），与下表的 m6 无关。[analysis.md](analysis.md) 第 5.5 节的 u1–u3 针对上游代码。
- 变异脚本 `mkmut.py`、`runmut.sh` 只在作者本机，不在仓库中。

| 变异 | 内容 | 失败次数 |
|---|---|---|
| m1 | `DataDumper::StopLocked` 去掉 `WaitWriters` | 2 |
| m2 | `Report` 去掉第二次 `start_` 检查 | 1 |
| m3 | `Push` 发布序号改为 relaxed | 8 |
| m4 | `Pop` 读取序号改为 relaxed | 8 |
| m5 | `Push` 先发布序号再写数据 | 10 |
| m6 | `Stop` 去掉 `control_mutex_` | 4 |
| m7 | `Pop` 先归还槽位再读数据 | 16 |
| m8 | `TraceDataDumper::StopLocked` 去掉 `WaitWriters` | 4 |
| m9 | `ReportHash` 去掉写入方登记 | 2 |
| m10 | `Run` 恢复原来的忙等写法 | 0 |
| m11 | 写入方的 seq_cst 改弱 | 0 |

出处：Y. Jia, M. Harman，"An Analysis and Survey of the Development of Mutation Testing"，IEEE TSE 37(5):649–678，2011，doi:10.1109/TSE.2010.62（题名、作者、卷期、页码经 Crossref 核对）。

## B. 经验

### B1. 修复本身多次引入新问题

第一版修改的自测全部通过。之后每一轮修复都引入了新问题。下表 9 条中，7 条由审阅者发现，2 条由第二轮修复者自测发现。

| 问题 | 引入它的修改 |
|---|---|
| 两个线程同时 `Stop` 时，`Flush` 在 `exchange` 分支外执行，同一缓冲区出现多个读取方 | 第一版问题 1 修复 |
| `Start` 先置 `start_ = true` 再创建线程，并发 `Stop` 可能对未赋值的 `pid_` 执行 `pthread_join` | 第一版问题 1 修复 |
| `Size()` 改为统计已占用槽位，`Run` 仍按 `Size()` 判断，写入方未写完时读取方忙等 | 第一版问题 2 修复 |
| 持有 `registration_mutex_` 时执行 Python 代码，终结器调用 `startOne`，同线程再次加锁，死锁 | 第一版问题 3 修复 |
| `startOne` 释放锁后才 `Py_XINCREF(ctx)`，窗口内 `clear()` 可释放 ctx | 第一轮修复 |
| `startOne` 的 `active_` 检查和安装回调分两步，`stop()` 可在中间执行完，之后回调读已释放的 `ThreadLocalResult`（TSan 报 `profiler_python.cpp:609` 与 `:884` 竞争）。修复报告称已解决，实际不成立。 | 第一轮修复 |
| 第一版 `session_` 没有同时提前置 `active_ = false`，两步之间注册的 `startOne` 不会判定为过期 | 第二轮修复的第一版（修复者自测发现） |
| `recordReturn` 等 `data_mutex_` 之后才读 ctx，等锁时 detach，其他线程可能已释放 ctx（TSan 报释放后访问） | 第二轮修复（修复者自测发现） |
| `startOne` 手动持有 ctx 引用，`recordStartStack` 抛异常时泄漏 32 字节。原代码不泄漏。 | 第一轮修复，`f36d74d` 已修 |

规律：

- 修改改变了某个量的含义，没有检查所有使用处。
- 等锁时 detach，之后读到的对象可能已释放。
- 修复报告的安全结论没有验证。

做法：

1. 修改改变变量或函数的含义时，写明新旧含义，grep 出所有使用处逐个检查。
2. FT 代码中，可能 detach 的调用包括等 `PyMutex`、`PyEval_SetProfile`、执行 Python 代码。在这些调用之后，重新确认之前读到的指针和状态。
3. 每轮修复后，由没有参与修改的人再检查一遍。修复者的自测不能代替这一步。

### B2. 压力测试一开始检不出回归

1. 审阅者删掉两个 `Stop` 中的 `WaitWriters`。TSan 构建下 dumper 场景 3/3 失败，trace 场景只有 1/3 失败。
   - 修复者增加写入方线程，加随机调用 `Stop` 的线程，在 stub 的 `C10_LIKELY` 中加暂停。trace 场景仍 0/3 检出。
   - trace 缓冲区容量改为 64 后 3/3 检出。
   - 修复者给出的原因：`Push` 失败时没有发布操作，与主线程之间没有其他 happens-before 关系。这时 TSan 才能报告。
2. stub 的暂停放在 `C10_LIKELY` 中，只有 `Report` 用这个宏。
   - 修复者把 `ReportHash/ReportParam` 改为 `if (C10_LIKELY(start_.load()))`。
   - 新变异“只去掉 `ReportHash` 的写入方登记”在第二轮修复时编号 m6。它在 TSan 下 0/3 检出，因为数据有锁保护。
   - 加语义检查 `encoded_after_stop`（`Stop()` 返回后不得再编码记录）后，plain 构建 3/3 检出，TSan 构建仍 0/3。
3. stub 的 `PyEval_SetProfile` 从不 detach。工作线程没有调用帧。所以 STW 不会发生在 `startOne` 执行期间。18/18 次 TSan 通过不能证明这条路径安全。
   - 修复者让 stub 在安装前 detach 0–2 ms，并给工作线程加调用帧。
   - 每次 `stop` 后检查 `stale_hooks_after_stop == 0`。
   - 之后第一轮修复的快照在 plain 构建下 3/3 失败。`stale_hooks_after_stop` 分别为 37、37、24。
4. stub 用 `std::shared_mutex` 模拟 STW。它在 glibc 上偏向读者，加压后写者长时间无法获得锁。
   - 一次 TSan 运行耗时 132 秒，plain 构建挂起。
   - 改为写者优先的读写锁后，TSan 单次约 1 秒（`stub_all.h` 第 59–101 行）。
5. 最终版本的检出能力仍有限（A4.3：m1 2/25、m2 1/25、m10 0/25、m11 0/25）。
   - 最终审阅者建议两条改法：在 `Stop()` 与 `UnInit()` 之间加 `RandomPause`；在提前返回检查与 `fetch_add` 之间加暂停。
   - 测试文件在 `663e8ba` 之后没有改动。
   - `RunDumper` 中 `first.Stop();` 之后紧接 `AfterStop([&] { first.UnInit(); });`。位置是 `profiler_thread_safety_test.cpp` 第 407–408 行。
   - 这两条建议没有实施。

做法：

1. 每写一处并发修复，同时写一个改回去的变异，确认测试稳定检出后再信任结果。
2. 除 TSan 外，加语义检查（计数、顺序、停止后不再写入）。
3. stub 要模拟被测代码依赖的阻塞行为（detach、STW），并检查 stub 自己的锁是否公平。
4. 接手后可以先实施第 5 条的两条建议，再重新运行 `mkmut.py` 的变异，对比失败次数。

### B3. stub 与真实 CPython 3.14 的差距

- 真实的 `_PyEval_SetProfile` 在替换回调前执行 Python 审计回调。stub 不执行 Python 代码。
- `ModuleInfo`、`parseModuleParams`、`parseOptimizerParams`、`py::repr` 调用的接口在 stub 中从不报错。
- 所以抛 Python 异常的路径没有测试覆盖。`f36d74d` 修复的泄漏就在这条路径上。仓库测试没有发现它，审阅者另写的 `exc_leak.cpp` 发现了它。
- GIL 路径只编译、不运行。“有 GIL 时行为不变”只来自阅读代码。
- `ProfilerMgr` 没有编译进测试。没有用真实 torch、CPython 3.14、CANN 头文件编译。
- stub 的 `py::handle::attr` 不访问对象。所以测试没有发现 [analysis.md](analysis.md) 第 4.2 节的空指针访问。

做法：

1. 在 stub 中加两个开关。这两条尚未实施。
   - `PyEval_SetProfile*` 替换回调前执行一个回调。
   - `py::object::attr` 可注入异常。
2. 真实实现对非法输入报错或崩溃时，stub 也要表现出来。
3. 在有真实 torch 和 CPython 3.14t 的机器上至少编译一次。环境见 [analysis.md](analysis.md) 第 7.1 节“准备”。
4. 在同一机器上运行一次 GIL 构建的原有 profiler 测试（[fix_details.md](fix_details.md) 第 7 节）。

### B4. 设计之前没有查上游最新代码

- 任务在 2026-11-02 的修复提交之前收到。具体日期没有可核对的出处。本机最早的记录是 2026-11-02 00:13 `-0800`。修复基于 r22 分支（2.13.0，基线 `46936d1`）设计。
- 上游在 2026-10-29 的 `ff5ad88d8a` 中已重构 `PythonTracer`。这件事在写文档时才查到。
- `675f2e9` 把上游方案概括为“FT 专用的加锁处理”。这一概括不准确，`da752e6` 已更正。
- 后果：[analysis.md](analysis.md) 第 7.3 节建议问题 3 改用上游方案。所以本分支问题 3 的修改可能不会合入。同一次对照才发现 `b2e659d62f` 和 issue #5658 两个候选。问题 1、2 在上游逻辑未变，仍要修复。

做法：

1. 设计前用 `gh api 'repos/Ascend/pytorch/commits?path=<文件>'` 查目标文件在上游的近期提交。
2. 查目标文件相关的 open issue。

### B5. 数字要从日志统计

1. 压力测试作者报告“25 次中 23 次失败”。`SUMMARY.json` 中三个构建为 9/8、9/7、7/6（运行数/失败数），25 行中 3 行是编译步骤且通过。实际是 22 次场景运行中 21 次失败。
2. “约 20 个问题、全部修复”没有逐条清单，无法核实。后来按逐条记录改为精确统计（[progress.md](progress.md)）。
3. 提交信息 `663e8ba`、`f36d74d` 写的“49/49 runs pass”包含 7 次编译检查。场景运行是 42 次。
4. 轮数这类计数曾写错，已按记录改正（[progress.md](progress.md)）。

做法：

1. 结果由脚本从 `SUMMARY.json` 统计。
2. 场景运行和编译步骤分开写。
3. 写“约”“全部”时要有逐条清单。
4. 计数前回到记录里数一遍。

### B6. 文档初稿的事实错误

- “行为变化只影响 FT 构建”：`data_dumper.cpp`、`data_dumper.h`、`ring_buffer.h` 中没有 `Py_GIL_DISABLED`。问题 1、2 的修改在两种构建中都生效。
- “持有这些锁时不调用 Python 代码”：`control_mutex_` 持有期间会执行 Python 代码。这条规则只适用于 `registration_mutex_` 和 `data_mutex_`。
- “改弱后在 aarch64 上会出错”：没有实测，改为“可能出错”。
- “会因空指针访问出现 SIGSEGV”：这是使用方的判断，没有调用栈支撑。
- `std::vector::clear()` 不释放缓冲区内存，只销毁元素并把大小置 0。
- “上游不存在全局锁争用”：上游每个事件仍获取一次 `results_mutex_`（[analysis.md](analysis.md) 第 4.3 节）。

做法：

1. 写“影响哪种构建”前，先 grep 条件编译宏。
2. 写锁规则时，逐把对照代码注释。
3. 每条结论标明来源。
4. 写“对方没有某个问题”前，逐行读对方的热路径。

### B7. 修复把崩溃换成了挂起

- fork 实验：8 个写入方在 `Report` 中时 fork 40 次，子进程在 2 秒超时内调用 `Stop`。
- 修复后的代码有 4/40 个子进程挂起。fork 时正在 `Report` 中的写入方没有复制到子进程，所以子进程继承的 `writers_` 永远大于 0。原代码在同一实验中 core dump：子进程对不存在的线程执行 join。
- 这一点没有修复。它写进了代码注释、`f36d74d` 的提交信息和 [fix_details.md](fix_details.md) 第 6 节。

做法：引入“等待计数归零”这类协议时，检查 fork、线程异常退出等计数无法归零的路径，并写明失败时的表现。

### B8. 中途修改测试会使已有日志作废

- 第一轮修复期间测试代码改了两次，对应旧测试的两组日志已作废，但没有删除。
- 上游对照的扩展场景第一次运行时，TSan 报告来自 stub 自身的问题。改正后报告消失，那次日志已删除（[analysis.md](analysis.md) 第 5.2 节）。

做法：

1. 日志目录名带上被测代码和测试代码的提交号。
2. 或者在日志中写入 `git rev-parse HEAD` 和 `git diff --stat`。
