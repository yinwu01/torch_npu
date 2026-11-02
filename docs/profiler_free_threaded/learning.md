# 学习笔记：这次修复需要的背景知识和过程经验

- 读者：接手 `fix/profiler-thread-safety` 分支（`663e8ba`、`f36d74d`）的工程师。
- 先读 [fix_details.md](fix_details.md) 了解三个问题和改法，再读 [analysis.md](analysis.md) 了解候选方案和与上游的对照；本文只补两件事：
  - A 部分：看懂、维护这次修改需要的背景知识，每条附官方出处；
  - B 部分：这次修复过程中有记录可查的经验。
- 所有出处的访问日期都是 2026-11-02。CPython 源码以当天 `python/cpython` 的 `3.14` 和 `3.13` 分支为准。
- 本文是文档，没有运行时依赖。整理时与本目录其他四个文件一起经过两轮文档审阅（AI 模型子代理，没有人工审阅）。
- **B 部分引用的证据路径（`/SSD1/wuyin/project/torch_npu-work/…`）都在作者本机，没有上传 GitHub，接手的工程师无法打开。** 保留这些路径是为了说明每条经验的依据出自哪里；审阅意见本身未公开。
- **B 部分提到的"审阅者""实现者""修复者"都是 AI 模型子代理，这次没有人工代码审阅。**

---

## A. 背景知识

每条按"是什么、为什么和这次的代码有关、在代码里的位置、出处"四项写。

### A1. free-threaded CPython

#### A1.1 PEP 703、PEP 779 和版本状态

- **是什么。**
  - PEP 703 提出让 CPython 可以在没有 GIL（全局解释器锁，同一时刻只允许一个线程执行 Python 字节码的锁）的情况下编译和运行。状态 Final，Python-Version 3.13，2023-10-24 决议。按 PEP 原文，用 `--disable-gil` 编译时，CPython 会定义宏 `Py_GIL_DISABLED`；GIL 构建仍是默认构建。
  - PEP 779 规定 free-threaded 构建进入"第二阶段"（官方支持，但仍是可选构建）的条件。状态 Final，Python-Version 3.14，2025-06-16 决议。
  - 3.14 的 What's New 写的是 "Free-threaded Python is officially supported"，并写明单线程代码在 free-threaded 模式下的性能损失"大约 5–10%"（官方自报，取决于平台和编译器）。
  - free-threaded 构建在运行时仍可能重新开启 GIL：环境变量 `PYTHON_GIL`、命令行 `-X gil`，或者导入一个没有声明支持 free threading 的 C 扩展（此时打印警告）。
- **为什么有关。** 使用方报告的 SIGSEGV 出现在 CPython 3.14 的 free-threaded 构建下（下文简称 FT）。有 GIL 时，多个 Python 线程基本依次执行，profiler 代码里的竞态很少暴露；FT 下多个线程同时执行，竞态就会暴露。更完整的说明（哪些写入方本来就不受 GIL 约束）见 [analysis.md](analysis.md) 第 1.3 节。
- **出处。**
  - https://peps.python.org/pep-0703/
  - https://peps.python.org/pep-0779/
  - https://docs.python.org/3.14/whatsnew/3.14.html
  - https://docs.python.org/3.14/howto/free-threading-python.html

#### A1.2 C 扩展如何识别 FT 构建、如何声明支持

- **是什么。**
  - `Py_GIL_DISABLED`：FT 构建中定义为 1，GIL 构建中不定义。扩展代码用 `#ifdef Py_GIL_DISABLED` 区分两种构建。
  - 声明支持：多阶段初始化的模块在模块定义里加 `{Py_mod_gil, Py_MOD_GIL_NOT_USED}`；单阶段初始化的模块调用 `PyUnstable_Module_SetGIL()`。不声明的话，导入时打印警告并在运行时开启 GIL。
- **为什么有关。** torch_npu 只有在声明支持后才会在真正无 GIL 的状态下运行，这也是问题暴露的前提。我们新增的锁只在 `Py_GIL_DISABLED` 下启用，GIL 构建中编译为空函数。
- **在代码里的位置。**
  - 声明：`torch_npu/csrc/InitNpuBindings.cpp:247`，`PyUnstable_Module_SetGIL(result, Py_MOD_GIL_NOT_USED)`，只在同时定义了 `Py_GIL_DISABLED` 和 `TORCH_NPU_EXPERIMENTAL_FREE_THREADED` 时编译。这段是 r22 分支原有的代码，我们没有改。
  - 条件编译：`torch_npu/csrc/profiler/profiler_python.cpp` 中的 `TracerMutex`、`TracerControlMutex`、`ctxKey()`、`start()`、`stop()`。
  - `ring_buffer.h`、`data_dumper.h/.cpp` 里没有 `Py_GIL_DISABLED` 条件编译，所以问题 1、2 的修改两种构建都生效。
- **出处。** https://docs.python.org/3.14/howto/free-threading-extensions.html

#### A1.3 线程状态的 attach/detach 和 stop-the-world

- **是什么。**
  - attached thread state：每个线程有一个指向 `PyThreadState` 的线程局部指针，这个指针指向的线程状态称为 attached（已附着）。只有 attached 的线程可以访问 Python 对象。GIL 构建里 attach 相当于持有 GIL。
  - detach（分离）：线程释放自己的线程状态，例如 `PyEval_SaveThread()` 或 `Py_BEGIN_ALLOW_THREADS`。官方文档写明 FT 构建中仍要 detach，因为解释器需要周期性地暂停所有线程。
  - stop-the-world：解释器暂停所有 attached 线程，以便在一致的状态下修改全局数据。文档中提到垃圾回收时会这样做。CPython 3.14 源码 `Python/pystate.c` 的 `_PyEval_StopTheWorld()` 只在 `Py_GIL_DISABLED` 下有实现，GIL 构建中是空函数。发起方要等其他线程全部 detach 或被暂停才能继续。
- **为什么有关。** 一个线程如果在 attached 状态下阻塞在普通的 `std::mutex` 上，它既不 detach，也不响应暂停请求，发起 stop-the-world 的线程就会一直阻塞，直到该线程 detach。持有这把 `std::mutex` 的线程如果也在等 stop-the-world 完成，就形成死锁。这是 `profiler_python.cpp` 改用 `PyMutex` 的原因（见 A1.4）。
- **在代码里的位置。** `profiler_python.cpp` 中 `TracerMutex` 上方的注释；stub 头文件 `test/cpp/profiler_thread_safety/python_tracer/stubs/stub_all.h` 用读写锁模拟 attach/detach 和 stop-the-world。
- **出处。**
  - https://docs.python.org/3.14/c-api/threads.html
  - https://github.com/python/cpython/blob/3.14/Python/pystate.c （`_PyEval_StopTheWorld`、`stop_the_world`、`tstate_wait_attach`）

#### A1.4 PyMutex

- **是什么。** CPython 提供的互斥锁，3.13 新增，目前占 1 字节（大小不保证）。官方文档对 `PyMutex_Lock` 的说明：如果锁已被占用，调用线程阻塞，阻塞期间临时 detach 线程状态。`PyMutex` 不可重入：同一线程第二次加锁会一直阻塞。
- **为什么有关。** 等锁时 detach，所以等锁的线程不会阻止 stop-the-world。代价是：等锁之后线程重新 attach，在等待期间其他线程可能已经完成了 stop、clear，释放了某些对象。第二轮修复中出现的 `recordReturn` 释放后访问就是这种情况（见 B1）。
- **在代码里的位置。** `profiler_python.cpp`：`TracerMutex`（`registration_mutex_`、`data_mutex_`）和 `TracerControlMutex`（`control_mutex_`，同一线程重入时抛异常，避免不可重入导致的死锁）。
- **出处。** https://docs.python.org/3.14/c-api/synchronization.html

#### A1.5 critical section（`Py_BEGIN_CRITICAL_SECTION`）

- **是什么。** 3.13 新增的宏，FT 构建中获取某个 Python 对象自带的锁；GIL 构建中只展开为 `{` 和 `}`，不加锁。为了避免死锁，线程在 critical section 内阻塞时（例如再等另一把锁、做 I/O），运行时会临时挂起这个 critical section，释放对象锁，之后再恢复。因此官方文档说明它不提供 `PyMutex` 那样的独占保证。
- **为什么有关。** 我们要保护的是 C++ 容器（`thread_local_results_`、各缓存、`events_`），不属于某个 Python 对象，而且需要在等锁时保持独占，所以没有用 critical section，用的是 `PyMutex`。维护时如果想改成 critical section，要考虑它会被挂起这一点。
- **在代码里的位置。** 本次修改没有使用。
- **出处。** https://docs.python.org/3.14/c-api/synchronization.html

#### A1.6 `PyEval_SetProfile` 和 `PyEval_SetProfileAllThreads`

- **是什么。**
  - `PyEval_SetProfile(func, obj)`：给当前线程设置 profile 回调，调用方必须持有 attached 线程状态。
  - `PyEval_SetProfileAllThreads(func, obj)`：给当前解释器所有正在运行的线程设置 profile 回调，3.12 新增。
  - 源码中的实际行为（`Python/legacy_tracing.c`，`3.14` 分支最近一次修改是 `f36da66`，2026-04-10）：
    - 两者都先在当前线程上触发审计事件 `sys.setprofile`（会执行已注册的 Python 审计回调），再调用 `_PyEval_StopTheWorld`，在暂停期间替换回调，然后 `_PyEval_StartTheWorld`；旧的 `obj` 在恢复运行后才 `Py_XDECREF`。
    - `3.13` 分支的这两个函数同样调用 `_PyEval_StopTheWorld`。
    - 公开函数 `PyEval_SetProfile*`（`Python/ceval.c`）忽略审计回调抛出的异常，只打印 unraisable 信息。
- **为什么有关。**
  - FT 构建里 `start()` 用 `PyEval_SetProfileAllThreads` 一次性给所有线程安装同一个共享 ctx，`stop()` 用 `PyEval_SetProfileAllThreads(nullptr, nullptr)` 清除所有线程的回调。因为替换发生在暂停期间，`stop()` 返回后不会再有线程以旧回调进入。前提是 `sys.setprofile` 审计回调没有抛异常：按 `Python/legacy_tracing.c`（3.14.3 的第 541–543、566–568 行），审计回调返回 -1 时函数直接返回，不进入 STW，也不替换回调，公开函数只打印 unraisable 信息。此时旧回调仍装着：`stop()` 之后由 `active_ == false` 保证不记录事件，`clear()` 之后由 `ctx->tracer_` 已置空保证（`tracer_` 只在 `~ThreadLocalResult` 中置空）；并且下一轮 `startOne` 会因该线程的 `c_profilefunc` 非空而跳过它（未测试）。
  - 代价：会清掉其他工具（如 cProfile）设置的回调（见 [fix_details.md](fix_details.md) 第 5.2 节）。
  - `startOne` 里的 `PyEval_SetProfile` 也会触发 stop-the-world，调用线程在这期间可能等待，其他线程的 `stop()`、`clear()` 可能在这段时间内执行完。第二轮审阅发现的"`stop()` 之后仍安装回调"就发生在这里（见 B1）。
  - stub 没有模拟审计回调（见 B3）。
- **在代码里的位置。** `profiler_python.cpp`：`start()`（约第 474、497 行，分别在 FT 分支和 GIL 分支）、`startOne()`（约第 549、561 行）、`stop()`（约第 589 行）。
- **出处。**
  - https://docs.python.org/3.14/c-api/profiling.html
  - https://github.com/python/cpython/blob/3.14/Python/legacy_tracing.c （`_PyEval_SetProfile`、`_PyEval_SetProfileAllThreads`）
  - https://github.com/python/cpython/blob/3.13/Python/legacy_tracing.c
  - https://github.com/python/cpython/blob/3.14/Python/ceval.c （`PyEval_SetProfile`）

### A2. C++ 内存模型中用到的部分

#### A2.1 数据竞争是未定义行为

- **是什么。** 两个求值访问同一内存位置，至少一个是写，至少一个不是原子操作，并且两者之间没有 happens-before 关系（一种由同步操作建立的先后关系），这就是数据竞争（data race），程序行为未定义。
- **为什么有关。** 原代码 `RingBuffer::is_inited_` 是普通 `bool`，被多个线程同时读写，按标准即为未定义行为，与"实际有没有崩溃"无关。TSan 报告的就是这类竞争。
- **在代码里的位置。** `ring_buffer.h` 的 `is_inited_` 已改为 `std::atomic<bool>`；`TraceContext::tracer_` 改为 `std::atomic<PythonTracer*>`（`profiler_python.cpp` 约第 140 行）。
- **出处。** https://en.cppreference.com/w/cpp/atomic/memory_order （"If one evaluation modifies a memory location…"一段）

#### A2.2 relaxed、acquire/release、seq_cst

- **是什么。**
  - `memory_order_relaxed`：只保证这次操作本身是原子的，并且对同一变量的修改有一个所有线程一致的顺序；不建立线程之间的先后关系。
  - release/acquire：线程 A 的 release 存储被线程 B 的 acquire 读取读到时，A 在这次存储之前的所有写入，对 B 在这次读取之后可见（称为 synchronizes-with）。前提是 B 读到了 A 写的值。
  - `memory_order_seq_cst`：在 release/acquire 的基础上，所有 seq_cst 操作还处于一个所有线程都认同的全局顺序中。cppreference 写明，在多核系统上全局顺序需要完整的内存屏障指令。
- **为什么有关。**
  - `ring_buffer.h`：写入方写完数据后对槽位序号做 release 存储，读取方对序号做 acquire 读取，读到 `pos + 1` 才取数据，所以读取方一定看到完整的数据。变异测试把这两处改成 relaxed（A4.3 表中的 m3、m4），各有 8/25 次运行失败。
  - `data_dumper.cpp`：`writers_` 与 `start_` 之间的四个操作必须是 seq_cst（见 A2.3）。
  - `Report()` 开头对 `start_` 的 relaxed 读取只用于提前返回，不承担同步。
- **出处。** https://en.cppreference.com/w/cpp/atomic/memory_order

#### A2.3 先写后读（Dekker 式）协议为什么需要 seq_cst

- **是什么。** 两个线程各自先写一个变量、再读对方的变量：
  - 写入方：`writers_.fetch_add(1)`，然后读 `start_`；
  - 停止方：`start_.exchange(false)`，然后读 `writers_`。

  目标是保证"写入方读到 `start_ == false`"和"停止方读到 `writers_ > 0`"至少一个成立。
  - 只用 acquire/release 时，两次读取都可能读到对方写入之前的旧值。此时没有任何一次 acquire 读到对应的 release 写，synchronizes-with 不成立，标准不禁止两边都读到旧值。
  - 四个操作都是 seq_cst 时，它们处于同一个全局顺序。无论两个写入在全局顺序中谁先，后写的一方随后的读取一定能看到先写的一方的写入（或在它之后的修改），所以上述两种情况至少一个成立。
  - 硬件上，这对应 x86-TSO 论文中的 SB（store buffering，存储缓冲）例子：两个处理器各写一个位置、再读另一个位置，可以同时读到 0。原因是写入先进入本处理器的存储缓冲区，读取可以先于写入对其他处理器可见。
- **为什么有关。** 这是问题 1（停止时释放正在写入的缓冲区）修复的核心。
- **在代码里的位置。** `data_dumper.cpp`：`DataDumper::Report`（第 157 行起）、`TraceDataDumper::Report/ReportHash/ReportParam`、`StopLocked`、`WaitWriters`（第 94 行，第 96–99 行的注释写明两边都必须保持 seq_cst）。
- **出处。**
  - https://en.cppreference.com/w/cpp/atomic/memory_order （Sequentially-consistent ordering 一节）
  - Peter Sewell 等，"x86-TSO: A Rigorous and Usable Programmer's Model for x86 Multiprocessors"，CACM 2010 最终稿：https://www.cl.cam.ac.uk/~pes20/weakmemory/cacm.pdf （SB 例子）

#### A2.4 release sequence（释放序列）

- **是什么。** 对原子变量 M 做一次 release 操作 A 之后，M 修改顺序中紧随其后、由任何线程执行的连续读-改-写（RMW）操作，与 A 一起构成 A 的 release sequence。读到这个序列中任一个值的 acquire 读取，都与 A synchronizes-with。C++20 起只算 RMW 操作，RMW 本身用什么内存顺序都可以。
- **为什么有关。** 多个写入方各自执行 `writers_.fetch_sub(1, release)`，最后一个把计数减到 0。停止方在 `WaitWriters` 里用 seq_cst（包含 acquire 语义）读到 0。因为中间的 `fetch_add`/`fetch_sub` 都是 RMW，较早写入方的 release 也通过释放序列延续到最后的 0，所以每个写入方在 `Push` 中的写入都在停止方的 `Flush`、`UnInit` 之前可见。第二轮并发审阅确认了这一推理（`review:concurrency:r2` 第 4 条）。
- **在代码里的位置。** `data_dumper.cpp`：`Report` 末尾的 `fetch_sub(1, std::memory_order_release)`，`WaitWriters`。
- **出处。** https://en.cppreference.com/w/cpp/atomic/memory_order （Release sequence 一段）

#### A2.5 为什么 x86 上测不出内存顺序改弱，aarch64 上可能出错

- **是什么。**
  - cppreference：x86 这类强内存序系统上，大多数操作的 release/acquire 不需要额外指令；ARM、PowerPC 这类弱内存序系统需要专门的读取指令或屏障指令。
  - Sewell 的 C/C++11 到处理器的映射表：x86 上 seq_cst 读取是普通 `MOV`，seq_cst 存储是带锁的 `XCHG`；AArch64 上 acquire/seq_cst 读取是 `LDAR`，release/seq_cst 存储是 `STLR`，RMW 用 `ldaxr`/`stlxr` 循环。
  - x86-TSO 论文：执行带 `LOCK` 前缀的指令时，线程在指令结束时清空自己的存储缓冲区。
- **为什么有关。** x86 上 `fetch_add` 无论写什么内存顺序参数，都编译为带 `LOCK` 的指令，指令结束时存储缓冲区已清空，后面的读取不会提前。所以把写入方改成 `fetch_add(relaxed)` 加 `load(acquire)`（A4.3 表中的 m11）在 x86 上 25 次运行全部通过，这与模型一致。AArch64 上 relaxed 的 RMW 不带 acquire/release 语义，按 C++ 模型上述"至少一个成立"不再有保证。**在 aarch64（如鲲鹏）上是否会实际出错，没有实测。**
- **在代码里的位置。** 同 A2.3。
- **出处。**
  - https://en.cppreference.com/w/cpp/atomic/memory_order （strongly-ordered / weakly-ordered systems 一段）
  - https://www.cl.cam.ac.uk/~pes20/cpp/cpp0xmappings.html
  - https://www.cl.cam.ac.uk/~pes20/weakmemory/cacm.pdf

### A3. 每个槽位带序号的有界队列（Vyukov bounded MPMC queue）

- **是什么。** Dmitry Vyukov 提出的有界多生产者多消费者队列（MPMC；我们只有一个读取方，所以是 MPSC 用法）：
  - 容量为 2 的幂；每个槽位带一个原子序号，初始化为槽位下标；
  - 写入方读到 `seq == pos`（acquire）表示槽位空闲，用 CAS（比较并交换）推进写位置占用它，写完数据后 `seq = pos + 1`（release）；
  - 读取方读到 `seq == pos + 1`（acquire）才取数据，取完 `seq = pos + 容量`（release），供下一轮同一下标的写入方使用。
  - 一个写入方占用槽位后如果迟迟不写完，读取方会停在这个槽位，直到它写完。
- **为什么有关。** 原代码用一个公共计数 `write_index_` 表示"已写完的条数"，无法知道具体哪个槽位已写完，读取方可能读到另一个写入方尚未写完的槽位（问题 2）。改为每个槽位一个序号后，读取方只读已写完的槽位，并按占用顺序读取。
  - 副作用 1：`Size()` 统计的是已占用的槽位，包括尚未写完的。后台线程 `Run` 原来根据 `Size()` 决定是否读取，改后在有写入方未写完时会忙等。这一点第一轮审阅才发现（见 B1）。
  - 副作用 2：每个槽位从 8 字节变为 16 字节。
- **在代码里的位置。** `torch_npu/csrc/toolkit/profiler/common/ring_buffer.h`：`Push`（第 89 行起）、`Pop`（第 127 行起）、`Size`、`struct Slot`。
- **出处。**
  - 原始页面：http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue 。2026-11-02 访问时这个地址返回 301，跳转到一个与作者无关的域名，所以没有打开跳转后的页面，原文也没有取到；web.archive.org 在本机无法访问。
  - 保留了原始代码和作者 Simplified BSD 许可证的副本（spdlog 早期版本，文件头写明 "A modified version of Bounded MPMC queue by Dmitry Vyukov"）：https://github.com/gabime/spdlog/blob/1bd10957cb47824c0e24859db2bd637e472ab97d/include/spdlog/details/mpmc_bounded_q.h 。上面的序号规则、初始值和内存顺序与这份代码一致。

### A4. 工具

#### A4.1 ThreadSanitizer（TSan）

- **是什么。** 编译期插桩加运行时检测的数据竞争检测器（`-fsanitize=thread`）。Clang 文档写明典型减速 5–15 倍、内存开销 5–10 倍（官方自报）。GCC 文档写明它不能与 `-fsanitize=address` 或 `-fsanitize=leak` 同时使用。
- **局限。**
  - 只检查实际执行到的交错。测试没有走到的时序窗口，TSan 不会报告。
  - Lidbury 和 Donaldson（POPL 2017）写道，当时的 tsan "considers only sequentially consistent program executions"，即只考察顺序一致的执行，不会产生 SB 这类弱内存序结果。当前 Clang 文档没有描述它对内存模型的支持范围，现在的版本是否仍是这样，没有核实。
  - 被互斥锁保护的数据不会出现数据竞争，所以"锁内的逻辑错误"TSan 不报。
- **为什么有关。**
  - 测试的 TSan 构建能检出原代码的竞争（`ring_buffer.h:105 in Pop`、`data_dumper.cpp:72 in GatherAndDumpData` 等）。
  - 但 TSan 通过不代表代码正确：变异 m11（seq_cst 改弱）在 TSan 下 25 次全部通过；B2 中提到的"`ReportHash` 去掉写入方登记、数据仍有锁保护"这一变异在 TSan 下 0/3 检出，只能靠后加的语义检查 `encoded_after_stop` 发现。
- **在代码里的位置。** `test/cpp/profiler_thread_safety/run.py` 的 `tsan` 构建。
- **出处。**
  - https://clang.llvm.org/docs/ThreadSanitizer.html
  - https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
  - C. Lidbury, A. F. Donaldson，"Dynamic Race Detection for C++11"，POPL 2017：https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2017/POPL.pdf

#### A4.2 AddressSanitizer（ASan）和 LeakSanitizer（LSan）

- **是什么。**
  - ASan（`-fsanitize=address`）检测堆、栈、全局变量越界访问，释放后访问（use-after-free），重复释放（double-free），无效释放。Clang 文档写明典型减速 2 倍（官方自报）。
  - LSan 检测内存泄漏；Clang 文档写明 Linux 上 ASan 构建默认开启泄漏检测。
- **为什么有关。**
  - 原代码在 ASan 构建下报 heap-use-after-free 和 double-free，这是问题 1 的直接证据。
  - 第二轮并发审阅者把窗口人为加宽后，用 ASan 复现了 `startOne` 中 `Py_XINCREF` 的释放后访问（3/3）。
  - 最终审阅者用 LSan 复现了异常路径上 32 字节的泄漏，`f36d74d` 修复后不再报告。
- **在代码里的位置。** `run.py` 的 `asan` 构建。
- **出处。**
  - https://clang.llvm.org/docs/AddressSanitizer.html
  - https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html

#### A4.3 变异测试

- **是什么。** 有意在被测代码中注入小改动（变异，例如删掉一行同步、把内存顺序改弱），检查测试能否发现。能发现的变异越多，说明测试检出回归的能力越强。
- **为什么有关。** 并发测试"在修复后的代码上通过"只说明没有发现问题，不说明测试能发现问题。

  **下表的编号 m1–m11 只用于我们自己的代码**（最终审阅者的编号，每项 25 次运行，来源 `torch_npu-work/final-review-upload-r1/mut-summary.txt`）。注意另有两套编号与此不同：B2 中第二轮修复时用过的 m6 是另一个变异；[analysis.md](analysis.md) 第 5.5 节的 u1–u3 是对上游代码做的变异。

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

- **在代码里的位置。** 变异脚本 `torch_npu-work/final-review-upload-r1/mkmut.py`、`runmut.sh`（本机路径，未上传 GitHub）。
- **出处。** Y. Jia, M. Harman，"An Analysis and Survey of the Development of Mutation Testing"，IEEE Transactions on Software Engineering 37(5):649–678，2011，doi:10.1109/TSE.2010.62（通过 Crossref 核对了题名、作者、卷期、页码）。

---

## B. 这次修复的经验

只写有记录可查的事。证据路径说明：

- `torch_npu-work/` 指本机 `/SSD1/wuyin/project/torch_npu-work/`，未上传 GitHub。
- 审阅意见和文档审阅意见都未公开。下文用标签引用其中的条目，例如 `review:concurrency` 第 1 条。标签的含义：代码审阅为 `review:concurrency`、`review:integration`（第一轮），`review:concurrency:r2`、`review:integration:r2`（第二轮），`final:tracer`、`final:upload`（第三轮）；对应的修复为 `repair:r1`、`repair:r2`。

### B1. 修复本身多次引入新问题，每一轮都靠另一个子代理的审阅发现

**发生了什么。** 两个实现者在第一版修改后都写明"只过了自测，没有做独立审阅"，各自的自测全部通过。之后每一轮审阅都在上一轮的修复中找到新问题：

| 轮次 | 发现的问题 | 由哪次修改引入 | 证据 |
|---|---|---|---|
| 第一轮审阅 | 两个线程同时 `Stop` 时，`Flush` 在 `exchange` 分支外执行，同一缓冲区出现多个读取方（critical） | 第一版问题 1 修复 | `review:concurrency` 第 1 条 |
| 第一轮审阅 | `Start` 先置 `start_ = true` 再创建线程，并发 `Stop` 可能对尚未赋值的 `pid_` 执行 `pthread_join` | 第一版问题 1 修复 | `review:concurrency` 第 3 条 |
| 第一轮审阅 | `Size()` 改为统计已占用的槽位，`Run` 仍按 `Size()` 判断，写入方未写完时后台线程忙等 | 第一版问题 2 修复 | `review:concurrency` 第 2 条、`review:integration` 第 1 条 |
| 第一轮审阅 | `registration_mutex_` 持有期间执行 Python 代码，终结器回调 `startOne` 时同一线程再次加锁，形成死锁 | 第一版问题 3 修复 | `review:concurrency` 第 5 条 |
| 第二轮审阅 | `startOne` 在释放锁之后才 `Py_XINCREF(ctx)`，窗口内 `clear()` 可释放 ctx；审阅者加宽窗口后 ASan 3/3 报释放后访问 | 第一轮修复 | `review:concurrency:r2` 第 1 条；`torch_npu-work/review-concurrency-r2/` |
| 第二轮审阅 | `startOne` 的 `active_` 检查和安装回调是两步，`stop()` 可在中间完整执行，之后回调读取已释放的 `ThreadLocalResult`；TSan 报 `clear() profiler_python.cpp:609` 与 `pyProfileFn profiler_python.cpp:884` 竞争 | 第一轮修复（修复报告声称已解决，审阅者指出不成立） | `review:integration:r2` 第 1 条；`torch_npu-work/review-integration-r2/pt/` |
| 第二轮修复自测 | 第一版 `session_` 只在移除回调前递增，没有同时提前置 `active_ = false`，在两步之间注册的 `startOne` 不会被判为过期 | 第二轮修复的第一版 | `repair:r2` 结果第 5 条 |
| 第二轮修复自测 | `recordReturn` 在等待 `data_mutex_` 之后才读 ctx，等锁期间线程 detach，ctx 可被释放；TSan 报释放后访问 | 第二轮修复（修复者自己写明由第 3 条修改引入） | `repair:r2` 结果第 5 条 |
| 最终审阅 | `startOne` 手动持有的 ctx 引用没有用 RAII 管理，`recordStartStack` 抛 Python 异常时泄漏；原代码在这里不泄漏；LSan 报 32 字节泄漏 | 第一轮修复 | `final:tracer` 第 1 条；`torch_npu-work/final-review-tracer-r1/out_exc_fix.txt`、`out_exc_orig.txt`；修复提交 `f36d74d` |

**规律。** 上表中的问题有三类：

- 修改改变了某个量的含义，但没有检查所有使用它的地方：`Size()` 的含义变了，`Run` 没改；`start_ == true` 原来意味着线程已创建，改后不再成立。
- 等锁会 detach（A1.4），等锁之后读到的对象可能已被释放：`Py_XINCREF` 和 `recordReturn` 两处都属于这种情况。
- 修复报告中的安全性结论没有经过验证：第二轮审阅指出第一轮修复报告的说法不成立。

**下次怎么做。**

1. 修改改变了一个变量或函数的含义时，在修改说明里写明新旧含义，并 grep 出所有使用处逐个检查。
2. FT 代码中，任何可能 detach 的调用（等 `PyMutex`、`PyEval_SetProfile`、执行 Python 代码）之后，都要重新确认之前读到的指针和状态是否仍然有效。
3. 每轮修复后都请没有参与修改的人重新审阅；修复者的自测通过不能代替审阅。

### B2. 压力测试一开始检不出回归，要调整暂停点、容量和 stub 的行为后才能检出

**发生了什么。**

1. **trace 场景检不出 `WaitWriters` 被删。** 第一轮集成审阅者删掉两个 `Stop` 中的 `WaitWriters`，TSan 构建下 dumper 场景 3/3 失败，trace 场景只有 1/3 失败（`review:integration` 第 2 条；`torch_npu-work/review-integration-r1/m1.console.log`、`logs-m1/`）。
   - 修复者先增加调用 `Report` 的线程、加一个随机调用 `Stop` 的控制线程、在 stub 的 `C10_LIKELY` 里加暂停，trace 场景仍然 0/3 检出。
   - 把 trace 缓冲区容量改为 64 后才 3/3 检出。修复者给出的原因：`Push` 失败时没有任何发布操作，与主线程之间没有其他 happens-before 关系，TSan 才能报告缺少 `WaitWriters` 的竞争（`repair:r1` 结果第 10 条）。
2. **暂停点只覆盖了 `Report`。** stub 的暂停挂在 `C10_LIKELY` 上，只有 `Report` 用了这个宏，`ReportHash/ReportParam` 的窗口从未被加宽（`review:integration:r2` 第 2 条）。
   - 修复者把这两个函数也改成 `if (C10_LIKELY(start_.load()))`。
   - 新增的变异"只去掉 `ReportHash` 的写入方登记"（第二轮修复时的编号 m6，与 A4.3 表中的 m6 不是同一个变异）在 TSan 下 0/3 检出，因为数据有锁保护，没有数据竞争。为此加了语义检查 `encoded_after_stop`（主线程 `Stop()` 返回后不得再有记录被编码），之后 plain 构建 3/3 检出，TSan 构建仍 0/3（`repair:r2` 结果第 6 条）。
3. **stub 的 `PyEval_SetProfile` 从不 detach。** 工作线程也没有调用帧，所以 `startOne` 始终处于 attached 状态，`stop()` 的 stop-the-world 无法落在它中间。审阅者因此指出"18/18 TSan 通过不能证明这条路径安全"（`review:integration:r2` 第 1 条）。
   - 修复者让 stub 在安装前 detach 0–2 ms，给工作线程加调用帧，每次 `stop` 后检查 `stale_hooks_after_stop == 0`。
   - 之后第一轮修复的快照在 plain 构建下 3/3 失败，`stale_hooks_after_stop` 分别为 37、37、24（`repair:r2` 结果；`torch_npu-work/tsan-logs/round2/`）。
4. **stub 自己的锁导致挂起。** 模拟 stop-the-world 的 `std::shared_mutex` 在 glibc 上优先读者，加压后写者长时间无法获得读写锁，模拟的 stop-the-world 无法完成：一次 TSan 运行耗时 132 秒，plain 构建挂起。改为写者优先的读写锁后，TSan 单次约 1 秒（`repair:r2` 结果"测试改动"；`stub_all.h` 第 59–101 行）。
5. **最终版本的检出能力仍然有限。** 见 A4.3 的表：m1 2/25、m2 1/25、m10 0/25、m11 0/25（`torch_npu-work/final-review-upload-r1/mut-summary.txt`）。
   - 最终审阅者给出的改法是在 `Stop()` 与 `UnInit()` 之间加 `RandomPause`，并在提前返回检查与 `fetch_add` 之间加暂停。
   - 测试文件在 `663e8ba` 之后没有改动，`RunDumper` 中 `first.Stop()` 之后紧接 `AfterStop([&] { first.UnInit(); })`（`profiler_thread_safety_test.cpp` 第 407–408 行），这两条建议没有实施。

**下次怎么做。**

1. 每写一处并发修复，同时写一个把它改回去的变异，确认测试能稳定检出（例如 N 次运行中 N 次失败）之后再信任测试结果。
2. 除了 TSan，加语义检查（计数、顺序、停止后不再写入），覆盖锁内的逻辑错误和 TSan 不建模的情况。
3. stub 要模拟被测代码依赖的阻塞行为（detach、stop-the-world）；stub 自己的锁也要检查公平性。
4. 接手后可以先实施上面第 5 条中最终审阅者的两条建议，再重新运行 `mkmut.py` 的变异，对比失败次数。

### B3. stub 与真实 CPython 3.14 的差距

**发生了什么。** 最终审阅者对照 `legacy_tracing.c` 列出了 stub 的差距（`final:tracer` 第 6 条）：

- 真实的 `_PyEval_SetProfile` 在替换回调之前执行 Python 审计回调，stub 不执行任何 Python 代码；
- `ModuleInfo`、`parseModuleParams`、`parseOptimizerParams`、`py::repr` 调用的 CPython 接口在 stub 中从不报错，所以抛 Python 异常的路径没有被测试覆盖。`f36d74d` 修复的泄漏正是在这条路径上，仓库测试没有发现，是审阅者另写的 `exc_leak.cpp` 发现的；
- GIL 路径只编译、不运行，"有 GIL 时行为不变"只来自阅读代码。

此外，`ProfilerMgr` 没有编译进测试；`profiler_python.cpp`、`profiler_mgr.cpp` 没有用真实的 torch、CPython 3.14、CANN 头文件编译过（[fix_details.md](fix_details.md) 第 1、4.5 节）。

做上游对照时，stub 的另一处差距也产生了后果：stub 的 `py::handle::attr` 不访问对象，传入 NULL 不会崩溃，所以压力测试没有发现 [analysis.md](analysis.md) 第 4.2 节那处空指针访问。

**下次怎么做。**

1. stub 里加两个开关：在 `PyEval_SetProfile*` 替换回调之前执行一个回调（模拟审计回调）；让 `py::object::attr` 可注入异常（这两条是最终审阅者的建议，尚未实施）。
2. stub 在模拟一个 C API 时，如果真实实现会对非法输入报错或崩溃，stub 也要能表现出来，否则调用方缺少判空的问题测不出来。
3. 在有真实 torch 和 CPython 3.14t 的机器上至少编译一次，并运行一次 GIL 构建的原有 profiler 测试（[fix_details.md](fix_details.md) 第 7 节第 3–5 步）。

### B4. 设计之前没有查上游最新代码

**发生了什么。**

- 任务在 2026-11-02 的修复提交（`663e8ba`、`f36d74d`）之前收到，具体日期没有可核对的出处（本机可核对的最早记录是 `torch_npu-work/clone.log`，2026-11-02 00:13 `-0800`）。修复基于 r22 分支（TorchNPU 2.13.0 基线 `46936d1`）设计。上游 `Ascend/pytorch` 在 2026-10-29 的 `ff5ad88d8a`（"fix:profiler fix with_stack memory bug"）中已经重构了 `PythonTracer`。
- 这件事是修复完成、写说明文档时才查到的（当时的进展汇报写明"这是新查出来的"；记录未公开）。
- 第一版文档（`675f2e9`）把上游方案概括为"FT 专用的加锁处理"。第二轮文档审阅者读了上游代码后指出概括不准确：上游把缓存和事件按线程存放，`control_mutex_`、`report_mutex_`、`callback_mutex_` 在两种构建中都启用，FT 且 3.14 以上直接调用 `_PyEval_StopTheWorld`（第二轮文档审阅结论第 1 条，记录未公开；更正提交 `da752e6`）。
- 代价不只是文档措辞。按后来做的对照（[analysis.md](analysis.md) 第 7.3 节），问题 3 的修改建议改用上游方案，也就是这部分工作可能不会被合入；而且同一次对照才发现基线缺少 `b2e659d62f`（[analysis.md](analysis.md) 第 4.2 节）和 issue #5658 所述路径，这两处在单线程、有 GIL 时同样成立，与三处竞态并列为使用方崩溃的候选原因（[analysis.md](analysis.md) 第 1.5 节）。问题 1、2 在上游 master 中逻辑未变，仍然需要修复；上游做过 clang-format，补丁不能直接 `git apply`。

**下次怎么做。** 动手设计之前，先用 `gh api 'repos/Ascend/pytorch/commits?path=<文件>'` 查要修改的文件在上游 master 的近期提交，读完相关提交再决定方案；同时查这个文件相关的 open issue，上游已修的问题可能正是手上要解决的现象。

### B5. 数字要从日志直接统计，并区分场景运行和编译步骤

**发生了什么。**

1. **23/25 和 21/22。** 压力测试作者报告"原始文件 25 次运行中 23 次失败"。第一轮集成审阅者核对 `SUMMARY.json`：三个构建分别为 9/8、9/7、7/6（运行数/失败数），共 25 行，其中 3 行是编译步骤且全部通过，所以实际是 22 次场景运行中 21 次失败（`review:integration` 第 3 条；`torch_npu-work/tsan-logs/baseline-*/`）。
2. **"约 20 个问题、全部修复"无法核实。** 第一版文档写"四轮审阅共发现约 20 个问题，现已全部修复"。第二轮文档审阅者表示没有找到逐条清单，无法核实。`da752e6` 改为精确统计："前两轮共提出 21 条意见：16 条修改了代码或测试，2 条更正措辞，1 条写进注释，2 条只做了性能测量"。这个统计与 `repair:r1`（14 条）、`repair:r2`（7 条）的逐条结果一致。
3. **49 和 42。** 提交信息 `663e8ba`、`f36d74d` 写的是"49/49 runs pass"，其中包含 7 次编译检查；[fix_details.md](fix_details.md) 写的是 42 次场景运行加 7 次编译检查。两者不矛盾，但单看提交信息容易误解。
4. **审阅轮数已核对并改正。** `675f2e9`、`da752e6` 版本的 `profiler_thread_safety.md` 第 4.4 节写"四轮独立审阅……第三轮发现第二轮修复中的两处问题"。
   - 审阅记录中的代码审阅是三轮，每轮两位审阅者，标签见 B 部分开头。另有两轮文档审阅。
   - 记录中与"第二轮修复中的两处问题"最接近的，是第二轮修复者用加强后的测试自己发现的两处（B1 表中"第二轮修复自测"两行），发现者是第二轮修复者本人。
   - 整理文档目录时已按这两点改写 [fix_details.md](fix_details.md) 第 4.4 节。

**下次怎么做。**

1. 结果数字由脚本从 `SUMMARY.json` 统计，不手工相加。
2. 报告时把场景运行和编译步骤分开写。
3. "约""全部"这类说法要有逐条清单支撑，否则不写。
4. 轮数、人数这类计数写之前，回到记录里数一遍标签。

### B6. 文档初稿的事实错误，集中在构建范围、锁规则和未实测的推断

**发生了什么。** 第一轮文档审阅提出 16 条意见（记录未公开），其中几条事实错误：

- 初稿写"行为变化只影响 FT 构建"。审阅者 grep 确认 `data_dumper.cpp`、`data_dumper.h`、`ring_buffer.h` 中没有 `Py_GIL_DISABLED`，问题 1、2 的修改两种构建都生效。
- 初稿写"持有这些锁时不调用 Python 代码"。代码注释写明 `control_mutex_` 持有期间会执行 Python 代码，这条规则只适用于 `registration_mutex_` 和 `data_mutex_`。
- 初稿写"改弱后在 aarch64（鲲鹏）上会出错"，没有实测，改为"可能出错，未实测"。
- 初稿写"会因空指针访问出现 SIGSEGV"，这个判断来自使用方，没有调用栈支撑，改为注明"使用方判断，未经核实"。
- 初稿把 `std::vector::clear()` 写成"释放缓冲区内存"；实际只销毁元素、把大小置 0，已分配的存储保留。

另外，`675f2e9`、`da752e6` 版本里"上游不存在我们方案中全局 `data_mutex_` 的争用"这一句也不成立：上游 FT 构建的回调每个事件仍要获取一次全局 `results_mutex_`，只是次数更少、持有时间更短。这是做上游对照时读代码发现的，整理文档目录时已改（见 [analysis.md](analysis.md) 第 4.5 节）。

**下次怎么做。**

1. 写"影响哪种构建"之前，先 grep 条件编译宏。
2. 写锁规则时对照代码注释逐把核对。
3. 每条结论标明来源：实测、读代码推断、还是他人判断。
4. 写"对方没有某个问题"之前，把对方的热路径逐行读一遍，不按设计方向推断。

### B7. 修复把一种失败形态换成了另一种

**发生了什么。** 最终审阅者做了 fork 实验（`torch_npu-work/final-review-upload-r1/fork/fork_test.cpp`：8 个写入方在 `Report` 中时 fork 40 次，子进程在 2 秒超时内调用 `Stop`）：

- 修复后的代码有 4/40 个子进程挂起：fork 时正在 `Report` 中的写入方没有复制到子进程，子进程继承的 `writers_` 永远大于 0；
- 原代码在同一实验中产生 core dump：子进程对不存在的线程执行 join。

这一点没有修复，只写进了代码注释和 [fix_details.md](fix_details.md) 第 6 节（`final:upload` 第 2 条；`f36d74d` 提交信息）。

**下次怎么做。** 引入"等待计数归零"这类协议时，检查 fork、线程异常退出等计数无法归零的路径，并写明失败时的表现（这里是从崩溃变为挂起，挂起更难诊断）。

### B8. 中途修改测试代码会使已有日志作废

**发生了什么。** 第一轮修复期间测试代码改动了两次，`torch_npu-work/tsan-logs/round1-attempt1-stale/` 和 `round1-attempt2-stale/` 中的日志对应旧测试代码，已作废。修复者最终没有删除这两个目录，它们至今保留（`repair:r1` 结果末尾）。

做上游对照时又出现一次同类情况：扩展场景第一次运行的 TSan 报告是 stub 自己的问题（新线程在 attach 之前写帧），改正后报告消失，那次的日志已删除（[analysis.md](analysis.md) 第 5.2 节）。

**下次怎么做。** 日志目录名带上被测代码和测试代码的提交号，或者每次运行时把 `git rev-parse HEAD` 和 `git diff --stat` 写进日志。

---

## 出处列表（访问日期均为 2026-11-02）

| 主题 | 出处 |
|---|---|
| PEP 703 | https://peps.python.org/pep-0703/ |
| PEP 779 | https://peps.python.org/pep-0779/ |
| 3.14 What's New | https://docs.python.org/3.14/whatsnew/3.14.html |
| free-threading 使用说明 | https://docs.python.org/3.14/howto/free-threading-python.html |
| C 扩展支持 free threading | https://docs.python.org/3.14/howto/free-threading-extensions.html |
| 线程状态、attach/detach | https://docs.python.org/3.14/c-api/threads.html |
| PyMutex、critical section | https://docs.python.org/3.14/c-api/synchronization.html |
| PyEval_SetProfile(AllThreads) 文档 | https://docs.python.org/3.14/c-api/profiling.html |
| CPython 3.14 源码 | https://github.com/python/cpython/blob/3.14/Python/legacy_tracing.c 、https://github.com/python/cpython/blob/3.14/Python/pystate.c 、https://github.com/python/cpython/blob/3.14/Python/ceval.c |
| CPython 3.13 源码 | https://github.com/python/cpython/blob/3.13/Python/legacy_tracing.c |
| C++ 内存顺序 | https://en.cppreference.com/w/cpp/atomic/memory_order |
| C/C++11 到处理器的指令映射 | https://www.cl.cam.ac.uk/~pes20/cpp/cpp0xmappings.html |
| x86-TSO | https://www.cl.cam.ac.uk/~pes20/weakmemory/cacm.pdf |
| Vyukov 有界队列 | 原始页面 http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue （当天跳转到无关域名，未打开）；代码副本 https://github.com/gabime/spdlog/blob/1bd10957cb47824c0e24859db2bd637e472ab97d/include/spdlog/details/mpmc_bounded_q.h |
| ThreadSanitizer | https://clang.llvm.org/docs/ThreadSanitizer.html |
| AddressSanitizer、LeakSanitizer | https://clang.llvm.org/docs/AddressSanitizer.html |
| GCC sanitizer 选项 | https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html |
| TSan 与 C++11 内存模型 | https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2017/POPL.pdf |
| 变异测试综述 | doi:10.1109/TSE.2010.62 |
