# 方案分析：问题怎么定位的、候选方案、与上游的对照

- 对应修复：`yinwu01/torch_npu` 分支 `fix/profiler-thread-safety`，提交 `663e8ba`、`f36d74d`。问题编号与 [fix_details.md](fix_details.md) 一致。
- 本文回答四件事：问题是怎么定位出来的；每个问题还有哪些修法、为什么选了现在这个；上游 `ff5ad88d8a` 的方案和我们的差别；接手的工程师该怎么做。
- 日期：2026-11-02。代码事实来自本仓库 `f36d74d` 和上游文件；测量在本机 stub 环境中完成。
- 时间顺序：第 2、3、5 节的测量（2026-11-02 03:58 至 04:23 `-0800`）都晚于修复提交（`663e8ba` 02:47、`f36d74d` 03:07）和最初的文档（`675f2e9` 03:48、`da752e6` 03:51），属于事后核对；当初的方案选择依据的是推理。
- 标注约定：**【已验证】** 表示读过原文、运行过命令或测试；**【推断】** 表示只根据阅读代码得出，没有运行验证。没有标注的句子是代码事实或本文的组织性说明。
- **文中引用的日志路径（`/SSD1/wuyin/project/torch_npu-work/…`）都在作者本机，没有上传 GitHub，接手的工程师无法打开。** 保留这些路径是为了说明每个数字出自哪一次运行。第 6.5 节列出的脚本（变体生成、基准、上游对照、`py314-checks/` 下的脚本）同样只在作者本机，没有进仓库，**所以本文第 2、3、5 节的数字、第 7.3 节建议的依据都无法由他人复现**。第 7.3 节第 4 条建议把这套测试整理进仓库，本次交付没有做。
- 本文所用测试工具的运行依赖：Linux，g++ 11.4（C++17，支持 `-fsanitize=thread`、`-fsanitize=address`），Python 3.8 以上（只用标准库），`systemd-run --user`。不需要 torch、CANN、NPU，不需要联网。第 4.2 节的 CPython 行为核对另外用了 CPython 3.14.3、3.14.3t（python-build-standalone 构建，解包在 `torch_npu-work/py314/`）和 CPython 3.12.13（本机已有的解释器，来源未记录）。
- 审阅：本目录五个文件在整理后经过三轮文档审阅（AI 模型子代理，没有人工审阅）；第三轮审阅意见已修改，修改后未再经独立审阅。

## 术语

第一次出现时在这里统一说明，后文不再解释。

- **FT 构建**：CPython 去掉 GIL 的构建（free-threaded，编译宏 `Py_GIL_DISABLED`）。**GIL 构建**：CPython 的默认构建。这两个词只用于区分 CPython 的两种构建，不用于区分编译选项。
- **stub**：只保留接口、实现最简的替代头文件，用来在没有 torch、CANN、CPython 3.14 的机器上单独编译被测源码。
- **TSan / ASan / LSan / UBSan**：ThreadSanitizer、AddressSanitizer、LeakSanitizer、UndefinedBehaviorSanitizer（编译选项依次为 `-fsanitize=thread`、`-fsanitize=address`、`-fsanitize=leak`、`-fsanitize=undefined`），分别检测数据竞争、内存错误（越界、释放后访问、重复释放）、内存泄漏和未定义行为（如有符号整数溢出、空指针解引用）。`run.py` 的 `asan` 构建用的是 `-fsanitize=address,undefined`，即同时开启 ASan 和 UBSan；`run.py` 还设置了 `ASAN_OPTIONS=detect_leaks=1`，开启 LSan（Linux 上 ASan 构建默认也开启）；UBSan 的报告中含 `runtime error:`，`run.py` 把它也算作失败，所以一次"ASan 构建失败"可能来自这三者中的任一个。**plain 构建**指 `-O2`、不带任何 sanitizer 的构建（`run.py` 中的 `plain`）。本文凡涉及编译方式的地方一律用 TSan 构建、ASan 构建、plain 构建这三个名称，不用"普通构建"。
- **变体**：把采用的修改按某个候选方案改写后的一份源码副本，用来比较开销。**参照变体**：去掉部分同步、不是有效修复的副本，只用于给出开销的参照值。
- **变异测试**：故意把某处修复改回错误写法，检查测试能否发现，用来衡量测试检出回归的能力。
- **`seq_cst`**：`std::memory_order_seq_cst`，C++ 最强的内存顺序，所有使用它的操作处于一个所有线程都认同的全局顺序中（详见 [learning.md](learning.md) A2.2）。
- **Vyukov 槽位格式 / Vyukov 队列**：Dmitry Vyukov 提出的有界队列写法，每个槽位带一个原子序号（详见第 3.4 节和 [learning.md](learning.md) A3）。
- **`PyMutex`**：CPython 3.13 起提供的互斥锁，等锁时会临时 detach 线程状态（详见 [learning.md](learning.md) A1.4）。
- **热路径**：采集期间每条记录、每个 Python 调用事件都会执行的代码。这里指 `DataDumper::Report`、`RingBuffer::Push`，以及 `PythonTracer` 的 profile 回调。
- **原子读-改-写**（RMW）：一次完成"读出、修改、写回"的原子操作，如 `fetch_add`、CAS。多个线程对同一变量执行时会相互等待。
- **CAS**（compare-and-swap，比较并交换）：变量等于期望值时才写入新值，否则失败并返回当前值。
- **缓存行**：CPU 缓存和内存之间交换数据的最小单位，本机为 64 字节。多个线程频繁写同一缓存行时，该缓存行在各 CPU 核之间来回传递，称为缓存行争用。
- **互斥锁**：同一时间只允许一个线程持有的锁，本文指 `std::mutex`。**读写锁**：允许多个线程同时以共享方式持有、或一个线程以独占方式持有的锁，本文指 `std::shared_mutex`。
- **MPSC / SPSC 队列**：多写入方单读取方 / 单写入方单读取方的队列。
- **线性一致**（linearizable）：每个操作的效果都等价于在调用到返回之间的某一时刻一次完成。对队列来说，含义是不同写入方的元素按这个时刻的先后出队。
- **tstate**：`PyThreadState`，CPython 给每个线程维护的线程状态结构体。profile 回调函数和它的参数就存在 tstate 里。
- **attach / detach**：FT 构建里，线程执行 Python 代码前要把自己的 tstate 绑定到解释器（attach），等锁或做阻塞操作前解除绑定（detach）。GIL 构建里对应获取和释放 GIL。
- **stop-the-world**（下文简称 STW）：CPython 暂停除调用线程以外的所有线程的机制。只有已经 detach 或运行到检查点的线程才会被暂停，所以一个线程如果在 attach 状态下阻塞等待，发起 STW 的线程会一直阻塞，直到该线程 detach。
- **重入**：同一线程在持有一把锁期间，又调用到需要同一把锁的函数。
- **ctx**：`TraceContext`，tracer 传给 `PyEval_SetProfile` 的 Python 对象，回调通过它找到 tracer。

## 1. 问题是怎么定位的

使用方提供的现场只有一句描述：FT 下开启 `torch_npu.profiler`，进程 SIGSEGV，没有调用栈。使用方自己判断是"上报模块初始化竞态导致空指针访问"，这个判断没有核实过。没有调用栈，也没有可以复现的脚本，所以定位方法是静态阅读上报链路，然后对读出来的每一处可疑点写压力测试。

### 1.1 阅读的范围

按记录从产生到落盘的顺序读一遍（链路本身见 [fix_details.md](fix_details.md) 第 2 节）：

1. 业务线程调用 `ProfilerMgr::Upload*` → `DataDumper::Report` → `RingBuffer::Push`；
2. 每个 `DataDumper` 的后台线程 `Pop` 出记录，编码后写进 `FRAMEWORK/` 下的文件；
3. 停止采集时 `ProfilerMgr::Stop` 对三个上报器依次调用 `Stop` 和 `UnInit`；
4. 开启 `with_stack` 等选项时，`PythonTracer` 在每个 Python 线程上安装 profile 回调，回调写入共享缓存，累积到阈值后交给 `TraceDataDumper`。

判断可疑的标准只有一条：同一块数据，是否存在两个线程可以同时访问、而代码里看不到任何同步。按这个标准读出三处。

### 1.2 读出的三处

| 编号 | 位置 | 同时访问同一数据的两方 |
|---|---|---|
| 问题 1 | `data_dumper.cpp`、`ring_buffer.h` | 写入方在 `Report`/`Push` 中访问缓冲区；停止方在 `UnInit` 中 `clear()` 销毁缓冲区元素 |
| 问题 2 | `ring_buffer.h` | 多个写入方各自占用槽位后分别写入；读取方按一个公共计数 `write_index_` 判断哪些槽位可读 |
| 问题 3 | `profiler_python.cpp` | 多个线程的 profile 回调写同一批缓存和 `events_`；`start/stop/clear` 与子线程的 `startOne/stopOne` 之间没有同步 |

三处的具体执行顺序和修改内容在 [fix_details.md](fix_details.md) 第 3 节，本文不重复。

### 1.3 为什么有 GIL 时很少暴露

有 GIL 时，执行 Python 字节码的线程同一时刻只有一个，`PythonTracer` 的回调和控制操作都在 Python 线程上，所以问题 3 的几处共享访问基本依次发生。有 GIL 时也会暴露，只是概率低：

- GIL 可以在字节码边界、以及 C 扩展主动释放时切换，所以两个 Python 线程的回调之间仍可能交错。
- 问题 1、2 的写入方不只有 Python 线程。算子下发、NPU 任务队列出队等记录来自 C++ 线程，这些线程不持有 GIL，GIL 对它们没有串行化作用。`ring_buffer.h`、`data_dumper.h/.cpp` 的修改也因此在两种构建中都生效（这两个文件里没有 `Py_GIL_DISABLED` 条件编译）。
- 停止流程本身是在已释放 GIL 的线程上执行的。

所以准确的说法是：FT 去掉 GIL 之后，问题 3 从"很少交错"变成"经常交错"；问题 1、2 原来就可能发生，FT 下并发程度更高。这一节是按阅读代码得出的【推断】，没有在真实环境中测量两种构建下的暴露概率。

### 1.4 验证手段

把这三处单独编译成压力测试（`test/cpp/profiler_thread_safety/`，用 stub 头文件替换日志、torch、CPython、`ProfilerMgr`），对原代码和修复后的代码运行同一套测试：原代码 42 次场景运行全部失败（含 SIGSEGV），修复后 42 次全部通过。细节、局限和复现命令见 [fix_details.md](fix_details.md) 第 4 节。

需要说明测试证明了什么：它证明这三条路径存在竞态，不能证明使用方遇到的崩溃来自其中哪一处。测试的调用方式（例如两个线程并发 `Stop`）比 `ProfilerMgr` 的实际调用更密集。

### 1.5 定位工作漏掉的问题，以及崩溃根因的三个候选

静态阅读只按"有没有同步"这一条标准筛，所以漏掉了一处与线程无关的问题（下面第 1 条）。另一处（第 2 条）在 `663e8ba` 中已随 `clear()` 一起修改，但当时没有把它列为使用方崩溃的候选。两处都是修复完成、做上游对照时才对应到上游的提交和 issue。

1. **基线缺少上游 `b2e659d62f`**（漏掉的一处）：在 CPython 3.13 及以上读取 `nn.Module` 帧的 `self` 会得到 NULL，随后对 NULL 取属性，见第 4.2 节。
2. **上游 issue #5658 所述路径**（已修改，当时没有列为候选）：【已验证，读代码】`d3216c2` 的 `PythonTracer::clear()`（原文第 420–433 行）直接执行 `thread_local_results_.clear()`，`~ThreadLocalResult` 中会 `Py_DECREF(ctx_)`；调用方 `stopNpuProfiler()`（`npu_profiler.cpp:375` 起）在第 377 行先执行 `pybind11::gil_scoped_release`，然后依次调用 `kStop`、`kClear`。所以原代码在 FT 和 GIL 两种构建下都会在没有 attach 的线程上释放 Python 对象。issue #5658 原文：开启 `with_modules=True` 后停止 profiler 时在没有持有 GIL 的情况下释放 `TraceContext`，开启 `PYTHONMALLOC=debug` 时必定出现 coredump。本分支的 `clear()` 已改为先取 GIL 再释放（见第 4.8 节表格对应行）。

因此使用方崩溃的根因有三个候选：上面这两处，以及三处竞态中的某一处。前两个候选在单线程、有 GIL 时同样成立，竞态只在并发时成立。

**按使用方的配置和崩溃时机排除候选**【推断，依据读 `d3216c2` 的代码和第 4.2 节】。在不知道使用方的 profiler 配置和崩溃时机的情况下，三个候选无法排序；知道这两项之后可以排除一部分，不需要 gdb：

- `PythonTracer` 只在开启 `with_stack` 或 `with_modules`、并且采集 CPU activity 时启动：`npu_profiler.cpp` 第 95–98 行的 `tracePython()` 返回 `(with_stack || with_modules) && activities_.count(CPU)`，第 354–355 行的 `kStartAll` 和第 391–393 行的 `kStop`、`kClear` 都受它控制。候选 1、2 和候选 3 中的问题 3 都在 `PythonTracer` 中，所以这两项都没开（或不采集 CPU activity）时只剩候选 3 中的问题 1、2。
- 在 3.13 及以上，`d3216c2` 的 `genPyCallHashId`（`profiler_python.cpp` 第 602–611 行）进入 `nn.Module.__call__` 帧时取得 NULL，`get_hash(NULL)` 不在缓存中，随即执行 `ModuleInfo(NULL)`。所以只要采集期间调用过任何 `nn.Module`，候选 1 就在第一次前向时发生，早于停止采集，候选 2 没有机会出现；候选 2 只可能出现在采集期间没有调用过 `nn.Module` 的情况。另外，`start()` 记录已有调用栈时也调用 `genPyCallHashId`（第 337 行），如果开始采集时调用栈里已有 `nn.Module.__call__` 帧，候选 1 在开始采集时就会发生。
- 对应关系：崩溃发生在开始采集或前向中，对应候选 1，或者候选 3 中的问题 2、3（这两处在采集期间也会发生，但需要多个写入方同时上报或多个 Python 线程同时执行回调）。只有一个 Python 线程时问题 3 不会发生，但问题 2 的写入方还包括算子下发、NPU 任务队列出队等 C++ 线程（第 1.3 节），所以不能据此排除问题 2。崩溃发生在停止采集时（`prof.stop()` 或退出 `with` 块），对应候选 2，或者候选 3 中的问题 1、问题 3 的 `stop/clear` 与回调之间的竞争。

所以接手后的第一步是向使用方要 profiler 配置（`with_stack`、`with_modules`、`activities` 是否含 CPU、`profile_memory`、`record_shapes`）、是否多线程，以及崩溃发生的时间点（第 7.1 节）。

## 2. 问题 1 的候选方案

要解决的事情：`Stop` + `UnInit` 释放缓冲区之前，必须确认没有写入方还在 `Report` 中访问它；同时 `Stop` 返回后，`Flush` 要能写完所有已被接受的记录。

### 2.1 候选 A：互斥锁包住 `Report` 和 `Stop`

- **做法。** `Report` 持锁检查 `start_` 并 `Push`；`Stop` 持锁把 `start_` 置 false（或置 false 后获取一次锁）。持锁期间检查和写入不会与停止交错。
- **热路径开销。** 每次 `Report` 一次加锁和解锁。所有写入线程串行通过同一把锁，锁所在缓存行在各核之间争用。
- **内存。** 一个 `std::mutex`（glibc 上 40 字节）。
- **复杂度。** 最低，正确性容易审阅；不依赖内存序细节，弱内存序平台上没有额外风险。
- **行为变化。** 写入线程之间串行化，业务线程在 `Report` 中等待的时间随线程数增加。
- **仓库中的先例。** `ProfilerMgr::UploadWithLock`（`profiler_mgr.cpp`）已经对 `dataReceiverWithLock_` 的每次 `Report` 持有 `reportDataMutex_`，调用方是 `npu_profiler.cpp` 中的内存记录和 `OpMark` 记录。但 `StopDataReceiver` 不获取这把锁，所以它防不住问题 1。如果只想修这一个上报器，可以让停止方也获取 `reportDataMutex_`；`dataReceiver_`（算子记录，`npu_profiler.cpp:298`）和 `traceDataReceiver_` 没有这把锁。这一做法没有实现和测试。
- **实测（`Report` 基准，见 2.7；下文所有速率的单位都是百万次/秒，括号外为 5 次重复的最小值到最大值，括号内为中位数）。** 8 个写入线程时 `Report` 调用速率 1.37 到 2.37（2.11），采用方案 3.86 到 4.32（3.98）；32 个时 1.17 到 1.31（1.26）对 2.40 到 4.05（3.78）。接受速率两者相近。
- **结论。** 放弃。正确性没有问题，但热路径上所有写入线程串行化，而 FT 下多线程上报正是这次要支持的场景。

### 2.2 候选 B：读写锁

- **做法。** `Report` 以共享方式持有 `std::shared_mutex`，多个写入方可以同时持有；`Stop` 以独占方式获取一次，等待所有共享持有者退出。
- **热路径开销。** 每次 `Report` 获取和释放一次共享锁。共享获取需要修改锁内的读者计数，所有写入线程仍争用同一缓存行；【推断】开销高于 `writers_` 的两次 `fetch_add/fetch_sub`。
- **内存。** 一个 `std::shared_mutex`（glibc 上 56 字节）。
- **复杂度。** 低。
- **行为变化。** glibc 读写锁的默认类型偏向读者（`/usr/include/pthread.h`：`PTHREAD_RWLOCK_DEFAULT_NP = PTHREAD_RWLOCK_PREFER_READER_NP`；libstdc++ 的 `std::shared_mutex` 是否使用默认类型没有核实）；写入方持续上报时，`Stop` 的独占获取可能等待较久。这一点没有测量。
- **实测。** 见第 2.7 节。这组数据不能区分它与采用方案的单次调用开销。
- **结论。** 放弃。它与采用的方案结构相同（写入方登记、停止方等待），只是把计数放进锁里；去掉锁的部分就是候选 E。选择候选 E 的依据是设计推理：`Report` 路径不取锁。如果维护者更看重"用标准库原语、不手写内存序"，这是最接近的替代方案。

### 2.3 候选 C：缓冲区在进程生命周期内只分配一次，从不释放

- **做法。** `UnInit` 不释放 `RingBuffer`，下一次 `Init` 复用。写入方即使在 `Stop` 之后才执行 `Push`，访问的也是有效内存，不会出现释放后访问。
- **热路径开销。** 与原代码相同。
- **内存。** `ProfilerMgr` 的容量是常量（`capacity_` = 2^20，`trace_capacity_` = 128），两个 2^20 槽位的缓冲区在首次采集后一直占用：原槽位格式 16 MB，Vyukov 槽位格式 32 MB。此外，在 `Stop` 检查之后才完成写入的记录对象会一直留在槽位中，直到被取出。
- **遗留问题。**
  - 在 `Stop` 检查之后才完成 `Push` 的记录不会被本次 `Flush` 写出，会留在缓冲区里，被下一次采集的读取方写进下一次采集的目录。结果文件中会混入上一次采集的记录。
  - 如果下一次 `Init` 清空槽位，清空操作又会和这些尚未完成的写入方竞争，问题只是从 `UnInit` 移到了 `Init`。
  - `RingBuffer::is_inited_` 等非原子字段的数据竞争仍需单独处理。
  - 容量变化时仍要重新分配；目前容量是常量，所以这一条在现有调用方式下不会发生。
- **结论。** 放弃。它消除了崩溃，但把"在 `Stop` 检查之后才完成的写入"变成了跨采集的数据错误，并且永久占用内存。

### 2.4 候选 D：用 `shared_ptr` 持有缓冲区，原子地替换

- **做法。** 缓冲区由 `std::shared_ptr<RingBuffer>` 持有。`Report` 先 `std::atomic_load` 取得一份引用再 `Push`；`Stop` 用 `std::atomic_exchange` 换成空指针。最后一个引用释放时缓冲区才被销毁，因此不会释放后访问。
- **热路径开销。**
  - 按 GCC 11 的 `bits/shared_ptr_atomic.h`，`std::atomic_load` 对 `shared_ptr` 的重载内部构造 `_Sp_locker`，即获取 libstdc++ 内部的一把互斥锁；GCC 11 的头文件中没有 C++20 的 `std::atomic<std::shared_ptr>`。
  - 复制和析构 `shared_ptr` 各有一次对引用计数的原子读-改-写，所有写入线程争用同一个控制块，与 `writers_` 的争用相同。
  - 所以热路径至少包含候选 A 的一次加锁和候选 E 的两次原子读-改-写。没有实测。
- **遗留问题。** `Stop` 换出指针后，仍可能有写入方持有旧引用并继续 `Push`。要让 `Flush` 写完所有已接受的记录，`Stop` 必须等待引用计数降到 1，这等价于候选 E 的等待；或者让缓冲区析构时自己写文件，那样写文件会发生在某个业务线程上，而且可能晚于 `UnInit` 关闭文件。
- **结论。** 放弃。开销不低于候选 E，还要额外的等待逻辑。

### 2.5 候选 E：`writers_` 计数 + `seq_cst` 双向检查（采用）

- **做法。** 见 [fix_details.md](fix_details.md) 第 3 节问题 1。写入方先 `writers_.fetch_add(1)` 再读 `start_`，停止方先 `start_.exchange(false)` 再读 `writers_`，四个操作都是 `seq_cst`，因此"写入方看到 false"和"停止方看到写入方"至少有一个成立。
- **热路径开销。** 采集期间每次 `Report` 多两次原子读-改-写。`writers_` 由所有写入线程共用，在同一缓存行上争用；已用 `alignas(64)` 与只读为主的 `start_` 分开。未采集时第一次 `relaxed` 读取就返回，开销与原代码相同。
- **内存。** 每个上报器 8 字节计数，按 64 字节对齐。
- **复杂度。** 中等。正确性依赖 `seq_cst`，改弱后在 x86 和 TSan 下都测不出（[fix_details.md](fix_details.md) 第 4.5 节的变异 25 次运行全部通过）。
- **行为变化。** `Stop` 中忙等写入方退出；fork 出的子进程如果继承了大于 0 的 `writers_`，`Stop` 会一直等待（[fix_details.md](fix_details.md) 第 6 节）。
- **实测。** 单位为百万次/秒，5 次重复的范围（中位数）。8 线程 3.86 到 4.32（3.98），32 线程 2.40 到 4.05（3.78）。与参照变体（只去掉 `writers_` 登记和 `WaitWriters` 等待、其余修改保留，不是有效修复；8 线程 3.78 到 5.27（4.89），32 线程 2.49 到 5.25（4.34））相比中位数较低，但范围重叠，5 次重复不足以确定这两次原子操作的开销。之前一轮测量（`torch_npu-work/tsan-logs/round2/bench-report-writers.log`，同一机器，每组 3 次）中，8 线程和 32 线程两组范围重叠；单线程一组范围不重叠，有 `writers_` 为 4.32 到 4.73，无 `writers_` 为 4.91 到 4.93，有 `writers_` 的一侧较慢。
- **为什么采用。** 依据是设计推理：`Report` 路径不取锁，只多两次原子读-改-写；改动只限 `data_dumper.h/.cpp`，`ProfilerMgr` 和调用方不需要修改。本节的测量在修复提交之后进行（见文首），不能区分它与读写锁的单次调用开销（第 2.7 节）。

### 2.6 候选 F：每个写入线程一个缓冲区

- **做法。** 每个写入线程首次 `Report` 时注册一个自己的 SPSC 缓冲区，读取方轮流读取各线程的缓冲区。释放时逐个检查每个线程自己的"正在写入"标志，避免共享计数。
- **热路径开销。** 写入方只写自己的缓冲区和标志，没有跨线程的缓存行争用；首次注册需要一次加锁。
- **内存。** 线程数乘以每线程容量。按现在的容量（2^20 槽位，Vyukov 格式 16 MB）每个线程一份不可接受，需要另定更小的每线程容量；容量减小后，单个线程短时间内大量上报时更容易写满丢弃。
- **复杂度。** 高：要处理线程退出时缓冲区的归属、采集之间的注册表重置、读取方的轮询策略。
- **行为变化。** 同一线程内的记录顺序保持；不同线程之间的写入先后顺序不再保留（采用的方案按占用槽位的顺序取出）。下游解析是否依赖跨线程顺序，没有核实。
- **结论。** 放弃。它同时解决问题 1 和问题 2 的争用，但改动范围和内存代价都大于这次修复的需要；就第 2.7 节 5 次重复的数据而言，`writers_` 的开销没有超出本机噪声（较早一轮的单线程测量显示出差别，见第 2.5 节）。如果将来在真实 NPU 场景中测出 `writers_` 争用明显，这是下一步可以考虑的方向。

### 2.7 实测：`Report` 基准

单位为百万次/秒，5 次重复，给出最小值到最大值，括号内为中位数。测量方法和局限见第 6 节。各变体之间同步修改的差别：采用方案、互斥锁、读写锁都是有效修复（第 6.3 节）；"参照（去掉 `writers_`）"只去掉 `writers_` 登记和 `WaitWriters` 等待，Vyukov 队列、`control_mutex_` 等其余修改保留，不是有效修复，只用于给出开销的参照值。

| 写入线程数 | 变体 | `Report` 调用速率 | 接受速率 |
|---|---|---|---|
| 1 | 采用方案 | 4.48 到 5.13（4.86） | 4.48 到 5.13（4.87） |
| 1 | 互斥锁 | 3.09 到 5.12（4.77） | 3.09 到 5.12（4.78） |
| 1 | 读写锁 | 3.68 到 5.07（4.45） | 3.69 到 5.07（4.45） |
| 1 | 参照（去掉 `writers_`） | 2.57 到 5.50（4.66） | 2.57 到 5.50（4.66） |
| 8 | 采用方案 | 3.86 到 4.32（3.98） | 2.02 到 2.16（2.08） |
| 8 | 互斥锁 | 1.37 到 2.37（2.11） | 1.37 到 2.37（2.11） |
| 8 | 读写锁 | 2.40 到 3.73（3.49） | 2.24 到 2.49（2.35） |
| 8 | 参照（去掉 `writers_`） | 3.78 到 5.27（4.89） | 1.91 到 2.21（2.08） |
| 32 | 采用方案 | 2.40 到 4.05（3.78） | 1.26 到 1.49（1.33） |
| 32 | 互斥锁 | 1.17 到 1.31（1.26） | 1.17 到 1.31（1.26） |
| 32 | 读写锁 | 1.79 到 3.20（3.06） | 1.41 到 1.61（1.46） |
| 32 | 参照（去掉 `writers_`） | 2.49 到 5.25（4.34） | 1.29 到 1.33（1.32） |

读法：

- 单线程时四个变体范围重叠，无法区分。
- 8、32 线程时，各变体的接受速率都在 1.1 到 2.5 之间，说明总速率受读取方（编码并写文件）限制，多出来的调用因缓冲区写满被丢弃。
- 调用速率包含缓冲区写满后 `Push` 直接失败返回的调用，而各变体的丢弃比例不同：按中位数估算（无出处），8 线程时采用方案约 48%、读写锁约 33%，32 线程时约 65% 对 52%；互斥锁几乎不丢弃。所以调用速率不能比较单次完整写入的开销，这组数据不能区分采用方案与读写锁的单次调用开销。
- 互斥锁的调用速率低于采用方案（8、32 线程范围不重叠），接受速率与采用方案相当（8 线程中位数 2.11 对 2.08，32 线程范围重叠）。
- 原始记录：`torch_npu-work/alt-bench/results/20261102T120205Z/report_bench.log`，汇总 `summary.txt`。

## 3. 问题 2 的候选方案

要解决的事情：读取方只能读到已经写完的槽位，并且同一写入方的记录按写入顺序取出。

### 3.1 候选 A：互斥锁 + `std::deque`

- **做法。** `Push`、`Pop`、`Size` 都在同一把锁内操作 `std::deque`；队列长度达到容量时 `Push` 返回 false。实现共 98 行。
- **热路径开销。** 每次 `Push` 一次加锁；读取方每次 `Pop` 也要加锁，与写入方竞争同一把锁。`std::deque` 按块分配内存，写入时偶尔在锁内分配。
- **内存。** 随队列中的元素数增长，空闲时很小；Vyukov 方案无论是否使用都预先分配全部槽位。
- **复杂度。** 最低。
- **行为变化。** 线性一致，按获取锁的顺序出队。
- **实测。**
  - 队列基准（容量 2^20）：单线程 3.57 到 5.48（4.13），采用方案 3.65 到 8.04（5.56），范围重叠。8 线程时写入成功 1.47 到 1.70，但读取方在 1 秒内只取出 0.42 到 0.65，采用方案取出 2.82 到 4.00；32 线程时读取方取出 0.09 到 0.16，采用方案 1.54 到 2.41。读取方与写入方竞争同一把锁，取出速率低于写入速率，队列积满后写入开始失败。
  - `Report` 基准：接受速率 8 线程 1.23 到 1.34，32 线程 1.09；采用方案为 2.02 到 2.16 和 1.26 到 1.49。单线程时调用速率也较低（1.84 到 3.27，采用方案 4.48 到 5.13），因为读取方与唯一的写入方在同一把锁上交替。
- **结论。** 放弃。正确且简单，但读取方与写入方共用一把锁，多线程下写文件的速率下降，丢弃增多。

### 3.2 候选 B：保留原结构，按占用顺序发布

- **做法。** 原 `Push` 在写完槽位后执行 `write_index_++`。改为等待 `write_index_` 等于自己占用的位置后再加 1：

  ```cpp
  while (write_index_.load(std::memory_order_acquire) != pos) { /* yield */ }
  write_index_.store(pos + 1, std::memory_order_release);
  ```

  这样 `write_index_` 只会越过已写完的槽位。
- **热路径开销。** 无竞争时多一次读；有竞争时写入方按占用顺序依次等待。
- **内存。** 不变（每槽位 8 字节）。
- **复杂度。** 改动约 3 行，是所有方案中最小的。
- **风险。** 一个写入方在占用槽位后、发布前被调度器挂起，后面所有已占用槽位的写入方都在循环中等待，等待时间没有上限（原代码的 `cycle_limit` 只限制占用阶段的重试次数）。FT 下线程数可能超过 CPU 核数，这种等待会出现在业务线程上。原代码中 `is_inited_` 的非原子访问、`Size()` 的计算等问题仍需另外处理。
- **实测。** 没有实现和测量。
- **结论。** 放弃。改动小，但把一个线程的调度延迟传递给所有写入线程。

### 3.3 候选 C：每个槽位一个"已写完"标志

- **做法。** 保留原来的 `idle_write_index_` 占用和 `read_index_` 判满；每个槽位增加 `atomic<bool> ready`。写入方写完后 `ready = true`（release），读取方只在 `ready[read_index_]` 为 true（acquire）时读取，读完置 false。
- **热路径开销。** 写入方判满时要读读取方的 `read_index_`，该变量由读取方频繁写入，写入方和读取方之间有缓存行争用。
- **内存。** 每槽位多 1 字节，对齐后与 Vyukov 槽位同为 16 字节。
- **复杂度。** 与 Vyukov 方案相近，但"空闲"和"已写完"分别由 `read_index_` 和 `ready` 两个变量表示，要单独论证两者之间的顺序。
- **结论。** 放弃。Vyukov 写法把这两种状态编码进同一个序号，写入方只读槽位自己的序号，不读 `read_index_`；它是 Dmitry Vyukov 公开的写法，审阅者可以对照原文检查。

### 3.4 候选 D：Vyukov 有界队列，单读取方（采用）

- **做法。** 见 [fix_details.md](fix_details.md) 第 3 节问题 2。每槽位 `{atomic<size_t> seq; T data;}`；写入方看到 `seq == pos` 时 CAS 占用，写完置 `seq = pos + 1`；读取方只在 `seq == pos + 1` 时读取，读完置 `seq = pos + capacity`。
- **热路径开销。** 一次 CAS 争用 `write_index_`，加上对本槽位序号的一次读取和一次写入。读取方不加锁。
- **内存。** 每槽位从 8 字节增至 16 字节；两个 2^20 槽位的缓冲区合计从 16 MB 增至 32 MB，并且始终预先分配。
- **复杂度。** 中等，`ring_buffer.h` 共 179 行，接口不变。
- **行为变化。** 容量向上取整到 2 的幂；可用容量从 capacity-1 变为 capacity；`Size()` 包含已占用但尚未写完的槽位（[fix_details.md](fix_details.md) 第 3、5.1 节）。
- **实测（队列基准）。** 8、32 线程、容量 2^20 时写入从未失败，读取方取出速率与写入速率一致（8 线程 2.82 到 4.00，32 线程 1.54 到 2.41）；所有运行中错序和丢失计数为 0。
- **为什么采用。** 读取方不与写入方争锁；第 3.7 节的事后测量（见文首）中，多线程下取出速率在测得的有效方案中最高；不需要第三方代码，接口不变，`data_dumper.cpp` 以外的调用方不需要修改。代价是内存翻倍。

### 3.5 候选 E：每个写入线程一个 SPSC 队列，由读取方合并

与 2.6 相同的方案，从问题 2 的角度看：每个队列只有一个写入方，不存在多写入方的发布顺序问题，CAS 争用也消失。代价同 2.6（内存随线程数增长、线程注册和退出、跨线程顺序不再保留）。没有实现和测量。放弃原因同 2.6。

### 3.6 候选 F：使用现成的并发队列库

| 库 | 许可证 | 最近推送（GitHub API，2026-11-02 查询） | 与本场景的差距 |
|---|---|---|---|
| moodycamel `ConcurrentQueue`（cameron314/concurrentqueue） | 简化 BSD 或 Boost Software License 1.0 二选一（LICENSE.md） | 2026-10-26 | README：单个头文件 `concurrentqueue.h`；不是线性一致的，只保证同一写入方的元素按顺序出队；默认按需分配内存，`try_enqueue` 只使用预分配空间 |
| folly `MPMCQueue`（meta/folly，原 facebook/folly） | Apache-2.0 | 2026-11-02 | 头文件注释说明它是线性一致的；`MPMCQueue.h` 包含 folly 的其他头文件（`Portability.h`、`CacheLocality.h`、`detail/TurnSequencer.h` 等），不能只复制一个文件；整体构建 folly 需要哪些依赖，本次没有核实 |
| `boost::lockfree::queue`（boostorg/lockfree） | BSL-1.0 | 2026-10-25 | `queue.hpp` 用静态断言要求元素类型的复制赋值和析构都是平凡的，`std::unique_ptr` 不满足，只能改存裸指针并手动管理释放 |
| rigtorp `MPMCQueue` | MIT | 2024-03-08 | 只查了许可证和推送日期，没有看代码 |
| max0x7ba `atomic_queue` | MIT | 2026-10-23 | 只查了许可证和推送日期，没有看代码 |

仓库中的依赖情况（本仓库可查到的部分）：

- `third_party/` 中有直接放入的头文件（`acl`、`hccl`、`dcmi`、`mspti`、`mstx`、`dlpack`、`shmem`），以及 `.gitmodules` 中的子模块（`op-plugin`、`googletest`、`torchair`、`Tensorpipe`、`fmt`、`nlohmann`、`torch-mlir`、`dvm`、`acl_src/*`），子模块地址都是 gitcode 上的镜像。
- 上述队列库都不在其中；`CMakeLists.txt` 和 `torch_npu/csrc` 中也没有引用 boost 或 folly。
- `CONTRIBUTING.md` 只把 `third_party/` 标为"第三方依赖"目录，没有写新增第三方代码的规则。上游对新增依赖的审批流程没有核实。

**结论。** 放弃。这个队列的文件共 179 行，需求固定（有界、单读取方、存 `unique_ptr`），而引入任何一个库都要增加子模块或复制外部代码，需要上游维护者同意。moodycamel 是其中差距最小的，但它会改变跨写入方的出队顺序，下游是否依赖这一顺序没有核实。

### 3.7 实测：队列基准

单位为百万次/秒，5 次重复，最小值到最大值，括号内中位数。

| 容量 | 写入线程数 | 变体 | 写入成功速率 | 1 秒内取出速率 |
|---|---|---|---|---|
| 2^20 | 1 | 采用方案 | 3.65 到 8.04（5.56） | 3.65 到 8.04（5.56） |
| 2^20 | 1 | mutex+deque | 3.57 到 5.48（4.13） | 3.57 到 5.48（4.13） |
| 2^20 | 8 | 采用方案 | 2.97 到 4.00（3.78） | 2.82 到 4.00（3.65） |
| 2^20 | 8 | mutex+deque | 1.47 到 1.70（1.64） | 0.42 到 0.65（0.59） |
| 2^20 | 32 | 采用方案 | 1.54 到 2.72（1.63） | 1.54 到 2.41（1.63） |
| 2^20 | 32 | mutex+deque | 1.14 到 1.21（1.16） | 0.09 到 0.16（0.11） |
| 1024 | 1 | 采用方案 | 5.19 到 8.28（6.35） | 5.19 到 8.28（6.35） |
| 1024 | 1 | mutex+deque | 2.37 到 5.39（3.84） | 2.37 到 5.39（3.84） |
| 1024 | 8 | 采用方案 | 2.18 到 3.90（3.65） | 2.18 到 3.90（3.65） |
| 1024 | 8 | mutex+deque | 0.41 到 0.77（0.65） | 0.41 到 0.77（0.65） |
| 1024 | 32 | 采用方案 | 0.38 到 2.66（1.14） | 0.38 到 2.66（1.14） |
| 1024 | 32 | mutex+deque | 0.11 到 0.18（0.14） | 0.11 到 0.18（0.13） |

- 两个变体所有运行的错序和丢失计数为 0。原 ring buffer 在容量 1024、8 和 32 线程时出现错序（单次最多 834 条）；8 和 32 线程的各组都出现取出数少于写入成功数的情况。它的数字不作为开销比较，只说明原写法本身也不快。丢失计数的具体成因没有分析。
- 1 秒内取出速率反映写文件一侧能跟上多少；8、32 线程时 mutex+deque 的取出速率中位数约为采用方案的 0.16 倍和 0.07 倍（容量 2^20），范围不重叠。
- 原始记录：`torch_npu-work/alt-bench/results/20261102T120205Z/queue_bench.log`。

## 4. 问题 3 的候选方案，以及与上游的对照

要解决的事情：FT 下多个线程同时执行 profile 回调、子线程注册（`startOne`）和控制操作（`start/stop/clear`）时，共享容器不被同时修改；`clear()` 释放的对象不被回调访问；`stop()` 返回后不再有线程安装着本 tracer 的回调。

问题 3 的候选方案之间没有做本机性能对比。采用方案的回调吞吐见 [fix_details.md](fix_details.md) 第 5.2 节；本文第 5.4 节另有一组与上游的对照，那组用的是扩展过的 stub 和另一次运行，两组数字不能互相换算。

### 4.1 上游改了什么、为什么

对照对象：`Ascend/pytorch` master（2026-11-02 查看时 HEAD 为 `790544fa`）的 `torch_npu/csrc/profiler/profiler_python.cpp`，该文件最后一次修改是 `ff5ad88d8a`（2026-10-29，"fix:profiler fix with_stack memory bug"，合并请求 `Ascend/pytorch!48607`，分支 `gilmemory`，只改这一个文件，+360/−240）。

关联 issue 是 `Ascend/pytorch` #5658（gitcode，2026-10-26 创建，查看时状态 open），标题"[Bug]: profiler存在违规内存操作，会导致采集堆栈时coredump"，正文原文：

> torch_npu 开启 `with_modules=True` 后，停止 profiler 时，在没有持有GIL 的情况下释放 `TraceContext`。
> 开启PYTHONMALLOC=debug的情况下必定出现coredump

PR 描述中的"修改方案"原文共五条，概括了上游的整个设计：给目标线程安装回调不再用 `PyThreadState_Swap`（3.13+ 用 `PyEval_SetProfileAllThreads`）；缓存和事件列表移进每线程的 `ThreadLocalResult`；FT 3.14+ 读取初始线程栈时用 STW 暂停解释器线程；`stop()` 先标记不活跃、清除回调，再释放 GIL 等待每个线程的回调锁；建立一个共享 `TraceContext` 传给所有线程，回调按当前 tstate 查回本线程的数据。

PR 的"功能验证"一栏只有一张截图，没有文字说明测试场景；截图内容没有查看。上游是否有针对 FT 的测试没有核实。

另外两个提交对这个文件各只改了一行 `#include`，逻辑没有变化【已验证】：`6ca3a36a15`（2026-10-26）把 `npu_profiler.h` 换成 `profiler_callbacks.h`，并把 `TraceTag`、`TraceEvent` 移到新文件 `python_tracer_types.h`；`94dd5bf251`（2026-10-27）把 `profiler/utils.h` 换成 `profiler/python/python_frame_utils.h`。【推断】移植到 2.13.0 时预计主要是这两行 `#include` 要换回旧路径，并确认 2.13.0 的 `profiler_python.h` 仍声明 `TraceEvent`；没有在 2.13.0 的头文件上编译验证。master 上的这个文件还新引用了 `torch/csrc/utils/pythoncapi_compat.h` 和 `torch/csrc/Exceptions.h`，这两个头文件在 2.13.0 中是否可用没有核实。

【已验证】`ff5ad88d8a` 之后，`profiler_python.cpp`、`profiler_python.h`、`python_tracer_types.h`、`python/python_frame_utils.h` 在 master 上没有新的提交（2026-11-02 用 GitHub API 按路径查询提交历史）。

### 4.2 一个和线程无关的问题：基线缺少 `b2e659d62f`

上游 `b2e659d62f`（2026-08-21，合并请求 `!43955`，关联 issue #3808"【bug】【Profiler】Python > 3.12 Profiler 修复适配"，已关闭）处理的是：

> PyFrame_GetLocals 不再返回字典值，而是返回 PyFrameLocalsProxy_Type实例, 所以 PyDict_GetItemString已失效，相关接口需替换为 PyMapping_GetItemString
> python 312 with_stack存在析构错误

【已验证】我们分支的 `profiler_python.cpp`（`d3216c2` 原文和 `f36d74d` 修复后都一样）用 `PyDict_GetItemString(f_locals, "self")` 读取 `nn.Module.__call__` 帧和 `Optimizer` step 帧的 `self`（`f36d74d` 第 837、853 行），取不到时没有判空：

```cpp
auto module_class = PyDict_GetItemString(f_locals, "self");
hash_id = c10::get_hash(module_class);
...
ModuleInfo module_info(module_class);   // 构造函数里执行 py::handle(module_class).attr("__class__")
```

【已验证】用 ctypes 在三个 CPython 上调用同样的 C API（脚本 `torch_npu-work/py314-checks/pep667_check.py`，输出 `pep667_check.out`）：

| Python | `PyFrame_GetLocals(frame)` 的类型 | `PyDict_GetItemString(f_locals, "self")` | `PyMapping_GetItemString(f_locals, "self")` |
|---|---|---|---|
| 3.14.3t（FT） | `FrameLocalsProxy` | NULL | 返回 `self` |
| 3.14.3（GIL） | `FrameLocalsProxy` | NULL | 返回 `self` |
| 3.12.13（GIL） | `dict` | 返回 `self` | 返回 `self` |

【已验证】在 3.14.3t 上调用 `PyObject_GetAttrString(NULL, "__class__")`，进程以 SIGSEGV 退出（`py314-checks/null_getattr.py`，退出码 139）。注意这次运行的输出文件 `null_getattr.out` 只记下了 `exit=139`，没有记录所用解释器的 `sys.version` 和 `Py_GIL_DISABLED`，所以"在 3.14.3t 上"这一点无法从日志核对；重新运行时应把这两项写进输出文件。pybind11 的 `handle(nullptr).attr("__class__")` 求值时调用的就是这个函数；【推断】这一点只读了 pybind11 的调用方式，没有在 pybind11 中单独运行。

【推断】综合以上：在 3.13 及以上开启 `with_stack` 或 `with_modules` 后（上表只实测了 3.12.13、3.14.3、3.14.3t；3.13 没有实测，结论依据 PEP 667 推断），第一次 `nn.Module` 调用进入回调时进程就会崩溃，与线程数无关，GIL 构建同样成立。Optimizer 分支在 `record_params_` 为真时（`profile_memory` 且 `record_shapes`，并且开启 `with_stack` 或 `with_modules`）对 NULL 调用 `parseOptimizerParams`，同样会对空指针取属性。使用方报告的现象是"FT 下 SIGSEGV、没有调用栈"，与这一路径相符，但**没有用真实的 torch_npu 运行确认，不能据此认定就是这一处**。

确认方法见第 7.1 节第 2 步。如果崩溃出在这一处，`gdb bt` 的栈顶应在 `PyObject_GetAttrString`，往上是 `ModuleInfo::ModuleInfo`。

我们的压力测试没有发现这个问题，原因是 stub 的 `py::handle::attr` 不访问对象，传入 NULL 不会崩溃。

【已验证】`b2e659d62f` 是 master 的祖先提交。`ff5ad88d8a` 保留了 `PyMapping_GetItemString` 的写法，并在取不到 `self` 时抛出 `KeyError`；同时去掉了 `b2e659d62f` 里"3.12+ 跳过 `Py_DECREF`"的写法，改为 `releaseContext()` 在持有 GIL 时释放唯一的共享 ctx。

### 4.3 候选 A：一把全局锁覆盖整个回调和控制操作

- **做法。** 回调入口加锁，返回时解锁；`start/stop/clear/startOne/stopOne` 获取同一把锁。
- **热路径开销。** 每次回调一次加锁，持锁时间包括 `PyCallInfo` 构造、`py::repr`、模块参数解析等 CPython 调用，所有 Python 线程的回调完全串行。
- **风险。**
  - 持锁期间执行的 CPython 调用可能运行任意 Python 代码（`__getattr__`、`__repr__`、终结器）。这些代码如果在同一线程上再次进入 tracer（例如创建线程触发 `startOne`），不可重入的锁会死锁；改用可重入锁则会在容器修改到一半时重入。
  - 如果用 `std::mutex`，等锁的线程保持 attach 状态，会阻塞 `PyEval_SetProfileAllThreads` 等操作所需的 STW；持锁线程如果正在等待这次 STW，就形成死锁。这一条是按代码分析得出，没有单独测试。用 `PyMutex` 可以避免这一条（等锁时 detach），但前一条仍在。
- **结论。** 放弃。采用的方案可以看作它的收窄版本：锁只覆盖容器操作，CPython 调用放在锁外。

### 4.4 候选 B：三把 `PyMutex` + 会话计数（采用）

- **做法。** 见 [fix_details.md](fix_details.md) 第 3 节问题 3。`control_mutex_` 串行化控制操作并检测同线程重入；`registration_mutex_` 保护注册表和 `session_`；`data_mutex_` 保护缓存和 `events_`。后两把锁持有期间不调用 Python、不分配 Python 对象。所有锁只在 FT 构建生效，GIL 构建中是空操作。
- **热路径开销。** 每次 Python 函数调用至少获取两次 `data_mutex_`（`genPyCallHashId` 和 `recordPyCall` 各一次），C 函数调用和返回各一次。所有线程共用这把锁。
- **内存。** 与原代码相同，缓存全进程一份。
- **复杂度。** 中等。改动集中在 `profiler_python.cpp`，原有数据结构不变；加锁顺序和"持锁不调 Python"两条规则需要后续修改者遵守。
- **行为变化。** 见 [fix_details.md](fix_details.md) 第 5.2 节：FT 下停止时清除所有线程的 profile 函数（包括其他工具的）；启动时只记录当前线程的调用栈。
- **为什么采用。** 在不改动数据结构的前提下覆盖了全部已知竞态，并且只使用 CPython 3.13 起公开的 `PyMutex`（`Doc/c-api/synchronization.rst`，3.14 分支）；GIL 构建不受影响。代价是 `with_stack` 时所有线程争用 `data_mutex_`。
- **这个方案是否值得移植到上游：不建议**，理由见第 7.3 节（上游已经重写了这个文件）。

### 4.5 候选 C：每线程缓存（上游 `ff5ad88d8a` 的方向）

以下按阅读上游 `ff5ad88d8a` 的 `profiler_python.cpp`（796 行）整理。

- **做法。**
  - 每个线程一个 `ThreadLocalResult`，各自持有 `py_call_cache_`、`module_info_cache_`、`events_`、参数缓存和起始调用栈；所有线程共用一个 `shared_ctx_`。
  - 回调通过 `findThreadLocalResult(PyThreadState_Get())` 在 `thread_local_results_map_` 中查找本线程的结果，FT 构建下查找时持有全局 `results_mutex_`，返回 `shared_ptr` 副本。
  - 回调随后对本线程的 `callback_mutex_` 执行 `try_lock`，拿不到就不记录本次事件。
  - `stop()` 先 `active_ = false` 并清除所有线程的 profile 函数，再逐个获取各线程的 `callback_mutex_`，等待进行中的回调结束。
  - `reportHashData`、`reportParamData` 在停止时遍历所有线程的缓存，用 `unordered_set` 去重后上报；`reportTraceData` 用 `report_mutex_` 串行化上传。
  - `start()` 在 FT 且 CPython ≥ 3.14 时调用 `_PyEval_StopTheWorld`，在暂停期间为每个线程创建 `ThreadLocalResult` 并记录其调用栈。
- **热路径开销。** 数据写入不再争用全局锁；但每次回调仍获取一次全局 `results_mutex_`（持有时间为一次哈希表查找），并对 `shared_ptr` 引用计数做两次原子操作（这个计数只被本线程的回调访问，没有跨线程争用）。与采用方案相比，全局锁的获取次数更少、持有时间更短，但没有消除跨线程争用。
- **内存。** 缓存按线程重复存放，同一个函数被 N 个线程调用就有 N 份条目，停止时去重。
- **行为变化。** 启动时记录所有线程的调用栈（我们的 FT 方案只记录当前线程）；`disableCallbacks` 在两种构建中都清除所有线程的 profile 函数。
- **复杂度。** 接近重写 `PythonTracer`。上游已经完成这一工作。
- **当时没有采用的原因。** 我们的修复基于 2.13.0（`d3216c2`），上游的重构不在这个基线中，当时也没有查到这个提交（见 [learning.md](learning.md) B4）；移植它等于替换整个文件。现在的建议见第 7.3 节。

### 4.6 候选 D：用 STW 私有 API 保护控制操作

- **做法。** `start/stop/clear` 在 `_PyEval_StopTheWorld` / `_PyEval_StartTheWorld` 之间执行。暂停期间其他线程都不在执行回调，控制操作可以直接读写所有共享状态，也可以遍历其他线程的调用栈。
- **热路径开销。** 只针对控制操作，回调之间对共享缓存的并发写入仍需另行保护（锁或每线程缓存）。所以它是候选 B 或 C 的补充手段，不能单独解决问题 3。
- **风险。** CPython 3.14 公开 C API 文档（`synchronization.rst`、`profiling.rst`）中没有这两个函数；上游用 `extern "C"` 自行声明。下划线开头的函数不在稳定接口承诺内，CPython 小版本之间签名或行为变化时编译能通过、运行可能出错。【已验证】3.14.3 中这两个声明在内部头文件 `internal/pycore_pystate.h`，标了 `PyAPI_FUNC`，`libpython3.14t.so` 导出这两个符号；3.15 是否保留没有核实，其他平台的符号导出情况也没有核实。
- **结论。** 放弃作为主要手段。采用的方案只用公开的 `PyEval_SetProfileAllThreads`（文档注明 3.12 起提供），由 CPython 内部处理其他线程的同步；代价是 FT 下无法遍历其他线程的调用栈。

### 4.7 候选 E：CPython critical section API

- **做法。** 用 `Py_BEGIN_CRITICAL_SECTION_MUTEX(&mutex)`（3.14 起）或以某个 Python 对象为锁的 `Py_BEGIN_CRITICAL_SECTION(op)`（3.13 起）包住共享容器的访问。GIL 构建中这些宏展开为普通代码块。
- **与需求的差距。** 按 `Doc/c-api/synchronization.rst`（3.14 分支），critical section 内调用的 C API 如果阻塞，运行时会临时挂起这个 critical section、释放锁，让其他线程进入，所以它"不提供 `PyMutex` 那样的互斥访问"。它适合锁内需要调用 C API 的情况。我们的容器操作不调用 C API，`PyMutex` 已能满足；如果把 CPython 调用放进锁内，又会遇到挂起期间容器状态被其他线程修改的问题。
- **结论。** 放弃。

### 4.8 两种设计逐项对照

行号：上游指 master `profiler_python.cpp`，我们指 `f36d74d` 的同一文件。

| 方面 | 我们的版本（`f36d74d`） | 上游版本（`ff5ad88d8a`） |
|---|---|---|
| 缓存和事件放在哪里 | `PythonTracer` 成员，所有线程共用：`py_call_cache_`、`pyc_call_cache_`、`module_info_cache_`、两个参数缓存、`events_`、`start_py_call_info_`、`ctx_tid_map_` | 每个线程一份 `ThreadLocalResult`，里面是同样的缓存和 `events_`、`start_py_call_info_`、`tid_`（上游 178–202 行）。`reportHashData` 在上报时按 hash 去重 |
| ctx | FT：`start()` 给所有线程装一个共享 ctx（`shared_ = true`），子线程 `startOne` 各建一个 ctx。GIL：每个线程一个 ctx。ctx 由 `ThreadLocalResult` 持有引用，里面有 `std::atomic<PythonTracer*> tracer_` | 两种构建都只有一个 `shared_ctx_`，由 `PythonTracer` 持有唯一的引用。`ThreadLocalResult::context_` 只是不持有引用的指针 |
| 回调如何找到本线程的数据 | 不按线程查找；直接写共享容器，用 `ctxKey(ctx)`（共享 ctx 用线程 id，其他 ctx 用地址）区分线程 | `findThreadLocalResult(PyThreadState_Get())`，在 `unordered_map<PyThreadState*, shared_ptr<ThreadLocalResult>>` 里查找并复制 `shared_ptr`（上游 458–464、721 行） |
| 锁 | `control_mutex_`（`start/stop/clear`）、`registration_mutex_`（线程登记、`session_`）、`data_mutex_`（缓存、事件）。都是 `PyMutex`，等锁时 detach | `control_mutex_`（五个入口都取）、`results_mutex_`（线程登记表）、`report_mutex_`（上传事件）、每个线程的 `callback_mutex_`。都是 `std::mutex`，等锁时不 detach；`control_mutex_` 在等锁前先释放 GIL（`lockControl`，上游 406–413 行） |
| 锁在哪种构建生效 | 只在 FT 构建；GIL 构建里是空函数 | `control_mutex_`、`report_mutex_`、`callback_mutex_` 两种构建都生效；`results_mutex_` 只在 FT 构建 |
| `start/stop/clear` 与其他线程的关系 | `control_mutex_` 让三者依次执行；`startOne/stopOne` 不取 `control_mutex_` | `start/stop/clear/startOne/stopOne` 全部取 `control_mutex_`，依次执行。`stop()` 在 `disableCallbacks()` 中先置 `active_ = false`、清除所有线程的回调，再释放 GIL，逐个获取每个线程的 `callback_mutex_` 并置 `enabled_ = false`（上游 446–456 行） |
| 子线程在 `stop()` 之后才完成登记 | `startOne` 不持有控制锁。它在 `registration_mutex_` 内登记并记下 `session_`，装好回调后再检查一次 `session_`；如果已变化就移除自己装的回调，最多重试一次 | `startOne` 整个函数体（包括 `PyEval_SetProfile`）都持有 `control_mutex_`，并在锁内检查 `active_`（上游 335–359 行）。因此它要么在 `stop()` 之前完成（之后被 `stop()` 清除），要么在 `stop()` 之后执行并直接返回 |
| 启动时记录已有调用栈 | FT：只记录调用 `start()` 的线程自己的栈。GIL：沿用原来的 `PyThreadState_Swap` 逐线程切换 | FT 3.14+：在 `_PyEval_StopTheWorld` 暂停期间枚举所有线程，用 `PyThreadState_GetFrame` 读每个线程的栈（上游 306–318 行）。GIL：同样枚举，靠 GIL 保证不并发。不再使用 `PyThreadState_Swap` |
| CPython 私有 API | 没有调用私有函数，用到的函数都是公开 API：`PyMutex`、`PyEval_SetProfileAllThreads`、`PyThread_get_thread_ident`。但直接读 `PyThreadState` 的 `c_profilefunc` 字段（第 518、592、628 行；上游和 `d3216c2` 原代码也读），第 560 行还读 `c_profileobj`（只有我们新加的代码读）。【已验证】这两个字段定义在 `Include/cpython/pystate.h` 的 `struct _ts` 中（3.14.3 头文件第 134、136 行，不在 `Py_BUILD_CORE` 保护内），C API 文档没有说明它们，结构体布局不属于稳定接口 | FT ≥ 3.14：`_PyEval_StopTheWorld`/`_PyEval_StartTheWorld`，源文件里自己写 `extern "C"` 声明（上游 60–62 行）。3.9–3.12：`_PyEval_SetProfile(thread_state, …)`（【已验证】3.11、3.12 的这个声明在公开头文件 `cpython/ceval.h`） |
| GIL 构建的行为变化 | 【推断】只有 `clear()` 改为获取 GIL；其余逻辑修改只在出现竞争时改变执行路径 | 【推断】`start()` 跟踪所有线程，不再有 `max_threads`（原来默认上限 256）。`stop()` 通过 `setProfileAllThreads(nullptr, nullptr)` 清除所有线程的 profile 函数，包括 cProfile 等其他工具设置的（原代码只清除自己装的）。每个事件多一次哈希表查找、一次 `shared_ptr` 复制和一次 `try_lock`。`start/stop/startOne/stopOne` 各多一次释放、重新获取 GIL |
| 热路径（FT） | `PyTrace_CALL`：至少两次获取全局 `data_mutex_`（`genPyCallHashId`、`recordPyCall` 各一次）；`nn.Module` 调用帧每次再多取一次（查 `module_info_cache_`，`f36d74d` 第 851–859 行），首次出现的模块再多取一到两次（插入缓存、写参数）。返回事件：一次。临界区内做缓存查找或插入、追加事件，事件数到阈值时在锁内上传 | 每个事件一次全局 `results_mutex_`（只做哈希表查找和 `shared_ptr` 复制），一次本线程 `callback_mutex_` 的 `try_lock`。缓存和事件写在本线程对象里，不取全局锁。上传到 `ProfilerMgr` 时取 `report_mutex_` |
| 3.13+ 读取 `self` | `PyDict_GetItemString`，3.13+ 上返回 NULL（第 4.2 节） | `PyMapping_GetItemString`，取不到时抛出 `KeyError`，由回调的 `catch (...)` 捕获并丢弃（上游 640–681、752 行） |
| ctx 释放时是否持有 GIL（#5658） | 是。`clear()` 获取 GIL 后析构 `ThreadLocalResult` | 是。`releaseContext()` 获取 GIL 后 `Py_XDECREF(shared_ctx_)`（上游 424–430 行） |
| 3.13t | 与 3.14t 走同一套代码，没有区分 | 只发出告警 `"The Python tracer is not thread-safe on Python 3.13t. Please use Python 3.14t or later."`，不做 STW |
| 同线程重入 `start/stop/clear` 或 `startOne/stopOne` | `control_mutex_` 检测到同线程重入时抛出异常；`startOne/stopOne` 不取 `control_mutex_`，可以从 `start()` 的帧遍历中调用 | 没有检测。`std::mutex` 被同一线程再次加锁属于未定义行为，glibc 上表现为死锁（第 5.3 节已复现） |

### 4.9 我们审阅中发现的竞态，上游是否处理

审阅过程中在我们自己的代码上发现六类竞态（经过见 [learning.md](learning.md) B1）。六类中有五类由上游的锁处理，"`recordReturn` 在等锁之后才读 ctx"在上游的结构中不存在。

| 竞态 | 上游的处理 | 依据 |
|---|---|---|
| 两个线程并发 `start/stop` | 已处理：五个入口第一步都是 `gil_scoped_acquire` 加 `lockControl()`（上游 281、337、363、380、392 行），`lockControl()` 先释放 GIL 再等锁（410 行），所以不会出现"A 持锁等 GIL、B 持 GIL 等锁"，FT 下等锁线程处于 detach 状态，不阻塞 STW | 【推断】压力测试覆盖了 `stop` 与 `startOne/stopOne` 并发，没有覆盖两个线程同时 `stop` |
| 子线程在 `stop()` 之后才 `startOne`，留下回调 | 已处理：`startOne` 在 `control_mutex_` 内检查 `active_`（338 行），装回调也在锁内（358 行），与 `stop()` 完全依次执行 | 【已验证】压力测试 18 次运行中 `stale_hooks_after_stop` 都是 0；去掉 `lockControl()` 后这项检查失败（第 5.5 节 u3） |
| `clear()` 释放 `ThreadLocalResult` 时回调仍在使用 | 已处理：回调持有一份 `shared_ptr`（721 行），`clear()` 只清空容器里的引用（396–398 行）；`stop()` 还通过每个线程的 `callback_mutex_` 等回调退出（451–455 行），回调拿到锁后再查 `enabled_` 和 `active_`（725–726 行） | 【推断】扩展场景中去掉 `callback_mutex_` 后 TSan 报竞争（第 5.5 节 u2），保留时 9 次运行没有报告 |
| `recordReturn` 在等锁之后才读 ctx | 结构上不存在：`recordReturn(ThreadLocalResult&, TraceTag)` 不使用 ctx，线程 id 存在 `tls.tid_`（555 行）。回调只在入口读一次 `ctx->tracer_`；在比较 `tls->context_` 之前唯一的等待是 `results_mutex_`，它是 `std::mutex`，等锁时不 detach，所以这段时间内 STW 无法完成，ctx 不会被释放 | 【推断】读代码；【已验证，压力测试】扩展场景中回调会中途 detach，上游 TSan、ASan 各 3 次运行都没有报告 |
| 线程 id 重用 | 已处理：线程数据按 tstate 地址登记（317、355 行），线程 id 在每次记录事件时写入本线程对象（555 行）。tstate 地址也可能被新线程重用，但 `PyEval_SetProfileAllThreads` 只作用于调用时已存在的线程，新建的 tstate 不继承 profile 函数，所以新线程在 `startOne` 之前没有回调；`startOne` 新建 `ThreadLocalResult` 并覆盖表中的旧项 | 【已验证，读 CPython v3.14.3 源码】`_PyEval_SetProfileAllThreads` 在 STW 期间遍历当前已有的 tstate 逐个设置（`Python/legacy_tracing.c` 576–594 行）；`Python/pystate.c` 中对 `c_profilefunc` 的写操作只有清零（1796–1805 行） |
| `recordStartStack` 中途抛出 Python 异常 | 已处理：`start()` 把整个 STW 和帧遍历包在 `try` 中，出错时清除回调、清空登记表、释放 ctx 再重新抛出（319–330 行），`StopTheWorldGuard` 在栈展开时先析构；`startOne` 的 `captureStartFrames` 在登记和安装回调之前执行（349 行先于 352–358 行），且不额外持有 ctx 引用；回调路径的异常由 `pyProfileFn` 的 `catch (...) { PyErr_Clear(); return 0; }`（752 行）捕获并丢弃 | 【推断】读代码。行为差异：上游 `start()` 中任何一个线程的帧解析失败都会让整次 `start()` 失败；我们的版本只遍历当前线程 |

**上游的一处残留问题**【推断】：`tls->context_ != ctx`（722 行）只比较指针，不解引用。如果旧 ctx 已释放、新一轮的 `shared_ctx_` 又恰好分配在同一地址，一个属于上一轮、执行较慢的回调会把一条事件写进新一轮的线程数据。后果是多出一条事件，不会访问已释放的内存。

### 4.10 上游独有的风险

1. **同线程重入导致死锁**【已验证，压力测试，见第 5.3 节】。`start()` 持有 `control_mutex_` 执行 `captureStartFrames → genPyCallHashId`，后者会构造 `PyCallInfo`、`ModuleInfo`，可能解析参数，这些都是 Python 对象操作。如果这些操作执行了用户 Python 代码（模块类上的 `__getattr__` 或 property、终结器），而用户代码又调用 `startOne/stopOne`，同一线程会再次获取 `std::mutex`。回调持有 `callback_mutex_` 时也有同样的问题：如果回调内的 Python 代码调用 `stop/startOne`，回调等 `control_mutex_`，而 `stop()` 持有 `control_mutex_` 等 `callback_mutex_`。后一种情况没有测试。实际代码中出现这种调用的可能性【推断】很低。
2. **STW 期间执行 Python 对象操作**【推断，未验证】。`start()` 在 `_PyEval_StopTheWorld` 与 `_PyEval_StartTheWorld` 之间调用 `captureStartFrames`，后者会取模块类名、读 `_parameters`、调用 `param_groups`、`state.get(...)` 并创建 `py::dict()`。
   - 【已验证，读 CPython v3.14.3 源码，未运行】`Python/pystate.c` 的 `stop_the_world()` 先获取 `stw->mutex`（`PyMutex_Lock(&stw->mutex)`，2372 行），`PyMutex` 不可重入，所以同一线程在 STW 期间再次请求 STW 会阻塞在这里。FT 构建的垃圾回收会调用 `_PyEval_StopTheWorld`（`Python/gc_free_threading.c` 2185 行等），并通过 `_Py_ScheduleGC` 推迟到线程下次处理待办事件时执行（2152 行）。
   - 【推断】因此，只要 STW 期间的属性访问执行了 Python 字节码（例如用户在模块类上定义的 property 或 `__getattr__`），并且这时有待执行的垃圾回收，就会嵌套请求 STW，导致死锁。实际代码中这种情况是否会出现没有验证。
   - 【推断】CPython 自己的 `sys._current_frames()` 也在 STW 期间读其他线程的帧对象（没有读这部分源码），所以"STW 期间读帧"有先例；"STW 期间执行可能运行字节码的属性访问"没有找到先例。
3. **依赖 CPython 私有 API**，见第 4.6 节。
4. **FT 构建每个事件仍取一次全局锁**：`findThreadLocalResult` 中的 `results_mutex_` 所有线程共用。临界区短，但没有消除跨线程争用。
5. **GIL 构建行为变化**：见第 4.8 节表格。其中"`stop()` 清除其他工具设置的 profile 函数"在 GIL 构建里也会发生，原代码没有这个行为。
6. **3.13t 只告警、不做 STW**，仍存在读其他线程栈的竞态。

### 4.11 两个方案的风险对照

| 方案 | 风险 | 依据 |
|---|---|---|
| 上游 | 同线程重入时死锁 | 已验证（压力测试，第 5.3 节） |
| 上游 | STW 期间执行可能运行字节码的属性访问；如果触发嵌套 STW，可能死锁 | 推断，未验证 |
| 上游 | 依赖 `_PyEval_StopTheWorld`，CPython 版本升级时可能失效 | 3.14.3 中已导出（已验证）；以后版本未核实 |
| 上游 | FT 构建每个事件取一次全局 `results_mutex_` | 读代码；stub 吞吐见第 5.4 节 |
| 上游 | GIL 构建行为变化：`stop()` 清除其他工具的 profile 函数，取消 `max_threads`，热路径多一次查表 | 读代码，GIL 构建没有运行 |
| 上游 | 3.13t 只告警，不做 STW，仍存在读其他线程栈的竞态 | 读代码 |
| 上游 | 等 `results_mutex_` 时不 detach，以后若在持有它时引入 Python 调用或 STW，会死锁 | 推断 |
| 我们 | 缺少 3.13+ 的 `PyMapping_GetItemString` 修改，3.14 上进入 `nn.Module` 分支时对 NULL 取属性 | C API 行为已验证；torch_npu 实际崩溃未验证 |
| 我们 | 直接读 `PyThreadState` 的 `c_profilefunc`、`c_profileobj`，这两个字段不属于稳定接口，CPython 小版本之间结构体布局可能变化（上游和 `d3216c2` 只读 `c_profilefunc`；`c_profileobj` 只有我们新加的代码读） | 读代码和 3.14.3 头文件（第 4.8 节） |
| 我们 | FT 构建每次 Python 调用至少取两次全局 `data_mutex_`，线程多时变慢 | stub 吞吐见第 5.4 节和 [fix_details.md](fix_details.md) 第 5.2 节 |
| 我们 | `session_` 和重试逻辑较复杂；一次 `startOne` 期间经历两轮 stop/start 时漏采该线程 | 代码注释已写明 |
| 我们 | 与上游 master 结构不同，无法直接合入，后续同步要手工合并 | 已验证（上游已重写该文件） |
| 我们 | 锁只在 FT 构建生效，GIL 构建依赖 GIL；GIL 构建只编译、没有运行 | [fix_details.md](fix_details.md) 第 4.5 节 |
| 两者 | FT 构建的 `stop()` 清除所有线程的 profile 函数 | 读代码 |
| 两者 | stub 与真实 CPython 3.14 不一致，压力测试通过不代表真实环境下没有问题；都没有在 NPU 上运行 | [fix_details.md](fix_details.md) 第 4.5 节 |

## 5. 用本仓库的压力测试运行上游版本

### 5.1 做法

- 工作目录：`torch_npu-work/upstream-harness/`，是测试代码的一份副本，仓库里的测试代码没有修改。
- 同一份测试代码和 stub 分别编译两个源文件：`src-upstream/`（上游 master 的 `profiler_python.cpp`、`profiler_python.h`、`python_tracer_types.h`、`profiler_callbacks.h`、`containers.h`，原样使用）和 `src-ours/`（`f36d74d` 的对应文件）。
- 场景与仓库 `test/cpp/profiler_thread_safety/python_tracer/python_tracer_test.cpp` 相同：worker 线程循环执行回调并调用 `kStartOne/kStopOne`；主线程在已释放 GIL 的状态下循环执行 `kStartAll/kStop/kClear`；stub 的 `PyEval_SetProfile` 会 detach 最多 2 ms，模拟 3.14 在这里做 STW（【已验证，读源码】3.14.3 的 `_PyEval_SetProfile` 在 `Python/legacy_tracing.c` 551–554 行调用 `_PyEval_StopTheWorld`）；每次 `kStop` 之后、进行中的 `kStartOne` 全部返回时，检查没有线程仍装着回调。
- 三种模式：
  - **含重入**（原场景）：主线程在 `kStartAll` 的帧遍历中调用 `kStartOne/kStopOne`，模拟用户 Python 代码在 tracer 内部再次调用 tracer。
  - **不含重入**：环境变量 `PYTRACER_NO_REENTRY=1` 关闭上面这一步，其余相同。
  - **扩展**：在"不含重入"的基础上打开 `PYTRACER_HOOK_DETACH=1`、`PYTRACER_LATE_THREADS=1`。前者让 worker 在回调中途（`genPyCallHashId` 读行号时）短暂 detach，模拟回调里的 Python 代码释放 GIL，或让 STW 在回调中途完成；后者让一个线程在运行期间不断创建短生命周期线程，每个新线程登记 tstate、调用 `kStartOne`、执行回调，部分调用 `kStopOne` 后退出，退出时在 STW 下清除并注销 tstate。glibc 会复用线程局部存储，所以这些线程的 tstate 地址会重复出现，同时覆盖了"地址重用"的情况。这两项是在第 5.5 节的变异检查发现原场景测不到上游两把锁之后补充的。
- 运行脚本 `run_compare.py`，判定规则与仓库 `run.py` 相同（退出码非 0、超时、没有结果行、检查失败、出现 sanitizer 报告都算失败）。参数：TSan、ASan 为 6 个 worker、100 轮；plain 构建为 8 个 worker、100 轮。单次超时：原场景 60 秒，扩展场景 90 秒。全部通过 `systemd-run --user --wait --collect` 运行。

为编译上游版本补充的 stub（都在副本里，注释以 `upstream-harness:` 开头）：定义 `PY_MAJOR_VERSION`、`PY_MINOR_VERSION`、`PY_VERSION_HEX` 为 3.14；在测试文件中实现 `_PyEval_StopTheWorld`/`_PyEval_StartTheWorld`（调用线程保持 attach，以独占方式获取模拟用的读写锁，其他线程在下次 attach 时暂停）；给 `PyThreadState` 加 `stub_frame`，让 `PyThreadState_GetFrame` 返回它、`PyInterpreterState_ThreadHead/Next` 遍历所有已登记的 tstate；`pybind11::gil_scoped_release`、`error_already_set`；让 `PyFrame_GetLocals` 返回非空、`PyDict_GetItemString` 返回 NULL、`PyMapping_GetItemString` 返回非空（与 3.14 实际行为一致，见第 4.2 节）；`Py_IsFinalizing`、`PyErr_Clear/Occurred/SetString`、`PyExc_KeyError`、`TORCH_WARN`；以及上游拆分依赖后新引用的几个头文件。

### 5.2 公平性说明

- stub 用 `std::mutex` 模拟 `PyMutex`，用读写锁模拟 attach 和 STW。它不模拟 GIL，所以 GIL 构建两边都没有运行。
- stub 的 Python 对象操作不执行任何用户代码，所以第 4.10 节第 2 条（STW 期间执行字节码）测不到。
- 扩展模式第一次运行时上游出现 TSan 报告：主线程在 STW 下读取新线程的帧时，新线程正在写入。排查发现是测试本身的问题：新线程在 attach 之前就写了帧。真实 CPython 中线程只在 attach 状态下修改自己的帧，所以 STW 会把这类写操作排除在外。把帧的初始化移到 attach 之后，报告消失。这次运行的日志已删除，没有计入结果。
- 上游 `start()` 会遍历 worker 线程的栈，我们的版本在 FT 构建中只遍历主线程，所以两边在同一场景下执行的路径不同。这是两种设计本身的差别。
- 两个版本用的是同一份扩展后的 stub，补充内容对两边相同。

### 5.3 结果（2026-11-02，本机）

表中为"失败次数 / 运行次数"。每次运行 100 轮 start/stop/clear。

| 模式 | 构建 | 上游版本 | 我们的版本 |
|---|---|---|---|
| 不含重入 | TSan | 0 / 3 | 0 / 3 |
| 不含重入 | ASan | 0 / 3 | 0 / 3 |
| 不含重入 | plain | 0 / 3 | 0 / 3 |
| 含重入 | TSan | 3 / 3（均超时） | 0 / 3 |
| 含重入 | ASan | 3 / 3（均超时） | 0 / 3 |
| 含重入 | plain | 3 / 3（均超时） | 0 / 3 |
| 扩展 | TSan | 0 / 3 | 0 / 3 |
| 扩展 | ASan | 0 / 3 | 0 / 3 |
| 扩展 | plain | 0 / 3 | 0 / 3 |

- 所有通过的运行中，`stale_hooks_after_stop`（`stop()` 之后仍装着回调的线程数）都是 0。
- 扩展模式中每次运行创建的短生命周期线程数：TSan 140–248 个，ASan 462–611 个，plain 构建 506–853 个。
- **含重入场景的挂起位置。** plain 构建下另编了一个诊断版本（`python_tracer_test_watchdog.cpp`，在副本中），10 秒没有进展就打印主线程所在位置并退出。上游 3 次都在第 0 轮挂起，输出相同：

  ```
  {"scenario": "pytracer-watchdog", "passed": false, "hang": true, "cycles_done": 0, "reentries": 1, "main_thread_in": "re-entrant kStartOne inside kStartAll"}
  ```

  也就是说，主线程在 `kStartAll` 持有 `control_mutex_` 时进入 `kStartOne`，再次获取同一把 `std::mutex`。我们的版本 3 次都通过（`reentries` 为 200）。TSan 构建在超时前没有输出死锁报告，日志里 stderr 为空。本机没有 gdb，没有抓线程栈。
- 日志：`upstream-harness/logs/full/`（不含重入、含重入，汇总在 `summary.json`；这组是在加入扩展模式之前编译运行的，扩展模式的代码默认关闭，不影响这两种模式）、`logs/full-ext/`（扩展）、`logs/watchdog-reentry.txt`（挂起位置）。

### 5.4 回调吞吐（stub 环境）

- 做法：`python_tracer_test bench <线程数> 1000`。worker 线程在 tracer 运行期间循环调用回调 1 秒，统计所有线程每秒合计的回调次数（百万次/秒）。每个点运行 3 次，表中给出 3 次的范围。plain 构建，2026-11-02，本机；运行时的系统负载没有记录。
- 帧类型分两种："模块帧"是 stub 默认情况，每一帧都被当作 `nn.Module.__call__` 帧，我们的版本因此每个调用事件多获取一次 `data_mutex_`（见第 4.8 节"热路径（FT）"一行）；"普通帧"（`PYTRACER_BENCH_PLAIN_FRAMES=1`）没有帧进入模块分支，【推断】真实场景中大部分帧属于这一类。

| 线程数 | 普通帧：上游 | 普通帧：我们 | 模块帧：上游 | 模块帧：我们 |
|---|---|---|---|---|
| 1 | 4.6–4.8 | 5.5–7.2 | 3.8 | 4.2–4.3 |
| 2 | 4.3–5.8 | 2.3–3.5 | 3.2–6.1 | 2.9–3.3 |
| 4 | 3.5–4.8 | 2.0–2.2 | 3.1–4.6 | 1.9–2.0 |
| 6 | 3.0–3.2 | 1.3–1.4 | 3.1–4.0 | 1.3–1.5 |
| 8 | 2.4–3.2 | 1.2–1.6 | 3.1–3.6 | 1.0–1.2 |

- 单线程时我们的版本更快，原因【推断】是上游每个事件多一次哈希表查找、一次 `shared_ptr` 复制和一次 `try_lock`。
- 普通帧从 2 个线程起、模块帧从 4 个线程起，上游更快，两边范围不重叠；模块帧 2 个线程时范围重叠（上游 3.198–6.137，我们 2.927–3.278）。6–8 个线程时，按两边范围的端点计算，上游的合计吞吐是我们的 1.57–3.51 倍（按原始记录计算：最小值为 8 线程普通帧 2.441/1.550，最大值为 8 线程模块帧 3.631/1.036）。
- 两个版本的合计吞吐都没有随线程数增长。【推断】上游的瓶颈是每个事件都要获取的全局 `results_mutex_`，我们的是 `data_mutex_`。
- 限制：stub 用 `std::mutex` 模拟 `PyMutex`，不含真实 CPython 挂起、恢复线程的开销；stub 中计算 hash 用 `ostringstream`，开销与真实实现不同；测试机是共享服务器，同一配置 3 次运行之间最多相差约一倍（例如上游模块帧 2 线程 3.2–6.1）。**这些数字只能用来比较两个版本的相对趋势，不能代表真实环境下的开销。** 这组与 [fix_details.md](fix_details.md) 第 5.2 节那组是不同的 stub、不同的运行，不能互相换算。
- 日志：`upstream-harness/logs/bench-20261102-042250.jsonl`，脚本 `run_bench.sh`。

### 5.5 用变异检查测试的检出能力

对上游 `profiler_python.cpp` 各做一处修改，检查测试能否发现。**本节的编号 u1–u3 只用于上游代码**，与 [fix_details.md](fix_details.md) 第 4.5 节、[learning.md](learning.md) A4.3 中针对我们自己代码的 m1–m11 不是同一套编号。变异文件在 `upstream-harness/mutants/`（文件名 `m1.diff`、`m2.diff`、`m3.diff` 沿用旧编号，依次对应 u1、u2、u3）。

| 变异 | 修改内容 | 原场景（不含重入）：TSan / plain | 扩展模式：TSan / ASan |
|---|---|---|---|
| u1 | `findThreadLocalResult` 不获取 `results_mutex_` | 0 / 3，0 / 3 | 3 / 3，0 / 3 |
| u2 | 回调不获取 `callback_mutex_`，同时去掉 `enabled_` 检查 | 0 / 3，0 / 3 | 3 / 3，0 / 3 |
| u3 | `startOne` 不获取 `control_mutex_` | 3 / 3，3 / 3 | 没有运行 |

表中为"失败次数 / 运行次数"，失败即测试发现了变异。

- **u2 的修改范围。** `m2.diff` 把 `std::unique_lock<std::mutex> callback(tls->callback_mutex_, std::try_to_lock); if (!callback.owns_lock() || !tls->enabled_ || !tracer->active_.load())` 整段换成 `if (!tracer->active_.load())`，即同时去掉 `try_lock` 和 `tls->enabled_` 检查。`enabled_` 由 `disableCallbacks` 在持有该线程的 `callback_mutex_` 时写入，只有在持锁时检查才有意义，所以两者必须一起去掉。下面对 u2 的 TSan 报告的解释也依赖这两处一起去掉。
- **原场景测不到 u1、u2。** u1：所有 worker 都在 `start()` 之前创建，`startOne` 只会覆盖哈希表中已有的键，哈希表结构从不在回调读取时改变。u2：worker 的回调从不在中途 detach，STW 本身就足以保证 `stop()` 读取事件时没有回调在写。这是扩展模式加入那两项内容的原因。
- **扩展模式下的报告内容。** u1：TSan 报告 `unordered_map::find`（回调中）与 `operator[]` 插入（新线程的 `startOne` 中）之间的数据竞争。u2：TSan 报告 `stop()` 读取事件列表与回调追加事件（`AppendOnlyList::emplace_back`）之间的数据竞争。ASan 两项都没有报告。
- **u3** 在原场景中就会失败：TSan 构建报数据竞争，`stale_hooks_after_stop` 为 11–23；plain 构建为 78–104（`upstream-harness/logs/mutants/summary.json`）。
- 日志：`upstream-harness/logs/mutants/`（原场景）、`logs/mutants-ext/`（扩展模式）。
- 结论：扩展模式能发现上游三把锁中任何一把被去掉；但只有 TSan 构建能发现 u1、u2，所以做回归测试时必须包含 TSan 构建。

## 6. 测量方法和局限

本节适用于第 2、3 节的全部数字。

### 6.1 环境

- 机器：Intel Xeon Platinum 8383C @ 2.70GHz，2 路，每路 40 核，每核 2 个硬件线程，`nproc` = 160，2 个 NUMA 节点。实验室共享服务器，测量期间 1 分钟平均负载在 14.8 到 21.3 之间（`uptime`）。
- 编译：g++ 11.4，`-O2 -g`，与 `run.py` 的 plain 构建相同。
- 每个可执行文件单独在 `systemd-run --user --wait --collect` 中运行；没有绑定 CPU 核。
- 所有变体都是复制出来的源码（`torch_npu-work/alt-bench/trees/<变体>/`），仓库文件没有修改。测试驱动直接使用仓库中的 `profiler_thread_safety_test.cpp` 和 `stubs/`。

### 6.2 两个基准

- **`Report` 基准**：仓库测试程序自带的 `bench` 模式。N 个线程对一个已启动的 `DataDumper` 循环调用 `Report`，每次调用先 `make_unique` 一个记录对象；后台线程同时取出、编码、写文件。容量 2^20（与 `ProfilerMgr` 的 `capacity_` 相同），每次 1 秒。输出两个数：`Report` 调用速率（包含缓冲区写满后 `Push` 直接失败返回的调用）和接受速率（成功放入缓冲区、最终写进文件的记录数除以时间）。
- **队列基准**：`alt-bench/scripts/qbench.cpp`。N 个线程向 `RingBuffer<uint64_t>` 写入 8 字节元素，失败不重试（与 `Report` 相同）；一个读取方连续取出并检查每个写入方的序号是否递增。不分配内存，不写文件，只测队列本身。容量 2^20 和 1024，每次 1 秒。
- 写入线程数 1、8、32，每种组合重复 5 次。每轮重复内各变体轮流运行，使负载变化同时影响所有变体。
- 比较时一律使用中位数之比（倍数），不使用百分比。

### 6.3 变体是不是有效修复

在用速率比较之前，先确认每个变体本身修复了问题：用仓库的 `run.py --source-root <变体目录> --builds tsan,plain --repeats 2` 运行压力测试，每个变体 16 次场景运行（`rb`、`dumper`、`trace` 各两种容量、TSan 和 plain 两种构建；`pytracer` 场景因变体目录缺少 `containers.h` 没有编译，与这些变体无关）。

| 变体 | 失败 / 运行 | 失败表现 |
|---|---|---|
| 互斥锁 | 0 / 16 | 无 |
| 读写锁 | 0 / 16 | 无 |
| mutex+deque | 0 / 16 | 无 |
| 参照（去掉 `writers_`） | 7 / 16 | TSan 数据竞争报告；`dumper`、`trace` 场景检查失败 |
| 原 ring buffer（参照） | 12 / 16 | TSan 报告；`rb` 场景丢数据或错序 |

日志：`alt-bench/stress/<变体>/`。后两个变体不是有效修复，只作参照，它们的速率数字不能当作可选方案的开销。

### 6.4 局限

- **共享服务器噪声大。** 同一变体 5 次之间常差 1.5 倍以上（例如参照变体单线程 2.57 到 5.50）。只有范围不重叠的差别才在本文中作为结论使用。
- **stub 环境。** 记录对象、编码和写文件都是测试代码中的简化实现；真实 `ProfilerMgr` 的记录更大、编码更慢，缓冲区写满的条件不同。
- **`Report` 基准中内存分配占比高。** 每次调用一次 `make_unique`，读取方释放；多线程下 glibc malloc 本身的开销混在结果里。
- **写满后的调用更便宜。** 8、32 线程时缓冲区很快写满，此后 `Push` 直接返回 false，所以 `Report` 调用速率是"完整写入"和"失败返回"两种路径的混合。
- **读取方调度。** 32 个写入线程空转时，读取方线程得到的 CPU 时间不稳定，采用方案在容量 1024 下 5 次结果从 0.38 到 2.66，呈两组分布。
- **没有在 aarch64 上测量。** 问题 1 中 `seq_cst` 的必要性、各方案在弱内存序平台上的表现都没有实测。
- 原 ring buffer 在一次 8 线程 `Report` 基准中没有输出（进程异常退出，退出码没有记录）。
- 问题 3 的候选方案之间没有做性能对比（第 5 节只对照上游方案与采用方案）。
- 问题 1 的候选 C、D、F 和问题 2 的候选 B、C、E、F 没有实现，开销分析来自阅读代码和头文件。
- 下游解析是否依赖跨线程的记录顺序没有核实，这影响问题 1 候选 F、问题 2 候选 E 和 moodycamel 的可行性。

### 6.5 复现

**这一节列出的脚本都在作者本机，没有进仓库，所以第 2、3、5 节的全部数字他人无法复现。** 第 7.3 节第 4 条建议把这套测试整理进仓库，本次交付没有做。

- 变体基准：`torch_npu-work/alt-bench/scripts/` 下按顺序运行 `make_variants.py`（生成互斥锁、读写锁、参照三个变体）、`build.sh`、`bench.sh`、`summarize.py`；正确性检查用仓库中的 `run.py --source-root`，这一个脚本在仓库里。
- 上游对照：`torch_npu-work/upstream-harness/run_compare.py`、`run_bench.sh`、`stubs/`、`src-upstream/`、`src-ours/`、`mutants/`。
- 第 4.2 节的 CPython 行为核对：`torch_npu-work/py314-checks/pep667_check.py`、`null_getattr.py`。
- 第 7.1 节 `-X faulthandler` 和 `PYTHONMALLOC=debug` 在 3.14.3、3.14.3t 上的输出：`torch_npu-work/py314-checks/crash_diag_check.sh`，输出 `crash_diag_check.out`。
- 以上都要放进 `systemd-run --user --wait --collect` 运行。

## 7. 给接手工程师的建议

### 7.1 第一步：向使用方确认配置和崩溃时机，确认崩溃出在哪一处，补上 `b2e659d62f`

按下面的顺序做。

1. **向使用方要 profiler 配置和崩溃时机**：`with_stack`、`with_modules`、`activities` 是否含 CPU（`tracePython()` 要求 `(with_stack || with_modules)` 且采集 CPU activity，`npu_profiler.cpp` 第 95–98 行）、`profile_memory`、`record_shapes`，是否多线程，崩溃发生在开始采集、前向中，还是停止采集时（`prof.stop()` 或退出 `with` 块）。按第 1.5 节的对应关系，这一步可以排除一部分候选。

2. **在使用方原来的构建上运行单线程最小脚本，检查候选 1。** 原来的构建指 `d3216c2` 或使用方实际使用的 wheel，不含本分支的修改，也不含 `b2e659d62f`。脚本用 3.14t，单线程，`with_stack=True`，采集 CPU activity，包含 `nn.Module` 前向，开启 `PYTHONFAULTHANDLER=1`。候选 1 成立时，进程在第一次前向时崩溃，到不了停止采集，所以这一步看不到候选 2。

3. **在原来的构建上只补 `b2e659d62f` 中 `PyMapping_GetItemString` 和判空的部分**（需要从源码编译），开启 `PYTHONMALLOC=debug`，重新运行并包含停止采集，检查候选 2。

4. **在本分支上补同一处修改，编译本分支，运行多线程用例**，检查候选 3 和本分支的修复。本分支同样缺少 `b2e659d62f`，不补的话这一步会先遇到候选 1。

   修改做法：3.13 以下保留 `PyDict_GetItemString`，3.13 及以上用 `PyMapping_GetItemString`，并用 `THPObjectPtr` 持有返回的新引用（`PyDict_GetItemString` 返回借用引用，只替换函数名会造成引用泄漏）；取不到时判空跳过（上游 `ff5ad88d8a` 的做法是取不到时抛 `KeyError`，由回调的 `catch (...)` 处理）。**不要整个 cherry-pick `b2e659d62f`**：它还在 `~ThreadLocalResult` 里用 `#if ... PY_MINOR_VERSION < 12` 跳过 `Py_DECREF(ctx_)`（提交中的注释原文是 "purposefully allowing for a small leak"），这一部分不要移植，因为本分支的 `clear()` 已改为持有 GIL 释放 ctx；它也没有对取到的 `self` 判空。按第 4.2 节已验证的 C API 行为，3.13 及以上 `PyDict_GetItemString` 必定返回 NULL，所以本分支上的这处修改不取决于第 2、3 步的结果，作为一个独立提交先做。

诊断手段：

- **`PYTHONFAULTHANDLER=1`（或 `python -X faulthandler`）**：SIGSEGV 时打印 Python 层调用栈，不需要 gdb，用来看崩溃发生在前向中还是停止采集时。【已验证，本机 3.14.3、3.14.3t，`py314-checks/crash_diag_check.out`，该输出经 `head -12` 截断】GIL 构建打印所有线程的 Python 调用栈；FT 构建只打印崩溃线程的，输出中有一行 `<Cannot show all threads while the GIL is disabled>`。
- **`PYTHONMALLOC=debug`**：安装 CPython 内存分配器的调试钩子。【已验证，同上】在没有 attach 线程状态的情况下调用 `PyObject_Free`，FT 构建立即以 `Fatal Python error: _PyMem_DebugFree: Python memory allocator called without an active thread state` 终止，GIL 构建的提示是 `called without holding the GIL`；不设这个环境变量时同一调用不报错。【推断】据此，候选 2（不持有 GIL 时经 `Py_DECREF` 释放对象）在开启它之后会在释放的位置终止，这条释放路径本身没有运行验证；这与 issue #5658"开启 `PYTHONMALLOC=debug` 时必定出现 coredump"一致。它只覆盖经过 CPython 分配器的释放；候选 3 涉及的 C++ 对象经 glibc malloc 分配，要用 ASan 构建检查。
- **gdb `bt`**：取得完整的 C++ 调用栈。读法如下：
  - 栈顶在 `PyObject_GetAttrString`、往上是 `ModuleInfo::ModuleInfo`，对应候选 (1)（缺少 `b2e659d62f`）；
  - 栈顶在 `_Py_Dealloc`、对象的 dealloc 函数或分配器中，往上是 `PythonTracer::clear` 或 `~ThreadLocalResult`，或者开启 `PYTHONMALLOC=debug` 后在这些位置报 `_PyMem_DebugFree` 错误，对应候选 (2)（issue #5658）；
  - 栈顶在上报链路（`DataDumper`、`RingBuffer`）或 profile 回调中，对应候选 (3)（三处竞态之一）；
  - 栈顶在 pymalloc、mimalloc、`free` 中，或者在与 profiler 无关的代码中，通常是堆损坏【推断】，崩溃位置通常晚于出错位置。开启 `PYTHONMALLOC=debug` 重新运行可以检查候选 2；候选 3 要用 ASan 构建检查。

### 7.2 问题 1、2：移植现有修改

`ring_buffer.h`、`data_dumper.h/.cpp` 的修改可以移植到上游 master：上游这两处的逻辑与 2.13.0 相同，只做过 clang-format，所以补丁不能直接 `git apply`，需要手工合入；没有在 master 上编译过。采用的方案和放弃的方案见第 2、3 节。如果上游维护者不接受手写内存序，最接近的替代是读写锁（第 2.2 节）。本机测量不能区分它与采用方案的单次调用开销（第 2.7 节）。

### 7.3 问题 3：建议以上游 `ff5ad88d8a` 为基础

理由：

1. **维护成本。** 上游已经合入了自己的重构，我们的补丁改的是重构之前的结构，与上游 master 冲突，不能直接合入。继续维护两套写法，以后每次同步上游都要重新合并。
2. **结构更简单。** 一把控制锁覆盖全部五个入口，`startOne` 与 `stop()` 完全依次执行，不需要 `session_` 计数和重试，也没有"两轮 stop/start 期间漏采一个线程"的限制。回调写的是本线程的对象，生命周期由 `shared_ptr` 和 `callback_mutex_` 保证，比"回调只能在第一次等锁前读 ctx"这类约束更容易检查。
3. **争用更少。** 缓存和事件不经过全局锁，全局锁只剩一次短临界区。stub 环境下，多线程时上游的回调吞吐更高（模块帧 2 个线程时范围重叠，第 5.4 节）。如果要进一步减少争用，可以把 `findThreadLocalResult` 改为线程局部缓存（按会话编号校验），这一步上游没有做，本文也没有实现。
4. **已包含 3.13+ 的 `PyMapping_GetItemString` 修改和 #5658 的修改。**

合入 2.13.0 或 r22 分支时建议同时做的修改：

1. **把参数解析移出 STW。** STW 期间只读取每个线程的帧对象和 `PyCallInfo` 所需的行号、文件名、函数名，保存帧的强引用；`_PyEval_StartTheWorld` 之后再对这些帧调用 `ModuleInfo`、`parseModuleParams`、`parseOptimizerParams`。这样可以去掉第 4.10 节第 2 条风险，代价是启动时多一次遍历。
2. **同线程重入时抛出异常。** 在 `lockControl()` 中记录持锁线程，同一线程重入时抛出异常，与我们版本的 `TracerControlMutex` 做法相同。也可以把 `start()` 中的 `captureStartFrames` 移到不持有 `control_mutex_` 的位置，但这样会重新引入登记与 `stop()` 的竞争，不建议。
3. **在 GIL 构建上确认两项行为变化是否可以接受**：`stop()` 清除其他工具的 profile 函数、不再限制线程数。如果不能接受，GIL 构建可以保留原来"只清除自己装的回调"的写法。
4. **测试。** 把 `upstream-harness` 整理进仓库，作为上游方案的回归测试。要包含扩展场景和 TSan 构建，原场景测不到 `results_mutex_`、`callback_mutex_` 被去掉（第 5.5 节）。重入场景在做了第 2 条之后，预期以异常结束。

如果不采用上游方案、继续使用我们的版本，至少要做第 7.1 节的各步，并接受第 4.11 节列出的风险。

## 8. 未解决的问题

- **使用方的崩溃到底出在哪一处，没有确定。** 候选有三个（见第 1.5 节）：第 4.2 节的空指针访问（缺少 `b2e659d62f`）；issue #5658 所述路径，原 `clear()` 在没有持有 GIL 的状态下 `Py_DECREF` `TraceContext`（本分支已改）；三个竞态中的某一处。前两个在单线程、有 GIL 时同样成立。在不知道使用方配置和崩溃时机的情况下三者无法排序，知道之后可以排除一部分（第 1.5 节）。确定根因的步骤和手段见第 7.1 节。
- 没有用真实的 torch、CPython 3.14t 头文件编译任何一个版本；没有在 NPU 上运行。
- 第 4.2 节的崩溃路径只验证了 CPython C API 的行为，没有运行 torch_npu。
- 上游 PR 的"功能验证"截图没有查看；上游是否有针对 FT 的测试没有核实。
- STW 嵌套会阻塞这一点只读了 CPython 源码，没有运行；上游 STW 期间的属性访问在实际模型上是否会执行字节码，没有验证。
- 上游 `init.cpp` 中绑定的 GIL 处理只依据 `6ca3a36a15` 的 PR 描述（原文一条：把 `warmup/start/stop/finalize` 的 GIL 释放操作移到 Python 层 `profiler/init.cpp` 的绑定处，用 `py::call_guard<py::gil_scoped_release>()` 管理）。按这条描述，上游 master 上 tracer 的 `start/stop/clear` 仍在已释放 GIL 的线程上被调用，和 2.13.0 相同；`init.cpp` 本身没有查看。
- 问题 3 的候选方案之间没有做性能对比。
- [fix_details.md](fix_details.md) 第 6 节列出的未处理问题（host regions 的 worker 线程采集不到、`ProfilerMgr` 的其他竞态、GIL 构建的 `record_params_`、fork 子进程挂起等）在本文中没有进一步分析。

## 附：引用的文件位置

都在作者本机 `/SSD1/wuyin/project/torch_npu-work/` 下，没有上传 GitHub；接手的工程师既无法打开日志，也无法运行其中的脚本。

| 内容 | 路径 |
|---|---|
| 候选方案的变体源码、脚本、压力测试日志、基准原始记录 | `alt-bench/`（`trees/`、`scripts/`、`stress/`、`results/20261102T120205Z/`、`upstream/`） |
| 上游源文件（master）、提交信息、issue 原文、`b2e659d62f` 补丁 | `upstream-src/` |
| CPython 3.14.3 源码片段 | `upstream-src/cpython-3.14.3/Python/`（`pystate.c`、`legacy_tracing.c`、`ceval.c`、`gc_free_threading.c`） |
| CPython 3.14 行为核对（第 4.2 节）；`faulthandler`、`PYTHONMALLOC=debug` 输出核对（第 7.1 节） | `py314-checks/` |
| 上游对照的测试副本、脚本、日志、变异 | `upstream-harness/`（`run_compare.py`、`run_bench.sh`、`stubs/`、`src-upstream/`、`src-ours/`、`mutants/`、`logs/`） |
| CPython 3.14.3 和 3.14.3t 解释器 | `py314/` |
