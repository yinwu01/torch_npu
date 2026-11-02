# torch_npu profiler 在 free-threaded Python 下的线程安全修复

- 代码：`yinwu01/torch_npu`，分支 `fix/profiler-thread-safety`。
- 基于 `113xiaoji/torch_npu` 的 `codex/r22-real-four-worker-20261027`（`d3216c2`，TorchNPU 2.13.0）。
- 修复提交：`663e8ba`（主体）、`f36d74d`（异常路径的引用泄漏）。
- 本文写给要在原仓库解决这个问题的工程师。读完应能知道：问题在哪、我们怎么改、为什么这样改、验证到了什么程度、接手后要做什么。
- 本文只讲这三个问题和对应修改。候选方案对比、与上游的逐项对照见 [analysis.md](analysis.md)；背景知识和过程经验见 [learning.md](learning.md)；进度和待办见 [progress.md](progress.md)。
- **先看两件和线程无关的事。** 使用方崩溃的根因没有确定，三个候选中有两个在单线程、有 GIL 时同样成立：
  - 本分支的基线缺少上游 `b2e659d62f`，在 CPython 3.13 及以上读取 `nn.Module` 帧的 `self` 会得到空指针，随后对空指针取属性（详见 [analysis.md](analysis.md) 第 4.2 节）；
  - 上游 issue #5658 所述路径：原 `clear()` 在没有持有 GIL 的状态下 `Py_DECREF` `TraceContext`，调用方 `stopNpuProfiler()` 先执行了 `gil_scoped_release`。本分支的 `clear()` 已改为先取 GIL 再释放（见第 3 节问题 3）。
  - 第三个候选是下面三处竞态中的某一处。在不知道使用方配置和崩溃时机的情况下三者无法排序，知道之后可以排除一部分（[analysis.md](analysis.md) 第 1.5 节）。接手后先向使用方要 profiler 配置（`with_stack`、`with_modules`、`profile_memory`、`record_shapes`）和崩溃发生的时间点，再按第 7 节和 [analysis.md](analysis.md) 第 7.1 节的步骤处理。

## 1. 结论

- **现象。** 在 free-threaded Python（CPython 3.14 去掉 GIL 的构建，下文简称 FT）下开启 `torch_npu.profiler`，进程出现 SIGSEGV。
  - 现场描述由使用方提供，没有调用栈。
  - 使用方判断是"上报模块初始化竞态导致空指针访问"，这一判断没有经过我们核实。
- **找到的问题。** profiler 的数据上报链路里有 3 处线程安全问题。
  - 对应文件和分支基线 `46936d1`（TorchNPU 2.13.0）逐字节相同，所以这些问题不是 r22 分支引入的。
  - 有 GIL 时，GIL 让多个 Python 线程基本依次执行，这些问题很少暴露；FT 下多个线程同时执行，就会暴露。
- **与上游的关系**（2026-11-02 查看 `Ascend/pytorch` master）：
  - 问题 1、2：上游 master 的 `ring_buffer.h`、`data_dumper.cpp` 只改过代码格式，逻辑与 2.13.0 相同，问题仍在。
  - 问题 3：上游在 `ff5ad88d8a`（2026-10-29，"fix:profiler fix with_stack memory bug"）里重构了 `PythonTracer`，设计和我们不同。两者的逐项对照见 [analysis.md](analysis.md) 第 4.8 节。上游方案的要点：
    - 缓存和事件按线程存放在各自的 `ThreadLocalResult` 中，所有线程共用一个 ctx（`shared_ctx_`）；
    - `control_mutex_`、`report_mutex_` 和每个线程的 `callback_mutex_` 在两种构建中都启用，`results_mutex_` 只在 FT 下启用；
    - FT 且 CPython ≥ 3.14 时，直接调用 CPython 私有 API `_PyEval_StopTheWorld`/`_PyEval_StartTheWorld`。
- **修改内容。** 改了 4 个源文件，新增一套压力测试。
- **验证结果。** 用 stub 头文件（只保留接口、实现最简的替代头文件）把改动的源码单独编译，做压力测试。同一套测试中：
  - 原代码 42 次场景运行全部失败，其中有 SIGSEGV；
  - 修复后 42 次全部通过。
  - 测试的调用方式比 `ProfilerMgr` 的实际调用更密集，例如两个线程并发调用 `Stop`。所以原代码失败只能说明这些路径存在竞态，不能证明使用方遇到的崩溃来自其中哪一处。
- **未完成的验证。**
  - 没有用真实的 torch、CPython 3.14、CANN 头文件编译；
  - 没有在 NPU 上运行；
  - 有 GIL 的构建只做了编译检查，没有运行。

## 2. 背景：profiler 记录从业务线程到文件的路径

开启 profiler 后，算子下发、内存分配、NPU 任务队列出队等都会生成记录，路径如下：

1. **写入。** 业务线程（下文称"写入方"）调用 `ProfilerMgr::Upload*`，再调用 `DataDumper::Report`，把记录放进环形缓冲区 `RingBuffer`。环形缓冲区是固定长度的数组，写到末尾后从开头继续写。
2. **读出。** 每个 `DataDumper` 有一个后台线程（下文称"读取方"），从缓冲区取出记录，写入 `FRAMEWORK/` 目录下的文件。
3. **停止。** 停止采集时，`ProfilerMgr::Stop` 对每个上报器依次调用两个函数：
   - `Stop`：停止后台线程，把剩余记录写完；
   - `UnInit`：释放缓冲区，关闭文件。

`ProfilerMgr` 里有三个上报器：`dataReceiver_`、`dataReceiverWithLock_`（这两个是 `DataDumper`）和 `traceDataReceiver_`（`TraceDataDumper`）。

开启 `with_stack` 等选项时，Python 侧还有一个 `PythonTracer`（`profiler_python.cpp`）：

- 它在每个 Python 线程上安装 profile 回调。
- 回调把函数调用和返回事件先存进 `events_`，数量达到 `TRACE_DUMP_THRESHOLD`、或者停止采集时，再批量交给 `TraceDataDumper`。
- 它内部有多个所有线程共用的容器和缓存。

## 3. 问题与修改

### 问题 1：停止采集时，写入方仍在访问正在被销毁的缓冲区

涉及文件：`toolkit/profiler/src/data_dumper.cpp`、`toolkit/profiler/inc/data_dumper.h`、`toolkit/profiler/common/ring_buffer.h`。

**原代码**（已简化）：

```cpp
// 写入方
void DataDumper::Report(data) {
    if (!start_.load()) return;             // ① 检查是否在采集
    data_chunk_buf_.Push(std::move(data));  // ② 写入缓冲区
}
// RingBuffer::Push 内部：循环里先检查 is_inited_，CAS 占用槽位成功后才写 data_queue_[k]

// 停止方
Stop();    // start_ = false，等待后台线程结束（join），再把剩余记录写完（Flush）
UnInit();  // data_queue_.clear()：销毁缓冲区中的元素，并把大小置 0
```

**出错的执行顺序：**

1. 写入方执行完 ①，或已进入 `Push` 并通过了 `is_inited_` 检查；
2. 停止方执行 `Stop` 和 `UnInit`，`clear()` 销毁缓冲区元素；
3. 写入方接着对已销毁的 `unique_ptr` 元素赋值，或与 `clear()` 同时操作同一个 `unique_ptr`。

后果是重复释放、释放后访问，或者记录泄漏。在测试中，原代码的 ASan 构建报出了 heap-use-after-free 和 double-free。

**其他相关问题：**

- `RingBuffer::is_inited_` 是普通 `bool`，被多个线程同时读写，按 C++ 标准属于未定义行为。
- `TraceDataDumper::ReportHash/ReportParam` 给 `unique_ptr` 成员赋值的同时，停止方可能在读这个成员，两者之间没有同步。
- `GatherAndDumpData` 里有一个函数内 `static bool create_flag`，三个上报器的后台线程共用这一个变量。

**修改：**

1. **写入方先登记、再检查。** 新增原子计数 `writers_`，记录当前处于 `Report` 内的写入方数量：

   ```cpp
   void DataDumper::Report(data) {
       if (data == nullptr || !start_.load(relaxed)) return;  // 未采集时提前返回，开销与原来相同
       writers_.fetch_add(1);            // 登记
       if (start_.load()) {              // 再次检查
           data_chunk_buf_.Push(std::move(data));
       }
       writers_.fetch_sub(1, release);   // 注销
   }
   ```

2. **停止方先关闭采集、再等待写入方退出：**

   ```cpp
   StopLocked():
       if (start_.exchange(false)) {  // 关闭采集
           WaitWriters();             // 忙等，直到 writers_ 为 0
           Thread::Stop();            // 等待后台线程结束
       }
       Flush();                       // 写完剩余记录
   UnInitLocked():
       StopLocked();                  // 确保已停止
       释放缓冲区，关闭文件
   ```

   **为什么这样是安全的：** 写入方先修改 `writers_` 再读 `start_`，停止方先修改 `start_` 再读 `writers_`，这四个操作都使用 `seq_cst`（C++ 最强的内存顺序，所有线程观察到同一个全局操作顺序）。在这个前提下，下面两种情况必有一种成立：

   - 写入方读到 `start_ == false`，不访问缓冲区；
   - 停止方读到 `writers_ > 0`，等待它注销。

   因此不会出现写入方访问缓冲区的同时缓冲区被释放。写入方 `fetch_sub` 使用 release，所以它对缓冲区的写入在 `Flush` 和 `UnInit` 之前可见。

   **注意：** 这两处的 `seq_cst` 不能改弱。按 C++ 内存模型，改弱以后上述"必有一种成立"不再有保证。在 x86 上，原子读-改-写指令本身带有完整内存屏障，所以改弱后压力测试检测不到（变异测试已确认：25 次运行全部通过）。在内存顺序更弱的平台（如 aarch64）上可能出错，这一点没有实测。代码注释已写明"不能改弱、x86 和 TSan 下测不出"。

3. **控制操作加锁。** `Init/Start/Stop/UnInit` 加互斥锁 `control_mutex_`，同一时间只有一个控制操作在执行。
   - 原因：第一轮审阅发现，两个线程同时调用 `Stop` 时，两个线程会同时从缓冲区读数据，而缓冲区（修改前后都一样）只支持一个读取方。
   - `Report` 路径不获取这把锁。

4. **其他修改：**
   - `Start` 改为先置 `start_ = true`，再创建后台线程。原代码顺序相反，后台线程可能启动后立即读到 false 并退出。
   - `ReportHash/ReportParam` 与对应的 Flush 用互斥锁 `extra_data_mutex_` 保护。这两个函数调用次数很少。
   - `create_flag` 改为每个实例一份（`create_dir_flag_`）。
   - `Flush` 改为按每次实际取出的条数循环，不再依赖 `Size()`，避免在尚未写完的槽位上忙等。后台线程的 `Run` 也做了同样的修改。

### 问题 2：多个写入方同时写入时，读取方可能读到尚未写完的槽位

涉及文件：`toolkit/profiler/common/ring_buffer.h`。

**原代码**（已简化，省略了 `is_quit_` 分支）：

```cpp
Push:
    CAS 占用 idle_write_index_，得到槽位 k   // CAS：比较并交换，一种原子操作
    data_queue_[k] = data                     // 写数据
    write_index_++                            // 可读条数加 1
Pop:
    if (read_index_ != write_index_) 读取 data_queue_[read_index_]
```

**出错的执行顺序：**

1. 写入方 A 占用槽位 5，写入方 B 占用槽位 6；
2. B 先写完，执行 `write_index_++`；
3. 读取方据此判断槽位 5 可读，但 A 尚未写完槽位 5。

后果是读到空数据，或者数据丢失、顺序错乱。根本原因是：`write_index_` 只记录已完成写入的总条数，无法据此判断具体哪个槽位已经写完。

**修改：** 改为每个槽位带一个序号的有界队列，即 Dmitry Vyukov 提出的 bounded MPMC queue 在单读取方场景下的写法。

- **槽位结构。** 每个槽位是 `{atomic<size_t> seq; T data;}`，初始化时 `seq` 等于槽位下标。
- **写入方。** 读到 `seq == pos` 时，说明槽位空闲，用 CAS 占用；写完数据后执行 `seq = pos + 1`（release），标记该槽位已写完。
- **读取方。** 只有读到 `seq == pos + 1`（acquire）才取数据；取完执行 `seq = pos + capacity`，供下一轮使用同一下标的写入方。
- **效果。** 读取方只会读到已写完的槽位，并按占用顺序读取。

**接口变化：**

- `Init/UnInit/Push/Pop/Size` 的签名不变，调用方不需要修改。
- `Size()` 的语义有变化：它统计已被占用但尚未取出的槽位，包括尚未写完的槽位；当 `write_index_ <= read_index_` 时返回 0。
- `RingBuffer::Init` 把容量向上取整到 2 的幂，最小为 2。容量为 0 或过大时，只记一条错误日志，不创建缓冲区，之后 `Push` 一律返回 false。

### 问题 3：PythonTracer 的共享状态在多线程下没有同步

涉及文件：`profiler/profiler_python.cpp`。

**原代码的问题：** 原代码依赖 GIL 保证同一时间只有一个线程执行。FT 下，以下几处会被多个线程同时访问：

- **子线程注册。** 子线程开始采集时（`startOne`），向共享的 `std::deque thread_local_results_` 执行 `emplace_back`，没有加锁。
- **回调写缓存。** 每次 Python 函数调用都会触发 profile 回调，回调写入 `events_`、`py_call_cache_`、`module_info_cache_` 等共享缓存。
- **控制操作之间。** `start/stop/clear` 与子线程的 `startOne/stopOne` 之间没有同步。
- **释放后访问。** 回调通过 `ctx->thread_local_result_->active_tracer_` 读取 `ThreadLocalResult`，而 `clear()` 会释放 `ThreadLocalResult`，于是回调可能访问已释放的对象。

**修改：** 下面分两部分说明。第一部分（锁）只在 FT 构建中生效；第二部分（逻辑修改）两种构建都生效。

**第一部分：锁，只在 FT 构建（`Py_GIL_DISABLED`）中启用。** 有 GIL 的构建里，这些锁是空函数。

| 锁 | 保护对象 | 规则 |
|---|---|---|
| `control_mutex_` | 让 `start/stop/clear` 依次执行；保护 `interpreter_` | 持有期间会执行 Python 代码。Python 代码中的终结器（如 `__del__`）或 `__getattr__` 可能再次调用 `start/stop/clear`，而 `PyMutex` 不可重入，所以同一线程重入时抛出异常，避免死锁 |
| `registration_mutex_` | `thread_local_results_`、`start_py_call_info_`、会话计数 `session_` | 持有期间不调用 Python、不分配 Python 对象 |
| `data_mutex_` | 回调写入的各缓存、`events_`、`ctx_tid_map_` | 同上 |

- **加锁顺序：** `control_mutex_` → `registration_mutex_` → `data_mutex_` → `ProfilerMgr` 的上报。修改代码时必须遵守这个顺序。
- **为什么用 `PyMutex`：** 等锁时线程会 detach（释放与解释器的绑定并进入等待），不阻塞 CPython 的 stop-the-world（暂停所有线程以修改全局状态）。

**第二部分：逻辑修改，两种构建都生效。**

- **回调不再访问 `ThreadLocalResult`。** `TraceContext` 中的 `thread_local_result_` 改为 `std::atomic<PythonTracer*> tracer_`，`~ThreadLocalResult` 会把它置空。回调只读取 ctx 和 tracer，不再访问 `ThreadLocalResult`。
- **会话计数 `session_`。** `stop()` 在移除回调之前，在 `registration_mutex_` 内同时执行 `active_ = false` 和 `++session_`。原代码在移除回调之后才置 `active_ = false`。
- **`startOne` 安装回调后再检查一次 `session_`。** 如果已经变化，说明 `stop()` 已经执行，就移除刚安装的回调；如果此时新会话已经开始，就重新注册一次（最多重试一次）。
- **结果。** `stop()` 返回、且所有进行中的 `startOne` 都返回之后，没有线程仍安装着本 tracer 的回调。前提是 `sys.setprofile` 审计回调没有抛异常：按 `Python/legacy_tracing.c`，审计回调返回 -1 时 CPython 直接返回，不进入 stop-the-world，也不替换回调，只打印 unraisable 信息，此时旧回调仍装着：`stop()` 之后由 `active_ == false` 保证不记录事件，`clear()` 之后由 `tracer_` 已置空保证（`tracer_` 只在 `~ThreadLocalResult` 中置空，即 `clear()` 时；未测试）。在此之前短暂存在的回调读到 `active_ == false`，不记录事件；ctx 由 `startOne` 持有的那份引用保证不会被提前释放。
- **其他：**
  - 回调的各记录函数（`genPyCallHashId`、`recordPyCall`、`recordCCall`、`recordReturn`）都新增了 `active_` 检查，`stop` 开始后不再记录。
  - `recordStartStack` 在会话已停止时丢弃记录。
  - `clear()` 改为在持有 GIL 时释放上下文对象（原代码不持有）。
- **ctx 引用的释放方式。** `startOne` 多持有的一份 ctx 引用是在 `663e8ba` 中加入的；`f36d74d` 把手动 `Py_XDECREF` 改为局部对象析构时释放。这样即使 `recordStartStack` 中途抛出 Python 异常（例如某个 `nn.Module` 子类没有调用 `super().__init__`），引用也会被释放。

**FT 构建中的结构变化：**

- `start()` 给所有线程安装同一个共享 ctx（`shared_ = true`），通过 `PyEval_SetProfileAllThreads` 在 stop-the-world 期间统一安装。
- 共享 ctx 的 `start_py_call_info_`、`ctx_tid_map_` 改用线程 id 作为键（见 `ctxKey()`），子线程自己注册的 ctx 仍用地址作为键。

按阅读代码判断，有 GIL 时外部可观察的行为不变：除 `clear()` 改为获取 GIL 外，其余逻辑修改只在出现竞争时才改变执行路径。这一结论没有经过运行验证。

## 4. 验证

运行依赖：Linux，g++（C++17，支持 `-fsanitize=thread`，本机为 11.4），Python 3.8 以上（只用标准库）。不需要 torch、CANN、NPU，不需要联网。

### 4.1 测试方法

测试代码在 `test/cpp/profiler_thread_safety/`：

- 真实的 `ring_buffer.h`、`thread.h`、`data_dumper.h/.cpp`、`profiler_python.cpp` 原样编译。
- 只把它们依赖的日志、torch、CPython、`ProfilerMgr` 头文件换成 `stubs/` 中的 stub。

| 场景 | 内容 |
|---|---|
| `rb` | 8 个线程同时写入，1 个线程同时读取；检查每条数据恰好读到一次、同一写入方的数据按顺序到达、内容完整 |
| `dumper` | 多个线程循环调用 `Report`；主线程反复执行 `Init/Start/Stop/UnInit`，另一个线程随机调用 `Stop` |
| `trace` | 同上，对象是 `TraceDataDumper`，同时调用 `ReportHash/ReportParam` |
| `pytracer` | 多个线程执行 profile 回调和 `startOne/stopOne`；主线程反复执行 `start/stop/clear`；每次 `stop` 之后、待进行中的 `startOne` 全部返回时，检查没有线程仍安装着回调 |

**编译方式和运行次数：**

- 三种编译方式：TSan（检测数据竞争）、ASan（越界、释放后访问、重复释放；同一构建还开了 UBSan，即 UndefinedBehaviorSanitizer，`-fsanitize=address,undefined`，报告中含 `runtime error:`；以及 LeakSanitizer（LSan），Linux 上 ASan 构建默认开启，`run.py` 也设置了 `detect_leaks=1`）、plain（`-O2`，不带 sanitizer）。`run.py` 把这几种 sanitizer 的报告都判为失败，所以一次"ASan 构建失败"可能来自 ASan、UBSan 或 LSan。
- 每个场景重复 3 次。`rb` 场景在不同编译方式下使用不同的缓冲区容量：TSan 用 64 和 1024，ASan 用 64，plain 构建用 64 和 4096。
- 合计 TSan 15 次、ASan 12 次、plain 构建 15 次，共 42 次场景运行。另有 7 次编译检查，包括一次有 GIL 路径的只编译检查，不计入下表。

### 4.2 结果（2026-11-02，本机，同一套测试）

| 被测代码 | 场景运行 | 失败 | 失败表现 |
|---|---|---|---|
| 原代码 `d3216c2` | 42 | 42 | TSan 数据竞争报告；ASan 报 heap-use-after-free 和 double-free；plain 构建下 6 次 SIGSEGV、3 次 SIGABRT；`rb` 丢数据或顺序错乱；部分运行挂起直至超时 |
| 修复后 `f36d74d` | 42 | 0 | 无 |

### 4.3 复现

在仓库根目录执行：

```bash
# 修复后的代码
python3 test/cpp/profiler_thread_safety/run.py --builds tsan,asan,plain --repeats 3

# 原代码：把 d3216c2 的源码导出到另一个目录，用 --source-root 指定
mkdir -p /tmp/orig && git archive d3216c2 torch_npu | tar -x -C /tmp/orig
python3 test/cpp/profiler_thread_safety/run.py --source-root /tmp/orig --builds tsan,asan,plain --repeats 3 --timeout 30
```

在共享服务器上，建议用 `systemd-run --user --wait --collect -- bash -c '...'` 单独运行，避免挂起的测试进程影响当前终端。

### 4.4 审阅

修改过程中做了三轮代码审阅，每轮两位审阅者。本文在 `675f2e9`、`da752e6` 期间经过两轮文档审阅；整理成 `docs/profiler_free_threaded/` 之后，与本目录其他四个文件一起又经过两轮文档审阅。**这些审阅和对应的修复都由 AI 模型子代理完成，没有人工代码审阅。**

- **第一、二轮**共提出 21 条意见：16 条修改了代码或测试，2 条更正了报告措辞，1 条在代码注释中说明（槽位内存增加），2 条只做了性能测量、没有改代码（`writers_` 计数开销、`with_stack` 锁争用，见第 5 节）。其中几个问题是前一轮修复本身引入的：
  - 两个线程同时 `Stop`；
  - `clear()` 重入导致死锁；
  - `startOne` 在 `stop()` 之后仍然安装回调。
- **第二轮的修复过程中**，修复者用加强后的测试自己发现了两处问题（`session_` 的第一版没有同时提前置 `active_ = false`；`recordReturn` 在等锁之后才读 ctx），已修复。这两处的发现者是第二轮修复者本人。
- **第三轮（最终审阅）**两位审阅者都判定没有严重或重要问题；其中确认的一处引用泄漏已在 `f36d74d` 修复，其余轻微问题写入了代码注释或本文第 6 节。

本节在整理文档目录时按审阅记录中的标签重新核对过：代码审阅标签为 `review:concurrency`、`review:integration`（第一轮），`review:concurrency:r2`、`review:integration:r2`（第二轮），`final:tracer`、`final:upload`（第三轮）。审阅记录未公开。原文写的"四轮独立审阅""第三轮发现第二轮修复中的两处问题"与记录不符，已按上面改写。逐条经过见 [learning.md](learning.md) B1、B5。

### 4.5 测试覆盖不到的内容

- **stub 与真实 CPython 3.14 不完全一致。** 例如真实的 `PyEval_SetProfile` 会触发审计回调；`ModuleInfo`、`parseModuleParams`（真实代码）调用的 CPython 接口在 stub 中从不报错，所以测试覆盖不到它们抛出 Python 异常的路径。
- **有 GIL 的路径**只编译、不运行。
- **测试检出回归的能力有限。** 变异测试（故意改回某个错误，看测试能否发现），每项 25 次运行：

  | 变异 | 失败次数 |
  |---|---|
  | 去掉 `WaitWriters` | 2 |
  | 去掉 `Report` 中的二次检查 | 1 |
  | 恢复原来的 `Run` 循环 | 0 |
  | 把 `seq_cst` 改弱 | 0 |

  按默认的每项 3 次运行做回归测试，很可能漏掉这类回归。修改问题 1 相关代码时，请对照本文第 3 节逐条检查。
- **ProfilerMgr 层面没有测到。** 测试的调用方式（并发 `Stop` 等）比 `ProfilerMgr` 的实际调用更密集；`ProfilerMgr` 本身没有编译进测试。

## 5. 行为变化

### 5.1 所有构建

| 变化 | 原因 |
|---|---|
| 采集期间每次上报多两次原子读-改-写操作（`writers_` 加 1、减 1） | 问题 1 的登记计数。`writers_` 是所有写入线程共用的一个计数器，多线程上报时这两次操作会在同一缓存行上争用；已用 `alignas(64)` 与 `start_` 分开存放 |
| 缓冲区每个槽位从 8 字节变为 16 字节；两个 2^20 槽位的缓冲区合计从 16 MB 增至 32 MB | 问题 2 的槽位序号 |
| 缓冲区可用容量从 capacity-1 变为 capacity；非 2 的幂的容量会被向上取整 | 新队列的写法 |
| 采集运行中调用 `Init`：记一条错误日志，先停止并把数据写完到旧路径，再按新路径初始化（原代码直接在运行中的缓冲区上重新分配） | 避免在写入方仍在访问时重新分配缓冲区 |
| `UnInit` 遇到仍在运行的上报器，会先停止并写完数据 | 析构路径同样需要保证安全 |

### 5.2 只影响 FT 构建

| 变化 | 原因 |
|---|---|
| 停止采集时，清除所有线程的 profile 函数，包括其他工具（如 cProfile）设置的 | FT 下无法安全读取其他正在运行的线程当前安装的回调，只能用 `PyEval_SetProfileAllThreads` 在 stop-the-world 期间统一清除 |
| 启动时只记录当前线程已有的调用栈，不应用 `max_threads` | 同上，FT 下不能切换到其他正在运行的线程去遍历它的调用栈 |
| 开启 `with_stack` 时 Python 调用变慢，线程越多越明显 | 每次 Python 调用至少获取两次 `data_mutex_`（`genPyCallHashId`、`recordPyCall` 各一次）。stub 环境中测回调吞吐（2026-11-02，3 次运行的范围，日志 `torch_npu-work/tsan-logs/round2/bench-pytracer-hook.log` 的 `pt_fix_plain` 各行，作者本机，未上传）：1 个线程每秒 418 万到 419 万次，6 个线程合计每秒 124 万到 141 万次。只测了修复后的代码，没有原代码对照；测试只走缓存命中路径；stub 用 `std::mutex` 模拟 `PyMutex`，不含真实 CPython 的线程挂起和恢复开销；测试机是共享服务器，有噪声。[analysis.md](analysis.md) 里有一组我们与上游的吞吐对照，那组用的是另一套扩展过的 stub 和另一次运行，两组数字不能互相换算 |

## 6. 未处理的问题

| 问题 | 说明 |
|---|---|
| host regions 的 worker 线程上的算子可能采集不到 | 算子回调通过 `addThreadLocalCallback` 注册（`npu_profiler.cpp:260`），只对持有 profiler 线程状态的线程生效。如果 worker 在 profiler 启动前已经创建，其算子不会被记录。worker 的数量和创建时机没有核实。影响是结果缺少这些线程的算子数据，没有观察到崩溃 |
| 基线缺少上游 `b2e659d62f`，3.13+ 读取 `self` 得到空指针 | 和线程无关，单线程、有 GIL 时同样成立。CPython C API 的行为已核实，torch_npu 上的崩溃未核实。接手后在第 7 节第 1 步中补上这处修改，做法见 [analysis.md](analysis.md) 第 7.1 节，背景见 [analysis.md](analysis.md) 第 4.2 节 |
| `with_stack` 在 FT 下的锁争用 | 彻底解决需要改为每个线程一份事件缓冲和缓存。上游 master 已采用这种结构（见第 1 节）。上游这种结构下每个事件仍要获取一次全局 `results_mutex_`，只是次数更少、持有时间更短，见 [analysis.md](analysis.md) 第 4.5、5.4 节 |
| `ProfilerMgr` 的 `mstx_domain_include_/exclude_` 存在竞态 | `Start/Stop` 修改这两个 `std::vector` 时，其他线程可能正在 `IsMstxDomainEnabled` 中读取 |
| `ProfilerMgr` 的其他非原子成员 | `path_`、`profConfig_` 等在 `Start/Stop/Init/Finalize` 中写入；多个线程同时调用这些控制函数时没有保护 |
| 有 GIL 时 `record_params_` 会沿用到下一次采集 | 上游原有的问题，修复会改变有 GIL 时的行为，所以没有改 |
| fork 出的子进程里停止采集可能无限等待 | fork 时如果有线程正在 `Report` 中，子进程继承到的 `writers_` 永远大于 0，`WaitWriters` 不会返回。已在代码注释中写明 |
| 一次 `startOne` 期间如果经历两轮完整的 stop/start，该线程在最新一轮会话中不会被采集 | `startOne` 只重试一次，代码注释已写明 |

## 7. 接手后的步骤

1. **确认崩溃出在哪一处，并补上 `b2e659d62f`。**
   - 先向使用方要 profiler 配置（`with_stack`、`with_modules`、`profile_memory`、`record_shapes`）、是否多线程，以及崩溃发生的时间点（开始采集、前向中、停止采集时）。按 [analysis.md](analysis.md) 第 1.5 节，这一步可以排除一部分候选。
   - 在使用方原来的构建（`d3216c2` 或使用方实际使用的 wheel，不含本分支的修改，也不含 `b2e659d62f`）上，用 3.14t 运行一个单线程、`with_stack=True`、含 `nn.Module` 前向并包含停止采集（退出 `with` 块）的最小脚本，开启 `PYTHONFAULTHANDLER=1`，需要时加 `PYTHONMALLOC=debug`，确认候选 1、2。不要在本分支上做这一步：候选 2 已在 `663e8ba` 中修改，补上 `b2e659d62f` 之后候选 1 也不存在，脚本不崩溃不能排除任何候选。
   - **无条件补上 `b2e659d62f` 的 `PyMapping_GetItemString` 修改**（3.13 以下保留 `PyDict_GetItemString`），用 `THPObjectPtr` 持有返回的新引用，取不到时判空跳过；不移植 `~ThreadLocalResult` 在 3.12+ 跳过 `Py_DECREF` 的部分，原因是本分支 `clear()` 已改为持有 GIL 释放 ctx。按 [analysis.md](analysis.md) 第 4.2 节已验证的 C API 行为，3.13 及以上 `PyDict_GetItemString` 必定返回 NULL，所以这处修改不取决于能否复现崩溃。
   - 补上之后，在本分支上再运行一次同一个脚本，确认不再崩溃。详见 [analysis.md](analysis.md) 第 7.1 节。
2. **决定合入方式。**
   - 问题 1、2 的修改（`ring_buffer.h`、`data_dumper.h/.cpp`）可以移植到上游 master：上游这两处的逻辑与 2.13.0 相同。上游做过 clang-format 格式化，补丁不能直接 `git apply`，需要手工合入；没有在 master 上编译过。
   - 问题 3 建议以上游 `ff5ad88d8a` 的方案为基础：它按线程存放缓存和事件，全局锁的获取次数比我们方案少、持有时间短（全局锁仍在：FT 构建下每个事件仍要获取一次 `results_mutex_`）；代价是依赖 CPython 私有 API `_PyEval_StopTheWorld`，并且同线程重入时会死锁。两套方案的逐项对照、上游方案在本仓库压力测试下的运行结果，以及合入时建议补的四处修改，见 [analysis.md](analysis.md) 第 4.8、7.3 节。
3. **编译。** 在 NPU 机器上按 [`../experimental_free_threaded/host_regions_r22.md`](../experimental_free_threaded/host_regions_r22.md) 的"Build and enable deliberately"一节和 [`../experimental_free_threaded/drain_r2.md`](../experimental_free_threaded/drain_r2.md) 的"Build and wheel identity"一节，分别编译 GIL 版和 FT 版。
4. **FT 版验证。** 运行一次多线程推理并开启 `torch_npu.profiler`，建议覆盖 `with_stack=True`、`profile_memory=True`、`schedule` 带 warmup 这几种配置。确认不再崩溃，并且能正常导出结果。
5. **GIL 版验证。** 运行原有的 profiler 测试（`test/profiler/`），确认有 GIL 时行为不变。
6. **如果仍然崩溃。** 先用 `PYTHONFAULTHANDLER=1` 获取 Python 层调用栈（FT 构建只打印崩溃线程的），看崩溃发生在前向中还是停止采集时；再用 `PYTHONMALLOC=debug` 运行，让不持有线程状态时的对象释放在释放的位置终止（`PyObject_Free` 已在本机 3.14.3、3.14.3t 上验证，经 `Py_DECREF` 的释放路径属于推断）；最后用 gdb 获取 C++ 调用栈（`bt`）。如果栈顶在 `PyObject_GetAttrString`、往上是 `ModuleInfo::ModuleInfo`，对应候选 (1)；如果栈顶在 `_Py_Dealloc`、对象的 dealloc 函数或分配器中，往上是 `PythonTracer::clear` 或 `~ThreadLocalResult`，对应候选 (2)；栈顶在上报链路或 profile 回调中，对应候选 (3)。栈顶在 pymalloc、mimalloc、`free` 或与 profiler 无关的代码中时，通常是堆损坏【推断】，崩溃位置不等于出错位置，应开启 `PYTHONMALLOC=debug` 重新运行（候选 3 中问题 1 的释放后访问和重复释放，在本仓库压力测试的 ASan 构建中对原代码复现过，见第 4.2 节）。读法详见 [analysis.md](analysis.md) 第 7.1 节。

## 8. 改动文件

| 文件 | 内容 |
|---|---|
| `torch_npu/csrc/toolkit/profiler/common/ring_buffer.h` | 问题 2：每个槽位带序号的有界队列 |
| `torch_npu/csrc/toolkit/profiler/inc/data_dumper.h` | 问题 1：新增 `writers_`、`control_mutex_`、`extra_data_mutex_`、`create_dir_flag_` |
| `torch_npu/csrc/toolkit/profiler/src/data_dumper.cpp` | 问题 1：登记、等待、加锁的停止流程 |
| `torch_npu/csrc/profiler/profiler_python.cpp` | 问题 3：FT 下加锁、会话计数、ctx 生命周期 |
| `test/cpp/profiler_thread_safety/` | 压力测试、stub 头文件、运行脚本 `run.py` |
