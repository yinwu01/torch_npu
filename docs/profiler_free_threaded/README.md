# torch_npu profiler 在 free-threaded Python 下的线程安全

- **问题。** 使用方在 free-threaded CPython 3.14（去掉 GIL 的构建，下文简称 FT）下开启 `torch_npu.profiler` 时进程 SIGSEGV，没有调用栈。
- **已做的工作。** 静态阅读上报链路，找到三处线程安全问题（停止采集时写入方仍在访问正在被销毁的环形缓冲区；多个写入方同时写入时读取方可能读到尚未写完的槽位；`PythonTracer` 的共享状态没有同步），在分支 `fix/profiler-thread-safety` 上修复。stub 环境的压力测试：原代码 42 次场景运行全部失败，修复后 42 次全部通过。做上游对照时另外发现两处在单线程、有 GIL 时同样成立的问题（其中一处本分支已修改）。所有修改都只在 stub 环境中验证过，没有用真实 torch、CPython 3.14t、CANN 头文件编译，也没有在 NPU 上运行。
- **接手后的步骤**（理由见 [analysis.md](analysis.md) 第 1.5、7.1 节）。准备：NPU 机器、CANN、面向 CPython 3.14t 编译的 torch；编译方法见 [`../experimental_free_threaded/host_regions_r22.md`](../experimental_free_threaded/host_regions_r22.md) 的“Build and enable deliberately”一节和 [`../experimental_free_threaded/drain_r2.md`](../experimental_free_threaded/drain_r2.md) 的“Build and wheel identity”一节。使用方只有 wheel、没有对应源码时，用 `d3216c2` 的源码编译一份作为下文的“原构建”。

  1. **向使用方要配置和崩溃时机。** profiler 配置（`with_stack`、`with_modules`、`activities` 是否含 CPU、`profile_memory`、`record_shapes`）、是否多线程，以及崩溃发生在开始采集、前向中，还是停止采集时。
  2. **在原构建上检查候选 1。** 原构建不含本分支的修改，也不含 `b2e659d62f`。用 3.14t 运行单线程、`with_stack=True`、采集 CPU activity、含 `nn.Module` 前向的最小脚本，开启 `PYTHONFAULTHANDLER=1`，并在 gdb 下运行。第一次前向时崩溃，且 `bt` 的栈顶在 `PyObject_GetAttrString`、往上是 `ModuleInfo::ModuleInfo`，对应候选 1。只看到前向崩溃不足以区分候选 1 和候选 3 中的问题 2（单个 Python 线程时仍有 C++ 写入线程）。候选 1 成立时，进程不会执行到停止采集，所以这一步看不到候选 2。
  3. **在原构建上只补 `b2e659d62f` 中 `PyMapping_GetItemString` 和判空的部分，重新编译，然后：**
     1. 重新运行单线程脚本，包含停止采集（`prof.stop()` 或退出 `with` 块），开启 `PYTHONMALLOC=debug`。前向不再崩溃，即确认候选 1；停止采集时终止，对应候选 2。
     2. 在同一构建上，按使用方的线程数运行多线程用例（有条件时用 ASan 构建）。这一构建仍含三处竞态，用来检查候选 3。
  4. **在本分支上补同一处修改（独立提交），编译本分支，运行第 3.2 步的同一个多线程用例，确认本分支的修复。** 本分支同样缺少 `b2e659d62f`，不补的话会先遇到候选 1。

使用方崩溃的根因没有确定，有三个候选，详见 [analysis.md](analysis.md) 第 1.5、7.1、8 节：

1. 基线缺少上游 `b2e659d62f`，CPython 3.13 及以上读取 `nn.Module` 帧的 `self` 得到空指针，随后对空指针取属性；
2. 上游 issue #5658 所述路径：原 `clear()` 在没有持有 GIL 的状态下 `Py_DECREF` `TraceContext`（本分支的 `clear()` 已改为先取 GIL 再释放）；
3. 上面三处竞态中的某一处。

候选 1、2 都在单线程、有 GIL 时成立，候选 3 只在并发时成立。知道使用方配置和崩溃时机之后可以排除一部分（推断，见 [analysis.md](analysis.md) 第 1.5 节）：候选 1、2 都要求开启 `with_stack` 或 `with_modules` 并采集 CPU activity，否则 `PythonTracer` 不启动，只剩候选 3 中的问题 1、2；在 3.13 及以上，只要采集期间调用过 `nn.Module`，候选 1 在第一次前向时就会发生，早于停止采集。诊断手段（[analysis.md](analysis.md) 第 7.1 节）：`PYTHONFAULTHANDLER=1` 在 SIGSEGV 时打印 Python 层调用栈（FT 构建只打印崩溃线程的）；`PYTHONMALLOC=debug` 在不持有线程状态时调用 CPython 分配器的释放函数时终止进程（经 `Py_DECREF` 释放 ctx 的路径属于推断），对候选 3 中 C++ 对象的释放没有作用，候选 3 要用 ASan 构建；gdb `bt` 取得 C++ 调用栈。

## 术语

- **GIL**：全局解释器锁，同一时刻只允许一个线程执行 Python 字节码的锁。**FT 构建**是去掉它的 CPython 构建（编译宏 `Py_GIL_DISABLED`）；**GIL 构建**是默认构建。
- **stub**：只保留接口、实现最简的替代头文件，用来在没有 torch、CANN、CPython 3.14 的机器上单独编译被测源码。
- **TSan / ASan**：编译选项 `-fsanitize=thread` / `-fsanitize=address`，分别检测数据竞争和内存错误（越界、释放后访问、重复释放）。测试的 ASan 构建实际用的是 `-fsanitize=address,undefined`，同时开启 **UBSan**（UndefinedBehaviorSanitizer，检测未定义行为，报告中含 `runtime error:`）。**plain 构建**指 `-O2`、不带任何 sanitizer 的构建。
- **LSan**：LeakSanitizer，检测内存泄漏；Linux 上 ASan 构建默认开启，测试脚本也显式设置了 `detect_leaks=1`。所以一次"ASan 构建失败"可能来自 ASan、UBSan 或 LSan。
- **`seq_cst`**：C++ 最强的内存顺序 `std::memory_order_seq_cst`，所有使用它的原子操作处于一个所有线程都认同的全局顺序中。
- **变异测试**：故意把某处修复改回错误写法，检查测试能否发现，用来衡量测试检出回归的能力。
- **ctx**：`TraceContext`，tracer 传给 `PyEval_SetProfile` 的 Python 对象，回调通过它找到 tracer。
- **host regions**、**r22 分支**：本分支基线所带的实验特性和它所在的分支，说明见 [`docs/experimental_free_threaded/host_regions_r22.md`](../experimental_free_threaded/host_regions_r22.md)。

## 当前状态

| 事项 | 状态 | 说明 |
|---|---|---|
| 三处线程安全问题的定位和修改 | 已完成 | `663e8ba`、`f36d74d` |
| stub 环境的压力测试 | 已完成 | 原代码 42/42 失败，修复后 0/42 失败 |
| 候选方案对比、与上游 `ff5ad88d8a` 的对照 | 已完成 | 见 [analysis.md](analysis.md) |
| 本文档目录 | 已完成 | `87dcd0e`、`d1bbbad` 及之后的文档提交；最后一次修改未再经独立审阅（见“交付状态”） |
| 向使用方要配置和崩溃时机；在原构建上检查候选 1、2、3 | 待办 | 接手后的第 1 至 3 步，做法见 [analysis.md](analysis.md) 第 7.1 节 |
| 在本分支补上游 `b2e659d62f` 的 `PyMapping_GetItemString` 和判空部分 | 待办 | 接手后的第 4 步，做法见 [analysis.md](analysis.md) 第 7.1 节 |
| 用真实 torch、CPython 3.14t、CANN 头文件编译，并在 NPU 上验证 | 待办 | 见 [fix_details.md](fix_details.md) 第 7 节 |
| 问题 3 的合入方式：继续用我们的写法，还是改用上游 `ff5ad88d8a` | 待办 | 需要接手的工程师或上游维护者决定，对照和建议见 [analysis.md](analysis.md) 第 4.8、4.11、7.3 节 |
| `fix_details.md` 第 6 节列出的其他未处理问题 | 待办 | 同节 |

逐项的证据和下一步见 [progress.md](progress.md)。

## 阅读顺序

1. **[fix_details.md](fix_details.md)** — 三个问题分别是什么、怎么改的、验证到什么程度、行为有哪些变化、哪些问题没处理、接手后的步骤。只想知道"改了什么"就读这一份。
2. **[analysis.md](analysis.md)** — 问题是怎么定位出来的；每个问题还有哪些修法、为什么选了现在这个（含实测数字和它们的局限）；与上游 `ff5ad88d8a` 的逐项对照和把上游版本放进本仓库测试运行的结果；给接手工程师的建议；未解决的问题。要决定"合入哪一套"就读这一份。
3. **[learning.md](learning.md)** — A 部分是维护这些代码需要的背景知识（free-threaded CPython、C++ 内存模型、Vyukov 队列、TSan/ASan、变异测试），每条附官方出处；B 部分是这次修复过程中有记录可查的经验，共 8 条。
4. **[progress.md](progress.md)** — 做过什么、现在停在哪里、待办清单。

本目录之外，FT 构建本身和 host regions 的说明在 `docs/experimental_free_threaded/`。

## 怎么运行测试

测试代码在 [`test/cpp/profiler_thread_safety/`](../../test/cpp/profiler_thread_safety)，另有一份说明在该目录的 `README.md`。运行依赖：Linux，g++（C++17，支持 `-fsanitize=thread`，本机为 11.4），Python 3.8 以上（只用标准库）。不需要 torch、CANN、NPU，不需要联网。

在仓库根目录执行：

```bash
# 修复后的代码
python3 test/cpp/profiler_thread_safety/run.py --builds tsan,asan,plain --repeats 3

# 原代码对照：把 d3216c2 的源码导出到另一个目录，用 --source-root 指定
mkdir -p /tmp/orig && git archive d3216c2 torch_npu | tar -x -C /tmp/orig
python3 test/cpp/profiler_thread_safety/run.py --source-root /tmp/orig --builds tsan,asan,plain --repeats 3 --timeout 30
```

在共享服务器上建议用 `systemd-run --user --wait --collect -- bash -c '...'` 单独运行，避免挂起的测试进程影响当前终端。

按默认的每项 3 次运行做回归测试，检出能力有限（变异测试的结果见 [learning.md](learning.md) A4.3）；修改问题 1 相关代码时请对照 [fix_details.md](fix_details.md) 第 3 节逐条检查。

## 分支上的提交

分支 `fix/profiler-thread-safety`，远端 <https://github.com/yinwu01/torch_npu>。

| 提交 | 内容 |
|---|---|
| [`663e8ba`](https://github.com/yinwu01/torch_npu/commit/663e8ba3da4b6db43ca04971b82bdb9bb1e96ba8) | 三个问题的修改加压力测试 |
| [`f36d74d`](https://github.com/yinwu01/torch_npu/commit/f36d74df1e4daf3af7e81387f1d367abc1fbaee0) | `startOne` 的 ctx 引用在异常路径上泄漏；并在 `data_dumper.cpp` 注释中写明 `seq_cst` 要求和 fork 限制 |
| [`675f2e9`](https://github.com/yinwu01/torch_npu/commit/675f2e9c5aa096da70cd649ca4dc00dd28b68a18) | 交接文档（即本目录的 `fix_details.md`） |
| [`da752e6`](https://github.com/yinwu01/torch_npu/commit/da752e6a12e87e79095da4f23050cb1f81833533) | 更正交接文档中对上游方案的概括和审阅条目统计 |
| `87dcd0e`、`d1bbbad` 及之后 | 本目录（方案分析、学习笔记、进展），交接文档移为 `fix_details.md` |

分支基线是 `113xiaoji/torch_npu` 的 `codex/r22-real-four-worker-20261027`（`d3216c2`，TorchNPU 2.13.0，基线 `46936d1`）。被修改的四个源文件与 2.13.0 逐字节相同，所以这些问题不是 r22 分支引入的。

## 交付状态

代码部分经过三轮审阅（每轮两位审阅者）。原交接文档（即 `fix_details.md`）在 `675f2e9`、`da752e6` 期间经过两轮文档审阅；整理成本目录之后，五个文件一起经过四轮文档审阅；第四轮审阅意见已修改，修改后未再经独立审阅。**这些审阅和对应的修复都由 AI 模型子代理完成，没有人工审阅。**
