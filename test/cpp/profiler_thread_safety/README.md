# Profiler upload path thread-safety stress test

What was fixed, why, and what is still open: see
[docs/experimental_free_threaded/profiler_thread_safety.md](../../../docs/experimental_free_threaded/profiler_thread_safety.md) (Chinese).

Standalone tests for the profiler under concurrent use, as happens with free-threaded
CPython (no GIL). Two binaries:

- `profiler_thread_safety_test.cpp` covers `RingBuffer`, `DataDumper` and
  `TraceDataDumper`. It compiles the real
  `torch_npu/csrc/toolkit/profiler/common/ring_buffer.h`, `common/thread.h`,
  `inc/data_dumper.h` and `src/data_dumper.cpp` unmodified. `stubs/` replaces only
  `npu_log.h` (silent log macros), `utils.h` (file helpers) and `data_reporter.h` (report
  types reduced to `tag` + `encode()`).
- `python_tracer/python_tracer_test.cpp` covers `PythonTracer` in
  `torch_npu/csrc/profiler/profiler_python.cpp`, compiled unmodified with
  `-DPy_GIL_DISABLED`. `python_tracer/stubs/` stands in for CPython, pybind11, torch,
  `ProfilerMgr` and the report types. It emulates attach/detach and the stop-the-world
  pause of `PyEval_SetProfileAllThreads` with a writer-preferring reader/writer lock,
  `PyMutex` detaching a waiting thread, and `PyEval_SetProfile` detaching for up to 2 ms
  before it installs the hook (3.14 does a stop-the-world there). The same file is also compiled once without `-DPy_GIL_DISABLED` as a
  build-only check of the GIL code path (built, not run).

No torch, CANN or NPU is needed.

Requirements: Linux, g++ with C++17 and `-fsanitize=thread` (tested with g++ 11.4),
Python 3.8+ (standard library only). No network.

## Run

```bash
python3 test/cpp/profiler_thread_safety/run.py                  # TSan + plain -O2, 3 repeats
python3 test/cpp/profiler_thread_safety/run.py --builds tsan,asan,plain --repeats 10
python3 test/cpp/profiler_thread_safety/run.py --source-root /path/to/other/tree
```

`--source-root` points at any tree that contains `torch_npu/csrc/toolkit/profiler/...`,
e.g. a copy of the original files for a before/after comparison. Logs go to
`results/<timestamp>/` (or `--log-dir`): one `.log` per run plus `SUMMARY.json`. Exit
code 0 means every run passed.

On a shared machine, run it in its own unit so a hung binary cannot block the shell:

```bash
systemd-run --user --wait --collect --quiet -- bash -c \
  'timeout 300 python3 test/cpp/profiler_thread_safety/run.py --log-dir /tmp/pts'
```

## Scenarios

- `rb`: 8 producers `Push` (retrying when full) while one consumer `Pop`s. Every item
  must be popped exactly once, in per-producer order, with a valid checksum.
- `dumper`: producers call `Report` on two `DataDumper` instances while the main thread
  cycles `Init/Start/Stop/UnInit` (the order `ProfilerMgr::StopDataReceiver` uses).
- `trace`: producers call `TraceDataDumper::Report` (two of every three producers),
  `ReportHash` and `ReportParam` while the main thread cycles `Init/Start/Stop/UnInit`.
- In `dumper` and `trace` a second control thread also calls `Stop()` at random times,
  so concurrent `Stop()` calls and `Stop()` against `Init/Start/UnInit` are exercised.
- `pytracer`: worker threads run the installed profile hook and call
  `kStartOne/kStopOne` (the child-thread path) while the main thread cycles
  `kStartAll/kStop/kClear` with the GIL released. Workers have frames, so `kStartOne`
  walks them. After each `kStop`, once no `kStartOne` is in flight, no worker may still
  have a hook installed (a hook installed after `stop()` removed all hooks would later
  run against state that `clear()` frees). During `start()` the frame walk runs a
  callback that calls `kStartOne` and `kStopOne`, standing in for user Python code
  (`__getattr__`, audit hook, finalizer) that re-enters the tracer; a tracer lock held
  across Python code deadlocks there and the run times out.

For `dumper` and `trace` the dumped files are read back: every record must be intact,
per-producer order must hold within each file, the number of records must equal the
number of `encode()` calls, no report object may leak, and no record may be encoded inside
the main thread's `UnInit()`/`Init()` (they run after its own `Stop()` returned, which must
have drained everything a `Report*` call accepted). `C10_LIKELY` in the stub
`data_reporter.h` pauses after the `start_` re-check in `Report`, `ReportHash` and
`ReportParam`, which widens the window `Stop()` has to close. A run fails on a non-zero
exit, a failed check, a timeout, or any sanitizer report on stderr.

## Benchmarks (no checks)

```bash
<binary>/test-dumper-plain bench <dir> <producers> <milliseconds> <capacity>   # Report() calls/s
<binary>/test-pytracer-plain bench <workers> <milliseconds>                     # hook calls/s
```

The stubs' `PyMutex` is a `std::mutex`, so the hook numbers show lock contention, not
CPython's park/attach cost.

## Not covered

`ProfilerMgr` state (`mstx_domain_*`, `path_`, concurrent `Start/Stop`), the real
CPython runtime (the `pytracer` stubs only emulate attach/detach and stop-the-world; no
real finalizers, GC or audit hooks run), and the real `aclAppLog`/`Utils` code.
`profiler_mgr.cpp` and `npu_profiler.cpp` are not compiled at all, and none of the sources
is compiled against the real torch/CPython headers. Passing here does not show the real
library builds; it only exercises these sources with stubs.
