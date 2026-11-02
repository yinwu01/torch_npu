// Standalone stress test for PythonTracer in torch_npu/csrc/profiler/profiler_python.cpp
// (bug 3: child threads registering into the shared tracer state without a lock).
// The real profiler_python.cpp is compiled unmodified; stubs/ stands in for CPython,
// pybind11, torch and ProfilerMgr and emulates free-threaded attach/detach and the
// stop-the-world pause of PyEval_SetProfileAllThreads with a reader/writer lock.
//
// Worker threads run the installed profile hook and call kStartOne/kStopOne, as
// _enable_profiler_in_child_thread does, while the main thread cycles
// kStartAll/kStop/kClear with the "GIL" released. Workers have frames on their stack,
// so kStartOne walks them, and the stub PyEval_SetProfile detaches for up to 2 ms
// before installing (3.14 does a stop-the-world there), so kStop and kClear can run
// to completion inside a worker's kStartOne. After each kStop, once no kStartOne is in
// flight, no worker may still have a profile hook installed. During start() the main
// thread's frame walk runs a callback that calls kStartOne and kStopOne, emulating
// user Python code that re-enters the tracer; a tracer lock held across Python code
// would deadlock here (caught by the timeout).
// Prints one JSON line; a failed check, a sanitizer report or a hang is a failure.
//
//   python_tracer_test <workers> <cycles>
//   python_tracer_test bench <workers> <milliseconds>   (hook throughput, no checks)
#include "stub_all.h"
#include <chrono>
#include <cstdio>
#include <random>

StubWorld g_world;
thread_local int t_attached = 0;
PyInterpreterState g_interp;
thread_local PyThreadState t_tstate{};
PyCodeObject g_code;
std::vector<PyThreadState*> g_tstates;
thread_local void (*t_frame_callback)() = nullptr;
thread_local PyFrameObject* t_current_frame = nullptr;

static std::atomic<bool> g_pause_set_profile{true};
long stub_set_profile_pause_us()
{
    thread_local std::mt19937 rng(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    if (!g_pause_set_profile.load(std::memory_order_relaxed)) {
        return -1;
    }
    return static_cast<long>(rng() % 2000);
}

namespace torch_npu { namespace profiler { namespace python_tracer {
static CallFn g_call = nullptr;
void registerFunctions(CallFn call) { g_call = call; }
}}}
using namespace torch_npu::profiler::python_tracer;

static std::atomic<long> g_reentries{0};
static void ReenterTracer()
{
    g_reentries++;
    g_call(Command::kStartOne);
    g_call(Command::kStopOne);
}

static double Seconds(std::chrono::steady_clock::time_point since)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

// Hook throughput with the tracer active: workers call the installed hook in a loop.
static int Bench(int workers_n, int ms)
{
    g_pause_set_profile = false;
    g_tstates.push_back(&t_tstate);
    init();
    std::atomic<bool> go{false}, done{false}, may_exit{false};
    std::atomic<int> ready{0}, finished{0};
    std::atomic<long> hook_calls{0};
    std::vector<std::thread> workers;
    for (int w = 0; w < workers_n; ++w) {
        workers.emplace_back([&, w] {
            { std::unique_lock<StubWorld> l(g_world); g_tstates.push_back(&t_tstate); }
            ready++;
            while (!go.load()) std::this_thread::yield();
            PyFrameObject frames[4];
            for (int i = 0; i < 4; ++i) { frames[i].line = w * 10 + i; frames[i].back = nullptr; }
            const int whats[] = {PyTrace_CALL, PyTrace_C_CALL, PyTrace_RETURN, PyTrace_C_RETURN};
            long n = 0;
            stub_attach();
            while (!done.load(std::memory_order_relaxed)) {
                for (int k = 0; k < 256; ++k, ++n) {
                    if (auto f = t_tstate.c_profilefunc) {
                        f(t_tstate.c_profileobj, &frames[n % 4], whats[n % 4], nullptr);
                    }
                }
                // Let the stop-the-world in kStop through now and then.
                stub_detach();
                stub_attach();
            }
            stub_detach();
            hook_calls += n;
            finished++;
            // Keep t_tstate (listed in g_tstates) alive until kStop has run.
            while (!may_exit.load()) std::this_thread::yield();
        });
    }
    while (ready.load() < workers_n) std::this_thread::yield();
    g_call(Command::kStartAll);
    auto t0 = std::chrono::steady_clock::now();
    go = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    done = true;
    while (finished.load() < workers_n) std::this_thread::yield();
    double secs = Seconds(t0);
    g_call(Command::kStop);
    g_call(Command::kClear);
    may_exit = true;
    for (auto& t : workers) t.join();
    printf("{\"scenario\": \"pytracer-bench\", \"passed\": true, \"workers\": %d, \"seconds\": %.3f, "
           "\"hook_calls\": %ld, \"calls_per_sec\": %.0f, \"calls_per_sec_per_thread\": %.0f}\n",
           workers_n, secs, hook_calls.load(), hook_calls.load() / secs, hook_calls.load() / secs / workers_n);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "bench") {
        return Bench(argc > 2 ? atoi(argv[2]) : 1, argc > 3 ? atoi(argv[3]) : 1000);
    }
    int workers_n = argc > 1 ? atoi(argv[1]) : 6;
    int cycles = argc > 2 ? atoi(argv[2]) : 40;
    auto t0 = std::chrono::steady_clock::now();
    // Main thread: registers its tstate, init() under "GIL".
    g_tstates.push_back(&t_tstate);
    init();
    std::atomic<bool> done{false};
    std::atomic<int> ready{0};
    std::atomic<int> start_one_in_flight{0};
    std::vector<std::thread> workers;
    std::atomic<long> hook_calls{0};
    for (int w = 0; w < workers_n; ++w) {
        workers.emplace_back([&, w] {
            { std::unique_lock<StubWorld> l(g_world); g_tstates.push_back(&t_tstate); }
            ready++;
            PyFrameObject frames[4];
            for (int i = 0; i < 4; ++i) { frames[i].line = w * 10 + i; frames[i].back = i > 0 ? &frames[i - 1] : nullptr; }
            // kStartOne walks these frames (genPyCallHashId takes the tracer locks).
            t_current_frame = &frames[3];
            const int whats[] = {PyTrace_CALL, PyTrace_C_CALL, PyTrace_RETURN, PyTrace_C_RETURN};
            long n = 0;
            while (!done.load()) {
                stub_attach();
                for (int k = 0; k < 64; ++k, ++n) {
                    if (w % 2 == 0 && n % 500 == 0) {
                        start_one_in_flight++;
                        g_call(Command::kStartOne);
                        start_one_in_flight--;
                    }
                    if (w % 2 == 0 && n % 1501 == 0) {
                        g_call(Command::kStopOne);
                    }
                    auto f = t_tstate.c_profilefunc;
                    auto o = t_tstate.c_profileobj;
                    if (f) {
                        f(o, &frames[n % 4], whats[n % 4], nullptr);
                        hook_calls++;
                    }
                }
                stub_detach();
            }
        });
    }
    while (ready.load() < workers_n) std::this_thread::yield();
    // Two frames on the main thread's stack, so start() walks them.
    PyFrameObject main_frames[2];
    main_frames[0].line = 1000; main_frames[0].back = nullptr;
    main_frames[1].line = 1001; main_frames[1].back = &main_frames[0];
    t_current_frame = &main_frames[1];
    long stale_hooks = 0;
    // Control thread runs with the "GIL" released, like stopNpuProfiler.
    for (int cycle = 0; cycle < cycles; ++cycle) {
        t_frame_callback = ReenterTracer;
        g_call(Command::kStartAll);
        t_frame_callback = nullptr;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        g_call(Command::kStop);
        // A kStartOne that began after kStop sees the tracer stopped and installs nothing;
        // one that began before has finished once the count reads 0.
        while (start_one_in_flight.load() != 0) std::this_thread::yield();
        {
            std::unique_lock<StubWorld> stw(g_world);
            for (size_t i = 1; i < g_tstates.size(); ++i) {
                if (g_tstates[i]->c_profilefunc != nullptr) {
                    stale_hooks++;
                }
            }
        }
        g_call(Command::kClear);
    }
    done = true;
    for (auto& t : workers) t.join();
    auto* mgr = torch_npu::profiler::ProfilerMgr::GetInstance();
    printf("{\"scenario\": \"pytracer\", \"passed\": %s, \"stale_hooks_after_stop\": %ld, \"hook_calls\": %ld, "
           "\"uploaded_events\": %zu, \"hash_uploads\": %zu, \"reentries\": %ld, \"seconds\": %.3f}\n",
           stale_hooks == 0 ? "true" : "false", stale_hooks, hook_calls.load(), mgr->events.load(),
           mgr->hashes.load(), g_reentries.load(), Seconds(t0));
    return stale_hooks == 0 ? 0 : 1;
}
