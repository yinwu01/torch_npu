#include "torch_npu/csrc/profiler/containers.h"
#include "torch_npu/csrc/profiler/profiler_python.h"

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <Python.h>
#include <frameobject.h>

#include "torch_npu/csrc/profiler/npu_profiler.h"
#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include "torch_npu/csrc/profiler/utils.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"

#include <c10/util/hash.h>
#include <ATen/record_function.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/profiler/api.h>

namespace torch_npu {
namespace profiler {
namespace python_tracer {

const std::string EXIT_EVENT_DESC = "__torch_npu_profiler_python_tracer_exit";  // Special hash value for exit event
const size_t EXIT_EVENT_HASH_ID = c10::get_hash(EXIT_EVENT_DESC);               // Special hash key for exit event
const std::string MODULE_NAME_DELIMITER = "######";
constexpr size_t TRACE_DUMP_THRESHOLD = 1024 * DEFAULT_BLOCK_SIZE;
constexpr size_t STACK_MAX_DEPTH = 128;

using TensorMetadata = torch_npu::toolkit::profiler::TensorMetadata;
using ModuleParam = torch_npu::toolkit::profiler::ModuleParam;
using OptimizerParam = torch_npu::toolkit::profiler::OptimizerParam;

std::vector<PyThreadState*> getInterpreterThreads(PyInterpreterState* interpreter)
{
    pybind11::gil_scoped_acquire gil;
    std::vector<PyThreadState*> threads;
    if (interpreter != nullptr) {
        auto* thread_state = PyInterpreterState_ThreadHead(interpreter);
        while (thread_state != nullptr) {
            threads.push_back(thread_state);
            thread_state = PyThreadState_Next(thread_state);
        }
    }
    return threads;
}

class GilAndRestoreThread {
public:
    GilAndRestoreThread() : gil_(), initial_thread_state_{PyThreadState_Get()} {}
    ~GilAndRestoreThread()
    {
        PyThreadState_Swap(initial_thread_state_);
        if (!Py_IsInitialized()) {
            gil_.disarm();
        }
    }

    PyThreadState* initial_thread_state() const
    {
        return initial_thread_state_;
    }

private:
    pybind11::gil_scoped_acquire gil_;
    PyThreadState* initial_thread_state_{nullptr};
};

#ifdef Py_GIL_DISABLED
// Without the GIL the tracer state is shared by all traced threads at once.
// PyMutex detaches the waiting thread state, so a thread blocked here never
// holds up a stop-the-world pause or another thread's attach. A std::mutex
// would deadlock in both cases.
class TracerMutex {
public:
    void lock()
    {
        PyMutex_Lock(&mutex_);
    }
    void unlock()
    {
        PyMutex_Unlock(&mutex_);
    }

private:
    PyMutex mutex_{};
};

// For start()/stop()/clear(), which run Python code while holding it. If that
// code re-enters one of them on the same thread, fail instead of deadlocking
// (PyMutex is not recursive).
class TracerControlMutex {
public:
    void lock()
    {
        unsigned long self = PyThread_get_thread_ident();
        TORCH_CHECK(owner_.load(std::memory_order_relaxed) != self,
            "PythonTracer start/stop/clear re-entered on the same thread.", PROF_ERROR(ErrCode::INTERNAL));
        PyMutex_Lock(&mutex_);
        owner_.store(self, std::memory_order_relaxed);
    }
    void unlock()
    {
        owner_.store(0, std::memory_order_relaxed);
        PyMutex_Unlock(&mutex_);
    }

private:
    PyMutex mutex_{};
    std::atomic<unsigned long> owner_{0};
};
#else
// With the GIL all callers are already serialized; keep the old behaviour.
class TracerMutex {
public:
    void lock() {}
    void unlock() {}
};
using TracerControlMutex = TracerMutex;
#endif
using TracerLockGuard = std::lock_guard<TracerMutex>;
using TracerControlGuard = std::lock_guard<TracerControlMutex>;

class PythonTracer;
struct TraceContext {
    PyObject_HEAD
    // Read lock-free by pyProfileFn on any traced thread. The hook only touches
    // ctx and the tracer singleton, never the ThreadLocalResult, which clear()
    // can free at any time. ctx is kept alive by the thread state's reference
    // only while this thread stays attached: without the GIL, read it before
    // the first tracer lock wait (which detaches) and not after.
    std::atomic<PythonTracer*> tracer_;
    // Set before publishing for the ctx that start() installs on all threads
    // (free-threaded build only); see ctxKey().
    bool shared_;
};

static PyTypeObject TraceContextType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "TraceContext",             /* tp_name */
    sizeof(TraceContext),       /* tp_basicsize */
    0,                          /* tp_itemsize */
    nullptr,                    /* tp_dealloc */
    0,                          /* tp_vectorcall_offset */  // NOLINT: modernize-use-nullptr
    nullptr,                    /* tp_getattr */
    nullptr,                    /* tp_setattr */
    nullptr,                    /* tp_reserved */
    nullptr,                    /* tp_repr */
    nullptr,                    /* tp_as_number */
    nullptr,                    /* tp_as_sequence */
    nullptr,                    /* tp_as_mapping */
    nullptr,                    /* tp_hash  */
    nullptr,                    /* tp_call */
    nullptr,                    /* tp_str */
    nullptr,                    /* tp_getattro */
    nullptr,                    /* tp_setattro */
    nullptr,                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,         /* tp_flags */
    "Python tracer TLS",        /* tp_doc */
    nullptr,                    /* tp_traverse */
    nullptr,                    /* tp_clear */
    nullptr,                    /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    nullptr,                    /* tp_iter */
    nullptr,                    /* tp_iternext */
    nullptr,                    /* tp_methods */
    nullptr,                    /* tp_members */
    nullptr,                    /* tp_getset */
    nullptr,                    /* tp_base */
    nullptr,                    /* tp_dict */
    nullptr,                    /* tp_descr_get */
    nullptr,                    /* tp_descr_set */
    0,                          /* tp_dictoffset */
    nullptr,                    /* tp_init */
    nullptr,                    /* tp_alloc */
    PyType_GenericNew,          /* tp_new */
    nullptr                     /* tp_free */
};

static TraceContext* newTraceContext()
{
    return (TraceContext*) TraceContextType.tp_alloc(&TraceContextType, 0);
}

struct ThreadLocalResult {
    // Takes over the reference to ctx (from newTraceContext(), may be null).
    // ctx is allocated by the caller so that no Python allocation happens
    // while registration_mutex_ is held.
    ThreadLocalResult(PythonTracer* active_tracer, TraceContext* ctx)
        : active_tracer_(active_tracer),
          ctx_(ctx)
    {
        if (ctx_) {
            ctx_->tracer_.store(active_tracer, std::memory_order_release);
        }
    }

    ThreadLocalResult() = delete;
    ThreadLocalResult(const ThreadLocalResult&) = delete;
    ThreadLocalResult(ThreadLocalResult&&) = delete;
    ThreadLocalResult& operator=(const ThreadLocalResult&) = delete;
    ThreadLocalResult& operator=(const ThreadLocalResult&&) = delete;

    ~ThreadLocalResult()
    {
        if (ctx_) {
            // Turns a hook that still holds ctx_ into a no-op.
            ctx_->tracer_.store(nullptr, std::memory_order_release);
            Py_DECREF((PyObject*)ctx_);
        }
        active_tracer_ = nullptr;
    }

    PythonTracer* active_tracer_{nullptr};
    TraceContext* ctx_{nullptr};
};

struct PyCallInfo {
    PyCallInfo() = default;
    explicit PyCallInfo(PyFrameObject* frame) : line_no_(PyFrame_GetLineNumber(frame))
    {
        auto f_code = PyFrame_GetCode_NPU(frame);
        file_name_ = THPUtils_unpackStringView(f_code->co_filename).data();
        func_name_ = THPUtils_unpackStringView(f_code->co_name).data();
    }

    size_t get_hash_id()
    {
        return c10::get_hash(line_no_, file_name_, func_name_);
    }

    std::string get_name()
    {
        std::stringstream name_stream;
        name_stream << file_name_ << "(" << line_no_ << "): " << func_name_;
        return name_stream.str();
    }

    int line_no_{0};
    const char* file_name_{nullptr};
    const char* func_name_{nullptr};
};

struct ModuleInfo {
    ModuleInfo() = default;
    explicit ModuleInfo(PyObject* module_class) : moudle_id_(reinterpret_cast<uintptr_t>(module_class))
    {
        auto py_class_name = py::handle(module_class).attr("__class__").attr("__name__");
        module_name_ = at::StringView(py::str(py_class_name));
    }

    std::string get_name()
    {
        std::stringstream name_stream;
        name_stream << module_name_ << MODULE_NAME_DELIMITER << moudle_id_;
        return name_stream.str();
    }

    uintptr_t moudle_id_{0};
    at::StringView module_name_;
};

constexpr size_t max_py_threads = std::numeric_limits<uint8_t>::max() + 1;

class PythonTracer final {
public:
    static void call(Command c);
    static int pyProfileFn(
        PyObject* obj,
        PyFrameObject* frame,
        int what,
        PyObject* arg);

    struct StartPyCall {
        explicit StartPyCall(size_t key)
            : hash_id_(static_cast<uint64_t>(key)), ts_(torch_npu::toolkit::profiler::Utils::GetClockTime()) {}

        uint64_t hash_id_{0};
        uint64_t ts_{0};
    };

private:
    PythonTracer();
    static PythonTracer& singleton();
    static PythonTracer* get_singleton_in_child_thread();

    void start(size_t max_threads = max_py_threads);
    void startOne();
    void stop();
    void stopOne();
    void clear();
    size_t genPyCallHashId(PyFrameObject* frame);
    void recordPyCall(TraceContext* ctx, PyFrameObject* frame);
    void recordCCall(TraceContext* ctx, PyFrameObject* frame, PyObject* arg);
    void recordReturn(TraceContext* ctx, PyFrameObject* frame, TraceTag tag);
    void recordEvent(TraceTag tag, size_t hash_key);
    void recordStartStack(TraceContext* ctx, uint64_t session);
    void reportTraceData();
    void reportHashData();
    void reportParamData();
    std::string trimPrefix(std::string s);

private:
    // Lock order: control_mutex_ -> registration_mutex_ -> data_mutex_ ->
    // ProfilerMgr upload (leaf, never calls back).
    // control_mutex_ serialises start()/stop()/clear() and is held while they
    // run Python; re-entering one of them on the same thread throws.
    // registration_mutex_ and data_mutex_ are only held around container
    // updates, never across Python calls or allocations, so a finalizer or
    // __getattr__ that calls startOne()/stopOne() cannot self-deadlock.
    // Waiting on any of them detaches the thread state (see TracerMutex).
    // All are no-ops with the GIL.
    TracerControlMutex control_mutex_;  // start()/stop()/clear(), interpreter_
    TracerMutex registration_mutex_;    // thread_local_results_, start_py_call_info_, session_
    TracerMutex data_mutex_;            // caches, events_, ctx_tid_map_
    std::atomic<bool> active_{false};
    std::atomic<bool> record_params_{false};
    // Bumped by stop() before it removes the hooks. A startOne() whose session
    // changed while it ran removes the hook it installed itself.
    uint64_t session_{0};
    PyInterpreterState* interpreter_{nullptr};
    std::deque<ThreadLocalResult> thread_local_results_;
    PyObject* module_call_code_{nullptr};
    PyObject* optimizer_call_code_{nullptr};
    std::vector<std::string> func_name_prefixes_;
    std::unordered_map<size_t, PyCallInfo> py_call_cache_;
    std::unordered_map<size_t, at::StringView> pyc_call_cache_;
    std::unordered_map<size_t, ModuleInfo> module_info_cache_;
    std::vector<std::pair<size_t, std::vector<ModuleParam>>> module_param_cache_;
    std::vector<std::pair<size_t, std::vector<OptimizerParam>>> optimizer_param_cache_;
    AppendOnlyList<TraceEvent> events_;
    std::unordered_map<uintptr_t, std::vector<StartPyCall>> start_py_call_info_;
    std::unordered_map<uintptr_t, uint64_t> ctx_tid_map_;
    static std::atomic<bool> instance_created_;
};

std::atomic<bool> torch_npu::profiler::python_tracer::PythonTracer::instance_created_ = false;

PythonTracer& PythonTracer::singleton()
{
    static PythonTracer singleton_;
    instance_created_ = true;
    return singleton_;
}

PythonTracer* PythonTracer::get_singleton_in_child_thread()
{
    if (instance_created_) {
        return &singleton();
    } else {
        return nullptr;
    }
}

PythonTracer::PythonTracer() : active_(false)
{
    pybind11::gil_scoped_acquire gil;
    module_call_code_ = py::module::import("torch.nn")
        .attr("Module")
        .attr("__call__")
        .attr("__code__")
        .ptr();
    optimizer_call_code_ = py::module::import("torch.optim")
        .attr("Optimizer")
        .attr("_optimizer_step_code")
        .attr("__code__")
        .ptr();
    func_name_prefixes_ = py::module::import("torch.profiler.python_tracer")
        .attr("_prefix_regex")()
        .cast<std::vector<std::string>>();
}

// Key for the per-thread maps (start_py_call_info_, ctx_tid_map_). A ctx
// registered for one thread is keyed by its address, which cannot be reused
// within a session (thread_local_results_ holds it until clear()). The shared
// ctx that start() installs on all threads without the GIL is keyed by thread id.
static uintptr_t ctxKey(TraceContext* ctx)
{
    if (ctx != nullptr && ctx->shared_) {
        return static_cast<uintptr_t>(torch_npu::toolkit::profiler::Utils::GetTid());
    }
    return reinterpret_cast<uintptr_t>(ctx);
}

// Record the frames that were already on the current thread's stack before
// profiling started. Must not be called with registration_mutex_ held: the
// frame walk runs Python code. Dropped if the session that registered ctx has
// been stopped meanwhile.
void PythonTracer::recordStartStack(TraceContext* ctx, uint64_t session)
{
    std::vector<THPFrameObjectPtr> current_stack;
    auto frame = PyEval_GetFrame_NPU();
    size_t depth = 0;  // Make sure we can't infinite loop.
    while (frame != nullptr && depth <= STACK_MAX_DEPTH) {
        current_stack.emplace_back(frame);
        frame = PyFrame_GetBack(frame);
        ++depth;
    }
    // record py call before profiler start
    std::vector<StartPyCall> calls;
    calls.reserve(current_stack.size());
    for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
        calls.emplace_back(genPyCallHashId(*it));
    }
    TracerLockGuard reg_guard(registration_mutex_);
    if (!active_.load() || session_ != session) {
        return;
    }
    auto& start_calls = start_py_call_info_[ctxKey(ctx)];
    start_calls.insert(start_calls.end(), calls.begin(), calls.end());
}

void PythonTracer::start(size_t max_threads)
{
    TracerControlGuard control_guard(control_mutex_);
    {
        TracerLockGuard reg_guard(registration_mutex_);
        TORCH_CHECK(thread_local_results_.empty(), "PythonTracer should not have active contexts", PROF_ERROR(ErrCode::INTERNAL));
    }
    TORCH_CHECK(max_threads > 0, "max_threads must be positive, got ", max_threads, PROF_ERROR(ErrCode::VALUE));
    TORCH_CHECK(max_threads <= max_py_threads, "max_threads must be less equal to ", max_py_threads, PROF_ERROR(ErrCode::VALUE));

    bool expected{false};
    bool active = active_.compare_exchange_strong(expected, true);
    if (!active) {
        ASCEND_LOGW("There is already an active PythonTracer. Refusing to register profile functions.");
        return;
    }

    auto config = torch::autograd::profiler::getProfilerConfig();
    const bool record_params = config.report_input_shapes
        && (config.with_stack || config.with_modules)
        && config.profile_memory;
#ifdef Py_GIL_DISABLED
    // Set before any hook is installed: other threads run the hook as soon as
    // it is installed, and a module or optimizer first seen with the flag off
    // never has its params parsed later. Stored every start, so a session
    // without param recording does not inherit the flag from an earlier one.
    record_params_.store(record_params, std::memory_order_release);
#endif

    GilAndRestoreThread gil;
    interpreter_ = PyInterpreterState_Get();
    if (!gil.initial_thread_state()) {
        ASCEND_LOGW("Failed to get main thread state, PythonTracer will not start.");
        return;
    }

#ifdef Py_GIL_DISABLED
    // Other threads keep running, so their thread states cannot be swapped in
    // (attach would spin until they detach) and their stacks cannot be walked.
    // Record only this thread's stack and let CPython install the hook on all
    // threads under a stop-the-world pause. max_threads is not applied here.
    // ctx stays valid below: clear() cannot run while control_mutex_ is held.
    auto* ctx = newTraceContext();
    if (ctx) {
        ctx->shared_ = true;
    }
    uint64_t session = 0;
    {
        TracerLockGuard reg_guard(registration_mutex_);
        thread_local_results_.emplace_back(this, ctx);
        session = session_;
    }
    recordStartStack(ctx, session);
    PyEval_SetProfileAllThreads(PythonTracer::pyProfileFn, (PyObject*)ctx);
#else
    std::vector<PyThreadState*> thread_states = getInterpreterThreads(interpreter_);
    if (thread_states.empty()) {
        ASCEND_LOGW("There is no active thread, PythonTracer will not start.")
        return;
    }
    if (thread_states.size() > max_threads) {
        ASCEND_LOGW("Can only trace %zu thread. %zu are currently active.", max_threads, thread_states.size());
        thread_states.resize(max_threads);
    }

    // Register the tracer in each thread.
    for (const auto thread_state : thread_states) {
        PyThreadState_Swap(thread_state);
        auto* ctx = newTraceContext();
        uint64_t session = 0;
        {
            TracerLockGuard reg_guard(registration_mutex_);
            thread_local_results_.emplace_back(this, ctx);
            session = session_;
        }
        recordStartStack(ctx, session);
        PyEval_SetProfile(PythonTracer::pyProfileFn, (PyObject*)ctx);
    }

    // With the GIL keep the original order and sticky behaviour.
    if (record_params) {
        record_params_ = true;
    }
#endif
}

void PythonTracer::startOne()
{
    if (!active_.load()) {
        return;
    }
    GilAndRestoreThread gil;
    auto thread_state = gil.initial_thread_state();
    // A second pass only runs after a stale install was removed, see below. It retries
    // once: if two full stop/start cycles pass during one startOne, this thread is
    // left untraced for the newest session.
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!active_.load() || thread_state == nullptr || thread_state->c_profilefunc != nullptr) {
            return;
        }
        PyThreadState_Swap(thread_state);
        auto* ctx = newTraceContext();
        // Drops our reference on every exit, including a Python error thrown
        // from recordStartStack.
        struct CtxRef {
            PyObject* obj = nullptr;
            ~CtxRef() { Py_XDECREF(obj); }
        } own;
        uint64_t session = 0;
        bool registered = false;
        {
            TracerLockGuard reg_guard(registration_mutex_);
            // stop() may have run since the check above.
            if (active_.load()) {
                thread_local_results_.emplace_back(this, ctx);
                // Take our own reference before the lock is released: from then on
                // stop() and clear() can run on another thread and drop the list's one.
                Py_XINCREF((PyObject*)ctx);
                own.obj = (PyObject*)ctx;
                session = session_;
                registered = true;
            }
        }
        if (!registered) {
            own.obj = (PyObject*)ctx;  // the creation reference, nobody else holds ctx
            return;
        }
        recordStartStack(ctx, session);
        PyEval_SetProfile(PythonTracer::pyProfileFn, (PyObject*)ctx);
        bool stale = false;
        {
            TracerLockGuard reg_guard(registration_mutex_);
            stale = session_ != session;
        }
        // stop() bumps session_ before it removes the hooks, so if it did not
        // change, stop() has not removed hooks yet and will remove ours too.
        // Otherwise our hook may have been installed after that removal: take it
        // off again. If a new session started meanwhile, our install replaced its
        // hook on this thread, so the next pass registers for the new session.
        if (stale && thread_state->c_profileobj == (PyObject*)ctx) {
            PyEval_SetProfile(nullptr, nullptr);
        }
        if (!stale) {
            return;
        }
    }
}

void PythonTracer::stop()
{
    TracerControlGuard control_guard(control_mutex_);
    TORCH_INTERNAL_ASSERT(active_.load(), "PythonTracer is not running.", PROF_ERROR(ErrCode::INTERNAL));

    GilAndRestoreThread gil;
    {
        // Both before the hooks are removed, in one step, so that a startOne()
        // either registered in this session and sees session_ change, or sees
        // active_ false and registers nothing; see startOne(). Hooks that still run
        // drop their events from here on. With the GIL, holding it orders this.
        TracerLockGuard reg_guard(registration_mutex_);
        active_ = false;
        ++session_;
    }
#ifdef Py_GIL_DISABLED
    // Another running thread's profile function cannot be read safely, so this
    // clears it on every thread, including one set by a different profiler.
    // CPython does it under a stop-the-world pause: once this returns no thread
    // can enter pyProfileFn with our ctx any more.
    PyEval_SetProfileAllThreads(nullptr, nullptr);
#else
    for (const auto thread_state : getInterpreterThreads(interpreter_)) {
        if (thread_state->c_profilefunc == &PythonTracer::pyProfileFn) {
            PyThreadState_Swap(thread_state);
            PyEval_SetProfile(nullptr, nullptr);
        }
    }
#endif
    // A hook that was already running may still be inside a record call; it
    // re-checks active_ under data_mutex_ and drops its event. No Python code
    // runs below.
    TracerLockGuard reg_guard(registration_mutex_);
    TracerLockGuard data_guard(data_mutex_);
    for (const auto& start_py_call : start_py_call_info_) {
        auto ctx_tid = ctx_tid_map_.find(start_py_call.first);
        if (ctx_tid != ctx_tid_map_.end()) {
            for (const auto& py_call : start_py_call.second) {
                events_.emplace_back(
                    ctx_tid->second,
                    py_call.ts_,
                    py_call.hash_id_,
                    TraceTag::kPy_Call);
            }
        }
    }
    reportTraceData();
    reportHashData();
    reportParamData();
}

void PythonTracer::stopOne()
{
    // Touches only this thread's own state, so no tracer lock is needed.
    if (!active_.load()) {
        return;
    }
    GilAndRestoreThread gil;
    auto thread_state = gil.initial_thread_state();
    if (thread_state && thread_state->c_profilefunc == &PythonTracer::pyProfileFn) {
        PyThreadState_Swap(thread_state);
        PyEval_SetProfile(nullptr, nullptr);
    }
}

void PythonTracer::clear()
{
    std::deque<ThreadLocalResult> results;
    {
        TracerControlGuard control_guard(control_mutex_);
        {
            TracerLockGuard reg_guard(registration_mutex_);
            TORCH_CHECK(!active_.load(), "Cannot clear state while PythonTracer is active.", PROF_ERROR(ErrCode::INTERNAL));
            results.swap(thread_local_results_);
            start_py_call_info_.clear();
        }
        {
            TracerLockGuard data_guard(data_mutex_);
            py_call_cache_.clear();
            pyc_call_cache_.clear();
            module_info_cache_.clear();
            module_param_cache_.clear();
            optimizer_param_cache_.clear();
            events_.clear();
            ctx_tid_map_.clear();
        }
        interpreter_ = nullptr;
    }
    if (!results.empty()) {
        // ~ThreadLocalResult drops a Python reference, which can run finalizers;
        // do it with no tracer lock held. The caller (stopNpuProfiler) has
        // released the GIL, so take it back here.
        pybind11::gil_scoped_acquire gil;
        results.clear();
    }
}

std::string PythonTracer::trimPrefix(std::string s)
{
    for (const auto& p : func_name_prefixes_) {
        if (s.compare(0, p.size(), p) == 0) {
            s.erase(0, p.size());
            return s;
        }
    }
    return s;
}

// reportTraceData/reportHashData/reportParamData and recordEvent: caller holds
// data_mutex_. Uploading under it keeps batches in order and none after stop().
void PythonTracer::reportTraceData()
{
    if (events_.size() > 0) {
        ProfilerMgr::GetInstance()->UploadTraceEventData(
            std::make_unique<torch_npu::toolkit::profiler::PythonTracerFuncData>(
                torch_npu::toolkit::profiler::Utils::GetPid(),
                std::move(events_)
            )
        );
        events_.clear();
    }
}

void PythonTracer::reportHashData()
{
    std::vector<std::pair<uint64_t, std::string>> hash_data;
    hash_data.resize(py_call_cache_.size() + pyc_call_cache_.size() + module_info_cache_.size() + 1);
    size_t idx = 0;
    for (auto& item : py_call_cache_) {
        hash_data[idx++] = std::make_pair(item.first, trimPrefix(std::move(item.second.get_name())));
    }
    for (auto& item : pyc_call_cache_) {
        hash_data[idx++] = std::make_pair(item.first, std::string(item.second.str()));
    }
    for (auto& item : module_info_cache_) {
        hash_data[idx++] = std::make_pair(item.first, item.second.get_name());
    }
    hash_data[idx] = std::make_pair(EXIT_EVENT_HASH_ID, EXIT_EVENT_DESC);

    ProfilerMgr::GetInstance()->UploadTraceHashData(
        std::make_unique<torch_npu::toolkit::profiler::PythonTracerHashData>(
            hash_data
        )
    );
}

void PythonTracer::reportParamData()
{
    if (module_param_cache_.size() > 0 || optimizer_param_cache_.size() > 0) {
        ProfilerMgr::GetInstance()->UploadParamData(
            std::make_unique<torch_npu::toolkit::profiler::ParamTensorData>(
                std::move(module_param_cache_),
                std::move(optimizer_param_cache_)
            )
        );
    }
    module_param_cache_.clear();
    optimizer_param_cache_.clear();
}

void PythonTracer::recordEvent(TraceTag tag, size_t hash_key)
{
    events_.emplace_back(
        torch_npu::toolkit::profiler::Utils::GetTid(),
        torch_npu::toolkit::profiler::Utils::GetClockTime(),
        hash_key,
        tag);
    if (events_.size() >= TRACE_DUMP_THRESHOLD) {
        reportTraceData();
    }
}

static TensorMetadata toTensorMetadata(PyObject* self)
{
    if (!THPVariable_CheckExact(self)) {
        TensorMetadata m;
        return m;
    }
    const auto& t = THPVariable_Unpack(self);
    TensorMetadata m{t};
    return m;
}

static c10::optional<TensorMetadata> recordIfTensor(py::handle p)
{
    return THPVariable_CheckExact(p.ptr())
        ? c10::optional<TensorMetadata>(toTensorMetadata(p.ptr()))
        : c10::nullopt;
}

static std::vector<std::pair<std::string, TensorMetadata>> unpackTensorMap(const py::dict& tensor_map)
{
    std::vector<std::pair<std::string, TensorMetadata>> out;
    for (auto& it : tensor_map) {
        auto* value = it.second.ptr();
        if (py::isinstance<py::str>(it.first) && THPVariable_CheckExact(value)) {
            out.emplace_back(py::cast<std::string>(it.first), toTensorMetadata(value));
        }
    }
    return out;
}

static void parseModuleParams(
    std::vector<std::pair<size_t, std::vector<ModuleParam>>> &module_param_cache,
    PyObject* cls,
    size_t hash_id)
{
    std::vector<ModuleParam> module_params;
    py::dict params = py::handle(cls).attr("_parameters");
    for (auto& it : params) {
        auto* p = it.second.ptr();
        if (py::isinstance<py::str>(it.first) && THPVariable_CheckExact(p)) {
            module_params.emplace_back(
                ModuleParam{
                    it.first.cast<std::string>(),
                    toTensorMetadata(p),
                    recordIfTensor(py::getattr(it.second, "grad", py::none()))});
        }
    }
    if (module_params.size() > 0) {
        module_param_cache.emplace_back(std::move(std::make_pair(hash_id, module_params)));
    }
}

static void parseOptimizerParams(
    std::vector<std::pair<size_t, std::vector<OptimizerParam>>> &optimizer_param_cache,
    PyObject* cls,
    size_t hash_id)
{
    std::vector<OptimizerParam> optimizer_params;
    const py::handle self{cls};
    for (const auto& it : (py::list)self.attr("param_groups")) {
        for (auto& param : py::cast<py::dict>(it).attr("get")("params")) {
            if (THPVariable_CheckExact(param.ptr())) {
                optimizer_params.emplace_back(
                    OptimizerParam{
                        toTensorMetadata(param.ptr()),
                        recordIfTensor(py::getattr(param, "grad", py::none())),
                        unpackTensorMap(py::cast<py::dict>(self.attr("state")).attr("get")(param, py::dict()))});
            }
        }
    }
    if (optimizer_params.size() > 0) {
        optimizer_param_cache.emplace_back(std::move(std::make_pair(hash_id, optimizer_params)));
    }
}

size_t PythonTracer::genPyCallHashId(PyFrameObject* frame)
{
    TORCH_INTERNAL_ASSERT(frame != nullptr, "frame can not be nullptr.", PTA_ERROR(ErrCode::PARAM));

    // Python calls run outside data_mutex_: they can run arbitrary code.
    auto call_info = PyCallInfo(frame);
    auto hash_id = call_info.get_hash_id();
    auto f_code = PyFrame_GetCode_NPU(frame);
    bool new_call = false;
    {
        TracerLockGuard data_guard(data_mutex_);
        if (active_.load(std::memory_order_relaxed) && py_call_cache_.find(hash_id) == py_call_cache_.end()) {
            py_call_cache_.insert({hash_id, call_info});
            new_call = true;
        }
    }

    // check optim.Optimizer call
    if (new_call && record_params_.load(std::memory_order_relaxed)
        && ((PyObject*)f_code.get() == optimizer_call_code_)) {
        auto f_locals = PyFrame_GetLocals_NPU(frame);
        auto optimizer_class = PyDict_GetItemString(f_locals, "self");
        std::vector<std::pair<size_t, std::vector<OptimizerParam>>> optimizer_params;
        parseOptimizerParams(optimizer_params, (PyObject*)optimizer_class, hash_id);
        if (!optimizer_params.empty()) {
            TracerLockGuard data_guard(data_mutex_);
            if (active_.load(std::memory_order_relaxed)) {
                for (auto& item : optimizer_params) {
                    optimizer_param_cache_.emplace_back(std::move(item));
                }
            }
        }
    }

    // check nn.Module call
    if ((PyObject*)f_code.get() == module_call_code_) {
        auto f_locals = PyFrame_GetLocals_NPU(frame);
        auto module_class = PyDict_GetItemString(f_locals, "self");
        hash_id = c10::get_hash(module_class);
        bool known = false;
        {
            TracerLockGuard data_guard(data_mutex_);
            known = module_info_cache_.find(hash_id) != module_info_cache_.end();
        }
        if (!known) {
            ModuleInfo module_info(module_class);
            bool inserted = false;
            {
                TracerLockGuard data_guard(data_mutex_);
                if (active_.load(std::memory_order_relaxed)) {
                    inserted = module_info_cache_.insert({hash_id, module_info}).second;
                }
            }
            if (inserted && record_params_.load(std::memory_order_relaxed)) {
                std::vector<std::pair<size_t, std::vector<ModuleParam>>> module_params;
                parseModuleParams(module_params, (PyObject*)module_class, hash_id);
                if (!module_params.empty()) {
                    TracerLockGuard data_guard(data_mutex_);
                    if (active_.load(std::memory_order_relaxed)) {
                        for (auto& item : module_params) {
                            module_param_cache_.emplace_back(std::move(item));
                        }
                    }
                }
            }
        }
    }
    return hash_id;
}

void PythonTracer::recordPyCall(TraceContext* ctx, PyFrameObject* frame)
{
    auto hash_id = genPyCallHashId(frame);
    TracerLockGuard data_guard(data_mutex_);
    if (!active_.load(std::memory_order_relaxed)) {
        return;
    }
    recordEvent(TraceTag::kPy_Call, hash_id);
}

void PythonTracer::recordCCall(TraceContext* ctx, PyFrameObject* frame, PyObject* arg)
{
    std::string call_info = py::repr(arg);
    auto hash_id = c10::get_hash(call_info);
    TracerLockGuard data_guard(data_mutex_);
    if (!active_.load(std::memory_order_relaxed)) {
        return;
    }
    if (pyc_call_cache_.find(hash_id) == pyc_call_cache_.end()) {
        pyc_call_cache_.insert({hash_id, at::StringView(std::move(call_info))});
    }
    recordEvent(TraceTag::kC_Call, hash_id);
}

void PythonTracer::recordReturn(TraceContext* ctx, PyFrameObject* frame, TraceTag tag)
{
    // Read ctx before taking the lock: waiting on it detaches this thread, and
    // stop() and clear() can then remove the hook and free ctx.
    auto ctx_addr = ctxKey(ctx);
    TracerLockGuard data_guard(data_mutex_);
    if (!active_.load(std::memory_order_relaxed)) {
        return;
    }
    recordEvent(tag, EXIT_EVENT_HASH_ID);

    // record ctx to thread id map
    if (ctx_tid_map_.find(ctx_addr) == ctx_tid_map_.end()) {
        ctx_tid_map_.insert({ctx_addr, torch_npu::toolkit::profiler::Utils::GetTid()});
    }
}

int PythonTracer::pyProfileFn(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg)
{
    auto ctx = reinterpret_cast<TraceContext*>(obj);
    if (ctx == nullptr) {
        return 0;
    }
    auto active_tracer = ctx->tracer_.load(std::memory_order_acquire);
    if (active_tracer == nullptr || !active_tracer->active_.load(std::memory_order_relaxed)) {
        return 0;
    }
    switch (what) {
        case PyTrace_CALL:
            active_tracer->recordPyCall(ctx, frame);
            break;

        case PyTrace_C_CALL:
            active_tracer->recordCCall(ctx, frame, arg);
            break;

        case PyTrace_EXCEPTION:
        case PyTrace_RETURN:
            active_tracer->recordReturn(ctx, frame, TraceTag::kPy_Return);
            break;

        case PyTrace_C_EXCEPTION:
        case PyTrace_C_RETURN:
            active_tracer->recordReturn(ctx, frame, TraceTag::kC_Return);
            break;

        default:
            break;
    }
    return 0;
}

void PythonTracer::call(Command c)
{
    switch (c) {
        case Command::kStartOne:
            if (auto tracer = get_singleton_in_child_thread()) {
                tracer->startOne();
            }
            break;

        case Command::kStartAll:
            PythonTracer::singleton().start();
            break;

        case Command::kStop:
            PythonTracer::singleton().stop();
            break;

        case Command::kStopOne:
            if (auto tracer = get_singleton_in_child_thread()) {
                tracer->stopOne();
            }
            break;

        case Command::kClear:
            PythonTracer::singleton().clear();
            break;

        default:
            break;
    }
};

void init()
{
    pybind11::gil_scoped_acquire gil;
    TORCH_CHECK(PyType_Ready(&TraceContextType) == 0, PROF_ERROR(ErrCode::INTERNAL));
    registerFunctions(
        &PythonTracer::call
    );
}
} // python_tracer
} // profiler
} // torch_npu
