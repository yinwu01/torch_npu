// Minimal stand-ins for CPython / pybind11 / torch / torch_npu so that
// profiler_python.cpp compiles and runs standalone. Emulates free-threaded
// attach/detach and stop-the-world with a writer-preferring reader/writer lock.
#pragma once
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "torch_npu/csrc/profiler/containers.h"
#include "torch_npu/csrc/profiler/profiler_python.h"

typedef long Py_ssize_t;
struct PyTypeObject;
struct PyObject { std::atomic<long> ob_refcnt; PyTypeObject* ob_type; };
struct PyVarObject { PyObject ob_base; Py_ssize_t ob_size; };
#define PyObject_HEAD PyObject ob_base;
#define PyVarObject_HEAD_INIT(type, size) { {{1}, type}, size },
typedef PyObject* (*allocfunc)(PyTypeObject*, Py_ssize_t);
typedef PyObject* (*newfunc)(PyTypeObject*, PyObject*, PyObject*);
struct PyTypeObject {
    PyVarObject ob_base;
    const char* tp_name; Py_ssize_t tp_basicsize; Py_ssize_t tp_itemsize;
    void* tp_dealloc; Py_ssize_t tp_vectorcall_offset;
    void* tp_getattr; void* tp_setattr; void* tp_as_async; void* tp_repr;
    void* tp_as_number; void* tp_as_sequence; void* tp_as_mapping; void* tp_hash;
    void* tp_call; void* tp_str; void* tp_getattro; void* tp_setattro; void* tp_as_buffer;
    unsigned long tp_flags; const char* tp_doc;
    void* tp_traverse; void* tp_clear; void* tp_richcompare; Py_ssize_t tp_weaklistoffset;
    void* tp_iter; void* tp_iternext; void* tp_methods; void* tp_members; void* tp_getset;
    void* tp_base; void* tp_dict; void* tp_descr_get; void* tp_descr_set; Py_ssize_t tp_dictoffset;
    void* tp_init; allocfunc tp_alloc; newfunc tp_new; void* tp_free;
};
#define Py_TPFLAGS_DEFAULT 0UL
inline PyObject* PyType_GenericNew(PyTypeObject*, PyObject*, PyObject*) { return nullptr; }
inline PyObject* stub_alloc(PyTypeObject* t, Py_ssize_t)
{
    auto* o = (PyObject*)calloc(1, t->tp_basicsize);
    o->ob_refcnt.store(1);
    o->ob_type = t;
    return o;
}
inline int PyType_Ready(PyTypeObject* t) { t->tp_alloc = stub_alloc; return 0; }
inline void Py_INCREF(PyObject* o) { o->ob_refcnt.fetch_add(1); }
inline void Py_DECREF(PyObject* o) { if (o->ob_refcnt.fetch_sub(1) == 1) { free(o); } }
inline void Py_XINCREF(PyObject* o) { if (o) Py_INCREF(o); }
inline void Py_XDECREF(PyObject* o) { if (o) Py_DECREF(o); }

// ---- emulated attach/detach and stop-the-world ----
// Attached threads hold it shared; a stop-the-world holds it exclusive. Writer-preferring
// like CPython: once a stop-the-world is requested, threads that try to attach wait for it
// (std::shared_mutex on glibc prefers readers, so busy threads could starve it forever).
class StubWorld {
public:
    void lock_shared()
    {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [this] { return !writer_ && writers_waiting_ == 0; });
        readers_++;
    }
    void unlock_shared()
    {
        std::lock_guard<std::mutex> l(m_);
        if (--readers_ == 0) cv_.notify_all();
    }
    void lock()
    {
        std::unique_lock<std::mutex> l(m_);
        writers_waiting_++;
        cv_.wait(l, [this] { return !writer_ && readers_ == 0; });
        writers_waiting_--;
        writer_ = true;
    }
    void unlock()
    {
        std::lock_guard<std::mutex> l(m_);
        writer_ = false;
        cv_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    int readers_{0};
    int writers_waiting_{0};
    bool writer_{false};
};
extern StubWorld g_world;
extern thread_local int t_attached;
inline void stub_attach() { g_world.lock_shared(); t_attached = 1; }
inline void stub_detach() { t_attached = 0; g_world.unlock_shared(); }

#ifdef Py_GIL_DISABLED
struct PyMutex { std::mutex m; };
inline void PyMutex_Lock(PyMutex* m)
{
    if (m->m.try_lock()) return;
    if (t_attached) { stub_detach(); m->m.lock(); stub_attach(); }
    else { m->m.lock(); }
}
inline void PyMutex_Unlock(PyMutex* m) { m->m.unlock(); }
#endif
inline unsigned long PyThread_get_thread_ident()
{
    return static_cast<unsigned long>(std::hash<std::thread::id>{}(std::this_thread::get_id())) | 1UL;
}

struct PyFrameObject { PyObject ob; int line; PyFrameObject* back; };
struct PyCodeObject { PyObject ob; PyObject* co_filename; PyObject* co_name; };
struct PyInterpreterState { int dummy; };
struct PyCodeObject;
extern PyCodeObject g_code;
typedef int (*Py_tracefunc)(PyObject*, PyFrameObject*, int, PyObject*);
struct PyThreadState { Py_tracefunc c_profilefunc; PyObject* c_profileobj; };
#define PyTrace_CALL 0
#define PyTrace_EXCEPTION 1
#define PyTrace_LINE 2
#define PyTrace_RETURN 3
#define PyTrace_C_CALL 4
#define PyTrace_C_EXCEPTION 5
#define PyTrace_C_RETURN 6
extern PyInterpreterState g_interp;
extern thread_local PyThreadState t_tstate;
extern PyCodeObject g_code;
inline PyInterpreterState* PyInterpreterState_Get() { return &g_interp; }
inline PyThreadState* PyInterpreterState_ThreadHead(PyInterpreterState*) { return &t_tstate; }
inline PyThreadState* PyThreadState_Next(PyThreadState*) { return nullptr; }
inline PyThreadState* PyThreadState_Get() { return &t_tstate; }
inline PyThreadState* PyThreadState_Swap(PyThreadState* t) { return t; }
inline int Py_IsInitialized() { return 1; }
// Defined by the test: whether this PyEval_SetProfile call pauses detached, and for how long.
long stub_set_profile_pause_us();
inline void PyEval_SetProfile(Py_tracefunc f, PyObject* o)
{
#ifdef Py_GIL_DISABLED
    // 3.14 updates the hook under a stop-the-world pause, during which the caller is
    // detached and stop()/clear()/start() on other threads can run to completion.
    if (t_attached) {
        long us = stub_set_profile_pause_us();
        if (us >= 0) {
            stub_detach();
            std::this_thread::sleep_for(std::chrono::microseconds(us));
            stub_attach();
        }
    }
#endif
    Py_XINCREF(o);
    PyObject* old = t_tstate.c_profileobj;
    t_tstate.c_profilefunc = f;
    t_tstate.c_profileobj = o;
    Py_XDECREF(old);
}
// Caller is attached. Emulates CPython's stop-the-world around the update.
extern std::vector<PyThreadState*> g_tstates;
inline void PyEval_SetProfileAllThreads(Py_tracefunc f, PyObject* o)
{
    stub_detach();
    {
        std::unique_lock<StubWorld> stw(g_world);
        for (auto* ts : g_tstates) {
            Py_XINCREF(o);
            PyObject* old = ts->c_profileobj;
            ts->c_profilefunc = f;
            ts->c_profileobj = o;
            Py_XDECREF(old);
        }
    }
    stub_attach();
}
// Emulates user Python code (a __getattr__, audit hook or finalizer) that runs
// while the tracer inspects a frame. Only runs on threads that set t_frame_callback.
extern thread_local void (*t_frame_callback)();
inline int PyFrame_GetLineNumber(PyFrameObject* f)
{
    if (t_frame_callback != nullptr) {
        auto cb = t_frame_callback;
        t_frame_callback = nullptr;  // no recursion
        cb();
        t_frame_callback = cb;
    }
    return f->line;
}
inline PyCodeObject* PyFrame_GetCode(PyFrameObject*) { return &g_code; }
inline PyFrameObject* PyFrame_GetBack(PyFrameObject* f) { return f->back; }
extern thread_local PyFrameObject* t_current_frame;
inline PyFrameObject* PyEval_GetFrame() { return t_current_frame; }
inline PyObject* PyFrame_GetLocals(PyFrameObject*) { return nullptr; }
inline PyObject* PyDict_GetItemString(PyObject*, const char*) { return nullptr; }

// ---- THP smart pointers (no refcounting needed in the stub) ----
template <class T> struct THPPointer {
    T* p{nullptr};
    THPPointer() = default;
    THPPointer(T* x) : p(x) {}
    T* get() const { return p; }
    T* operator->() const { return p; }
    operator T*() const { return p; }
};
using THPObjectPtr = THPPointer<PyObject>;
using THPCodeObjectPtr = THPPointer<PyCodeObject>;
using THPFrameObjectPtr = THPPointer<PyFrameObject>;
struct StubStringView { const char* s; const char* data() const { return s; } };
inline StubStringView THPUtils_unpackStringView(PyObject*) { return {"file.py"}; }

// ---- pybind11 ----
namespace pybind11 {
struct object;
struct object_iter;
struct object {
    object() = default;
    object(PyObject*) {}
    object attr(const char*) const { return {}; }
    template <class... A> object operator()(A&&...) const { return {}; }
    PyObject* ptr() const { return (PyObject*)&g_code; }
    template <class T> T cast() const { return T{}; }
    operator std::string() const { return "repr"; }
    struct iter {
        bool operator!=(const iter&) const { return false; }
        void operator++() {}
        object& operator*() const { static object o; return o; }
    };
    iter begin() const { return {}; }
    iter end() const { return {}; }
};
struct pair_t { object first, second; };
using handle = object;
struct dict : object {
    using object::object;
    dict(const object&) {}
    struct iter {
        bool operator!=(const iter&) const { return false; }
        void operator++() {}
        pair_t& operator*() const { static pair_t p; return p; }
    };
    iter begin() const { return {}; }
    iter end() const { return {}; }
};
struct list : object { using object::object; list(const object&) {} };
struct str : object { using object::object; str(const object&) {} };
struct none : object {};
struct module : object { static object import(const char*) { return {}; } };
template <class T> bool isinstance(const object&) { return false; }
template <class T> T cast(const object&) { return T{}; }
inline object getattr(const object&, const char*, const object&) { return {}; }
inline std::string repr(PyObject*) { return "repr"; }
struct gil_scoped_acquire {
    bool took{false};
    gil_scoped_acquire() { if (!t_attached) { stub_attach(); took = true; } }
    ~gil_scoped_acquire() { if (took) stub_detach(); }
    void disarm() {}
};
}
namespace py = pybind11;

// ---- c10 / at / torch ----
namespace c10 {
template <class... A> size_t get_hash(const A&... a)
{
    std::ostringstream os;
    ((os << a << '|'), ...);
    return std::hash<std::string>{}(os.str());
}
template <class T> using optional = std::optional<T>;
constexpr auto nullopt = std::nullopt;
}
namespace at {
struct StringView {
    std::string s;
    StringView() = default;
    StringView(std::string x) : s(std::move(x)) {}
    StringView(const py::object&) : s("cls") {}
    const char* str() const { return s.c_str(); }
};
inline std::ostream& operator<<(std::ostream& o, const StringView& v) { return o << v.s; }
struct Tensor {};
}
inline bool THPVariable_CheckExact(PyObject*) { return false; }
inline at::Tensor THPVariable_Unpack(PyObject*) { return {}; }
namespace torch { namespace autograd { namespace profiler {
struct ProfilerConfig { bool report_input_shapes{true}, with_stack{true}, with_modules{false}, profile_memory{true}; };
inline ProfilerConfig getProfilerConfig() { return {}; }
}}}
enum class ErrCode { INTERNAL, VALUE, PARAM };
#define PROF_ERROR(x) ""
#define PTA_ERROR(x) ""
#define TORCH_CHECK(cond, ...) do { if (!(cond)) throw std::runtime_error("check"); } while (0)
#define TORCH_INTERNAL_ASSERT(cond, ...) TORCH_CHECK(cond)
#define ASCEND_LOGW(...) do {} while (0);

namespace torch_npu {
namespace toolkit { namespace profiler {
struct TensorMetadata { TensorMetadata() = default; explicit TensorMetadata(const at::Tensor&) {} };
struct ModuleParam { std::string name; TensorMetadata m; std::optional<TensorMetadata> g; };
struct OptimizerParam { TensorMetadata m; std::optional<TensorMetadata> g; std::vector<std::pair<std::string, TensorMetadata>> s; };
struct PythonTracerFuncData {
    torch_npu::profiler::AppendOnlyList<torch_npu::profiler::python_tracer::TraceEvent> events;
    PythonTracerFuncData(uint64_t, torch_npu::profiler::AppendOnlyList<torch_npu::profiler::python_tracer::TraceEvent>&& e)
        : events(std::move(e)) {}
};
struct PythonTracerHashData { std::vector<std::pair<uint64_t, std::string>> d; PythonTracerHashData(std::vector<std::pair<uint64_t, std::string>> x) : d(std::move(x)) {} };
struct ParamTensorData {
    ParamTensorData(std::vector<std::pair<uint64_t, std::vector<ModuleParam>>>, std::vector<std::pair<uint64_t, std::vector<OptimizerParam>>>) {}
};
struct Utils {
    static uint64_t GetTid() { return std::hash<std::thread::id>{}(std::this_thread::get_id()); }
    static uint64_t GetClockTime() { return 1; }
    static uint64_t GetPid() { return 1; }
};
}}
namespace profiler {
struct ProfilerMgr {
    static ProfilerMgr* GetInstance() { static ProfilerMgr m; return &m; }
    std::atomic<size_t> events{0}, hashes{0};
    void UploadTraceEventData(std::unique_ptr<torch_npu::toolkit::profiler::PythonTracerFuncData> d) { events += d->events.size(); }
    void UploadTraceHashData(std::unique_ptr<torch_npu::toolkit::profiler::PythonTracerHashData>) { hashes++; }
    void UploadParamData(std::unique_ptr<torch_npu::toolkit::profiler::ParamTensorData>) {}
};
namespace python_tracer {
enum class Command { kStartOne = 0, kStartAll, kStop, kStopOne, kClear };
using CallFn = void (*)(Command);
void registerFunctions(CallFn call);
}
}
}
