#include <unistd.h>

#include <vector>
#include <thread>
#include <map>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

#include "torch_npu/csrc/toolkit/profiler/inc/data_dumper.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace torch_npu {
namespace toolkit {
namespace profiler {
DataDumper::DataDumper() : path_(""), start_(false), init_(false) {}

DataDumper::~DataDumper()
{
    UnInit();
}

void DataDumper::Init(const std::string &path, size_t capacity = kDefaultRingBuffer)
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (start_.load()) {
        ASCEND_LOGE("DataDumper is still running on %s, stop it and restart on %s.", path_.c_str(), path.c_str());
    }
    // Drain to the old path and close its files so the new path starts clean.
    UnInitLocked();
    path_ = path;
    create_dir_flag_ = true;
    data_chunk_buf_.Init(capacity);
    init_.store(true);
}

void DataDumper::UnInit()
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    UnInitLocked();
}

void DataDumper::UnInitLocked()
{
    // Never free the buffer while producers or the dump thread may still use it.
    StopLocked();
    if (init_.load()) {
        init_.store(false);
        data_chunk_buf_.UnInit();
        for (auto &f : fd_map_) {
            if (f.second != nullptr) {
                fclose(f.second);
                f.second = nullptr;
            }
        }
        fd_map_.clear();
    }
}

void DataDumper::Start()
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!init_.load() || start_.load()) {
        return;
    }
    // Set start_ before the thread exists so Run() does not see false and exit at once.
    // control_mutex_ keeps Stop() from joining before pthread_create has set pid_.
    start_.store(true);
    if (Thread::Start() != 0) {
        start_.store(false);
        WaitWriters();
    }
}

void DataDumper::Stop()
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    StopLocked();
}

void DataDumper::StopLocked()
{
    // Order: stop accepting data, wait for in-flight Report calls, join, then drain.
    // The drain runs with control_mutex_ held and the dump thread joined (or never
    // started), so it is the only consumer.
    if (start_.exchange(false)) {
        WaitWriters();
        Thread::Stop();
    }
    Flush();
}

void DataDumper::WaitWriters()
{
    // Pairs with Report(): writers_ is raised before start_ is read (both seq_cst), so after
    // start_ is false and writers_ reads 0, no Report call can still touch the buffer.
    while (writers_.load() != 0) {
        std::this_thread::yield();
    }
}

size_t DataDumper::GatherAndDumpData()
{
    std::map<std::string, std::vector<uint8_t>> dataMap;
    uint64_t batchSize = 0;
    size_t count = 0;
    while (batchSize < kBatchMaxLen) {
        std::unique_ptr<BaseReportData> data = nullptr;
        if (!data_chunk_buf_.Pop(data) || data == nullptr) {
            break;
        }
        count++;
        std::vector<uint8_t> encodeData = data->encode();
        batchSize += encodeData.size();
        const std::string &key = data->tag;
        auto iter = dataMap.find(key);
        if (iter == dataMap.end()) {
            dataMap.insert({ key, encodeData });
        } else {
            iter->second.insert(iter->second.end(), encodeData.cbegin(), encodeData.cend());
        }
    }
    if (dataMap.size() > 0) {
        if (create_dir_flag_) {
            create_dir_flag_ = !Utils::CreateDir(this->path_);
        }
        Dump(dataMap);
    }
    return count;
}

void DataDumper::Run()
{
    for (;;) {
        if (!start_.load()) {
            break;
        }
        // Size() also counts claimed slots that are not published yet, so sleep when
        // nothing could be popped instead of spinning on Size().
        if (data_chunk_buf_.Size() <= kNotifyInterval || GatherAndDumpData() == 0) {
            usleep(kMaxWaitTimeUs);
        }
    }
}

void DataDumper::Flush()
{
    // Loop on Pop results, not Size(), so a half-written slot can never make this spin.
    while (GatherAndDumpData() > 0) {
    }
}

void DataDumper::Report(std::unique_ptr<BaseReportData> data)
{
    // Cheap early exit when not running; the real check is the one after fetch_add.
    if (C10_UNLIKELY(data == nullptr || !start_.load(std::memory_order_relaxed))) {
        return;
    }
    // Register as a writer before checking start_ again; see WaitWriters().
    writers_.fetch_add(1);
    if (C10_LIKELY(start_.load())) {
        data_chunk_buf_.Push(std::move(data));
    }
    writers_.fetch_sub(1, std::memory_order_release);
}

void DataDumper::Dump(const std::map<std::string, std::vector<uint8_t>> &dataMap)
{
    for (auto &data : dataMap) {
        FILE *fd = nullptr;
        const std::string dump_file = path_ + "/" + data.first;
        auto iter = fd_map_.find(dump_file);
        if (iter == fd_map_.end()) {
            if (!Utils::IsFileExist(dump_file) && !Utils::CreateFile(dump_file)) {
                ASCEND_LOGE("DataDumper create file failed: %s", dump_file.c_str());
                continue;
            }
            fd = fopen(dump_file.c_str(), "ab");
            if (fd == nullptr) {
                ASCEND_LOGE("DataDumper open file failed: %s", dump_file.c_str());
                continue;
            }
            fd_map_.insert({ dump_file, fd });
        } else {
            fd = iter->second;
        }
        fwrite(reinterpret_cast<const char *>(data.second.data()), sizeof(char), data.second.size(), fd);
    }
}

TraceDataDumper::TraceDataDumper() : path_(""), start_(false), init_(false), trace_hash_data_(nullptr) {}

TraceDataDumper::~TraceDataDumper()
{
    UnInit();
}

void TraceDataDumper::Init(const std::string &path, size_t capacity)
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (start_.load()) {
        ASCEND_LOGE("TraceDataDumper is still running on %s, stop it and restart on %s.", path_.c_str(), path.c_str());
    }
    // Drain to the old path and close its files so the new path starts clean.
    UnInitLocked();
    path_ = path;
    create_dir_flag_ = true;
    trace_data_buf_.Init(capacity);
    init_.store(true);
}

void TraceDataDumper::UnInit()
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    UnInitLocked();
}

void TraceDataDumper::UnInitLocked()
{
    // Never free the buffer while producers or the dump thread may still use it.
    StopLocked();
    if (init_.load()) {
        init_.store(false);
        trace_data_buf_.UnInit();
        for (auto &f : fd_map_) {
            if (f.second != nullptr) {
                fclose(f.second);
                f.second = nullptr;
            }
        }
        fd_map_.clear();
    }
}

void TraceDataDumper::Start()
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!init_.load() || start_.load()) {
        return;
    }
    // Set start_ before the thread exists so Run() does not see false and exit at once.
    // control_mutex_ keeps Stop() from joining before pthread_create has set pid_.
    start_.store(true);
    if (Thread::Start() != 0) {
        start_.store(false);
        WaitWriters();
    }
}

void TraceDataDumper::Stop()
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    StopLocked();
}

void TraceDataDumper::StopLocked()
{
    // Order: stop accepting data, wait for in-flight Report calls, join, then drain.
    // The drain runs with control_mutex_ held and the dump thread joined (or never
    // started), so it is the only consumer.
    if (start_.exchange(false)) {
        WaitWriters();
        Thread::Stop();
    }
    while (FlushTraceData()) {
    }
    FlushHashData();
    FlushParamData();
}

void TraceDataDumper::WaitWriters()
{
    // Same protocol as DataDumper::WaitWriters().
    while (writers_.load() != 0) {
        std::this_thread::yield();
    }
}

void TraceDataDumper::Run()
{
    while (true) {
        if (!start_.load()) {
            break;
        }
        // Drive by Pop results: a claimed but unpublished slot must not cause a spin.
        if (!FlushTraceData()) {
            usleep(kMaxWaitTimeUs);
        }
    }
}

void TraceDataDumper::Report(std::unique_ptr<PythonTracerFuncData> data)
{
    // Cheap early exit when not running; the real check is the one after fetch_add.
    if (C10_UNLIKELY(data == nullptr || !start_.load(std::memory_order_relaxed))) {
        return;
    }
    writers_.fetch_add(1);
    if (C10_LIKELY(start_.load())) {
        trace_data_buf_.Push(std::move(data));
    }
    writers_.fetch_sub(1, std::memory_order_release);
}

void TraceDataDumper::ReportHash(std::unique_ptr<PythonTracerHashData> data)
{
    // Cheap early exit when not running; the real check is the one after fetch_add.
    if (C10_UNLIKELY(data == nullptr || !start_.load(std::memory_order_relaxed))) {
        return;
    }
    writers_.fetch_add(1);
    if (C10_LIKELY(start_.load())) {
        std::lock_guard<std::mutex> lock(extra_data_mutex_);
        trace_hash_data_ = std::move(data);
    }
    writers_.fetch_sub(1, std::memory_order_release);
}

void TraceDataDumper::ReportParam(std::unique_ptr<ParamTensorData> data)
{
    // Cheap early exit when not running; the real check is the one after fetch_add.
    if (C10_UNLIKELY(data == nullptr || !start_.load(std::memory_order_relaxed))) {
        return;
    }
    writers_.fetch_add(1);
    if (C10_LIKELY(start_.load())) {
        std::lock_guard<std::mutex> lock(extra_data_mutex_);
        param_data_ = std::move(data);
    }
    writers_.fetch_sub(1, std::memory_order_release);
}

void TraceDataDumper::CreateDumpDir()
{
    if (create_dir_flag_) {
        create_dir_flag_ = !Utils::CreateDir(this->path_);
    }
}

bool TraceDataDumper::FlushTraceData()
{
    std::unique_ptr<PythonTracerFuncData> data = nullptr;
    if (!trace_data_buf_.Pop(data) || data == nullptr) {
        return false;
    }
    auto encode_data = data->encode();
    if (!encode_data.empty()) {
        CreateDumpDir();
        Dump(data->tag, encode_data);
    }
    return true;
}

void TraceDataDumper::FlushHashData()
{
    std::unique_ptr<PythonTracerHashData> data = nullptr;
    {
        std::lock_guard<std::mutex> lock(extra_data_mutex_);
        data = std::move(trace_hash_data_);
    }
    if (data == nullptr) {
        return;
    }
    auto encode_data = data->encode();
    if (!encode_data.empty()) {
        CreateDumpDir();
        Dump(data->tag, encode_data);
    }
}

void TraceDataDumper::FlushParamData()
{
    std::unique_ptr<ParamTensorData> data = nullptr;
    {
        std::lock_guard<std::mutex> lock(extra_data_mutex_);
        data = std::move(param_data_);
    }
    if (data == nullptr) {
        return;
    }
    auto encode_data = data->encode();
    if (!encode_data.empty()) {
        CreateDumpDir();
        Dump(data->tag, encode_data);
    }
}

void TraceDataDumper::Dump(const std::string &file_name, const std::vector<uint8_t> &encode_data)
{
    FILE *fd = nullptr;
    const std::string dump_file = path_ + "/" + file_name;
    auto iter = fd_map_.find(dump_file);
    if (iter == fd_map_.end()) {
        if (!Utils::IsFileExist(dump_file) && !Utils::CreateFile(dump_file)) {
            ASCEND_LOGE("TraceDataDumper create file failed: %s", dump_file.c_str());
            return;
        }
        fd = fopen(dump_file.c_str(), "ab");
        if (fd == nullptr) {
            ASCEND_LOGE("TraceDataDumper open file failed: %s", dump_file.c_str());
            return;
        }
        fd_map_.insert({ dump_file, fd });
    } else {
        fd = iter->second;
    }
    fwrite(reinterpret_cast<const char *>(encode_data.data()), sizeof(char), encode_data.size(), fd);
}
} // profiler
} // toolkit
} // torch_npu
