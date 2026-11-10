#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <condition_variable>

#include "torch_npu/csrc/toolkit/profiler/common/thread.h"
#include "torch_npu/csrc/toolkit/profiler/common/ring_buffer.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"

namespace torch_npu {
namespace toolkit {
namespace profiler {
constexpr uint32_t kDefaultRingBuffer = 1024;
constexpr uint32_t kBatchMaxLen = 5 * 1024 * 1024; // 5 MB
constexpr uint32_t kMaxWaitTimeUs = 1024;
constexpr uint32_t kNotifyInterval = 256;

class DataDumper : public Thread {
public:
    explicit DataDumper();
    virtual ~DataDumper();
    void Init(const std::string &path, size_t capacity);
    void UnInit();
    void Report(std::unique_ptr<BaseReportData> data);
    void Start();
    void Stop();

private:
    void Flush();
    void Dump(const std::map<std::string, std::vector<uint8_t>> &dataMap);
    void Run();
    size_t GatherAndDumpData();
    void WaitWriters();
    void StopLocked();
    void UnInitLocked();

private:
    std::string path_;
    std::atomic<bool> start_;
    std::atomic<bool> init_;
    // Own cache line so writer RMWs do not invalidate the read-mostly start_.
    alignas(64) std::atomic<size_t> writers_{ 0 }; // Report calls inside the start_ check / Push
    std::mutex control_mutex_; // serialises Init/Start/Stop/UnInit; never taken by Report
    bool create_dir_flag_{ true };
    RingBuffer<std::unique_ptr<BaseReportData>> data_chunk_buf_;
    std::map<std::string, FILE *> fd_map_;
};

class TraceDataDumper : public Thread {
public:
    explicit TraceDataDumper();
    virtual ~TraceDataDumper();
    void Init(const std::string &path, size_t capacity);
    void UnInit();
    void Report(std::unique_ptr<PythonTracerFuncData> data);
    void ReportHash(std::unique_ptr<PythonTracerHashData> data);
    void ReportParam(std::unique_ptr<ParamTensorData> data);
    void Start();
    void Stop();

private:
    void CreateDumpDir();
    bool FlushTraceData();
    void FlushHashData();
    void FlushParamData();
    void Dump(const std::string &file_name, const std::vector<uint8_t> &encode_data);
    void Run();
    void WaitWriters();
    void StopLocked();
    void UnInitLocked();

private:
    std::string path_;
    std::atomic<bool> start_;
    std::atomic<bool> init_;
    // Own cache line so writer RMWs do not invalidate the read-mostly start_.
    alignas(64) std::atomic<size_t> writers_{ 0 }; // Report* calls inside the start_ check / write
    std::mutex control_mutex_; // serialises Init/Start/Stop/UnInit; never taken by Report*
    bool create_dir_flag_{ true };
    std::mutex extra_data_mutex_; // guards trace_hash_data_ and param_data_
    std::unique_ptr<PythonTracerHashData> trace_hash_data_{ nullptr };
    std::unique_ptr<ParamTensorData> param_data_{ nullptr };
    RingBuffer<std::unique_ptr<PythonTracerFuncData>> trace_data_buf_;
    std::map<std::string, FILE *> fd_map_;
};
} // profiler
} // toolkit
} // torch_npu
