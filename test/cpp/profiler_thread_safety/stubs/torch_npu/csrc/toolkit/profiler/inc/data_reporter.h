#pragma once
// Test stub for torch_npu/csrc/toolkit/profiler/inc/data_reporter.h. The real header needs
// ATen. Only the members data_dumper.cpp touches (tag, encode) are kept; the test file
// defines the constructors and encode() of the concrete types.
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifndef C10_LIKELY
// Test hook: in data_dumper.cpp C10_LIKELY wraps only the start_ re-checks in Report(),
// ReportHash() and ReportParam(), right before the write. StubAfterCheck (defined in the
// test) sometimes pauses after a true result, which widens the window that Stop() has to
// close by waiting for writers.
bool StubAfterCheck(bool value);
#define C10_LIKELY(expr) (__builtin_expect(StubAfterCheck(static_cast<bool>(expr)), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#endif

namespace torch_npu {
namespace toolkit {
namespace profiler {
struct BaseReportData {
    int32_t device_id{ 0 };
    std::string tag;
    BaseReportData(int32_t device_id, std::string tag) : device_id(device_id), tag(std::move(tag)) {}
    virtual ~BaseReportData() = default;
    virtual std::vector<uint8_t> encode() = 0;
};

// Payload shared by the test types: who produced the record and its sequence number.
struct StubPayload {
    uint32_t producer{ 0 };
    uint64_t seq{ 0 };
};

struct PythonTracerFuncData : BaseReportData {
    StubPayload payload;
    explicit PythonTracerFuncData(StubPayload p);
    ~PythonTracerFuncData() override;
    std::vector<uint8_t> encode() override;
};

struct PythonTracerHashData : BaseReportData {
    StubPayload payload;
    explicit PythonTracerHashData(StubPayload p);
    ~PythonTracerHashData() override;
    std::vector<uint8_t> encode() override;
};

struct ParamTensorData : BaseReportData {
    StubPayload payload;
    explicit ParamTensorData(StubPayload p);
    ~ParamTensorData() override;
    std::vector<uint8_t> encode() override;
};
} // profiler
} // toolkit
} // torch_npu
