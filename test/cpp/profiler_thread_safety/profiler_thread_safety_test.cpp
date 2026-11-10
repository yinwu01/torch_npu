// Standalone stress test for the profiler upload path: RingBuffer, DataDumper and
// TraceDataDumper. The real ring_buffer.h, data_dumper.h, data_dumper.cpp and thread.h are
// compiled unmodified; only npu_log.h, utils.h and data_reporter.h come from stubs/.
//
// Usage:
//   profiler_thread_safety_test rb <producers> <items_per_producer> <capacity>
//   profiler_thread_safety_test dumper <dir> <producers> <cycles> <capacity>
//   profiler_thread_safety_test trace <dir> <producers> <cycles> <capacity>
//   profiler_thread_safety_test bench <dir> <producers> <milliseconds> <capacity>
//     (DataDumper::Report throughput while started; no checks, no test pauses)
// Prints one JSON line and exits 0 on pass, 1 on a detected violation.
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "torch_npu/csrc/toolkit/profiler/common/ring_buffer.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_dumper.h"

using namespace torch_npu::toolkit::profiler;

namespace {
constexpr uint32_t kMagic = 0xA5A5F00Du;
constexpr size_t kRecordSize = 24;

std::atomic<uint64_t> g_encoded{ 0 }; // encode() calls, i.e. records handed to Dump
std::atomic<int64_t> g_live{ 0 };     // report objects alive; must be 0 at the end
// Records encoded inside the main thread's UnInit()/Init() calls. Those run after the main
// thread's own Stop() returned, which must have drained everything a Report* accepted, so
// any record found there was written after Stop() (e.g. a Report* not counted in writers_).
std::atomic<uint64_t> g_late_encodes{ 0 };
thread_local bool t_after_stop = false;

uint64_t Checksum(uint64_t producer, uint64_t seq)
{
    return seq ^ (producer * 0x9E3779B97F4A7C15ull) ^ kMagic;
}

std::vector<uint8_t> EncodeRecord(const StubPayload &p)
{
    g_encoded.fetch_add(1, std::memory_order_relaxed);
    if (t_after_stop) {
        g_late_encodes.fetch_add(1, std::memory_order_relaxed);
    }
    std::vector<uint8_t> out(kRecordSize);
    uint64_t check = Checksum(p.producer, p.seq);
    memcpy(out.data(), &kMagic, 4);
    memcpy(out.data() + 4, &p.producer, 4);
    memcpy(out.data() + 8, &p.seq, 8);
    memcpy(out.data() + 16, &check, 8);
    return out;
}

struct OpData : BaseReportData {
    StubPayload payload;
    OpData(std::string tag, StubPayload p) : BaseReportData(0, std::move(tag)), payload(p)
    {
        g_live.fetch_add(1, std::memory_order_relaxed);
    }
    ~OpData() override
    {
        g_live.fetch_sub(1, std::memory_order_relaxed);
    }
    std::vector<uint8_t> encode() override
    {
        return EncodeRecord(payload);
    }
};
} // namespace

// See stubs/.../data_reporter.h: pause for 50-300 us after about one in 256 passed checks.
std::atomic<bool> g_no_pause{ false };

bool StubAfterCheck(bool value)
{
    if (g_no_pause.load(std::memory_order_relaxed)) {
        return value;
    }
    thread_local std::mt19937 rng(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    if (value && (rng() & 255) == 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50 + rng() % 251));
    }
    return value;
}

namespace torch_npu {
namespace toolkit {
namespace profiler {
PythonTracerFuncData::PythonTracerFuncData(StubPayload p) : BaseReportData(0, "torch.python_tracer_func"), payload(p)
{
    g_live.fetch_add(1, std::memory_order_relaxed);
}
PythonTracerFuncData::~PythonTracerFuncData()
{
    g_live.fetch_sub(1, std::memory_order_relaxed);
}
std::vector<uint8_t> PythonTracerFuncData::encode()
{
    return EncodeRecord(payload);
}

PythonTracerHashData::PythonTracerHashData(StubPayload p) : BaseReportData(0, "torch.python_tracer_hash"), payload(p)
{
    g_live.fetch_add(1, std::memory_order_relaxed);
}
PythonTracerHashData::~PythonTracerHashData()
{
    g_live.fetch_sub(1, std::memory_order_relaxed);
}
std::vector<uint8_t> PythonTracerHashData::encode()
{
    return EncodeRecord(payload);
}

ParamTensorData::ParamTensorData(StubPayload p) : BaseReportData(0, "torch.param_tensor_info"), payload(p)
{
    g_live.fetch_add(1, std::memory_order_relaxed);
}
ParamTensorData::~ParamTensorData()
{
    g_live.fetch_sub(1, std::memory_order_relaxed);
}
std::vector<uint8_t> ParamTensorData::encode()
{
    return EncodeRecord(payload);
}
} // profiler
} // toolkit
} // torch_npu

namespace {
struct FileStats {
    uint64_t records = 0;
    uint64_t files = 0;
    uint64_t bad_size = 0;   // file length not a multiple of the record size
    uint64_t torn = 0;       // magic or checksum mismatch
    uint64_t misordered = 0; // per-producer sequence not strictly increasing within one file
};

// Reads every regular file under dir (one level of subdirectories) and checks each record.
void VerifyDir(const std::string &dir, FileStats &st)
{
    DIR *d = opendir(dir.c_str());
    if (d == nullptr) {
        return;
    }
    std::vector<std::string> entries;
    while (dirent *e = readdir(d)) {
        std::string name = e->d_name;
        if (name != "." && name != "..") {
            entries.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    for (const auto &path : entries) {
        struct stat s;
        if (stat(path.c_str(), &s) != 0) {
            continue;
        }
        if (S_ISDIR(s.st_mode)) {
            VerifyDir(path, st);
            continue;
        }
        st.files++;
        std::ifstream in(path, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (bytes.size() % kRecordSize != 0) {
            st.bad_size++;
        }
        std::map<uint32_t, uint64_t> next_min;
        for (size_t off = 0; off + kRecordSize <= bytes.size(); off += kRecordSize) {
            uint32_t magic = 0;
            uint32_t producer = 0;
            uint64_t seq = 0;
            uint64_t check = 0;
            memcpy(&magic, bytes.data() + off, 4);
            memcpy(&producer, bytes.data() + off + 4, 4);
            memcpy(&seq, bytes.data() + off + 8, 8);
            memcpy(&check, bytes.data() + off + 16, 8);
            st.records++;
            if (magic != kMagic || check != Checksum(producer, seq)) {
                st.torn++;
                continue;
            }
            auto it = next_min.find(producer);
            if (it != next_min.end() && seq < it->second) {
                st.misordered++;
            }
            next_min[producer] = seq + 1;
        }
    }
}

struct RbItem {
    uint64_t producer = 0;
    uint64_t seq = 0;
    uint64_t check = 0;
};

// (a) N producers Push while one consumer Pops. Producers retry when Push fails, so every
// item must come out exactly once, in per-producer order, with an intact checksum.
int RunRingBuffer(int producers, uint64_t per_producer, size_t capacity)
{
    RingBuffer<RbItem> rb;
    rb.Init(capacity);
    std::atomic<bool> go{ false };
    std::atomic<int> finished{ 0 };
    std::vector<std::thread> threads;
    for (int p = 0; p < producers; ++p) {
        threads.emplace_back([&, p]() {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint64_t s = 0; s < per_producer; ++s) {
                RbItem item{ static_cast<uint64_t>(p), s, Checksum(p, s) };
                while (!rb.Push(item)) {
                    std::this_thread::yield();
                }
            }
            finished.fetch_add(1);
        });
    }

    const uint64_t target = static_cast<uint64_t>(producers) * per_producer;
    std::vector<uint64_t> next(producers, 0);
    uint64_t popped = 0;
    uint64_t torn = 0;
    uint64_t stale = 0; // seq lower than expected: duplicate or stale slot
    uint64_t gaps = 0;  // seq higher than expected: earlier items skipped
    uint64_t skipped = 0;
    bool stalled = false;
    auto last_progress = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    while (popped < target) {
        RbItem item;
        if (!rb.Pop(item)) {
            if (std::chrono::steady_clock::now() - last_progress > std::chrono::seconds(10)) {
                stalled = true;
                break;
            }
            std::this_thread::yield();
            continue;
        }
        last_progress = std::chrono::steady_clock::now();
        popped++;
        if (item.producer >= static_cast<uint64_t>(producers) || item.check != Checksum(item.producer, item.seq)) {
            if (torn < 5) {
                fprintf(stderr, "TORN item producer=%" PRIu64 " seq=%" PRIu64 "\n", item.producer, item.seq);
            }
            torn++;
            continue;
        }
        uint64_t &want = next[item.producer];
        if (item.seq < want) {
            if (stale < 5) {
                fprintf(stderr, "STALE producer=%" PRIu64 " got=%" PRIu64 " want=%" PRIu64 "\n", item.producer,
                    item.seq, want);
            }
            stale++;
        } else {
            if (item.seq > want) {
                if (gaps < 5) {
                    fprintf(stderr, "GAP producer=%" PRIu64 " got=%" PRIu64 " want=%" PRIu64 "\n", item.producer,
                        item.seq, want);
                }
                gaps++;
                skipped += item.seq - want;
            }
            want = item.seq + 1;
        }
    }
    if (stalled) {
        // Keep draining so producers blocked on a full queue can finish and be joined.
        RbItem item;
        while (finished.load() < producers) {
            while (rb.Pop(item)) {
            }
            std::this_thread::yield();
        }
    }
    for (auto &t : threads) {
        t.join();
    }
    uint64_t extra = 0;
    RbItem item;
    while (rb.Pop(item)) {
        extra++;
    }
    // Items never seen: skipped over inside the stream, or missing at the tail.
    uint64_t lost = skipped;
    for (int p = 0; p < producers; ++p) {
        lost += per_producer - std::min(per_producer, next[p]);
    }
    rb.UnInit();
    bool passed = !stalled && torn == 0 && stale == 0 && gaps == 0 && extra == 0 && lost == 0 && popped == target;
    printf("{\"scenario\":\"rb\",\"producers\":%d,\"per_producer\":%" PRIu64 ",\"capacity\":%zu,"
           "\"popped\":%" PRIu64 ",\"target\":%" PRIu64 ",\"torn\":%" PRIu64 ",\"stale\":%" PRIu64
           ",\"gaps\":%" PRIu64 ",\"extra\":%" PRIu64 ",\"lost\":%" PRIu64 ",\"stalled\":%s,\"passed\":%s}\n",
        producers, per_producer, capacity, popped, target, torn, stale, gaps, extra, lost, stalled ? "true" : "false",
        passed ? "true" : "false");
    return passed ? 0 : 1;
}

// Runs f with t_after_stop set; see g_late_encodes.
template <typename F> void AfterStop(F f)
{
    t_after_stop = true;
    f();
    t_after_stop = false;
}

void RandomPause(std::mt19937 &rng, unsigned max_us)
{
    std::this_thread::sleep_for(std::chrono::microseconds(rng() % (max_us + 1)));
}

// Calls Stop() on the given dumpers at random times from a second control thread, so two
// Stop() calls (or Stop() and Init/Start/UnInit) overlap, as when nothing serialises
// ProfilerMgr::Stop.
template <typename... D> std::thread StartStopper(std::atomic<bool> &stop, unsigned seed, D &...dumpers)
{
    return std::thread([&stop, seed, &dumpers...]() {
        std::mt19937 rng(seed);
        while (!stop.load(std::memory_order_relaxed)) {
            RandomPause(rng, 1500);
            (dumpers.Stop(), ...);
        }
    });
}

int Report(const char *scenario, const std::string &dir, int producers, int cycles, size_t capacity,
    uint64_t reported)
{
    FileStats st;
    VerifyDir(dir, st);
    uint64_t encoded = g_encoded.load();
    int64_t live = g_live.load();
    uint64_t late = g_late_encodes.load();
    // Every encoded record must reach a file, nothing torn or reordered, nothing leaked,
    // nothing left behind after Stop().
    bool passed = st.bad_size == 0 && st.torn == 0 && st.misordered == 0 && st.records == encoded && live == 0 &&
        late == 0 && st.records > 0;
    printf("{\"scenario\":\"%s\",\"producers\":%d,\"cycles\":%d,\"capacity\":%zu,\"reported\":%" PRIu64
           ",\"encoded\":%" PRIu64 ",\"records_in_files\":%" PRIu64 ",\"files\":%" PRIu64 ",\"bad_size\":%" PRIu64
           ",\"torn\":%" PRIu64 ",\"misordered\":%" PRIu64 ",\"live_objects\":%" PRId64
           ",\"encoded_after_stop\":%" PRIu64 ",\"passed\":%s}\n",
        scenario, producers, cycles, capacity, reported, encoded, st.records, st.files, st.bad_size, st.torn,
        st.misordered, live, late, passed ? "true" : "false");
    return passed ? 0 : 1;
}

// (b) Producers call Report in a loop on two DataDumpers while the main thread cycles
// Init/Start/Stop/UnInit in the order ProfilerMgr::StopDataReceiver uses, and a second
// control thread calls Stop() at random times. Two instances run at once, as in the real
// profiler, so state shared between instances is exercised.
int RunDumper(const std::string &dir, int producers, int cycles, size_t capacity)
{
    std::atomic<bool> stop{ false };
    std::atomic<uint64_t> reported{ 0 };
    {
        DataDumper first;
        DataDumper second;
        std::vector<std::thread> threads;
        for (int p = 0; p < producers; ++p) {
            threads.emplace_back([&, p]() {
                uint64_t seq = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    StubPayload payload{ static_cast<uint32_t>(p), seq };
                    const char *tag = (seq & 1) ? "op_range.bin" : "op_mark.bin";
                    DataDumper &target = (seq & 2) ? second : first;
                    target.Report(std::make_unique<OpData>(tag, payload));
                    seq++;
                    if ((seq & 63) == 0) {
                        std::this_thread::yield();
                    }
                }
                reported.fetch_add(seq);
            });
        }
        std::atomic<bool> stop_stopper{ false };
        std::thread stopper = StartStopper(stop_stopper, 777, first, second);
        std::mt19937 rng(12345);
        for (int c = 0; c < cycles; ++c) {
            AfterStop([&] {
                first.Init(dir + "/first", capacity);
                second.Init(dir + "/second", capacity);
            });
            first.Start();
            second.Start();
            RandomPause(rng, 2000);
            first.Stop();
            AfterStop([&] { first.UnInit(); });
            second.Stop();
            AfterStop([&] { second.UnInit(); });
            RandomPause(rng, 200);
        }
        stop_stopper.store(true);
        stopper.join();
        stop.store(true);
        for (auto &t : threads) {
            t.join();
        }
    }
    return Report("dumper", dir, producers, cycles, capacity, reported.load());
}

// (c) TraceDataDumper: producers call Report, ReportHash and ReportParam while the main
// thread cycles Init/Start/Stop/UnInit and a second control thread calls Stop() at random
// times, so Report* races with Stop's wait and flush. Two of every three producers call
// Report, so the ring buffer path sees most of the traffic.
int RunTrace(const std::string &dir, int producers, int cycles, size_t capacity)
{
    std::atomic<bool> stop{ false };
    std::atomic<uint64_t> reported{ 0 };
    {
        TraceDataDumper dumper;
        std::vector<std::thread> threads;
        for (int p = 0; p < producers; ++p) {
            threads.emplace_back([&, p]() {
                uint64_t seq = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    StubPayload payload{ static_cast<uint32_t>(p), seq };
                    if (p % 3 != 2) {
                        dumper.Report(std::make_unique<PythonTracerFuncData>(payload));
                    } else if (seq & 1) {
                        dumper.ReportHash(std::make_unique<PythonTracerHashData>(payload));
                    } else {
                        dumper.ReportParam(std::make_unique<ParamTensorData>(payload));
                    }
                    seq++;
                    if ((seq & 255) == 0) {
                        std::this_thread::yield();
                    }
                }
                reported.fetch_add(seq);
            });
        }
        std::atomic<bool> stop_stopper{ false };
        std::thread stopper = StartStopper(stop_stopper, 888, dumper);
        std::mt19937 rng(54321);
        for (int c = 0; c < cycles; ++c) {
            AfterStop([&] { dumper.Init(dir + "/trace", capacity); });
            dumper.Start();
            RandomPause(rng, 1000);
            dumper.Stop();
            AfterStop([&] { dumper.UnInit(); });
            RandomPause(rng, 100);
        }
        stop_stopper.store(true);
        stopper.join();
        stop.store(true);
        for (auto &t : threads) {
            t.join();
        }
    }
    return Report("trace", dir, producers, cycles, capacity, reported.load());
}

// Report() calls per second from <producers> threads into one started DataDumper. The
// dump thread drains and writes files meanwhile; a full buffer makes Push fail fast, which
// still runs the whole Report() path.
int RunBench(const std::string &dir, int producers, int ms, size_t capacity)
{
    g_no_pause = true;
    std::atomic<bool> go{ false };
    std::atomic<bool> stop{ false };
    std::atomic<uint64_t> reported{ 0 };
    double secs = 0;
    {
        DataDumper dumper;
        dumper.Init(dir + "/bench", capacity);
        dumper.Start();
        std::vector<std::thread> threads;
        for (int p = 0; p < producers; ++p) {
            threads.emplace_back([&, p]() {
                while (!go.load()) {
                    std::this_thread::yield();
                }
                uint64_t seq = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    dumper.Report(std::make_unique<OpData>("op_range.bin", StubPayload{ static_cast<uint32_t>(p), seq }));
                    seq++;
                }
                reported.fetch_add(seq);
            });
        }
        auto t0 = std::chrono::steady_clock::now();
        go = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        stop = true;
        for (auto &t : threads) {
            t.join();
        }
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        dumper.Stop();
        dumper.UnInit();
    }
    printf("{\"scenario\":\"bench\",\"producers\":%d,\"capacity\":%zu,\"seconds\":%.3f,\"reports\":%" PRIu64
           ",\"reports_per_sec\":%.0f,\"encoded\":%" PRIu64 ",\"passed\":true}\n",
        producers, capacity, secs, reported.load(), reported.load() / secs, g_encoded.load());
    return 0;
}

int Usage()
{
    fprintf(stderr,
        "usage: profiler_thread_safety_test rb <producers> <items_per_producer> <capacity>\n"
        "       profiler_thread_safety_test dumper|trace <dir> <producers> <cycles> <capacity>\n"
        "       profiler_thread_safety_test bench <dir> <producers> <milliseconds> <capacity>\n");
    return 2;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        return Usage();
    }
    std::string mode = argv[1];
    if (mode == "rb" && argc == 5) {
        return RunRingBuffer(atoi(argv[2]), strtoull(argv[3], nullptr, 10), strtoull(argv[4], nullptr, 10));
    }
    if ((mode == "dumper" || mode == "trace") && argc == 6) {
        std::string dir = argv[2];
        int producers = atoi(argv[3]);
        int cycles = atoi(argv[4]);
        size_t capacity = strtoull(argv[5], nullptr, 10);
        return mode == "dumper" ? RunDumper(dir, producers, cycles, capacity) : RunTrace(dir, producers, cycles, capacity);
    }
    if (mode == "bench" && argc == 6) {
        return RunBench(argv[2], atoi(argv[3]), atoi(argv[4]), strtoull(argv[5], nullptr, 10));
    }
    return Usage();
}
