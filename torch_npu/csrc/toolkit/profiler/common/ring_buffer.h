#pragma once

#include <atomic>
#include <memory>
#include <cstdint>
#include <cstddef>

#include "torch_npu/csrc/core/npu/npu_log.h"

namespace torch_npu {
namespace toolkit {
namespace profiler {
// Bounded multi-producer / single-consumer queue (per-slot sequence numbers, Vyukov style).
// A slot is visible to Pop only after its producer has finished writing it, and slots are
// consumed in claim order.
// Lifecycle contract: Init/UnInit must not run concurrently with Push/Pop/Size. The owner
// must stop producers and the consumer (and wait for in-flight Push calls) before UnInit.
template <typename T> class RingBuffer {
public:
    RingBuffer()
        : is_inited_(false),
          read_index_(0),
          write_index_(0),
          capacity_(0),
          mask_(0),
          cycles_exceed_cnt_(0),
          full_cnt_(0)
    {}

    ~RingBuffer()
    {
        UnInit();
    }

    void Init(size_t capacity)
    {
        if (is_inited_.load(std::memory_order_relaxed)) {
            UnInit();
        }
        // Round up to a power of two (at least 2). Reject sizes that would overflow; the
        // buffer then stays uninited and Push/Pop return false.
        const size_t max_cap = (SIZE_MAX >> 1) + 1;
        if (capacity == 0 || capacity > max_cap / sizeof(Slot)) {
            ASCEND_LOGE("RingBuffer capacity %zu is invalid, buffer not created", capacity);
            return;
        }
        size_t cap = 2;
        while (cap < capacity) {
            cap <<= 1;
        }
        if (cap != capacity) {
            ASCEND_LOGW("RingBuffer capacity %zu is not a power of two, use %zu", capacity, cap);
        }
        slots_.reset(new Slot[cap]);
        for (size_t i = 0; i < cap; ++i) {
            slots_[i].seq.store(i, std::memory_order_relaxed);
        }
        capacity_ = cap;
        mask_ = cap - 1;
        read_index_.store(0, std::memory_order_relaxed);
        write_index_.store(0, std::memory_order_relaxed);
        cycles_exceed_cnt_.store(0, std::memory_order_relaxed);
        full_cnt_.store(0, std::memory_order_relaxed);
        // Publish storage and indexes before anyone can see is_inited_ == true.
        is_inited_.store(true, std::memory_order_release);
    }

    void UnInit()
    {
        if (!is_inited_.load(std::memory_order_acquire)) {
            return;
        }
        is_inited_.store(false, std::memory_order_release);
        slots_.reset();
        read_index_.store(0, std::memory_order_relaxed);
        write_index_.store(0, std::memory_order_relaxed);
        capacity_ = 0;
        mask_ = 0;
        auto final_cycles_exceed_cnt = cycles_exceed_cnt_.exchange(0, std::memory_order_relaxed);
        if (final_cycles_exceed_cnt > 0) {
            ASCEND_LOGE("RingBuffer cycles exceed %zu times", final_cycles_exceed_cnt);
        }
        auto final_full_cnt = full_cnt_.exchange(0, std::memory_order_relaxed);
        if (final_full_cnt > 0) {
            ASCEND_LOGE("RingBuffer full %zu times", final_full_cnt);
        }
    }

    bool Push(T data)
    {
        if (!is_inited_.load(std::memory_order_acquire)) {
            return false;
        }
        static const size_t cycle_limit = 1024;
        size_t cycles = 0;
        Slot *slot = nullptr;
        size_t pos = write_index_.load(std::memory_order_relaxed);
        for (;;) {
            cycles++;
            if (cycles >= cycle_limit) {
                cycles_exceed_cnt_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            slot = &slots_[pos & mask_];
            size_t seq = slot->seq.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                // Slot is free for this lap: try to claim it.
                if (write_index_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                // Slot still holds data from the previous lap: queue is full.
                full_cnt_.fetch_add(1, std::memory_order_relaxed);
                return false;
            } else {
                pos = write_index_.load(std::memory_order_relaxed);
            }
        }
        slot->data = std::move(data);
        // Hand the slot to the consumer.
        slot->seq.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Single consumer only.
    bool Pop(T &data)
    {
        if (!is_inited_.load(std::memory_order_acquire)) {
            return false;
        }
        size_t pos = read_index_.load(std::memory_order_relaxed);
        Slot *slot = &slots_[pos & mask_];
        size_t seq = slot->seq.load(std::memory_order_acquire);
        if (seq != pos + 1) {
            // Empty, or the producer of this slot has not finished writing yet.
            return false;
        }
        data = std::move(slot->data);
        // Free the slot for the producer of the next lap.
        slot->seq.store(pos + capacity_, std::memory_order_release);
        read_index_.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Number of claimed but not yet popped slots. May include slots whose producer is still
    // writing; it is exact once all producers have returned.
    size_t Size()
    {
        size_t curr_read_index = read_index_.load(std::memory_order_acquire);
        size_t curr_write_index = write_index_.load(std::memory_order_acquire);
        if (curr_write_index <= curr_read_index) {
            return 0;
        }
        return curr_write_index - curr_read_index;
    }

private:
    // 16 bytes per slot for T = unique_ptr (the old vector<T> used 8), so a 2^20-slot
    // buffer takes 16 MB instead of 8 MB. Not padded: padding would cost far more memory.
    struct Slot {
        std::atomic<size_t> seq{0};
        T data{};
    };

    std::atomic<bool> is_inited_;
    std::atomic<size_t> read_index_;   // next slot to pop, written by the consumer only
    std::atomic<size_t> write_index_;  // next slot to claim, shared by producers
    size_t capacity_;
    size_t mask_;
    std::unique_ptr<Slot[]> slots_;

    // Ringbuffer push failed info
    std::atomic<size_t> cycles_exceed_cnt_;
    std::atomic<size_t> full_cnt_;
};
} // profiler
} // toolkit
} // torch_npu
