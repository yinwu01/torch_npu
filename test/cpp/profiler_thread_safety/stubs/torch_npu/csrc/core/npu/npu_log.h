#pragma once
// Test stub for torch_npu/csrc/core/npu/npu_log.h: the real header needs CANN.
// Logging is silent; the format string is still type-checked.
#include <cstdio>
#include <cstring>

#define ASCEND_LOG_STUB(fmt, ...)                                                                      \
    do {                                                                                                \
        if (false) {                                                                                    \
            fprintf(stderr, fmt, ##__VA_ARGS__);                                                        \
        }                                                                                               \
    } while (0)
#define ASCEND_LOGE(fmt, ...) ASCEND_LOG_STUB(fmt, ##__VA_ARGS__)
#define ASCEND_LOGW(fmt, ...) ASCEND_LOG_STUB(fmt, ##__VA_ARGS__)
#define ASCEND_LOGI(fmt, ...) ASCEND_LOG_STUB(fmt, ##__VA_ARGS__)
#define ASCEND_LOGD(fmt, ...) ASCEND_LOG_STUB(fmt, ##__VA_ARGS__)
