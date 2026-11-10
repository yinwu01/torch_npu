#pragma once
// Test stub for torch_npu/csrc/toolkit/profiler/common/utils.h: only the file helpers
// that data_dumper.cpp uses. The real header pulls in LibAscendHal.h.
#include <cerrno>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace torch_npu {
namespace toolkit {
namespace profiler {
class Utils {
public:
    static bool IsFileExist(const std::string &path)
    {
        return !path.empty() && access(path.c_str(), F_OK) == 0;
    }

    static bool CreateDir(const std::string &path)
    {
        if (path.empty()) {
            return false;
        }
        for (size_t pos = 1; pos <= path.size(); ++pos) {
            if (pos != path.size() && path[pos] != '/') {
                continue;
            }
            std::string sub = path.substr(0, pos);
            if (mkdir(sub.c_str(), 0750) != 0 && errno != EEXIST) {
                return false;
            }
        }
        struct stat st;
        return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    }

    static bool CreateFile(const std::string &path)
    {
        size_t slash = path.rfind('/');
        if (slash != std::string::npos && !CreateDir(path.substr(0, slash))) {
            return false;
        }
        int fd = creat(path.c_str(), S_IRUSR | S_IWUSR | S_IRGRP);
        if (fd < 0) {
            return false;
        }
        close(fd);
        return true;
    }
};
} // profiler
} // toolkit
} // torch_npu
