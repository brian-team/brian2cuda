#include "logging.h"

#include <stdarg.h>
#include <stdio.h>
#include <string>

namespace brian {

static FILE* b2c_log_fp = nullptr;
static std::string b2c_log_path;
static int b2c_file_log_level = B2C_LOG_LEVEL_WARNING;

static void b2c_vlog(FILE* fp, LogLevel level, const char* fmt, va_list ap)
{
    fprintf(fp, "[brian2cuda][%s] ", b2c_log_level_name(level));
    vfprintf(fp, fmt, ap);
    fprintf(fp, "\n");
}

static bool b2c_should_persist(LogLevel level)
{
    return static_cast<int>(level) >= b2c_file_log_level;
}

void b2c_log_reset(const char* results_dir, int file_log_level)
{
    b2c_log_close();
    b2c_file_log_level = file_log_level;
    b2c_log_path = std::string(results_dir) + "cuda_log.txt";
    remove(b2c_log_path.c_str());
}

void b2c_log_close()
{
    if (b2c_log_fp != nullptr)
    {
        fclose(b2c_log_fp);
        b2c_log_fp = nullptr;
    }
    b2c_log_path.clear();
}

void b2c_log_message(LogLevel level, const char* fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    b2c_vlog(stderr, level, fmt, ap);
    va_end(ap);

    if (!b2c_should_persist(level) || b2c_log_path.empty())
        return;

    if (b2c_log_fp == nullptr)
    {
        b2c_log_fp = fopen(b2c_log_path.c_str(), "a");
        if (b2c_log_fp == nullptr)
            return;
    }

    va_start(ap, fmt);
    b2c_vlog(b2c_log_fp, level, fmt, ap);
    fflush(b2c_log_fp);
    va_end(ap);
}

}  // namespace brian
