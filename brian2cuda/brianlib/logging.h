#ifndef BRIAN2CUDA_LOGGING_H
#define BRIAN2CUDA_LOGGING_H

#include <stdio.h>

#define B2C_LOG_LEVEL_DEBUG 10
#define B2C_LOG_LEVEL_INFO 20
#define B2C_LOG_LEVEL_WARNING 30
#define B2C_LOG_LEVEL_ERROR 40

#ifndef B2C_LOG_LEVEL
#define B2C_LOG_LEVEL B2C_LOG_LEVEL_WARNING
#endif

#if defined(__CUDACC__)
#define B2C_HOST_DEVICE __host__ __device__
#else
#define B2C_HOST_DEVICE
#endif

// Declarations must stay visible in nvcc's device pass (__CUDA_ARCH__ set).
namespace brian {

enum class LogLevel {
    Debug = B2C_LOG_LEVEL_DEBUG,
    Info = B2C_LOG_LEVEL_INFO,
    Warning = B2C_LOG_LEVEL_WARNING,
    Error = B2C_LOG_LEVEL_ERROR,
};

B2C_HOST_DEVICE inline const char* b2c_log_level_name(LogLevel level)
{
    switch (level)
    {
    case LogLevel::Debug:
        return "DEBUG";
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Warning:
        return "WARNING";
    case LogLevel::Error:
        return "ERROR";
    }
    return "UNKNOWN";
}

void b2c_log_reset(const char* results_dir, int file_log_level);
void b2c_log_close();
void b2c_log_message(LogLevel level, const char* fmt, ...);

}

#if defined(__CUDA_ARCH__)
#define B2C_LOG_EMIT(level, ...)                                               \
    do {                                                                       \
        printf("[brian2cuda][%s] ",                                            \
               ::brian::b2c_log_level_name(::brian::LogLevel::level));         \
        printf(__VA_ARGS__);                                                   \
        printf("\n");                                                          \
    } while (0)
#else
#define B2C_LOG_EMIT(level, ...)                                               \
    do {                                                                       \
        ::brian::b2c_log_message(::brian::LogLevel::level, __VA_ARGS__);       \
    } while (0)
#endif

#if B2C_LOG_LEVEL <= B2C_LOG_LEVEL_ERROR
#define B2C_LOG_ERROR(...) B2C_LOG_EMIT(Error, __VA_ARGS__)
#else
#define B2C_LOG_ERROR(...) ((void)0)
#endif

#if B2C_LOG_LEVEL <= B2C_LOG_LEVEL_WARNING
#define B2C_LOG_WARN(...) B2C_LOG_EMIT(Warning, __VA_ARGS__)
#else
#define B2C_LOG_WARN(...) ((void)0)
#endif

#if B2C_LOG_LEVEL <= B2C_LOG_LEVEL_INFO
#define B2C_LOG_INFO(...) B2C_LOG_EMIT(Info, __VA_ARGS__)
#else
#define B2C_LOG_INFO(...) ((void)0)
#endif

#if B2C_LOG_LEVEL <= B2C_LOG_LEVEL_DEBUG
#define B2C_LOG_DEBUG(...) B2C_LOG_EMIT(Debug, __VA_ARGS__)
#else
#define B2C_LOG_DEBUG(...) ((void)0)
#endif

#endif  // BRIAN2CUDA_LOGGING_H
