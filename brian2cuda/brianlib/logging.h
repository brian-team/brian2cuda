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

// Declarations must stay visible in nvcc's device pass (__CUDA_ARCH__ set).
namespace brian {
void b2c_log_open(const char* results_dir);
void b2c_log_close();
void b2c_log_message(const char* level, const char* fmt, ...);
}

#if defined(__CUDA_ARCH__)
#define B2C_LOG_EMIT(tag, ...)                                                 \
    do {                                                                       \
        printf("[brian2cuda][" tag "] " __VA_ARGS__);                          \
        printf("\n");                                                          \
    } while (0)
#else
#define B2C_LOG_EMIT(tag, ...)                                                 \
    do { ::brian::b2c_log_message(tag, __VA_ARGS__); } while (0)
#endif

#if B2C_LOG_LEVEL <= B2C_LOG_LEVEL_ERROR
#define B2C_LOG_ERROR(...) B2C_LOG_EMIT("ERROR", __VA_ARGS__)
#else
#define B2C_LOG_ERROR(...) ((void)0)
#endif

#if B2C_LOG_LEVEL <= B2C_LOG_LEVEL_WARNING
#define B2C_LOG_WARN(...) B2C_LOG_EMIT("WARNING", __VA_ARGS__)
#else
#define B2C_LOG_WARN(...) ((void)0)
#endif

#if B2C_LOG_LEVEL <= B2C_LOG_LEVEL_INFO
#define B2C_LOG_INFO(...) B2C_LOG_EMIT("INFO", __VA_ARGS__)
#else
#define B2C_LOG_INFO(...) ((void)0)
#endif

#if B2C_LOG_LEVEL <= B2C_LOG_LEVEL_DEBUG
#define B2C_LOG_DEBUG(...) B2C_LOG_EMIT("DEBUG", __VA_ARGS__)
#else
#define B2C_LOG_DEBUG(...) ((void)0)
#endif

#endif  // BRIAN2CUDA_LOGGING_H
