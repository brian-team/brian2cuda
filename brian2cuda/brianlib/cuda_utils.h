#ifndef BRIAN2CUDA_ERROR_CHECK_H
#define BRIAN2CUDA_ERROR_CHECK_H
// Define this to turn on error checking
#define BRIAN2CUDA_ERROR_CHECK
// Define this to synchronize device before checking errors
//#define BRIAN2CUDA_ERROR_CHECK_BLOCKING

// Define this to turn on memory checking
//#define BRIAN2CUDA_MEMORY_CHECK
// Define this to synchronize device before checking memory
//#define BRIAN2CUDA_MEMORY_CHECK_BLOCKING

// Choose which LOG macros to define based on LOG_LEVEL_<level> macro defined
// during compilation, <level> is one of DEBUG, INFO, WARNING, ERROR, CRITICAL
// For now we treat CRITICAL the same as ERROR

// TODO Could make this with less code with if !defined? Though this is easier
// to understand.
#if defined LOG_LEVEL_CRITICAL || defined LOG_LEVEL_ERROR
    #define DEF_LOG_CUDA_ERROR
    #define DEF_LOG_ERROR
#elif defined LOG_LEVEL_WARNING
    #define DEF_LOG_CUDA_ERROR
    #define DEF_LOG_ERROR
    #define DEF_LOG_WARNING
#elif defined LOG_LEVEL_INFO
    #define DEF_LOG_CUDA_ERROR
    #define DEF_LOG_ERROR
    #define DEF_LOG_WARNING
    #define DEF_LOG_INFO
#elif defined LOG_LEVEL_DEBUG
    #define DEF_LOG_CUDA_ERROR
    #define DEF_LOG_ERROR
    #define DEF_LOG_WARNING
    #define DEF_LOG_INFO
    #define DEF_LOG_DEBUG
#elif defined LOG_LEVEL_DIAGNOSTIC
    #define DEF_LOG_CUDA_ERROR
    #define DEF_LOG_ERROR
    #define DEF_LOG_WARNING
    #define DEF_LOG_INFO
    #define DEF_LOG_DEBUG
    #define DEF_LOG_DIAGNOSTIC
#endif

// DEFINE the LOG macros as printf statements or no_ops if not defined
// LOG_CUDA_ERROR is the only macro usable in device code currently and will
//   be printed to stdout when CUDA ring buffer is flushed at host/device
//   serialization (this sometimes does not happen when the program crashes).
// TODO: All other LOG macros could in principle be redirected to the Brian2
// log file via fprintf (not implemented yet)
#ifdef DEF_LOG_CUDA_ERROR
    #define LOG_CUDA_ERROR(fmt, ...)    printf("GPU ERROR\t"        fmt, __VA_ARGS__)
#else
    #define LOG_CUDA_ERROR(fmt, ...)    do {} while(0)
#endif

#ifdef DEF_LOG_ERROR
    #define LOG_ERROR(fmt, ...)         printf("CUDA ERROR\t"       fmt, __VA_ARGS__); fflush(stdout);
#else
    #define LOG_ERROR(fmt, ...)         do {} while(0)
#endif

#ifdef DEF_LOG_WARNING
    #define LOG_WARNING(fmt, ...)       printf("CUDA WARNING\t"     fmt, __VA_ARGS__); fflush(stdout);
#else
    #define LOG_WARNING(fmt, ...)       do {} while(0)
#endif

#ifdef DEF_LOG_INFO
    #define LOG_INFO(fmt, ...)          printf("CUDA INFO\t"        fmt, __VA_ARGS__); fflush(stdout);
#else
    #define LOG_INFO(fmt, ...)          do {} while(0)
#endif

#ifdef DEF_LOG_DEBUG
    #define LOG_DEBUG(fmt, ...)         printf("CUDA DEBUG\t"       fmt, __VA_ARGS__); fflush(stdout);
#else
    #define LOG_DEBUG(fmt, ...)         do {} while(0)
#endif

#ifdef DEF_LOG_DIAGNOSTIC
    #define LOG_DIAGNOSTIC(fmt, ...)    printf("CUDA DIAGNOSTIC\t"  fmt, __VA_ARGS__); fflush(stdout);
#else
    #define LOG_DIAGNOSTIC(fmt, ...)    do {} while(0)
#endif

// partly adapted from https://gist.github.com/ashwin/2652488
#define CUDA_SAFE_CALL(err)     _cudaSafeCall(err, __FILE__, __LINE__, #err)
#define CUDA_CHECK_ERROR(msg)   _cudaCheckError(__FILE__, __LINE__, #msg)
#define CUDA_CHECK_MEMORY()     _cudaCheckMemory(__FILE__, __LINE__)
#define THRUST_CHECK_ERROR(code)  { try {code;} \
    catch(...) {_thrustCheckError(__FILE__, __LINE__, #code);} }

// Place includes after macro definitions to avoid circular includes
#include <stdio.h>
#include <thrust/system_error.h>
#include "objects.h"
#include "curand.h"

// adapted from NVIDIA cuda samples, shipped with cuda 10.1 (common/inc/helper_cuda.h)
#ifdef CURAND_H_
// cuRAND API errors
static const char *_curandGetErrorEnum(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif


inline void _cudaSafeCall(cudaError err, const char *file, const int line, const char *call = "")
{
#ifdef BRIAN2CUDA_ERROR_CHECK
    if (cudaSuccess != err)
    {
        LOG_ERROR("%s failed at %s:%i : %s\n",
                  call, file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}


inline void _cudaSafeCall(curandStatus_t err, const char *file, const int line, const char *call = "")
{
#ifdef BRIAN2CUDA_ERROR_CHECK
    if (CURAND_STATUS_SUCCESS != err)
    {
        LOG_ERROR("%s failed at %s:%i : %s\n",
                  call, file, line, _curandGetErrorEnum(err));
        exit(-1);
    }
#endif

    return;
}


inline void _cudaCheckError(const char *file, const int line, const char *msg)
{
#ifdef BRIAN2CUDA_ERROR_CHECK_BLOCKING
    // More careful checking. However, this will affect performance.
    cudaError err = cudaDeviceSynchronize();
    if(cudaSuccess != err)
    {
        LOG_ERROR("CUDA_CHECK_ERROR() failed after %s at %s:%i : %s\n",
                  msg, file, line, cudaGetErrorString(err));
        exit(-1);
    }
#else
#ifdef BRIAN2CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        LOG_ERROR("CUDA_CHECK_ERROR() failed at %s:%i : %s\n",
                  file, line, cudaGetErrorString(err));
        exit(-1);
    }

#endif
#endif

    return;
}


// Report device memory usage. The memory diff is reported with respect to the
// global brian::used_device_memory as reference, which was set in the last
// _cudaCheckMemory call.
inline void _cudaCheckMemory(const char *file, const int line)
{
#ifdef BRIAN2CUDA_MEMORY_CHECK
#ifdef BRIAN2CUDA_MEMORY_CHECK_BLOCKING
    cudaDeviceSynchronize();
#endif
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    size_t avail, total, used, diff;
    cudaMemGetInfo(&avail, &total);
    used = total - avail;
    diff = used - brian::used_device_memory;
    // print memory information only if device memory usage changed
    // NOTE: Device memory is allocated in chunks. When allocating only little
    // memory, the memory usage reported by cudaMemGetInfo might not change if
    // the previously allocated chunk has enough free memory to be used for the
    // newly requested allocation.
    if (diff > 0)
    {
        LOG_DEBUG("CUDA device memory usage in %s:%i\n"
                  "\t\t\t used:  \t %f MB\n"
                  "\t\t\t avail: \t %f MB\n"
                  "\t\t\t total: \t %f MB\n"
                  "\t\t\t diff:  \t %f MB \t (%zu bytes)\n",
                  file, line,
                  double(used) * to_MB,
                  double(avail) * to_MB,
                  double(total) * to_MB,
                  double(diff) * to_MB, diff);
        brian::used_device_memory = used;
    }
#endif
}


inline void _thrustCheckError(const char *file, const int line,
        const char *code)
{
    LOG_ERROR("THRUST_CHECK_ERROR() caught an exception from %s at %s:%i\n",
            code, file, line);
    throw;
}

#endif  // BRIAN2CUDA_ERROR_CHECK_H
