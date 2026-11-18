#ifndef BRIAN2CUDA_ERROR_CHECK_H
#define BRIAN2CUDA_ERROR_CHECK_H
#include <stdio.h>
#include <thrust/system_error.h>
#include "objects.h"
#include "curand.h"

// Define this to turn on error checking
#define BRIAN2CUDA_ERROR_CHECK
// Define this to synchronize device before checking errors
//#define BRIAN2CUDA_ERROR_CHECK_BLOCKING

// Define this to turn on memory checking
//#define BRIAN2CUDA_MEMORY_CHECK
// Define this to synchronize device before checking memory
//#define BRIAN2CUDA_MEMORY_CHECK_BLOCKING


// partly adapted from https://gist.github.com/ashwin/2652488
#define CUDA_SAFE_CALL(err)     _cudaSafeCall(err, __FILE__, __LINE__, #err)
#define CUDA_CHECK_ERROR(msg)   _cudaCheckError(__FILE__, __LINE__, #msg)
#define CUDA_CHECK_MEMORY()     _cudaCheckMemory(__FILE__, __LINE__)
#define THRUST_CHECK_ERROR(code)  { try {code;} \
    catch(...) {_thrustCheckError(__FILE__, __LINE__, #code);} }


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
        fprintf(stderr, "ERROR: %s failed at %s:%i : %s\n",
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
        fprintf(stderr, "ERROR: %s failed at %s:%i : %s\n",
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
        fprintf(stderr, "ERROR: CUDA_CHECK_ERROR() failed after %s at %s:%i : %s\n",
                msg, file, line, cudaGetErrorString(err));
        exit(-1);
    }
#else
#ifdef BRIAN2CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "ERROR: CUDA_CHECK_ERROR() failed at %s:%i : %s\n",
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
        fprintf(stdout, "INFO: cuda device memory usage in %s:%i\n"
               "\t used:  \t %f MB\n"
               "\t avail: \t %f MB\n"
               "\t total: \t %f MB\n"
               "\t diff:  \t %f MB \t (%zu bytes)\n",
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
    fprintf(stderr, "ERROR: THRUST_CHECK_ERROR() caught an exception from %s at %s:%i\n",
            code, file, line);
    throw;
}

#endif  // BRIAN2CUDA_ERROR_CHECK_H
