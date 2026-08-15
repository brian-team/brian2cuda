#ifndef BRIAN2CUDA_CURAND_UTILS_H
#define BRIAN2CUDA_CURAND_UTILS_H

#include <curand.h>
#include "brianlib/cuda_utils.h"
#include "brianlib/logging.h"

// adapted from NVIDIA cuda samples, shipped with cuda 10.1 (common/inc/helper_cuda.h)
// cuRAND API errors
inline const char *_curandGetErrorEnum(curandStatus_t error) {
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

inline void _cudaSafeCall(curandStatus_t err, const char *file, const int line, const char *call = "")
{
#ifdef BRIAN2CUDA_ERROR_CHECK
    if (CURAND_STATUS_SUCCESS != err)
    {
        B2C_LOG_ERROR("%s failed at %s:%i : %s",
                call, file, line, _curandGetErrorEnum(err));
        exit(-1);
    }
#endif

    return;
}

#endif  // BRIAN2CUDA_CURAND_UTILS_H
