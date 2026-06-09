/*
 * CUDA to HIP compatibility header for Brian2CUDA
 *
 * This header provides CUDA->HIP symbol mapping when building with HIP/ROCm.
 * On NVIDIA platforms, this header is a no-op.
 *
 * Usage: Include this header first in any .cu file that uses CUDA runtime/library calls.
 * The generated code continues to use CUDA spelling (cudaMalloc, curand*, etc.);
 * this header translates them to HIP equivalents at compile time.
 */

#ifndef BRIAN2CUDA_CUDA_TO_HIP_H
#define BRIAN2CUDA_CUDA_TO_HIP_H

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)

// HIP runtime replaces CUDA runtime
#include <hip/hip_runtime.h>

// hipRAND replaces cuRAND
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

// rocThrust replaces Thrust (thrust headers work directly with HIP)
// No explicit include needed - thrust/ headers are provided by rocThrust

// CUDA runtime API -> HIP runtime API
#define cudaMalloc                  hipMalloc
#define cudaFree                    hipFree
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice
#define cudaMemset                  hipMemset
#define cudaMemGetInfo              hipMemGetInfo
// hipMemcpyToSymbol requires HIP_SYMBOL() around the device symbol name
// but we can use a helper macro that wraps the symbol
#define cudaMemcpyToSymbol(symbol, src, count, ...) \
    hipMemcpyToSymbol(HIP_SYMBOL(symbol), src, count, ##__VA_ARGS__)
#define cudaMemcpyFromSymbol(dst, symbol, count, ...) \
    hipMemcpyFromSymbol(dst, HIP_SYMBOL(symbol), count, ##__VA_ARGS__)

#define cudaSetDevice               hipSetDevice
#define cudaGetDevice               hipGetDevice
#define cudaGetDeviceCount          hipGetDeviceCount
#define cudaGetDeviceProperties     hipGetDeviceProperties
#define cudaDeviceSetLimit          hipDeviceSetLimit
#define cudaDeviceSynchronize       hipDeviceSynchronize
#define cudaDeviceReset             hipDeviceReset

#define cudaStream_t                hipStream_t
#define cudaStreamCreate            hipStreamCreate
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize

#define cudaEvent_t                 hipEvent_t
#define cudaEventCreate             hipEventCreate
#define cudaEventDestroy            hipEventDestroy
#define cudaEventRecord             hipEventRecord
#define cudaEventSynchronize        hipEventSynchronize
#define cudaEventElapsedTime        hipEventElapsedTime

#define cudaError_t                 hipError_t
#define cudaError                   hipError_t
#define cudaSuccess                 hipSuccess
#define cudaGetLastError            hipGetLastError
#define cudaGetErrorString          hipGetErrorString
#define cudaGetErrorName            hipGetErrorName

#define cudaDeviceProp              hipDeviceProp_t
#define cudaLimitMallocHeapSize     hipLimitMallocHeapSize

// cuRAND host API -> hipRAND host API
#define curandGenerator_t           hiprandGenerator_t
#define curandStatus_t              hiprandStatus_t
#define curandRngType_t             hiprandRngType_t

#define curandCreateGenerator       hiprandCreateGenerator
#define curandDestroyGenerator      hiprandDestroyGenerator
#define curandSetPseudoRandomGeneratorSeed  hiprandSetPseudoRandomGeneratorSeed
#define curandSetGeneratorOrdering  hiprandSetGeneratorOrdering
#define curandSetStream             hiprandSetStream
#define curandSetGeneratorOffset    hiprandSetGeneratorOffset
#define curandGenerate              hiprandGenerate
#define curandGenerateUniform       hiprandGenerateUniform
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
#define curandGenerateNormal        hiprandGenerateNormal
#define curandGenerateNormalDouble  hiprandGenerateNormalDouble
#define curandGeneratePoisson       hiprandGeneratePoisson

// cuRAND status codes -> hipRAND status codes
#define CURAND_STATUS_SUCCESS                   HIPRAND_STATUS_SUCCESS
#define CURAND_STATUS_VERSION_MISMATCH          HIPRAND_STATUS_VERSION_MISMATCH
#define CURAND_STATUS_NOT_INITIALIZED           HIPRAND_STATUS_NOT_INITIALIZED
#define CURAND_STATUS_ALLOCATION_FAILED         HIPRAND_STATUS_ALLOCATION_FAILED
#define CURAND_STATUS_TYPE_ERROR                HIPRAND_STATUS_TYPE_ERROR
#define CURAND_STATUS_OUT_OF_RANGE              HIPRAND_STATUS_OUT_OF_RANGE
#define CURAND_STATUS_LENGTH_NOT_MULTIPLE       HIPRAND_STATUS_LENGTH_NOT_MULTIPLE
#define CURAND_STATUS_DOUBLE_PRECISION_REQUIRED HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define CURAND_STATUS_LAUNCH_FAILURE            HIPRAND_STATUS_LAUNCH_FAILURE
#define CURAND_STATUS_PREEXISTING_FAILURE       HIPRAND_STATUS_PREEXISTING_FAILURE
#define CURAND_STATUS_INITIALIZATION_FAILED     HIPRAND_STATUS_INITIALIZATION_FAILED
#define CURAND_STATUS_ARCH_MISMATCH             HIPRAND_STATUS_ARCH_MISMATCH
#define CURAND_STATUS_INTERNAL_ERROR            HIPRAND_STATUS_INTERNAL_ERROR

// cuRAND generator types -> hipRAND generator types
#define CURAND_RNG_PSEUDO_DEFAULT       HIPRAND_RNG_PSEUDO_DEFAULT
#define CURAND_RNG_PSEUDO_XORWOW        HIPRAND_RNG_PSEUDO_XORWOW
#define CURAND_RNG_PSEUDO_MRG32K3A      HIPRAND_RNG_PSEUDO_MRG32K3A
#define CURAND_RNG_PSEUDO_MTGP32        HIPRAND_RNG_PSEUDO_MTGP32
#define CURAND_RNG_PSEUDO_PHILOX4_32_10 HIPRAND_RNG_PSEUDO_PHILOX4_32_10
#define CURAND_RNG_PSEUDO_MT19937       HIPRAND_RNG_PSEUDO_MT19937
#define CURAND_RNG_QUASI_DEFAULT        HIPRAND_RNG_QUASI_DEFAULT
#define CURAND_RNG_QUASI_SOBOL32        HIPRAND_RNG_QUASI_SOBOL32
#define CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
#define CURAND_RNG_QUASI_SOBOL64        HIPRAND_RNG_QUASI_SOBOL64
#define CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64

// cuRAND ordering types -> hipRAND ordering types
#define CURAND_ORDERING_PSEUDO_DEFAULT  HIPRAND_ORDERING_PSEUDO_DEFAULT
#define CURAND_ORDERING_PSEUDO_BEST     HIPRAND_ORDERING_PSEUDO_BEST
#define CURAND_ORDERING_PSEUDO_SEEDED   HIPRAND_ORDERING_PSEUDO_SEEDED
#define CURAND_ORDERING_QUASI_DEFAULT   HIPRAND_ORDERING_QUASI_DEFAULT

// cuRAND device API -> hipRAND device API
#define curandState                 hiprandState
#define curandState_t               hiprandState_t
#define curandStateXORWOW           hiprandStateXORWOW
#define curandStateXORWOW_t         hiprandStateXORWOW_t
#define curandStateMRG32k3a         hiprandStateMRG32k3a
#define curandStateMRG32k3a_t       hiprandStateMRG32k3a_t
#define curandStatePhilox4_32_10    hiprandStatePhilox4_32_10
#define curandStatePhilox4_32_10_t  hiprandStatePhilox4_32_10_t

#define curand_init                 hiprand_init
#define curand                      hiprand
#define curand_uniform              hiprand_uniform
#define curand_uniform_double       hiprand_uniform_double
#define curand_normal               hiprand_normal
#define curand_normal_double        hiprand_normal_double
#define curand_log_normal           hiprand_log_normal
#define curand_log_normal_double    hiprand_log_normal_double
#define curand_poisson              hiprand_poisson

// CUDA profiler API (no-op on HIP, profiling done via rocprof)
// Return hipSuccess to make CUDA_SAFE_CALL work
#define cudaProfilerStart()         hipSuccess
#define cudaProfilerStop()          hipSuccess

// Atomic operations (HIP has native support)
// These are already defined in HIP headers, but we ensure they're available
#ifndef atomicCAS
#define atomicCAS                   atomicCAS
#endif
#ifndef atomicAdd
#define atomicAdd                   atomicAdd
#endif
#ifndef atomicExch
#define atomicExch                  atomicExch
#endif
#ifndef atomicMin
#define atomicMin                   atomicMin
#endif
#ifndef atomicMax
#define atomicMax                   atomicMax
#endif

// Thread synchronization
#ifndef __syncthreads
#define __syncthreads               __syncthreads
#endif
#ifndef __threadfence
#define __threadfence               __threadfence
#endif
#ifndef __threadfence_block
#define __threadfence_block         __threadfence_block
#endif

// Occupancy API
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxPotentialBlockSize            hipOccupancyMaxPotentialBlockSize
#define cudaOccupancyMaxPotentialBlockSizeWithFlags   hipOccupancyMaxPotentialBlockSizeWithFlags

// Device properties
#define cudaFuncCachePreferNone     hipFuncCachePreferNone
#define cudaFuncCachePreferShared   hipFuncCachePreferShared
#define cudaFuncCachePreferL1       hipFuncCachePreferL1
#define cudaFuncCachePreferEqual    hipFuncCachePreferEqual
#define cudaFuncSetCacheConfig      hipFuncSetCacheConfig

// Function attributes
#define cudaFuncAttributes          hipFuncAttributes
// hipFuncGetAttributes needs (void*) cast for function pointer
#define cudaFuncGetAttributes(attr, func) hipFuncGetAttributes(attr, (const void*)(func))

#else  // NVIDIA CUDA platform

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#endif  // __HIP_PLATFORM_AMD__ || USE_HIP

#endif  // BRIAN2CUDA_CUDA_TO_HIP_H
