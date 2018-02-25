#ifndef BRIAN2CUDA_ERROR_CHECK_H
#define BRIAN2CUDA_ERROR_CHECK_H

// Define this to turn on error checking
#define BRIAN2CUDA_ERROR_CHECK
// Define this to make kernel calls block CPU execution
//#define BRIAN2CUDA_ERROR_CHECK_BLOCKING

// Define this to turn on memory checking
//#define BRIAN2CUDA_MEMORY_CHECK
// Define this to synchronize device before checking memory
//#define BRIAN2CUDA_MEMORY_CHECK_BLOCKING

// partly adapted from https://gist.github.com/ashwin/2652488
#define CudaSafeCall(err)       __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError()        __cudaCheckError(__FILE__, __LINE__)
#define CudaCheckMemory(param)  __cudaCheckMemory(param, __FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef BRIAN2CUDA_ERROR_CHECK
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef BRIAN2CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }

#ifdef BRIAN2CUDA_ERROR_CHECK_BLOCKING
    // More careful checking. However, this will affect performance.
    err = cudaDeviceSynchronize();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
#endif

    return;
}

#endif


// Report cuda device memory usage
inline void __cudaCheckMemory(const char *msg, const char *file, const int line)
{
#ifdef BRIAN2CUDA_MEMORY_CHECK
#ifdef BRIAN2CUDA_MEMORY_CHECK_BLOCKING
    cudaDeviceSynchronize();
#endif
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    size_t avail;
    size_t total;
    cudaMemGetInfo(&avail, &total);
    size_t used = total - avail;
    printf("INFO: cuda device memory usage in %s:%i (%s)\n"
           "\t used:  \t %f MB\n"
           "\t avail: \t %f MB\n"
           "\t total: \t %f MB\n",
           file, line, msg,
           double(used) * to_MB,
           double(avail) * to_MB,
           double(total) * to_MB);
#endif
}


// In this version the memory difference is always reported with respect to
// some reference memory, e.g. from a previous cudaMemGetInfo call
inline void __cudaCheckMemory(size_t &reference_memory, const char *file, const int line)
{
#ifdef BRIAN2CUDA_MEMORY_CHECK
#ifdef BRIAN2CUDA_MEMORY_CHECK_BLOCKING
    cudaDeviceSynchronize();
#endif
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    size_t avail;
    size_t total;
    cudaMemGetInfo(&avail, &total);
    size_t used = total - avail;
    size_t diff = used - reference_memory;
    if (diff > 0)
    {
        if (reference_memory == 0)
            diff = NAN;
        printf("INFO: cuda device memory usage in %s:%i\n"
               "\t used:  \t %f MB\n"
               "\t avail: \t %f MB\n"
               "\t total: \t %f MB\n"
               "\t diff:  \t %f MB \t (%zu bytes)\n",
               file, line,
               double(used) * to_MB,
               double(avail) * to_MB,
               double(total) * to_MB,
               double(diff) * to_MB, diff);
        reference_memory = used;
    }
#endif
}
