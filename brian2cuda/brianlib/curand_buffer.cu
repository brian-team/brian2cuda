#define BRIAN2CUDA_CURAND_HOST
#include "brianlib/curand_buffer.h"
#include "brianlib/cuda_utils.h"
#include <cstdio>
#include <cstdlib>
#include <curand.h>

namespace {

template <class T>
void cb_uniform(curandGenerator_t gen, T* out, size_t n);
template <>
void cb_uniform<float>(curandGenerator_t gen, float* out, size_t n)
{
    CUDA_SAFE_CALL(curandGenerateUniform(gen, out, n));
}
template <>
void cb_uniform<double>(curandGenerator_t gen, double* out, size_t n)
{
    CUDA_SAFE_CALL(curandGenerateUniformDouble(gen, out, n));
}

template <class T>
void cb_normal(curandGenerator_t gen, T* out, size_t n);
template <>
void cb_normal<float>(curandGenerator_t gen, float* out, size_t n)
{
    CUDA_SAFE_CALL(curandGenerateNormal(gen, out, n, 0.f, 1.f));
}
template <>
void cb_normal<double>(curandGenerator_t gen, double* out, size_t n)
{
    CUDA_SAFE_CALL(curandGenerateNormalDouble(gen, out, n, 0., 1.));
}

}  // namespace

template <class randomNumber_t>
void CurandBuffer<randomNumber_t>::generate_numbers()
{
    if (current_idx != buffer_size && memory_allocated)
    {
        printf("WARNING: CurandBuffer::generate_numbers() called before "
               "buffer was empty (current_idx = %u, buffer_size = %u)",
               current_idx, buffer_size);
    }
    if (!memory_allocated)
    {
        host_data = new randomNumber_t[buffer_size];
        if (!host_data)
        {
            printf("ERROR allocating host_data for CurandBuffer (size %ld)\n",
                   sizeof(randomNumber_t) * buffer_size);
            exit(EXIT_FAILURE);
        }
        CUDA_SAFE_CALL(cudaMalloc((void**)&dev_data, buffer_size * sizeof(randomNumber_t)));
        memory_allocated = true;
    }
    if (distribution == RAND)
        cb_uniform<randomNumber_t>(*generator, dev_data, buffer_size);
    else
        cb_normal<randomNumber_t>(*generator, dev_data, buffer_size);
    CUDA_SAFE_CALL(cudaMemcpy(
            host_data, dev_data, buffer_size * sizeof(randomNumber_t),
            cudaMemcpyDeviceToHost));
    current_idx = 0;
}

template <class randomNumber_t>
CurandBuffer<randomNumber_t>::CurandBuffer(curandGenerator_t* gen, ProbDistr distr)
{
    generator = gen;
    distribution = distr;
    buffer_size = 10000;
    current_idx = 0;
    memory_allocated = false;
    host_data = nullptr;
    dev_data = nullptr;
}

template <class randomNumber_t>
CurandBuffer<randomNumber_t>::~CurandBuffer()
{
    if (memory_allocated)
        free_memory();
}

template <class randomNumber_t>
void CurandBuffer<randomNumber_t>::free_memory()
{
    delete[] host_data;
    host_data = nullptr;
    if (dev_data)
    {
        CUDA_SAFE_CALL(cudaFree(dev_data));
        dev_data = nullptr;
    }
    memory_allocated = false;
}

template <class randomNumber_t>
randomNumber_t CurandBuffer<randomNumber_t>::operator[](const int dummy)
{
    (void)dummy;
    if (current_idx == buffer_size || !memory_allocated)
        generate_numbers();
    randomNumber_t number = host_data[current_idx];
    current_idx += 1;
    return number;
}

template class CurandBuffer<float>;
template class CurandBuffer<double>;
