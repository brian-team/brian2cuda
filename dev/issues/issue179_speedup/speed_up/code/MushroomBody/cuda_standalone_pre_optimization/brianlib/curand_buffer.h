#ifndef _CURAND_BUFFER_H
#define _CURAND_BUFFER_H

#include <stdio.h>
#include <curand.h>
#include <cuda.h>


// XXX: for some documentation on random number generation, check out our wiki:
//      https://github.com/brian-team/brian2cuda/wiki/Random-number-generation


enum ProbDistr
{
    RAND,  // uniform distribution over [0,1)
    RANDN  // standard normal distribution with mean 0 and std 1
};


template <class randomNumber_t>  // random number type
// only float and double are supported as template types
class CurandBuffer
/* This class generates a fixed sized buffer of random numbers on a cuda device,
 * copies them to the host and whenever the operater[] is called from the host
 * it returns the next random number. After all random numbers returned once,
 * a new set of numbers is generated.
 */
{
private:
    int buffer_size;
    int current_idx;
    bool memory_allocated;
    randomNumber_t* host_data;
    randomNumber_t* dev_data;
    curandGenerator_t* generator;
    ProbDistr distribution;

    void generate_numbers()
    {
        if (current_idx != buffer_size && memory_allocated)
        {
            printf("WARNING: CurandBuffer::generate_numbers() called before "
                    "buffer was empty (current_idx = %u, buffer_size = %u)",
                    current_idx, buffer_size);
        }
        // TODO: should we allocate the memory in the constructor (even if we end up not using it)?
        if (!memory_allocated)
        {
            // allocate host memory
            host_data = new randomNumber_t[buffer_size];
            if (!host_data)
            {
                printf("ERROR allocating host_data for CurandBuffer (size %ld)\n", sizeof(randomNumber_t)*buffer_size);
                exit(EXIT_FAILURE);
            }
            // allocate device memory
            cudaError_t status = cudaMalloc((void **)&dev_data, buffer_size*sizeof(randomNumber_t));
            if (status != cudaSuccess)
            {
                printf("ERROR allocating memory on device (size = %ld) in %s(%d):\n\t%s\n",
                        buffer_size*sizeof(randomNumber_t), __FILE__, __LINE__,
                        cudaGetErrorString(status));
                exit(EXIT_FAILURE);
            }
            memory_allocated = true;
        }
        // generate random numbers on device
        if (distribution == RAND)
        {
            curandStatus_t status = generateUniform(*generator, dev_data, buffer_size);
            if (status != CURAND_STATUS_SUCCESS)
            {
                printf("ERROR generating random numbers in %s(%d):\n", __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }
        }
        else  // distribution == RANDN
        {
            curandStatus_t status = generateNormal(*generator, dev_data, buffer_size, 0, 1);
            if (status != CURAND_STATUS_SUCCESS)
            {
                printf("ERROR generating normal distributed random numbers in %s(%d):\n",
                        __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }
        }
        // copy random numbers to host
        cudaError_t status = cudaMemcpy(host_data, dev_data, buffer_size*sizeof(randomNumber_t), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess)
        {
            printf("ERROR copying device to host memory (size = %ld) in %s(%d):\n\t%s\n",
                    buffer_size*sizeof(randomNumber_t), __FILE__, __LINE__,
                    cudaGetErrorString(status));
            exit(EXIT_FAILURE);
        }
        // reset buffer index
        current_idx = 0;
    }

    curandStatus_t generateUniform(curandGenerator_t generator, randomNumber_t *outputPtr, size_t num)
    {
        printf("ERROR curand can only generate random numbers as 'float' or 'double' types.\n");
        exit(EXIT_FAILURE);
    }

    curandStatus_t generateNormal(curandGenerator_t generator, randomNumber_t *outputPtr,
            size_t n, randomNumber_t mean, randomNumber_t stddev)
    {
        printf("ERROR curand can only generate random numbers as 'float' or 'double' types.\n");
        exit(EXIT_FAILURE);
    }

public:
    CurandBuffer(curandGenerator_t* gen, ProbDistr distr)
    {
        generator = gen;
        distribution = distr;
        buffer_size = 10000;
        current_idx = 0;
        memory_allocated = false;
    }

    ~CurandBuffer()
    {
        if (memory_allocated)
        {
            free_memory();
        }
    }

    // We declare the CurandBuffer in anonymous namespace (file global
    // variable) in the synapses_create_generator template, therefore its
    // declaration scope only ends at program termination, but then the CUDA
    // device is already detached, which results in an error when freeing the
    // device memory in the destructor. This method can be called to free
    // device memory manually before the destructor is called.
    void free_memory()
    {
        delete[] host_data;
        cudaError_t status = cudaFree(dev_data);
        if (status != cudaSuccess)
        {
            printf("ERROR freeing device memory in %s(%d):%s\n",
                    __FILE__, __LINE__, cudaGetErrorString(status));
            exit(EXIT_FAILURE);
        }
        memory_allocated = false;
    }

    // don't return reference to prohibit assignment
    randomNumber_t operator[](const int dummy)
    {
        // we ignore dummy and just return the next number in the buffer
        if (current_idx == buffer_size || !memory_allocated)
            generate_numbers();
        randomNumber_t number = host_data[current_idx];
        current_idx += 1;
        return number;
    }
};  // class CurandBuffer


// define generator functions depending on curand float type
// normal (RANDN)
template <> inline
curandStatus_t CurandBuffer<float>::generateNormal(curandGenerator_t generator,
        float *outputPtr, size_t n, float mean, float stddev)
{
    return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

template <> inline
curandStatus_t CurandBuffer<double>::generateNormal(curandGenerator_t generator,
        double *outputPtr, size_t n, double mean, double stddev)
{
    return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}

// uniform (RAND)
template <> inline
curandStatus_t CurandBuffer<float>::generateUniform(curandGenerator_t generator,
        float *outputPtr, size_t num)
{
    return curandGenerateUniform(generator, outputPtr, num);
}

template <> inline
curandStatus_t CurandBuffer<double>::generateUniform(curandGenerator_t generator,
        double *outputPtr, size_t num)
{
    return curandGenerateUniformDouble(generator, outputPtr, num);
}

#endif
