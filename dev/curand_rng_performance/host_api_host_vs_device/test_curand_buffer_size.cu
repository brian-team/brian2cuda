/*
 ============================================================================
 Name        : test_curand_buffer_size.cu
 Author      : Denis Alevi
 ============================================================================
 */

/*
 * This program uses the cuRAND host API to generate
 * pseudorandom floats on host and device and compares
 * execution times.
 */
#include <ctime>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <assert.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

// switch below btwn float and double rng

/* FLOAT */
typedef float size_type;
#define CURAND_GENERATE(x,y,z) curandGenerateUniform(x,y,z)

/* DOUBLE */
//typedef double size_type;
//#define CURAND_GENERATE(x,y,z) curandGenerateUniformDouble(x,y,z)

int rngOnDevice(size_t N, size_t max_bs)
{
    std::cout << "Creating random numbers on device and copying to host." << std::endl;
    std::cout << "[execution time [ms], buffer size]:" << std::endl;

    /* Create pseudo-random number generator */
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_DEFAULT));

    size_t bs, alloc_size, i;
    size_type *devData, *hostData, *compareData;

    alloc_size = (N < max_bs) ? max_bs : N;  // get max(N, bs)

    /* Allocate alloc_size size_types on host */
    hostData = (size_type *)calloc(alloc_size, sizeof(size_type));

    /* Allocate alloc_size size_types for comparison on host */
    compareData = (size_type *)calloc(alloc_size, sizeof(size_type));

    /* Allocate N size_types on device */
    CUDA_CALL(cudaMalloc((void **)&devData, alloc_size*sizeof(size_type)));

    for (bs=1; bs<=max_bs; bs*=10)  // how many random numbers per curand call
    {
        /* Set seed */
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
                    1234ULL));

        /* Reset offset */
        CURAND_CALL(curandSetGeneratorOffset(gen, 0ULL));

        size_t start = std::clock();

        for(i=0; i<N; i+=bs)
        {
            /* Generate bs size_types on device */
            CURAND_CALL(CURAND_GENERATE(gen, devData+i, bs));

            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostData+i, devData+i, bs * sizeof(size_type), cudaMemcpyDeviceToHost));
        }


        size_t stop = std::clock();
        std::cout << "[" << (stop - start)/double(CLOCKS_PER_SEC)*1000 << ", " << bs << "]," << std::endl;

        if (bs==1)
        {
            // copy results
            for (i=0; i<N; i++)
            {
                compareData[i] = hostData[i];
            }
        }
        else
        {
            // check results
            for (i=0; i<N; i++)
            {
                assert (hostData[i] == compareData[i]);
            }
        }

    }  // for bs

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    free(hostData);
    CUDA_CALL(cudaFree(devData));
    return EXIT_SUCCESS;
}

int rngOnHost(size_t N, size_t max_bs)
{
    std::cout << "Creating random numbers on host." << std::endl;
    std::cout << "[execution time [ms], buffer size]:" << std::endl;

    /* Create pseudo-random number generator */
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGeneratorHost(&gen,
                CURAND_RNG_PSEUDO_DEFAULT));

    size_t bs, alloc_size, i;
    size_type *hostData, *compareData;

    alloc_size = (N < max_bs) ? max_bs : N;  // get max(N, bs)

    /* Allocate alloc_size size_types on host */
    hostData = (size_type *)calloc(alloc_size, sizeof(size_type));

    /* Allocate alloc_size size_types for comparison on host */
    compareData = (size_type *)calloc(alloc_size, sizeof(size_type));

    for (bs=1; bs<=max_bs; bs*=10)  // how many random numbers per curand call
    {
        /* Set seed */
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
                    1234ULL));

        /* Reset offset */
        CURAND_CALL(curandSetGeneratorOffset(gen, 0ULL));

        size_t start = std::clock();

        for(i=0; i<N; i+=bs)
        {
            /* Generate bs size_types on host */
            CURAND_CALL(CURAND_GENERATE(gen, hostData+i, bs));
        }

        size_t stop = std::clock();
        std::cout << "[" << (stop - start)/double(CLOCKS_PER_SEC)*1000 << ", " << bs << "]," << std::endl;

        if (bs==1)
        {
            // copy results
            for (i=0; i<N; i++)
            {
                compareData[i] = hostData[i];
            }
        }
        else
        {
            // check results
            for (i=0; i<N; i++)
            {
                assert(hostData[i] == compareData[i]);
            }
        }

    }  // for bs

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    free(hostData);
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    size_t N_max = 1000000;

    for (size_t N=1; N<=N_max; N*=10)
    {
        std::cout << std::endl;
        std::cout << "Total number of random numbers to create is N = " << N << std::endl << std::endl;

        rngOnHost(N, N_max);
        rngOnDevice(N, N_max);
    }

    return EXIT_SUCCESS;
}
