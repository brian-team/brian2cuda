#ifndef _CURAND_BUFFER_H
#define _CURAND_BUFFER_H

#include <cstddef>

struct curandGenerator_st;
typedef struct curandGenerator_st* curandGenerator_t;

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

    void generate_numbers();

public:
    CurandBuffer(curandGenerator_t* gen, ProbDistr distr);
    ~CurandBuffer();
    // We declare the CurandBuffer in anonymous namespace (file global
    // variable) in the synapses_create_generator template, therefore its
    // declaration scope only ends at program termination, but then the CUDA
    // device is already detached, which results in an error when freeing the
    // device memory in the destructor. This method can be called to free
    // device memory manually before the destructor is called.
    void free_memory();
    // don't return reference to prohibit assignment
    randomNumber_t operator[](const int dummy);
};

#endif
