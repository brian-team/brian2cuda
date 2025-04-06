
#ifndef _BRIAN_RAND_H
#define _BRIAN_RAND_H

#include <curand.h>

void _run_random_number_buffer();

class RandomNumberBuffer
{
    // TODO let all random number pointers be class members of this class ->
    //      check which ones are needed as global variables, maybe have both,
    //      global and member variables? or change parameters in codeobjects?

    // before each run, buffers need to be reinitialized
    bool needs_init = true;
    // how many 'run' calls have finished
    int run_counter = 0;
    // number of needed cuRAND states
    int num_curand_states = 0;
    // number of threads and blocks to set curand states
    int num_threads_curand_init, num_blocks_curand_init;

    // how many random numbers we want to create at once (tradeoff memory usage <-> generation overhead)
    double mb_per_obj = 50;  // MB per codeobject and rand / randn
    // TODO: This assumes all random number have randomNumber_t type, but poisson
    //       objects have different type
    int floats_per_obj = (mb_per_obj * 1024.0 * 1024.0) / sizeof(randomNumber_t);

    // The number of needed random numbers per clock cycle, the generation interval, and the number generated per curand call.
    //
    // needed random numbers per clock cycle
    // int num_per_cycle_rand_{};
    //
    // number of time steps after which buffer needs to be refilled
    // int rand_interval_{};
    //
    // buffer size
    // int num_per_gen_rand_{};
    //
    // number of time steps since last buffer refill
    // int idx_rand_{};
    //
    // maximum number of random numbers fitting given allocated memory
    // int rand_floats_per_obj_{};

    // For each call of brians `run`, a new set of codeobjects (with different
    // suffixes) is generated. The following are variables for all codeobjects
    // for all runs that need random numbers.

    ////// run 0


    void init();
    void allocate_device_curand_states();
    void update_needed_number_curand_states();
    void set_curand_device_api_states(bool);
    void refill_uniform_numbers(randomNumber_t*, randomNumber_t*&, int, int&);
    void refill_normal_numbers(randomNumber_t*, randomNumber_t*&, int, int&);
    void refill_poisson_numbers(double lambda, unsigned int*, unsigned int*&, int, int&);

public:
    void next_time_step();
    void set_seed(unsigned long long);
    void run_finished();
    void ensure_enough_curand_states();
};

#endif

