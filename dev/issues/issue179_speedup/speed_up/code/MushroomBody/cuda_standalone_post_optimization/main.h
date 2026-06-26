// objects.h starts here 

#include <ctime>
// typedefs need to be outside the include guards to
// be visible to all files including objects.h
typedef double randomNumber_t;  // random number type

#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "network.h"
#include "rand.h"

#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>

namespace brian {

extern size_t used_device_memory;

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;

//////////////// dynamic arrays ///////////
extern thrust::host_vector<int32_t> _dynamic_array_spikegeneratorgroup__timebins;
extern thrust::device_vector<int32_t> dev_dynamic_array_spikegeneratorgroup__timebins;
extern thrust::host_vector<int32_t> _dynamic_array_spikegeneratorgroup_neuron_index;
extern thrust::device_vector<int32_t> dev_dynamic_array_spikegeneratorgroup_neuron_index;
extern thrust::host_vector<int32_t> _dynamic_array_spikegeneratorgroup_spike_number;
extern thrust::device_vector<int32_t> dev_dynamic_array_spikegeneratorgroup_spike_number;
extern thrust::host_vector<double> _dynamic_array_spikegeneratorgroup_spike_time;
extern thrust::device_vector<double> dev_dynamic_array_spikegeneratorgroup_spike_time;
extern thrust::host_vector<int32_t> _dynamic_array_spikemonitor_1_i;
extern thrust::device_vector<int32_t> dev_dynamic_array_spikemonitor_1_i;
extern thrust::host_vector<double> _dynamic_array_spikemonitor_1_t;
extern thrust::device_vector<double> dev_dynamic_array_spikemonitor_1_t;
extern thrust::host_vector<int32_t> _dynamic_array_spikemonitor_2_i;
extern thrust::device_vector<int32_t> dev_dynamic_array_spikemonitor_2_i;
extern thrust::host_vector<double> _dynamic_array_spikemonitor_2_t;
extern thrust::device_vector<double> dev_dynamic_array_spikemonitor_2_t;
extern thrust::host_vector<int32_t> _dynamic_array_spikemonitor_i;
extern thrust::device_vector<int32_t> dev_dynamic_array_spikemonitor_i;
extern thrust::host_vector<double> _dynamic_array_spikemonitor_t;
extern thrust::device_vector<double> dev_dynamic_array_spikemonitor_t;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_1__synaptic_post;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_1__synaptic_pre;
extern thrust::host_vector<double> _dynamic_array_synapses_1_Apost;
extern thrust::device_vector<double> dev_dynamic_array_synapses_1_Apost;
extern thrust::host_vector<double> _dynamic_array_synapses_1_Apre;
extern thrust::device_vector<double> dev_dynamic_array_synapses_1_Apre;
extern thrust::host_vector<double> _dynamic_array_synapses_1_delay;
extern thrust::device_vector<double> dev_dynamic_array_synapses_1_delay;
extern thrust::host_vector<double> _dynamic_array_synapses_1_delay_1;
extern thrust::device_vector<double> dev_dynamic_array_synapses_1_delay_1;
extern thrust::host_vector<double> _dynamic_array_synapses_1_g_raw;
extern thrust::device_vector<double> dev_dynamic_array_synapses_1_g_raw;
extern thrust::host_vector<double> _dynamic_array_synapses_1_lastupdate;
extern thrust::device_vector<double> dev_dynamic_array_synapses_1_lastupdate;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_1_N_incoming;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_1_N_incoming;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_1_N_outgoing;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_2__synaptic_post;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_2__synaptic_post;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_2__synaptic_pre;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_2__synaptic_pre;
extern thrust::host_vector<double> _dynamic_array_synapses_2_delay;
extern thrust::device_vector<double> dev_dynamic_array_synapses_2_delay;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_2_N_incoming;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_2_N_incoming;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_2_N_outgoing;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_2_N_outgoing;
extern thrust::host_vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses__synaptic_post;
extern thrust::host_vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses__synaptic_pre;
extern thrust::host_vector<double> _dynamic_array_synapses_delay;
extern thrust::device_vector<double> dev_dynamic_array_synapses_delay;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_N_incoming;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_N_incoming;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_N_outgoing;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_N_outgoing;
extern thrust::host_vector<double> _dynamic_array_synapses_weight;
extern thrust::device_vector<double> dev_dynamic_array_synapses_weight;

//////////////// arrays ///////////////////
extern double * _array_defaultclock_dt;
extern double * dev_array_defaultclock_dt;
extern __device__ double *d_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double * _array_defaultclock_t;
extern double * dev_array_defaultclock_t;
extern __device__ double *d_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t * _array_defaultclock_timestep;
extern int64_t * dev_array_defaultclock_timestep;
extern __device__ int64_t *d_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern double * _array_neurongroup_1_g_eKC_eKC;
extern double * dev_array_neurongroup_1_g_eKC_eKC;
extern __device__ double *d_array_neurongroup_1_g_eKC_eKC;
extern const int _num__array_neurongroup_1_g_eKC_eKC;
extern double * _array_neurongroup_1_g_iKC_eKC;
extern double * dev_array_neurongroup_1_g_iKC_eKC;
extern __device__ double *d_array_neurongroup_1_g_iKC_eKC;
extern const int _num__array_neurongroup_1_g_iKC_eKC;
extern double * _array_neurongroup_1_h;
extern double * dev_array_neurongroup_1_h;
extern __device__ double *d_array_neurongroup_1_h;
extern const int _num__array_neurongroup_1_h;
extern int32_t * _array_neurongroup_1_i;
extern int32_t * dev_array_neurongroup_1_i;
extern __device__ int32_t *d_array_neurongroup_1_i;
extern const int _num__array_neurongroup_1_i;
extern double * _array_neurongroup_1_lastspike;
extern double * dev_array_neurongroup_1_lastspike;
extern __device__ double *d_array_neurongroup_1_lastspike;
extern const int _num__array_neurongroup_1_lastspike;
extern double * _array_neurongroup_1_m;
extern double * dev_array_neurongroup_1_m;
extern __device__ double *d_array_neurongroup_1_m;
extern const int _num__array_neurongroup_1_m;
extern double * _array_neurongroup_1_n;
extern double * dev_array_neurongroup_1_n;
extern __device__ double *d_array_neurongroup_1_n;
extern const int _num__array_neurongroup_1_n;
extern char * _array_neurongroup_1_not_refractory;
extern char * dev_array_neurongroup_1_not_refractory;
extern __device__ char *d_array_neurongroup_1_not_refractory;
extern const int _num__array_neurongroup_1_not_refractory;
extern double * _array_neurongroup_1_V;
extern double * dev_array_neurongroup_1_V;
extern __device__ double *d_array_neurongroup_1_V;
extern const int _num__array_neurongroup_1_V;
extern double * _array_neurongroup_g_PN_iKC;
extern double * dev_array_neurongroup_g_PN_iKC;
extern __device__ double *d_array_neurongroup_g_PN_iKC;
extern const int _num__array_neurongroup_g_PN_iKC;
extern double * _array_neurongroup_h;
extern double * dev_array_neurongroup_h;
extern __device__ double *d_array_neurongroup_h;
extern const int _num__array_neurongroup_h;
extern int32_t * _array_neurongroup_i;
extern int32_t * dev_array_neurongroup_i;
extern __device__ int32_t *d_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double * _array_neurongroup_lastspike;
extern double * dev_array_neurongroup_lastspike;
extern __device__ double *d_array_neurongroup_lastspike;
extern const int _num__array_neurongroup_lastspike;
extern double * _array_neurongroup_m;
extern double * dev_array_neurongroup_m;
extern __device__ double *d_array_neurongroup_m;
extern const int _num__array_neurongroup_m;
extern double * _array_neurongroup_n;
extern double * dev_array_neurongroup_n;
extern __device__ double *d_array_neurongroup_n;
extern const int _num__array_neurongroup_n;
extern char * _array_neurongroup_not_refractory;
extern char * dev_array_neurongroup_not_refractory;
extern __device__ char *d_array_neurongroup_not_refractory;
extern const int _num__array_neurongroup_not_refractory;
extern double * _array_neurongroup_V;
extern double * dev_array_neurongroup_V;
extern __device__ double *d_array_neurongroup_V;
extern const int _num__array_neurongroup_V;
extern int32_t * _array_spikegeneratorgroup__lastindex;
extern int32_t * dev_array_spikegeneratorgroup__lastindex;
extern __device__ int32_t *d_array_spikegeneratorgroup__lastindex;
extern const int _num__array_spikegeneratorgroup__lastindex;
extern int32_t * _array_spikegeneratorgroup__period_bins;
extern int32_t * dev_array_spikegeneratorgroup__period_bins;
extern __device__ int32_t *d_array_spikegeneratorgroup__period_bins;
extern const int _num__array_spikegeneratorgroup__period_bins;
extern int32_t * _array_spikegeneratorgroup_i;
extern int32_t * dev_array_spikegeneratorgroup_i;
extern __device__ int32_t *d_array_spikegeneratorgroup_i;
extern const int _num__array_spikegeneratorgroup_i;
extern double * _array_spikegeneratorgroup_period;
extern double * dev_array_spikegeneratorgroup_period;
extern __device__ double *d_array_spikegeneratorgroup_period;
extern const int _num__array_spikegeneratorgroup_period;
extern int32_t * _array_spikemonitor_1__source_idx;
extern int32_t * dev_array_spikemonitor_1__source_idx;
extern __device__ int32_t *d_array_spikemonitor_1__source_idx;
extern const int _num__array_spikemonitor_1__source_idx;
extern int32_t * _array_spikemonitor_1_count;
extern int32_t * dev_array_spikemonitor_1_count;
extern __device__ int32_t *d_array_spikemonitor_1_count;
extern const int _num__array_spikemonitor_1_count;
extern int32_t * _array_spikemonitor_1_N;
extern int32_t * dev_array_spikemonitor_1_N;
extern __device__ int32_t *d_array_spikemonitor_1_N;
extern const int _num__array_spikemonitor_1_N;
extern int32_t * _array_spikemonitor_2__source_idx;
extern int32_t * dev_array_spikemonitor_2__source_idx;
extern __device__ int32_t *d_array_spikemonitor_2__source_idx;
extern const int _num__array_spikemonitor_2__source_idx;
extern int32_t * _array_spikemonitor_2_count;
extern int32_t * dev_array_spikemonitor_2_count;
extern __device__ int32_t *d_array_spikemonitor_2_count;
extern const int _num__array_spikemonitor_2_count;
extern int32_t * _array_spikemonitor_2_N;
extern int32_t * dev_array_spikemonitor_2_N;
extern __device__ int32_t *d_array_spikemonitor_2_N;
extern const int _num__array_spikemonitor_2_N;
extern int32_t * _array_spikemonitor__source_idx;
extern int32_t * dev_array_spikemonitor__source_idx;
extern __device__ int32_t *d_array_spikemonitor__source_idx;
extern const int _num__array_spikemonitor__source_idx;
extern int32_t * _array_spikemonitor_count;
extern int32_t * dev_array_spikemonitor_count;
extern __device__ int32_t *d_array_spikemonitor_count;
extern const int _num__array_spikemonitor_count;
extern int32_t * _array_spikemonitor_N;
extern int32_t * dev_array_spikemonitor_N;
extern __device__ int32_t *d_array_spikemonitor_N;
extern const int _num__array_spikemonitor_N;
extern int32_t * _array_synapses_1_N;
extern int32_t * dev_array_synapses_1_N;
extern __device__ int32_t *d_array_synapses_1_N;
extern const int _num__array_synapses_1_N;
extern int32_t * _array_synapses_2_N;
extern int32_t * dev_array_synapses_2_N;
extern __device__ int32_t *d_array_synapses_2_N;
extern const int _num__array_synapses_2_N;
extern int32_t * _array_synapses_N;
extern int32_t * dev_array_synapses_N;
extern __device__ int32_t *d_array_synapses_N;
extern const int _num__array_synapses_N;

//////////////// eventspaces ///////////////
extern int32_t * _array_neurongroup_1__spikespace;
extern thrust::host_vector<int32_t*> dev_array_neurongroup_1__spikespace;
extern const int _num__array_neurongroup_1__spikespace;
extern int current_idx_array_neurongroup_1__spikespace;
extern int32_t * _array_neurongroup__spikespace;
extern thrust::host_vector<int32_t*> dev_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;
extern int current_idx_array_neurongroup__spikespace;
extern int32_t * _array_spikegeneratorgroup__spikespace;
extern thrust::host_vector<int32_t*> dev_array_spikegeneratorgroup__spikespace;
extern const int _num__array_spikegeneratorgroup__spikespace;
extern int current_idx_array_spikegeneratorgroup__spikespace;
extern int previous_idx_array_spikegeneratorgroup__spikespace;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
extern int32_t *_static_array__dynamic_array_spikegeneratorgroup__timebins;
extern int32_t *dev_static_array__dynamic_array_spikegeneratorgroup__timebins;
extern __device__ int32_t *d_static_array__dynamic_array_spikegeneratorgroup__timebins;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup__timebins;
extern int64_t *_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
extern int64_t *dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
extern __device__ int64_t *d_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup_neuron_index;
extern int64_t *_static_array__dynamic_array_spikegeneratorgroup_spike_number;
extern int64_t *dev_static_array__dynamic_array_spikegeneratorgroup_spike_number;
extern __device__ int64_t *d_static_array__dynamic_array_spikegeneratorgroup_spike_number;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup_spike_number;
extern double *_static_array__dynamic_array_spikegeneratorgroup_spike_time;
extern double *dev_static_array__dynamic_array_spikegeneratorgroup_spike_time;
extern __device__ double *d_static_array__dynamic_array_spikegeneratorgroup_spike_time;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup_spike_time;
extern double *_timedarray_1_values;
extern double *dev_timedarray_1_values;
extern __device__ double *d_timedarray_1_values;
extern const int _num__timedarray_1_values;
extern double *_timedarray_2_values;
extern double *dev_timedarray_2_values;
extern __device__ double *d_timedarray_2_values;
extern const int _num__timedarray_2_values;
extern double *_timedarray_3_values;
extern double *dev_timedarray_3_values;
extern __device__ double *d_timedarray_3_values;
extern const int _num__timedarray_3_values;
extern double *_timedarray_4_values;
extern double *dev_timedarray_4_values;
extern __device__ double *d_timedarray_4_values;
extern const int _num__timedarray_4_values;
extern double *_timedarray_values;
extern double *dev_timedarray_values;
extern __device__ double *d_timedarray_values;
extern const int _num__timedarray_values;

//////////////// synapses /////////////////
// synapses
extern bool synapses_multiple_pre_post;
extern __device__ int* synapses_pre_num_synapses_by_pre;
extern __device__ int* synapses_pre_num_synapses_by_bundle;
extern __device__ int* synapses_pre_unique_delays;
extern __device__ int* synapses_pre_synapses_offset_by_bundle;
extern __device__ int* synapses_pre_global_bundle_id_start_by_pre;
extern int synapses_pre_max_bundle_size;
extern int synapses_pre_mean_bundle_size;
extern int synapses_pre_max_size;
extern __device__ int* synapses_pre_num_unique_delays_by_pre;
extern int synapses_pre_max_num_unique_delays;
extern __device__ int32_t** synapses_pre_synapse_ids_by_pre;
extern __device__ int32_t* synapses_pre_synapse_ids;
extern __device__ int* synapses_pre_unique_delay_start_idcs;
extern __device__ int* synapses_pre_unique_delays_offset_by_pre;
extern __device__ SynapticPathway synapses_pre;
extern int synapses_pre_eventspace_idx;
extern int synapses_pre_delay;
extern bool synapses_pre_scalar_delay;
// synapses_1
extern bool synapses_1_multiple_pre_post;
extern __device__ int* synapses_1_post_num_synapses_by_pre;
extern __device__ int* synapses_1_post_num_synapses_by_bundle;
extern __device__ int* synapses_1_post_unique_delays;
extern __device__ int* synapses_1_post_synapses_offset_by_bundle;
extern __device__ int* synapses_1_post_global_bundle_id_start_by_pre;
extern int synapses_1_post_max_bundle_size;
extern int synapses_1_post_mean_bundle_size;
extern int synapses_1_post_max_size;
extern __device__ int* synapses_1_post_num_unique_delays_by_pre;
extern int synapses_1_post_max_num_unique_delays;
extern __device__ int32_t** synapses_1_post_synapse_ids_by_pre;
extern __device__ int32_t* synapses_1_post_synapse_ids;
extern __device__ int* synapses_1_post_unique_delay_start_idcs;
extern __device__ int* synapses_1_post_unique_delays_offset_by_pre;
extern __device__ SynapticPathway synapses_1_post;
extern int synapses_1_post_eventspace_idx;
extern int synapses_1_post_delay;
extern bool synapses_1_post_scalar_delay;
extern __device__ int* synapses_1_pre_num_synapses_by_pre;
extern __device__ int* synapses_1_pre_num_synapses_by_bundle;
extern __device__ int* synapses_1_pre_unique_delays;
extern __device__ int* synapses_1_pre_synapses_offset_by_bundle;
extern __device__ int* synapses_1_pre_global_bundle_id_start_by_pre;
extern int synapses_1_pre_max_bundle_size;
extern int synapses_1_pre_mean_bundle_size;
extern int synapses_1_pre_max_size;
extern __device__ int* synapses_1_pre_num_unique_delays_by_pre;
extern int synapses_1_pre_max_num_unique_delays;
extern __device__ int32_t** synapses_1_pre_synapse_ids_by_pre;
extern __device__ int32_t* synapses_1_pre_synapse_ids;
extern __device__ int* synapses_1_pre_unique_delay_start_idcs;
extern __device__ int* synapses_1_pre_unique_delays_offset_by_pre;
extern __device__ SynapticPathway synapses_1_pre;
extern int synapses_1_pre_eventspace_idx;
extern int synapses_1_pre_delay;
extern bool synapses_1_pre_scalar_delay;
// synapses_2
extern bool synapses_2_multiple_pre_post;
extern __device__ int* synapses_2_pre_num_synapses_by_pre;
extern __device__ int* synapses_2_pre_num_synapses_by_bundle;
extern __device__ int* synapses_2_pre_unique_delays;
extern __device__ int* synapses_2_pre_synapses_offset_by_bundle;
extern __device__ int* synapses_2_pre_global_bundle_id_start_by_pre;
extern int synapses_2_pre_max_bundle_size;
extern int synapses_2_pre_mean_bundle_size;
extern int synapses_2_pre_max_size;
extern __device__ int* synapses_2_pre_num_unique_delays_by_pre;
extern int synapses_2_pre_max_num_unique_delays;
extern __device__ int32_t** synapses_2_pre_synapse_ids_by_pre;
extern __device__ int32_t* synapses_2_pre_synapse_ids;
extern __device__ int* synapses_2_pre_unique_delay_start_idcs;
extern __device__ int* synapses_2_pre_unique_delays_offset_by_pre;
extern __device__ SynapticPathway synapses_2_pre;
extern int synapses_2_pre_eventspace_idx;
extern int synapses_2_pre_delay;
extern bool synapses_2_pre_scalar_delay;

// Profiling information for each code object

//////////////// random numbers /////////////////
extern curandGenerator_t curand_generator;
extern unsigned long long* dev_curand_seed;
extern __device__ unsigned long long* d_curand_seed;

extern curandState* dev_curand_states;
extern __device__ curandState* d_curand_states;
extern RandomNumberBuffer random_number_buffer;

//CUDA
extern int num_parallel_blocks;
extern int max_threads_per_block;
extern int max_threads_per_sm;
extern int max_shared_mem_size;
extern int num_threads_per_warp;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


// objects.h ends here

// run.h starts here


void brian_start();
void brian_end();

// run.h ends here

// rand.h starts here

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


// rand.h ends here

// synapses_classes.h starts here 


#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>

#include "brianlib/spikequeue.h"

class SynapticPathway
{
public:
    int32_t* dev_sources;
    int32_t* dev_targets;

    // first and last index in source NeuronGroup corresponding to Subgroup in SynapticPathway
    // important for Subgroups created with syntax: NeuronGroup(N=4000,...)[:3200]
    int32_t spikes_start;
    int32_t spikes_stop;

    double dt;
    CudaSpikeQueue* queue;
    bool no_or_const_delay_mode;

    //our real constructor
    __device__ void init(
            int32_t* _sources,
            int32_t* _targets,
            double _dt,
            int32_t _spikes_start,
            int32_t _spikes_stop)
    {
        dev_sources = _sources;
        dev_targets = _targets;
        dt = _dt;
        spikes_start = _spikes_start;
        spikes_stop = _spikes_stop;
        queue = new CudaSpikeQueue;
    };

    //our real destructor
    __device__ void destroy()
    {
        queue->destroy();
        delete queue;
    }
};
#endif

// synapses_classes.h ends here

// network.h starts here 

#ifndef _BRIAN_NETWORK_H
#define _BRIAN_NETWORK_H

#include <vector>
#include <utility>
#include <set>
#include <ctime>
#include "brianlib/clocks.h"

typedef void (*codeobj_func)();

class Network
{
    std::set<Clock*> clocks, curclocks;
    void compute_clocks();
    Clock* next_clocks();
public:
    std::vector< std::pair< Clock*, codeobj_func > > objects;
    double t;
    static double _last_run_time;
    static double _last_run_completed_fraction;

    Network();
    void clear();
    void add(Clock *clock, codeobj_func func);
    void run(const double duration, void (*report_func)(const double, const double, const double, const double), const double report_period);
};

#endif

// network.h ends here

// neurongroup_1_stateupdater_codeobject starts here

#ifndef _INCLUDED_neurongroup_1_stateupdater_codeobject
#define _INCLUDED_neurongroup_1_stateupdater_codeobject

#include "objects.h"

void _run_neurongroup_1_stateupdater_codeobject();


#endif

// neurongroup_1_stateupdater_codeobject ends here

// neurongroup_1_thresholder_codeobject starts here 

#ifndef _INCLUDED_neurongroup_1_thresholder_codeobject
#define _INCLUDED_neurongroup_1_thresholder_codeobject

#include "objects.h"

void _run_neurongroup_1_thresholder_codeobject();


#endif

// neurongroup_1_thresholder_codeobject ends here

// neurongroup_thresholder_codeobject starts here 

#ifndef _INCLUDED_neurongroup_thresholder_codeobject
#define _INCLUDED_neurongroup_thresholder_codeobject

#include "objects.h"

void _run_neurongroup_thresholder_codeobject();


#endif

// neurongroup_thresholder_codeobject ends here

// neurongroup_stateupdater_codeobject starts here

#ifndef _INCLUDED_neurongroup_stateupdater_codeobject
#define _INCLUDED_neurongroup_stateupdater_codeobject

#include "objects.h"

void _run_neurongroup_stateupdater_codeobject();


#endif

// neurongroup_stateupdater_codeobject ends here

// spikegeneratorgroup_codeobject.h starts here

#ifndef _INCLUDED_spikegeneratorgroup_codeobject
#define _INCLUDED_spikegeneratorgroup_codeobject

#include "objects.h"

void _run_spikegeneratorgroup_codeobject();


#endif

// spikegeneratorgroup_codeobject.h ends here


// spikemonitor_1_codeobject starts here 

#ifndef _INCLUDED_spikemonitor_1_codeobject
#define _INCLUDED_spikemonitor_1_codeobject

#include "objects.h"

void _run_spikemonitor_1_codeobject();

void _copyToHost_spikemonitor_1_codeobject();
void _debugmsg_spikemonitor_1_codeobject();

#endif

// spikemonitor_1_codeobject ends here

// spikemonitor_2_codeobject starts here 

#ifndef _INCLUDED_spikemonitor_2_codeobject
#define _INCLUDED_spikemonitor_2_codeobject

#include "objects.h"

void _run_spikemonitor_2_codeobject();

void _copyToHost_spikemonitor_2_codeobject();
void _debugmsg_spikemonitor_2_codeobject();

#endif

// spikemonitor_2_codeobject ends here

// spikemonitor_codeobject starts here 

#ifndef _INCLUDED_spikemonitor_codeobject
#define _INCLUDED_spikemonitor_codeobject

#include "objects.h"

void _run_spikemonitor_codeobject();

void _copyToHost_spikemonitor_codeobject();
void _debugmsg_spikemonitor_codeobject();

#endif

// spikemonitor code object ends here 

// synapses_1_group_variable_set_conditional_codeobject starts here

#ifndef _INCLUDED_synapses_1_group_variable_set_conditional_codeobject
#define _INCLUDED_synapses_1_group_variable_set_conditional_codeobject

#include "objects.h"

void _run_synapses_1_group_variable_set_conditional_codeobject();


#endif

// synapses_1_group_variable_set_conditional_codeobject ends here

// synapses_1_group_variable_set_conditional_codeobject_1 starts here

#ifndef _INCLUDED_synapses_1_group_variable_set_conditional_codeobject_1
#define _INCLUDED_synapses_1_group_variable_set_conditional_codeobject_1

#include "objects.h"

void _run_synapses_1_group_variable_set_conditional_codeobject_1();


#endif

// synapses_1_group_variable_set_conditional_codeobject_1 ends here

// synapses_1_post_codeobject starts here 

#ifndef _INCLUDED_synapses_1_post_codeobject
#define _INCLUDED_synapses_1_post_codeobject

#include "objects.h"

void _run_synapses_1_post_codeobject();

void _debugmsg_synapses_1_post_codeobject();

#endif

// synapses_1_post_codeobject ends here 

// synapses_1_post_initialise_queue starts here 

#ifndef _INCLUDED_synapses_1_post_initialise_queue
#define _INCLUDED_synapses_1_post_initialise_queue

void _run_synapses_1_post_initialise_queue();

#endif

// synapses_1_post_initialise_queue ends here 

// synapses_1_post_push_spikes star here

#ifndef _INCLUDED_synapses_1_post_push_spikes
#define _INCLUDED_synapses_1_post_push_spikes

#include "objects.h"

void _run_synapses_1_post_push_spikes();

#endif

// synapses_1_post_push_spikes ends here


// synapses_1_pre_codeobject starts here

#ifndef _INCLUDED_synapses_1_pre_codeobject
#define _INCLUDED_synapses_1_pre_codeobject

#include "objects.h"

void _run_synapses_1_pre_codeobject();

void _debugmsg_synapses_1_pre_codeobject();

#endif

// synapses_1_pre_codeobject ends here 


// synapses_1_pre_initialise queue startes here 

#ifndef _INCLUDED_synapses_1_pre_initialise_queue
#define _INCLUDED_synapses_1_pre_initialise_queue

void _run_synapses_1_pre_initialise_queue();

#endif

// synapses_1_preinitialise queue ends here 

// synapses_1_pre_push_spikes starts here 

#ifndef _INCLUDED_synapses_1_pre_push_spikes
#define _INCLUDED_synapses_1_pre_push_spikes

#include "objects.h"

void _run_synapses_1_pre_push_spikes();

#endif

// synapses_1_pre_push_spikes ends here 


// synapses_1_synapses_create_generator_codeobject starts here 

#ifndef _INCLUDED_synapses_1_synapses_create_generator_codeobject
#define _INCLUDED_synapses_1_synapses_create_generator_codeobject

#include "objects.h"

void _run_synapses_1_synapses_create_generator_codeobject();


#endif

// synapses_1_synapse_create_generator_codeobject ends here

