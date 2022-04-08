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

//////////////// synapses /////////////////
// synapses
extern cudaStream_t stream;
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
extern cudaStream_t stream1;
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
extern cudaStream_t stream2;
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

// neurongroup
extern cudaStream_t neurongroup_stream1;
extern cudaStream_t neurongroup_stream;

//spike generator
extern cudaStream_t spikegenerator_stream;

// spike monitor
extern cudaStream_t spikemonitor_stream1;
extern cudaStream_t spikemonitor_stream;
extern cudaStream_t spikemonitor_stream2;

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


