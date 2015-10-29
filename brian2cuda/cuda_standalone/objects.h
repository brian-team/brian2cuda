
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "network.h"

#include <thrust/device_vector.h>
#include <curand.h>

namespace brian {

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;
extern Network magicnetwork;

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_spikegeneratorgroup_neuron_index;
extern thrust::device_vector<int32_t> dev_dynamic_array_spikegeneratorgroup_neuron_index;
extern std::vector<int32_t> _dynamic_array_spikegeneratorgroup_spike_number;
extern thrust::device_vector<int32_t> dev_dynamic_array_spikegeneratorgroup_spike_number;
extern std::vector<double> _dynamic_array_spikegeneratorgroup_spike_time;
extern thrust::device_vector<double> dev_dynamic_array_spikegeneratorgroup_spike_time;
extern std::vector<int32_t> _dynamic_array_spikemonitor_i;
extern thrust::device_vector<int32_t> dev_dynamic_array_spikemonitor_i;
extern std::vector<double> _dynamic_array_spikemonitor_t;
extern thrust::device_vector<double> dev_dynamic_array_spikemonitor_t;
extern std::vector<double> _dynamic_array_statemonitor_1_t;
extern thrust::device_vector<double> dev_dynamic_array_statemonitor_1_t;
extern std::vector<double> _dynamic_array_statemonitor_t;
extern thrust::device_vector<double> dev_dynamic_array_statemonitor_t;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_1__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_1__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_1_lastupdate;
extern thrust::device_vector<double> dev_dynamic_array_synapses_1_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_1_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_1_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_1_pre_delay;
extern thrust::device_vector<double> dev_dynamic_array_synapses_1_pre_delay;
extern std::vector<int32_t> _dynamic_array_synapses_1_pre_spiking_synapses;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_1_pre_spiking_synapses;
extern std::vector<double> _dynamic_array_synapses_1_w;
extern thrust::device_vector<double> dev_dynamic_array_synapses_1_w;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_Apost;
extern thrust::device_vector<double> dev_dynamic_array_synapses_Apost;
extern std::vector<double> _dynamic_array_synapses_Apre;
extern thrust::device_vector<double> dev_dynamic_array_synapses_Apre;
extern std::vector<double> _dynamic_array_synapses_lastupdate;
extern thrust::device_vector<double> dev_dynamic_array_synapses_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_N_incoming;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_N_outgoing;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_post_delay;
extern thrust::device_vector<double> dev_dynamic_array_synapses_post_delay;
extern std::vector<int32_t> _dynamic_array_synapses_post_spiking_synapses;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_post_spiking_synapses;
extern std::vector<double> _dynamic_array_synapses_pre_delay;
extern thrust::device_vector<double> dev_dynamic_array_synapses_pre_delay;
extern std::vector<int32_t> _dynamic_array_synapses_pre_spiking_synapses;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_pre_spiking_synapses;
extern std::vector<double> _dynamic_array_synapses_w;
extern thrust::device_vector<double> dev_dynamic_array_synapses_w;

//////////////// arrays ///////////////////
extern double * _array_defaultclock_dt;
extern double * dev_array_defaultclock_dt;
extern __device__ double *d_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double * _array_defaultclock_t;
extern double * dev_array_defaultclock_t;
extern __device__ double *d_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern uint64_t * _array_defaultclock_timestep;
extern uint64_t * dev_array_defaultclock_timestep;
extern __device__ uint64_t *d_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t * _array_neurongroup__spikespace;
extern int32_t * dev_array_neurongroup__spikespace;
extern __device__ int32_t *d_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;
extern double * _array_neurongroup_g;
extern double * dev_array_neurongroup_g;
extern __device__ double *d_array_neurongroup_g;
extern const int _num__array_neurongroup_g;
extern int32_t * _array_neurongroup_i;
extern int32_t * dev_array_neurongroup_i;
extern __device__ int32_t *d_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double * _array_neurongroup_lastspike;
extern double * dev_array_neurongroup_lastspike;
extern __device__ double *d_array_neurongroup_lastspike;
extern const int _num__array_neurongroup_lastspike;
extern char * _array_neurongroup_not_refractory;
extern char * dev_array_neurongroup_not_refractory;
extern __device__ char *d_array_neurongroup_not_refractory;
extern const int _num__array_neurongroup_not_refractory;
extern double * _array_neurongroup_v;
extern double * dev_array_neurongroup_v;
extern __device__ double *d_array_neurongroup_v;
extern const int _num__array_neurongroup_v;
extern int32_t * _array_spikegeneratorgroup__lastindex;
extern int32_t * dev_array_spikegeneratorgroup__lastindex;
extern __device__ int32_t *d_array_spikegeneratorgroup__lastindex;
extern const int _num__array_spikegeneratorgroup__lastindex;
extern int32_t * _array_spikegeneratorgroup__spikespace;
extern int32_t * dev_array_spikegeneratorgroup__spikespace;
extern __device__ int32_t *d_array_spikegeneratorgroup__spikespace;
extern const int _num__array_spikegeneratorgroup__spikespace;
extern int32_t * _array_spikegeneratorgroup_i;
extern int32_t * dev_array_spikegeneratorgroup_i;
extern __device__ int32_t *d_array_spikegeneratorgroup_i;
extern const int _num__array_spikegeneratorgroup_i;
extern double * _array_spikegeneratorgroup_period;
extern double * dev_array_spikegeneratorgroup_period;
extern __device__ double *d_array_spikegeneratorgroup_period;
extern const int _num__array_spikegeneratorgroup_period;
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
extern int32_t * _array_statemonitor_1__indices;
extern int32_t * dev_array_statemonitor_1__indices;
extern __device__ int32_t *d_array_statemonitor_1__indices;
extern const int _num__array_statemonitor_1__indices;
extern int32_t * _array_statemonitor_1_N;
extern int32_t * dev_array_statemonitor_1_N;
extern __device__ int32_t *d_array_statemonitor_1_N;
extern const int _num__array_statemonitor_1_N;
extern double * _array_statemonitor_1_v;
extern double * dev_array_statemonitor_1_v;
extern __device__ double *d_array_statemonitor_1_v;
extern const int _num__array_statemonitor_1_v;
extern int32_t * _array_statemonitor__indices;
extern int32_t * dev_array_statemonitor__indices;
extern __device__ int32_t *d_array_statemonitor__indices;
extern const int _num__array_statemonitor__indices;
extern int32_t * _array_statemonitor_N;
extern int32_t * dev_array_statemonitor_N;
extern __device__ int32_t *d_array_statemonitor_N;
extern const int _num__array_statemonitor_N;
extern double * _array_statemonitor_w;
extern double * dev_array_statemonitor_w;
extern __device__ double *d_array_statemonitor_w;
extern const int _num__array_statemonitor_w;
extern int32_t * _array_synapses_1_N;
extern int32_t * dev_array_synapses_1_N;
extern __device__ int32_t *d_array_synapses_1_N;
extern const int _num__array_synapses_1_N;
extern int32_t * _array_synapses_N;
extern int32_t * dev_array_synapses_N;
extern __device__ int32_t *d_array_synapses_N;
extern const int _num__array_synapses_N;

//////////////// dynamic arrays 2d /////////
extern thrust::device_vector<double*> addresses_monitor__dynamic_array_statemonitor_1_v;
extern thrust::device_vector<double>* _dynamic_array_statemonitor_1_v;
extern thrust::device_vector<double*> addresses_monitor__dynamic_array_statemonitor_w;
extern thrust::device_vector<double>* _dynamic_array_statemonitor_w;

/////////////// static arrays /////////////
extern double *_static_array__array_neurongroup_v;
extern double *dev_static_array__array_neurongroup_v;
extern __device__ double *d_static_array__array_neurongroup_v;
extern const int _num__static_array__array_neurongroup_v;
extern int32_t *_static_array__array_statemonitor_1__indices;
extern int32_t *dev_static_array__array_statemonitor_1__indices;
extern __device__ int32_t *d_static_array__array_statemonitor_1__indices;
extern const int _num__static_array__array_statemonitor_1__indices;
extern int32_t *_static_array__array_statemonitor__indices;
extern int32_t *dev_static_array__array_statemonitor__indices;
extern __device__ int32_t *d_static_array__array_statemonitor__indices;
extern const int _num__static_array__array_statemonitor__indices;
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
extern double *_static_array__dynamic_array_synapses_w;
extern double *dev_static_array__dynamic_array_synapses_w;
extern __device__ double *d_static_array__dynamic_array_synapses_w;
extern const int _num__static_array__dynamic_array_synapses_w;

//////////////// synapses /////////////////
// synapses
extern Synapses<double> synapses;
extern __device__ unsigned* synapses_post_size_by_pre;
extern __device__ int32_t** synapses_post_synapses_id_by_pre;
extern __device__ unsigned int** synapses_post_delay_by_pre;
extern __device__ SynapticPathway<double> synapses_post;
extern __device__ unsigned* synapses_pre_size_by_pre;
extern __device__ int32_t** synapses_pre_synapses_id_by_pre;
extern __device__ unsigned int** synapses_pre_delay_by_pre;
extern __device__ SynapticPathway<double> synapses_pre;
// synapses_1
extern Synapses<double> synapses_1;
extern __device__ unsigned* synapses_1_pre_size_by_pre;
extern __device__ int32_t** synapses_1_pre_synapses_id_by_pre;
extern __device__ unsigned int** synapses_1_pre_delay_by_pre;
extern __device__ SynapticPathway<double> synapses_1_pre;

//////////////// random numbers /////////////////
extern curandGenerator_t random_float_generator;


//CUDA
extern unsigned int num_parallel_blocks;
extern unsigned int max_threads_per_block;
extern unsigned int max_shared_mem_size;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


