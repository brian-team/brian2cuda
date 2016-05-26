
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
extern std::vector<int32_t> _dynamic_array_spikemonitor_i;
extern thrust::device_vector<int32_t> dev_dynamic_array_spikemonitor_i;
extern std::vector<double> _dynamic_array_spikemonitor_t;
extern thrust::device_vector<double> dev_dynamic_array_spikemonitor_t;

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
extern int32_t * _array_neurongroup_i;
extern int32_t * dev_array_neurongroup_i;
extern __device__ int32_t *d_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double * _array_neurongroup_v;
extern double * dev_array_neurongroup_v;
extern __device__ double *d_array_neurongroup_v;
extern const int _num__array_neurongroup_v;
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

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////

//////////////// synapses /////////////////

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


