
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<vector>
#include <omp.h>

namespace brian {

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;
extern Network magicnetwork;

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_spikegeneratorgroup_neuron_index;
extern std::vector<int32_t> _dynamic_array_spikegeneratorgroup_spike_number;
extern std::vector<double> _dynamic_array_spikegeneratorgroup_spike_time;
extern std::vector<int32_t> _dynamic_array_spikemonitor_i;
extern std::vector<double> _dynamic_array_spikemonitor_t;
extern std::vector<double> _dynamic_array_statemonitor_1_t;
extern std::vector<double> _dynamic_array_statemonitor_t;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_1_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_1_pre_delay;
extern std::vector<int32_t> _dynamic_array_synapses_1_pre_spiking_synapses;
extern std::vector<double> _dynamic_array_synapses_1_w;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_Apost;
extern std::vector<double> _dynamic_array_synapses_Apre;
extern std::vector<double> _dynamic_array_synapses_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_post_delay;
extern std::vector<int32_t> _dynamic_array_synapses_post_spiking_synapses;
extern std::vector<double> _dynamic_array_synapses_pre_delay;
extern std::vector<int32_t> _dynamic_array_synapses_pre_spiking_synapses;
extern std::vector<double> _dynamic_array_synapses_w;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern uint64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;
extern double *_array_neurongroup_g;
extern const int _num__array_neurongroup_g;
extern int32_t *_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double *_array_neurongroup_lastspike;
extern const int _num__array_neurongroup_lastspike;
extern char *_array_neurongroup_not_refractory;
extern const int _num__array_neurongroup_not_refractory;
extern double *_array_neurongroup_v;
extern const int _num__array_neurongroup_v;
extern int32_t *_array_spikegeneratorgroup__lastindex;
extern const int _num__array_spikegeneratorgroup__lastindex;
extern int32_t *_array_spikegeneratorgroup__spikespace;
extern const int _num__array_spikegeneratorgroup__spikespace;
extern int32_t *_array_spikegeneratorgroup_i;
extern const int _num__array_spikegeneratorgroup_i;
extern double *_array_spikegeneratorgroup_period;
extern const int _num__array_spikegeneratorgroup_period;
extern int32_t *_array_spikemonitor__source_idx;
extern const int _num__array_spikemonitor__source_idx;
extern int32_t *_array_spikemonitor_count;
extern const int _num__array_spikemonitor_count;
extern int32_t *_array_spikemonitor_N;
extern const int _num__array_spikemonitor_N;
extern int32_t *_array_statemonitor_1__indices;
extern const int _num__array_statemonitor_1__indices;
extern int32_t *_array_statemonitor_1_N;
extern const int _num__array_statemonitor_1_N;
extern double *_array_statemonitor_1_v;
extern const int _num__array_statemonitor_1_v;
extern int32_t *_array_statemonitor__indices;
extern const int _num__array_statemonitor__indices;
extern int32_t *_array_statemonitor_N;
extern const int _num__array_statemonitor_N;
extern double *_array_statemonitor_w;
extern const int _num__array_statemonitor_w;
extern int32_t *_array_synapses_1_N;
extern const int _num__array_synapses_1_N;
extern int32_t *_array_synapses_N;
extern const int _num__array_synapses_N;

//////////////// dynamic arrays 2d /////////
extern DynamicArray2D<double> _dynamic_array_statemonitor_1_v;
extern DynamicArray2D<double> _dynamic_array_statemonitor_w;

/////////////// static arrays /////////////
extern double *_static_array__array_neurongroup_v;
extern const int _num__static_array__array_neurongroup_v;
extern int32_t *_static_array__array_statemonitor_1__indices;
extern const int _num__static_array__array_statemonitor_1__indices;
extern int32_t *_static_array__array_statemonitor__indices;
extern const int _num__static_array__array_statemonitor__indices;
extern int64_t *_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup_neuron_index;
extern int64_t *_static_array__dynamic_array_spikegeneratorgroup_spike_number;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup_spike_number;
extern double *_static_array__dynamic_array_spikegeneratorgroup_spike_time;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup_spike_time;
extern double *_static_array__dynamic_array_synapses_w;
extern const int _num__static_array__dynamic_array_synapses_w;

//////////////// synapses /////////////////
// synapses
extern SynapticPathway<double> synapses_post;
extern SynapticPathway<double> synapses_pre;
// synapses_1
extern SynapticPathway<double> synapses_1_pre;

// Profiling information for each code object
extern double neurongroup_resetter_codeobject_profiling_info;
extern double neurongroup_stateupdater_codeobject_profiling_info;
extern double neurongroup_thresholder_codeobject_profiling_info;
extern double spikegeneratorgroup_codeobject_profiling_info;
extern double spikemonitor_codeobject_profiling_info;
extern double statemonitor_1_codeobject_profiling_info;
extern double statemonitor_codeobject_profiling_info;
extern double synapses_1_group_variable_set_conditional_codeobject_profiling_info;
extern double synapses_1_pre_codeobject_profiling_info;
extern double synapses_1_pre_initialise_queue_profiling_info;
extern double synapses_1_pre_push_spikes_profiling_info;
extern double synapses_1_synapses_create_codeobject_profiling_info;
extern double synapses_group_variable_set_conditional_codeobject_profiling_info;
extern double synapses_post_codeobject_profiling_info;
extern double synapses_post_initialise_queue_profiling_info;
extern double synapses_post_push_spikes_profiling_info;
extern double synapses_pre_codeobject_profiling_info;
extern double synapses_pre_initialise_queue_profiling_info;
extern double synapses_pre_push_spikes_profiling_info;
extern double synapses_synapses_create_codeobject_profiling_info;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


