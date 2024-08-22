
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/cuda_utils.h"
#include "network.h"
#include "rand.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <utility>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>

size_t brian::used_device_memory = 0;

//////////////// clocks ///////////////////
Clock brian::defaultclock;

//////////////// networks /////////////////
Network brian::magicnetwork;

//////////////// arrays ///////////////////
double * brian::_array_defaultclock_dt;
double * brian::dev_array_defaultclock_dt;
__device__ double * brian::d_array_defaultclock_dt;
const int brian::_num__array_defaultclock_dt = 1;

double * brian::_array_defaultclock_t;
double * brian::dev_array_defaultclock_t;
__device__ double * brian::d_array_defaultclock_t;
const int brian::_num__array_defaultclock_t = 1;

int64_t * brian::_array_defaultclock_timestep;
int64_t * brian::dev_array_defaultclock_timestep;
__device__ int64_t * brian::d_array_defaultclock_timestep;
const int brian::_num__array_defaultclock_timestep = 1;

double * brian::_array_neurongroup_1_g_eKC_eKC;
double * brian::dev_array_neurongroup_1_g_eKC_eKC;
__device__ double * brian::d_array_neurongroup_1_g_eKC_eKC;
const int brian::_num__array_neurongroup_1_g_eKC_eKC = 100;

double * brian::_array_neurongroup_1_g_iKC_eKC;
double * brian::dev_array_neurongroup_1_g_iKC_eKC;
__device__ double * brian::d_array_neurongroup_1_g_iKC_eKC;
const int brian::_num__array_neurongroup_1_g_iKC_eKC = 100;

double * brian::_array_neurongroup_1_h;
double * brian::dev_array_neurongroup_1_h;
__device__ double * brian::d_array_neurongroup_1_h;
const int brian::_num__array_neurongroup_1_h = 100;

int32_t * brian::_array_neurongroup_1_i;
int32_t * brian::dev_array_neurongroup_1_i;
__device__ int32_t * brian::d_array_neurongroup_1_i;
const int brian::_num__array_neurongroup_1_i = 100;

double * brian::_array_neurongroup_1_lastspike;
double * brian::dev_array_neurongroup_1_lastspike;
__device__ double * brian::d_array_neurongroup_1_lastspike;
const int brian::_num__array_neurongroup_1_lastspike = 100;

double * brian::_array_neurongroup_1_m;
double * brian::dev_array_neurongroup_1_m;
__device__ double * brian::d_array_neurongroup_1_m;
const int brian::_num__array_neurongroup_1_m = 100;

double * brian::_array_neurongroup_1_n;
double * brian::dev_array_neurongroup_1_n;
__device__ double * brian::d_array_neurongroup_1_n;
const int brian::_num__array_neurongroup_1_n = 100;

char * brian::_array_neurongroup_1_not_refractory;
char * brian::dev_array_neurongroup_1_not_refractory;
__device__ char * brian::d_array_neurongroup_1_not_refractory;
const int brian::_num__array_neurongroup_1_not_refractory = 100;

double * brian::_array_neurongroup_1_V;
double * brian::dev_array_neurongroup_1_V;
__device__ double * brian::d_array_neurongroup_1_V;
const int brian::_num__array_neurongroup_1_V = 100;

double * brian::_array_neurongroup_g_PN_iKC;
double * brian::dev_array_neurongroup_g_PN_iKC;
__device__ double * brian::d_array_neurongroup_g_PN_iKC;
const int brian::_num__array_neurongroup_g_PN_iKC = 2500;

double * brian::_array_neurongroup_h;
double * brian::dev_array_neurongroup_h;
__device__ double * brian::d_array_neurongroup_h;
const int brian::_num__array_neurongroup_h = 2500;

int32_t * brian::_array_neurongroup_i;
int32_t * brian::dev_array_neurongroup_i;
__device__ int32_t * brian::d_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 2500;

double * brian::_array_neurongroup_lastspike;
double * brian::dev_array_neurongroup_lastspike;
__device__ double * brian::d_array_neurongroup_lastspike;
const int brian::_num__array_neurongroup_lastspike = 2500;

double * brian::_array_neurongroup_m;
double * brian::dev_array_neurongroup_m;
__device__ double * brian::d_array_neurongroup_m;
const int brian::_num__array_neurongroup_m = 2500;

double * brian::_array_neurongroup_n;
double * brian::dev_array_neurongroup_n;
__device__ double * brian::d_array_neurongroup_n;
const int brian::_num__array_neurongroup_n = 2500;

char * brian::_array_neurongroup_not_refractory;
char * brian::dev_array_neurongroup_not_refractory;
__device__ char * brian::d_array_neurongroup_not_refractory;
const int brian::_num__array_neurongroup_not_refractory = 2500;

double * brian::_array_neurongroup_V;
double * brian::dev_array_neurongroup_V;
__device__ double * brian::d_array_neurongroup_V;
const int brian::_num__array_neurongroup_V = 2500;

int32_t * brian::_array_spikegeneratorgroup__lastindex;
int32_t * brian::dev_array_spikegeneratorgroup__lastindex;
__device__ int32_t * brian::d_array_spikegeneratorgroup__lastindex;
const int brian::_num__array_spikegeneratorgroup__lastindex = 1;

int32_t * brian::_array_spikegeneratorgroup__period_bins;
int32_t * brian::dev_array_spikegeneratorgroup__period_bins;
__device__ int32_t * brian::d_array_spikegeneratorgroup__period_bins;
const int brian::_num__array_spikegeneratorgroup__period_bins = 1;

int32_t * brian::_array_spikegeneratorgroup_i;
int32_t * brian::dev_array_spikegeneratorgroup_i;
__device__ int32_t * brian::d_array_spikegeneratorgroup_i;
const int brian::_num__array_spikegeneratorgroup_i = 100;

double * brian::_array_spikegeneratorgroup_period;
double * brian::dev_array_spikegeneratorgroup_period;
__device__ double * brian::d_array_spikegeneratorgroup_period;
const int brian::_num__array_spikegeneratorgroup_period = 1;

int32_t * brian::_array_spikemonitor_1__source_idx;
int32_t * brian::dev_array_spikemonitor_1__source_idx;
__device__ int32_t * brian::d_array_spikemonitor_1__source_idx;
const int brian::_num__array_spikemonitor_1__source_idx = 2500;

int32_t * brian::_array_spikemonitor_1_count;
int32_t * brian::dev_array_spikemonitor_1_count;
__device__ int32_t * brian::d_array_spikemonitor_1_count;
const int brian::_num__array_spikemonitor_1_count = 2500;

int32_t * brian::_array_spikemonitor_1_N;
int32_t * brian::dev_array_spikemonitor_1_N;
__device__ int32_t * brian::d_array_spikemonitor_1_N;
const int brian::_num__array_spikemonitor_1_N = 1;

int32_t * brian::_array_spikemonitor_2__source_idx;
int32_t * brian::dev_array_spikemonitor_2__source_idx;
__device__ int32_t * brian::d_array_spikemonitor_2__source_idx;
const int brian::_num__array_spikemonitor_2__source_idx = 100;

int32_t * brian::_array_spikemonitor_2_count;
int32_t * brian::dev_array_spikemonitor_2_count;
__device__ int32_t * brian::d_array_spikemonitor_2_count;
const int brian::_num__array_spikemonitor_2_count = 100;

int32_t * brian::_array_spikemonitor_2_N;
int32_t * brian::dev_array_spikemonitor_2_N;
__device__ int32_t * brian::d_array_spikemonitor_2_N;
const int brian::_num__array_spikemonitor_2_N = 1;

int32_t * brian::_array_spikemonitor__source_idx;
int32_t * brian::dev_array_spikemonitor__source_idx;
__device__ int32_t * brian::d_array_spikemonitor__source_idx;
const int brian::_num__array_spikemonitor__source_idx = 100;

int32_t * brian::_array_spikemonitor_count;
int32_t * brian::dev_array_spikemonitor_count;
__device__ int32_t * brian::d_array_spikemonitor_count;
const int brian::_num__array_spikemonitor_count = 100;

int32_t * brian::_array_spikemonitor_N;
int32_t * brian::dev_array_spikemonitor_N;
__device__ int32_t * brian::d_array_spikemonitor_N;
const int brian::_num__array_spikemonitor_N = 1;

int32_t * brian::_array_synapses_1_N;
int32_t * brian::dev_array_synapses_1_N;
__device__ int32_t * brian::d_array_synapses_1_N;
const int brian::_num__array_synapses_1_N = 1;

int32_t * brian::_array_synapses_2_N;
int32_t * brian::dev_array_synapses_2_N;
__device__ int32_t * brian::d_array_synapses_2_N;
const int brian::_num__array_synapses_2_N = 1;

int32_t * brian::_array_synapses_N;
int32_t * brian::dev_array_synapses_N;
__device__ int32_t * brian::d_array_synapses_N;
const int brian::_num__array_synapses_N = 1;


//////////////// eventspaces ///////////////
// we dynamically create multiple eventspaces in no_or_const_delay_mode
// for initiating the first spikespace, we need a host pointer
// for choosing the right spikespace, we need a global index variable
int32_t * brian::_array_neurongroup_1__spikespace;
const int brian::_num__array_neurongroup_1__spikespace = 101;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_1__spikespace(1);
int brian::current_idx_array_neurongroup_1__spikespace = 0;
int32_t * brian::_array_neurongroup__spikespace;
const int brian::_num__array_neurongroup__spikespace = 2501;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup__spikespace(1);
int brian::current_idx_array_neurongroup__spikespace = 0;
int32_t * brian::_array_spikegeneratorgroup__spikespace;
const int brian::_num__array_spikegeneratorgroup__spikespace = 101;
thrust::host_vector<int32_t*> brian::dev_array_spikegeneratorgroup__spikespace(1);
int brian::current_idx_array_spikegeneratorgroup__spikespace = 0;
int brian::previous_idx_array_spikegeneratorgroup__spikespace;

//////////////// dynamic arrays 1d /////////
thrust::host_vector<int32_t> brian::_dynamic_array_spikegeneratorgroup__timebins;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup__timebins;
thrust::host_vector<int32_t> brian::_dynamic_array_spikegeneratorgroup_neuron_index;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup_neuron_index;
thrust::host_vector<int32_t> brian::_dynamic_array_spikegeneratorgroup_spike_number;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup_spike_number;
thrust::host_vector<double> brian::_dynamic_array_spikegeneratorgroup_spike_time;
thrust::device_vector<double> brian::dev_dynamic_array_spikegeneratorgroup_spike_time;
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_1_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_1_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_1_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_1_t;
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_2_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_2_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_2_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_2_t;
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_t;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_Apost;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_Apost;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_Apre;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_Apre;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_delay;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_delay_1;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_delay_1;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_g_raw;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_g_raw;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_lastupdate;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_lastupdate;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_outgoing;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_2_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_2_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2_N_outgoing;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_weight;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_weight;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
int32_t * brian::_static_array__dynamic_array_spikegeneratorgroup__timebins;
int32_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup__timebins;
__device__ int32_t * brian::d_static_array__dynamic_array_spikegeneratorgroup__timebins;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup__timebins = 19676;
int64_t * brian::_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
int64_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
__device__ int64_t * brian::d_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_neuron_index = 19676;
int64_t * brian::_static_array__dynamic_array_spikegeneratorgroup_spike_number;
int64_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup_spike_number;
__device__ int64_t * brian::d_static_array__dynamic_array_spikegeneratorgroup_spike_number;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_spike_number = 19676;
double * brian::_static_array__dynamic_array_spikegeneratorgroup_spike_time;
double * brian::dev_static_array__dynamic_array_spikegeneratorgroup_spike_time;
__device__ double * brian::d_static_array__dynamic_array_spikegeneratorgroup_spike_time;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_spike_time = 19676;
double * brian::_timedarray_1_values;
double * brian::dev_timedarray_1_values;
__device__ double * brian::d_timedarray_1_values;
const int brian::_num__timedarray_1_values = 250000;
double * brian::_timedarray_2_values;
double * brian::dev_timedarray_2_values;
__device__ double * brian::d_timedarray_2_values;
const int brian::_num__timedarray_2_values = 250000;
double * brian::_timedarray_3_values;
double * brian::dev_timedarray_3_values;
__device__ double * brian::d_timedarray_3_values;
const int brian::_num__timedarray_3_values = 250000;
double * brian::_timedarray_4_values;
double * brian::dev_timedarray_4_values;
__device__ double * brian::d_timedarray_4_values;
const int brian::_num__timedarray_4_values = 250000;
double * brian::_timedarray_values;
double * brian::dev_timedarray_values;
__device__ double * brian::d_timedarray_values;
const int brian::_num__timedarray_values = 250000;

//////////////// synapses /////////////////
// synapses
int32_t synapses_source_start_index;
int32_t synapses_source_stop_index;
bool brian::synapses_multiple_pre_post = false;
// synapses_pre
__device__ int* brian::synapses_pre_num_synapses_by_pre;
__device__ int* brian::synapses_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_pre_unique_delays;
__device__ int* brian::synapses_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_pre_global_bundle_id_start_by_pre;
int brian::synapses_pre_max_bundle_size = 0;
int brian::synapses_pre_mean_bundle_size = 0;
int brian::synapses_pre_max_size = 0;
__device__ int* brian::synapses_pre_num_unique_delays_by_pre;
int brian::synapses_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_pre_synapse_ids;
__device__ int* brian::synapses_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_pre;
int brian::synapses_pre_eventspace_idx = 0;
int brian::synapses_pre_delay;
bool brian::synapses_pre_scalar_delay;
// synapses_1
int32_t synapses_1_source_start_index;
int32_t synapses_1_source_stop_index;
bool brian::synapses_1_multiple_pre_post = false;
// synapses_1_post
__device__ int* brian::synapses_1_post_num_synapses_by_pre;
__device__ int* brian::synapses_1_post_num_synapses_by_bundle;
__device__ int* brian::synapses_1_post_unique_delays;
__device__ int* brian::synapses_1_post_synapses_offset_by_bundle;
__device__ int* brian::synapses_1_post_global_bundle_id_start_by_pre;
int brian::synapses_1_post_max_bundle_size = 0;
int brian::synapses_1_post_mean_bundle_size = 0;
int brian::synapses_1_post_max_size = 0;
__device__ int* brian::synapses_1_post_num_unique_delays_by_pre;
int brian::synapses_1_post_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_1_post_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_1_post_synapse_ids;
__device__ int* brian::synapses_1_post_unique_delay_start_idcs;
__device__ int* brian::synapses_1_post_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_1_post;
int brian::synapses_1_post_eventspace_idx = 0;
int brian::synapses_1_post_delay;
bool brian::synapses_1_post_scalar_delay;
// synapses_1_pre
__device__ int* brian::synapses_1_pre_num_synapses_by_pre;
__device__ int* brian::synapses_1_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_1_pre_unique_delays;
__device__ int* brian::synapses_1_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_1_pre_global_bundle_id_start_by_pre;
int brian::synapses_1_pre_max_bundle_size = 0;
int brian::synapses_1_pre_mean_bundle_size = 0;
int brian::synapses_1_pre_max_size = 0;
__device__ int* brian::synapses_1_pre_num_unique_delays_by_pre;
int brian::synapses_1_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_1_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_1_pre_synapse_ids;
__device__ int* brian::synapses_1_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_1_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_1_pre;
int brian::synapses_1_pre_eventspace_idx = 0;
int brian::synapses_1_pre_delay;
bool brian::synapses_1_pre_scalar_delay;
// synapses_2
int32_t synapses_2_source_start_index;
int32_t synapses_2_source_stop_index;
bool brian::synapses_2_multiple_pre_post = false;
// synapses_2_pre
__device__ int* brian::synapses_2_pre_num_synapses_by_pre;
__device__ int* brian::synapses_2_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_2_pre_unique_delays;
__device__ int* brian::synapses_2_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_2_pre_global_bundle_id_start_by_pre;
int brian::synapses_2_pre_max_bundle_size = 0;
int brian::synapses_2_pre_mean_bundle_size = 0;
int brian::synapses_2_pre_max_size = 0;
__device__ int* brian::synapses_2_pre_num_unique_delays_by_pre;
int brian::synapses_2_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_2_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_2_pre_synapse_ids;
__device__ int* brian::synapses_2_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_2_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_2_pre;
int brian::synapses_2_pre_eventspace_idx = 0;
int brian::synapses_2_pre_delay;
bool brian::synapses_2_pre_scalar_delay;

int brian::num_parallel_blocks;
int brian::max_threads_per_block;
int brian::max_threads_per_sm;
int brian::max_shared_mem_size;
int brian::num_threads_per_warp;

__global__ void synapses_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_1_post_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_1_post.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_1_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_1_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_2_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_2_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}

// Profiling information for each code object

//////////////random numbers//////////////////
curandGenerator_t brian::curand_generator;
__device__ unsigned long long* brian::d_curand_seed;
unsigned long long* brian::dev_curand_seed;
// dev_{co.name}_{rng_type}_allocator
//      pointer to start of generated random numbers array
//      at each generation cycle this array is refilled
// dev_{co.name}_{rng_type}
//      pointer moving through generated random number array
//      until it is regenerated at the next generation cycle
curandState* brian::dev_curand_states;
__device__ curandState* brian::d_curand_states;
RandomNumberBuffer brian::random_number_buffer;

void _init_arrays()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );

    num_parallel_blocks = props.multiProcessorCount * 1;
    printf("objects cu num par blocks %d\n", num_parallel_blocks);
    max_threads_per_block = props.maxThreadsPerBlock;
    max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    max_shared_mem_size = props.sharedMemPerBlock;
    num_threads_per_warp = props.warpSize;

    // Random seeds might be overwritten in main.cu
    unsigned long long seed = time(0);

    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_curand_seed,
                sizeof(unsigned long long))
            );

    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_curand_seed, &dev_curand_seed,
                sizeof(unsigned long long*))
            );

    CUDA_SAFE_CALL(
            curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT)
            );


    // this sets seed for host and device api RNG
    random_number_buffer.set_seed(seed);

    synapses_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_pre_init");
    synapses_1_post_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_1_post_init");
    synapses_1_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2500
            );
    CUDA_CHECK_ERROR("synapses_1_pre_init");
    synapses_2_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_2_pre_init");

    // Arrays initialized to 0
            _array_defaultclock_dt = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_dt, _array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_t = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_t, _array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_timestep = new int64_t[1];
            for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_timestep, _array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_g_eKC_eKC = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_g_eKC_eKC[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_g_eKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_eKC_eKC)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_g_eKC_eKC, _array_neurongroup_1_g_eKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_eKC_eKC, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_g_iKC_eKC = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_g_iKC_eKC[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_g_iKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_iKC_eKC)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_g_iKC_eKC, _array_neurongroup_1_g_iKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_iKC_eKC, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_h = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_h[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_h, sizeof(double)*_num__array_neurongroup_1_h)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_h, _array_neurongroup_1_h, sizeof(double)*_num__array_neurongroup_1_h, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_i = new int32_t[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_i, _array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_lastspike = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_lastspike[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_lastspike, sizeof(double)*_num__array_neurongroup_1_lastspike)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_lastspike, _array_neurongroup_1_lastspike, sizeof(double)*_num__array_neurongroup_1_lastspike, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_m = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_m[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_m, sizeof(double)*_num__array_neurongroup_1_m)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_m, _array_neurongroup_1_m, sizeof(double)*_num__array_neurongroup_1_m, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_n = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_n[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_n, sizeof(double)*_num__array_neurongroup_1_n)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_n, _array_neurongroup_1_n, sizeof(double)*_num__array_neurongroup_1_n, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_not_refractory = new char[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_not_refractory[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_not_refractory, sizeof(char)*_num__array_neurongroup_1_not_refractory)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_not_refractory, _array_neurongroup_1_not_refractory, sizeof(char)*_num__array_neurongroup_1_not_refractory, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_V = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_V[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_V, sizeof(double)*_num__array_neurongroup_1_V)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_V, _array_neurongroup_1_V, sizeof(double)*_num__array_neurongroup_1_V, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_g_PN_iKC = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_g_PN_iKC[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_g_PN_iKC, sizeof(double)*_num__array_neurongroup_g_PN_iKC)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_g_PN_iKC, _array_neurongroup_g_PN_iKC, sizeof(double)*_num__array_neurongroup_g_PN_iKC, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_h = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_h[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_h, sizeof(double)*_num__array_neurongroup_h)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_h, _array_neurongroup_h, sizeof(double)*_num__array_neurongroup_h, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_i = new int32_t[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_lastspike = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_lastspike[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_lastspike, _array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_m = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_m[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_m, sizeof(double)*_num__array_neurongroup_m)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_m, _array_neurongroup_m, sizeof(double)*_num__array_neurongroup_m, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_n = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_n[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_n, sizeof(double)*_num__array_neurongroup_n)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_n, _array_neurongroup_n, sizeof(double)*_num__array_neurongroup_n, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_not_refractory = new char[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_not_refractory[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_not_refractory, sizeof(char)*_num__array_neurongroup_not_refractory)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_not_refractory, _array_neurongroup_not_refractory, sizeof(char)*_num__array_neurongroup_not_refractory, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_V = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_V[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_V, sizeof(double)*_num__array_neurongroup_V)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_V, _array_neurongroup_V, sizeof(double)*_num__array_neurongroup_V, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup__lastindex = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikegeneratorgroup__lastindex[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup__lastindex, _array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup__period_bins = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikegeneratorgroup__period_bins[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup__period_bins, sizeof(int32_t)*_num__array_spikegeneratorgroup__period_bins)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup__period_bins, _array_spikegeneratorgroup__period_bins, sizeof(int32_t)*_num__array_spikegeneratorgroup__period_bins, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup_i = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikegeneratorgroup_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup_i, _array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup_period = new double[1];
            for(int i=0; i<1; i++) _array_spikegeneratorgroup_period[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup_period, _array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_1__source_idx = new int32_t[2500];
            for(int i=0; i<2500; i++) _array_spikemonitor_1__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_1__source_idx, _array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_1_count = new int32_t[2500];
            for(int i=0; i<2500; i++) _array_spikemonitor_1_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_1_count, sizeof(int32_t)*_num__array_spikemonitor_1_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_1_count, _array_spikemonitor_1_count, sizeof(int32_t)*_num__array_spikemonitor_1_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_1_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_1_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_1_N, sizeof(int32_t)*_num__array_spikemonitor_1_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_1_N, _array_spikemonitor_1_N, sizeof(int32_t)*_num__array_spikemonitor_1_N, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_2__source_idx = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikemonitor_2__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_2__source_idx, _array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_2_count = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikemonitor_2_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_2_count, sizeof(int32_t)*_num__array_spikemonitor_2_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_2_count, _array_spikemonitor_2_count, sizeof(int32_t)*_num__array_spikemonitor_2_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_2_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_2_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_2_N, sizeof(int32_t)*_num__array_spikemonitor_2_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_2_N, _array_spikemonitor_2_N, sizeof(int32_t)*_num__array_spikemonitor_2_N, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor__source_idx = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_count = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikemonitor_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_count, _array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_N, _array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_1_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_1_N, _array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_2_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_2_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_2_N, _array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_N, _array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyHostToDevice)
                    );
            _dynamic_array_spikegeneratorgroup__timebins.resize(19676);
            THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup__timebins.resize(19676));
            for(int i=0; i<19676; i++)
            {
                _dynamic_array_spikegeneratorgroup__timebins[i] = 0;
                dev_dynamic_array_spikegeneratorgroup__timebins[i] = 0;
            }
            _dynamic_array_synapses_1_delay.resize(1);
            THRUST_CHECK_ERROR(dev_dynamic_array_synapses_1_delay.resize(1));
            for(int i=0; i<1; i++)
            {
                _dynamic_array_synapses_1_delay[i] = 0;
                dev_dynamic_array_synapses_1_delay[i] = 0;
            }
            _dynamic_array_synapses_2_delay.resize(1);
            THRUST_CHECK_ERROR(dev_dynamic_array_synapses_2_delay.resize(1));
            for(int i=0; i<1; i++)
            {
                _dynamic_array_synapses_2_delay[i] = 0;
                dev_dynamic_array_synapses_2_delay[i] = 0;
            }

    // Arrays initialized to an "arange"
    _array_neurongroup_1_i = new int32_t[100];
    for(int i=0; i<100; i++) _array_neurongroup_1_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_1_i, _array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_i = new int32_t[2500];
    for(int i=0; i<2500; i++) _array_neurongroup_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
            );
    _array_spikegeneratorgroup_i = new int32_t[100];
    for(int i=0; i<100; i++) _array_spikegeneratorgroup_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikegeneratorgroup_i, _array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor_1__source_idx = new int32_t[2500];
    for(int i=0; i<2500; i++) _array_spikemonitor_1__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor_1__source_idx, _array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor_2__source_idx = new int32_t[100];
    for(int i=0; i<100; i++) _array_spikemonitor_2__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor_2__source_idx, _array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor__source_idx = new int32_t[100];
    for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice)
            );

    // static arrays
    _static_array__dynamic_array_spikegeneratorgroup__timebins = new int32_t[19676];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup__timebins, sizeof(int32_t)*19676)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup__timebins, &dev_static_array__dynamic_array_spikegeneratorgroup__timebins, sizeof(int32_t*))
            );
    _static_array__dynamic_array_spikegeneratorgroup_neuron_index = new int64_t[19676];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t)*19676)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_neuron_index, &dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t*))
            );
    _static_array__dynamic_array_spikegeneratorgroup_spike_number = new int64_t[19676];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int64_t)*19676)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_spike_number, &dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int64_t*))
            );
    _static_array__dynamic_array_spikegeneratorgroup_spike_time = new double[19676];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double)*19676)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_spike_time, &dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double*))
            );
    _timedarray_1_values = new double[250000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_1_values, sizeof(double)*250000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_1_values, &dev_timedarray_1_values, sizeof(double*))
            );
    _timedarray_2_values = new double[250000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_2_values, sizeof(double)*250000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_2_values, &dev_timedarray_2_values, sizeof(double*))
            );
    _timedarray_3_values = new double[250000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_3_values, sizeof(double)*250000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_3_values, &dev_timedarray_3_values, sizeof(double*))
            );
    _timedarray_4_values = new double[250000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_4_values, sizeof(double)*250000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_4_values, &dev_timedarray_4_values, sizeof(double*))
            );
    _timedarray_values = new double[250000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_values, sizeof(double)*250000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_values, &dev_timedarray_values, sizeof(double*))
            );


    // eventspace_arrays
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_1__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_1__spikespace)
            );
    _array_neurongroup_1__spikespace = new int32_t[101];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup__spikespace[0], sizeof(int32_t)*_num__array_neurongroup__spikespace)
            );
    _array_neurongroup__spikespace = new int32_t[2501];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikegeneratorgroup__spikespace[0], sizeof(int32_t)*_num__array_spikegeneratorgroup__spikespace)
            );
    _array_spikegeneratorgroup__spikespace = new int32_t[101];

    CUDA_CHECK_MEMORY();
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    double tot_memory_MB = (used_device_memory - used_device_memory_start) * to_MB;
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: _init_arrays() took " <<  time_passed << "s";
    if (tot_memory_MB > 0)
        std::cout << " and used " << tot_memory_MB << "MB of device memory.";
    std::cout << std::endl;
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__dynamic_array_spikegeneratorgroup__timebins;
    f_static_array__dynamic_array_spikegeneratorgroup__timebins.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup__timebins", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup__timebins.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup__timebins.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup__timebins), 19676*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup__timebins." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup__timebins, _static_array__dynamic_array_spikegeneratorgroup__timebins, sizeof(int32_t)*19676, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
    f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_neuron_index", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_neuron_index), 19676*sizeof(int64_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_neuron_index." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, _static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t)*19676, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_number;
    f_static_array__dynamic_array_spikegeneratorgroup_spike_number.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_number", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_spike_number.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_spike_number.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_number), 19676*sizeof(int64_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_number." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, _static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int64_t)*19676, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_time;
    f_static_array__dynamic_array_spikegeneratorgroup_spike_time.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_time", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_spike_time.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_spike_time.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_time), 19676*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_time." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, _static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double)*19676, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_1_values;
    f_timedarray_1_values.open("static_arrays/_timedarray_1_values", ios::in | ios::binary);
    if(f_timedarray_1_values.is_open())
    {
        f_timedarray_1_values.read(reinterpret_cast<char*>(_timedarray_1_values), 250000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_1_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_1_values, _timedarray_1_values, sizeof(double)*250000, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_2_values;
    f_timedarray_2_values.open("static_arrays/_timedarray_2_values", ios::in | ios::binary);
    if(f_timedarray_2_values.is_open())
    {
        f_timedarray_2_values.read(reinterpret_cast<char*>(_timedarray_2_values), 250000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_2_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_2_values, _timedarray_2_values, sizeof(double)*250000, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_3_values;
    f_timedarray_3_values.open("static_arrays/_timedarray_3_values", ios::in | ios::binary);
    if(f_timedarray_3_values.is_open())
    {
        f_timedarray_3_values.read(reinterpret_cast<char*>(_timedarray_3_values), 250000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_3_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_3_values, _timedarray_3_values, sizeof(double)*250000, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_4_values;
    f_timedarray_4_values.open("static_arrays/_timedarray_4_values", ios::in | ios::binary);
    if(f_timedarray_4_values.is_open())
    {
        f_timedarray_4_values.read(reinterpret_cast<char*>(_timedarray_4_values), 250000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_4_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_4_values, _timedarray_4_values, sizeof(double)*250000, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_values;
    f_timedarray_values.open("static_arrays/_timedarray_values", ios::in | ios::binary);
    if(f_timedarray_values.is_open())
    {
        f_timedarray_values.read(reinterpret_cast<char*>(_timedarray_values), 250000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_values, _timedarray_values, sizeof(double)*250000, cudaMemcpyHostToDevice)
            );
}

void _write_arrays()
{
    using namespace brian;

    CUDA_SAFE_CALL(
            cudaMemcpy(_array_defaultclock_dt, dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_-2498286751126143934", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(double));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_defaultclock_t, dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open("results/_array_defaultclock_t_-2737290110509227905", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(double));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_defaultclock_timestep, dev_array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_-8079704882989719448", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(int64_t));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_g_eKC_eKC, dev_array_neurongroup_1_g_eKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_eKC_eKC, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_g_eKC_eKC;
    outfile__array_neurongroup_1_g_eKC_eKC.open("results/_array_neurongroup_1_g_eKC_eKC_-5583994226418931441", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_g_eKC_eKC.is_open())
    {
        outfile__array_neurongroup_1_g_eKC_eKC.write(reinterpret_cast<char*>(_array_neurongroup_1_g_eKC_eKC), 100*sizeof(double));
        outfile__array_neurongroup_1_g_eKC_eKC.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_g_eKC_eKC." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_g_iKC_eKC, dev_array_neurongroup_1_g_iKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_iKC_eKC, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_g_iKC_eKC;
    outfile__array_neurongroup_1_g_iKC_eKC.open("results/_array_neurongroup_1_g_iKC_eKC_698381179621967883", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_g_iKC_eKC.is_open())
    {
        outfile__array_neurongroup_1_g_iKC_eKC.write(reinterpret_cast<char*>(_array_neurongroup_1_g_iKC_eKC), 100*sizeof(double));
        outfile__array_neurongroup_1_g_iKC_eKC.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_g_iKC_eKC." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_h, dev_array_neurongroup_1_h, sizeof(double)*_num__array_neurongroup_1_h, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_h;
    outfile__array_neurongroup_1_h.open("results/_array_neurongroup_1_h_5344468206502301276", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_h.is_open())
    {
        outfile__array_neurongroup_1_h.write(reinterpret_cast<char*>(_array_neurongroup_1_h), 100*sizeof(double));
        outfile__array_neurongroup_1_h.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_h." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_i, dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_i;
    outfile__array_neurongroup_1_i.open("results/_array_neurongroup_1_i_-3789611295489125583", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_i.is_open())
    {
        outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 100*sizeof(int32_t));
        outfile__array_neurongroup_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_lastspike, dev_array_neurongroup_1_lastspike, sizeof(double)*_num__array_neurongroup_1_lastspike, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_lastspike;
    outfile__array_neurongroup_1_lastspike.open("results/_array_neurongroup_1_lastspike_7243448513092147373", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_lastspike.is_open())
    {
        outfile__array_neurongroup_1_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_1_lastspike), 100*sizeof(double));
        outfile__array_neurongroup_1_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_lastspike." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_m, dev_array_neurongroup_1_m, sizeof(double)*_num__array_neurongroup_1_m, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_m;
    outfile__array_neurongroup_1_m.open("results/_array_neurongroup_1_m_-2120397286057579911", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_m.is_open())
    {
        outfile__array_neurongroup_1_m.write(reinterpret_cast<char*>(_array_neurongroup_1_m), 100*sizeof(double));
        outfile__array_neurongroup_1_m.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_m." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_n, dev_array_neurongroup_1_n, sizeof(double)*_num__array_neurongroup_1_n, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_n;
    outfile__array_neurongroup_1_n.open("results/_array_neurongroup_1_n_-8428575863822087678", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_n.is_open())
    {
        outfile__array_neurongroup_1_n.write(reinterpret_cast<char*>(_array_neurongroup_1_n), 100*sizeof(double));
        outfile__array_neurongroup_1_n.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_n." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_not_refractory, dev_array_neurongroup_1_not_refractory, sizeof(char)*_num__array_neurongroup_1_not_refractory, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_not_refractory;
    outfile__array_neurongroup_1_not_refractory.open("results/_array_neurongroup_1_not_refractory_7368107638204237228", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_not_refractory.is_open())
    {
        outfile__array_neurongroup_1_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_1_not_refractory), 100*sizeof(char));
        outfile__array_neurongroup_1_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_not_refractory." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_V, dev_array_neurongroup_1_V, sizeof(double)*_num__array_neurongroup_1_V, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_V;
    outfile__array_neurongroup_1_V.open("results/_array_neurongroup_1_V_-6137459464929975667", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_V.is_open())
    {
        outfile__array_neurongroup_1_V.write(reinterpret_cast<char*>(_array_neurongroup_1_V), 100*sizeof(double));
        outfile__array_neurongroup_1_V.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_V." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_g_PN_iKC, dev_array_neurongroup_g_PN_iKC, sizeof(double)*_num__array_neurongroup_g_PN_iKC, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_g_PN_iKC;
    outfile__array_neurongroup_g_PN_iKC.open("results/_array_neurongroup_g_PN_iKC_1280755048330296697", ios::binary | ios::out);
    if(outfile__array_neurongroup_g_PN_iKC.is_open())
    {
        outfile__array_neurongroup_g_PN_iKC.write(reinterpret_cast<char*>(_array_neurongroup_g_PN_iKC), 2500*sizeof(double));
        outfile__array_neurongroup_g_PN_iKC.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_g_PN_iKC." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_h, dev_array_neurongroup_h, sizeof(double)*_num__array_neurongroup_h, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_h;
    outfile__array_neurongroup_h.open("results/_array_neurongroup_h_-3092669250522089359", ios::binary | ios::out);
    if(outfile__array_neurongroup_h.is_open())
    {
        outfile__array_neurongroup_h.write(reinterpret_cast<char*>(_array_neurongroup_h), 2500*sizeof(double));
        outfile__array_neurongroup_h.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_h." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_i, dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_i;
    outfile__array_neurongroup_i.open("results/_array_neurongroup_i_-5243747817659055085", ios::binary | ios::out);
    if(outfile__array_neurongroup_i.is_open())
    {
        outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 2500*sizeof(int32_t));
        outfile__array_neurongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_lastspike, dev_array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_lastspike;
    outfile__array_neurongroup_lastspike.open("results/_array_neurongroup_lastspike_5887856098539198058", ios::binary | ios::out);
    if(outfile__array_neurongroup_lastspike.is_open())
    {
        outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), 2500*sizeof(double));
        outfile__array_neurongroup_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_m, dev_array_neurongroup_m, sizeof(double)*_num__array_neurongroup_m, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_m;
    outfile__array_neurongroup_m.open("results/_array_neurongroup_m_-1318907606515658718", ios::binary | ios::out);
    if(outfile__array_neurongroup_m.is_open())
    {
        outfile__array_neurongroup_m.write(reinterpret_cast<char*>(_array_neurongroup_m), 2500*sizeof(double));
        outfile__array_neurongroup_m.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_m." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_n, dev_array_neurongroup_n, sizeof(double)*_num__array_neurongroup_n, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_n;
    outfile__array_neurongroup_n.open("results/_array_neurongroup_n_-8712203438137663296", ios::binary | ios::out);
    if(outfile__array_neurongroup_n.is_open())
    {
        outfile__array_neurongroup_n.write(reinterpret_cast<char*>(_array_neurongroup_n), 2500*sizeof(double));
        outfile__array_neurongroup_n.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_n." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_not_refractory, dev_array_neurongroup_not_refractory, sizeof(char)*_num__array_neurongroup_not_refractory, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_not_refractory;
    outfile__array_neurongroup_not_refractory.open("results/_array_neurongroup_not_refractory_2056821189576763631", ios::binary | ios::out);
    if(outfile__array_neurongroup_not_refractory.is_open())
    {
        outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), 2500*sizeof(char));
        outfile__array_neurongroup_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_V, dev_array_neurongroup_V, sizeof(double)*_num__array_neurongroup_V, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_V;
    outfile__array_neurongroup_V.open("results/_array_neurongroup_V_1012544495117118507", ios::binary | ios::out);
    if(outfile__array_neurongroup_V.is_open())
    {
        outfile__array_neurongroup_V.write(reinterpret_cast<char*>(_array_neurongroup_V), 2500*sizeof(double));
        outfile__array_neurongroup_V.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_V." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup__lastindex, dev_array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup__lastindex;
    outfile__array_spikegeneratorgroup__lastindex.open("results/_array_spikegeneratorgroup__lastindex_-4562001451387750606", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup__lastindex.is_open())
    {
        outfile__array_spikegeneratorgroup__lastindex.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__lastindex), 1*sizeof(int32_t));
        outfile__array_spikegeneratorgroup__lastindex.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup__lastindex." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup__period_bins, dev_array_spikegeneratorgroup__period_bins, sizeof(int32_t)*_num__array_spikegeneratorgroup__period_bins, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup__period_bins;
    outfile__array_spikegeneratorgroup__period_bins.open("results/_array_spikegeneratorgroup__period_bins_3814106930303245699", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup__period_bins.is_open())
    {
        outfile__array_spikegeneratorgroup__period_bins.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__period_bins), 1*sizeof(int32_t));
        outfile__array_spikegeneratorgroup__period_bins.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup__period_bins." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup_i, dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup_i;
    outfile__array_spikegeneratorgroup_i.open("results/_array_spikegeneratorgroup_i_-397239631630174056", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup_i.is_open())
    {
        outfile__array_spikegeneratorgroup_i.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_i), 100*sizeof(int32_t));
        outfile__array_spikegeneratorgroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup_period, dev_array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup_period;
    outfile__array_spikegeneratorgroup_period.open("results/_array_spikegeneratorgroup_period_7725887948559186023", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup_period.is_open())
    {
        outfile__array_spikegeneratorgroup_period.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_period), 1*sizeof(double));
        outfile__array_spikegeneratorgroup_period.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup_period." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_1__source_idx, dev_array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_1__source_idx;
    outfile__array_spikemonitor_1__source_idx.open("results/_array_spikemonitor_1__source_idx_1057894290895839432", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1__source_idx.is_open())
    {
        outfile__array_spikemonitor_1__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_1__source_idx), 2500*sizeof(int32_t));
        outfile__array_spikemonitor_1__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_1_count, dev_array_spikemonitor_1_count, sizeof(int32_t)*_num__array_spikemonitor_1_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_1_count;
    outfile__array_spikemonitor_1_count.open("results/_array_spikemonitor_1_count_-9137222686626857787", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1_count.is_open())
    {
        outfile__array_spikemonitor_1_count.write(reinterpret_cast<char*>(_array_spikemonitor_1_count), 2500*sizeof(int32_t));
        outfile__array_spikemonitor_1_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1_count." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_1_N, dev_array_spikemonitor_1_N, sizeof(int32_t)*_num__array_spikemonitor_1_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_1_N;
    outfile__array_spikemonitor_1_N.open("results/_array_spikemonitor_1_N_7812431033937921391", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1_N.is_open())
    {
        outfile__array_spikemonitor_1_N.write(reinterpret_cast<char*>(_array_spikemonitor_1_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_2__source_idx, dev_array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_2__source_idx;
    outfile__array_spikemonitor_2__source_idx.open("results/_array_spikemonitor_2__source_idx_1362393325657685696", ios::binary | ios::out);
    if(outfile__array_spikemonitor_2__source_idx.is_open())
    {
        outfile__array_spikemonitor_2__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_2__source_idx), 100*sizeof(int32_t));
        outfile__array_spikemonitor_2__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_2__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_2_count, dev_array_spikemonitor_2_count, sizeof(int32_t)*_num__array_spikemonitor_2_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_2_count;
    outfile__array_spikemonitor_2_count.open("results/_array_spikemonitor_2_count_7626239611073350978", ios::binary | ios::out);
    if(outfile__array_spikemonitor_2_count.is_open())
    {
        outfile__array_spikemonitor_2_count.write(reinterpret_cast<char*>(_array_spikemonitor_2_count), 100*sizeof(int32_t));
        outfile__array_spikemonitor_2_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_2_count." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_2_N, dev_array_spikemonitor_2_N, sizeof(int32_t)*_num__array_spikemonitor_2_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_2_N;
    outfile__array_spikemonitor_2_N.open("results/_array_spikemonitor_2_N_4348563937531214317", ios::binary | ios::out);
    if(outfile__array_spikemonitor_2_N.is_open())
    {
        outfile__array_spikemonitor_2_N.write(reinterpret_cast<char*>(_array_spikemonitor_2_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_2_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor__source_idx, dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor__source_idx;
    outfile__array_spikemonitor__source_idx.open("results/_array_spikemonitor__source_idx_4492520236303501020", ios::binary | ios::out);
    if(outfile__array_spikemonitor__source_idx.is_open())
    {
        outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 100*sizeof(int32_t));
        outfile__array_spikemonitor__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_count, dev_array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_count;
    outfile__array_spikemonitor_count.open("results/_array_spikemonitor_count_-8939381533980173332", ios::binary | ios::out);
    if(outfile__array_spikemonitor_count.is_open())
    {
        outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 100*sizeof(int32_t));
        outfile__array_spikemonitor_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_N, dev_array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_N;
    outfile__array_spikemonitor_N.open("results/_array_spikemonitor_N_-4176126692401210049", ios::binary | ios::out);
    if(outfile__array_spikemonitor_N.is_open())
    {
        outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_1_N, dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_1_N;
    outfile__array_synapses_1_N.open("results/_array_synapses_1_N_293610487777644577", ios::binary | ios::out);
    if(outfile__array_synapses_1_N.is_open())
    {
        outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(int32_t));
        outfile__array_synapses_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_2_N, dev_array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_2_N;
    outfile__array_synapses_2_N.open("results/_array_synapses_2_N_6367731152988484261", ios::binary | ios::out);
    if(outfile__array_synapses_2_N.is_open())
    {
        outfile__array_synapses_2_N.write(reinterpret_cast<char*>(_array_synapses_2_N), 1*sizeof(int32_t));
        outfile__array_synapses_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_2_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_N, dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_N;
    outfile__array_synapses_N.open("results/_array_synapses_N_-4267478581729905340", ios::binary | ios::out);
    if(outfile__array_synapses_N.is_open())
    {
        outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(int32_t));
        outfile__array_synapses_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_N." << endl;
    }

    _dynamic_array_spikegeneratorgroup__timebins = dev_dynamic_array_spikegeneratorgroup__timebins;
    ofstream outfile__dynamic_array_spikegeneratorgroup__timebins;
    outfile__dynamic_array_spikegeneratorgroup__timebins.open("results/_dynamic_array_spikegeneratorgroup__timebins_5801608633011989326", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup__timebins.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup__timebins.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup__timebins[0])), _dynamic_array_spikegeneratorgroup__timebins.size()*sizeof(int32_t));
        outfile__dynamic_array_spikegeneratorgroup__timebins.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup__timebins." << endl;
    }
    _dynamic_array_spikegeneratorgroup_neuron_index = dev_dynamic_array_spikegeneratorgroup_neuron_index;
    ofstream outfile__dynamic_array_spikegeneratorgroup_neuron_index;
    outfile__dynamic_array_spikegeneratorgroup_neuron_index.open("results/_dynamic_array_spikegeneratorgroup_neuron_index_3327538713134410383", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup_neuron_index.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup_neuron_index[0])), _dynamic_array_spikegeneratorgroup_neuron_index.size()*sizeof(int32_t));
        outfile__dynamic_array_spikegeneratorgroup_neuron_index.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_neuron_index." << endl;
    }
    _dynamic_array_spikegeneratorgroup_spike_number = dev_dynamic_array_spikegeneratorgroup_spike_number;
    ofstream outfile__dynamic_array_spikegeneratorgroup_spike_number;
    outfile__dynamic_array_spikegeneratorgroup_spike_number.open("results/_dynamic_array_spikegeneratorgroup_spike_number_-4677374471276286859", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_spike_number.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup_spike_number.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup_spike_number[0])), _dynamic_array_spikegeneratorgroup_spike_number.size()*sizeof(int32_t));
        outfile__dynamic_array_spikegeneratorgroup_spike_number.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_number." << endl;
    }
    _dynamic_array_spikegeneratorgroup_spike_time = dev_dynamic_array_spikegeneratorgroup_spike_time;
    ofstream outfile__dynamic_array_spikegeneratorgroup_spike_time;
    outfile__dynamic_array_spikegeneratorgroup_spike_time.open("results/_dynamic_array_spikegeneratorgroup_spike_time_5281423122437343888", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_spike_time.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup_spike_time.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup_spike_time[0])), _dynamic_array_spikegeneratorgroup_spike_time.size()*sizeof(double));
        outfile__dynamic_array_spikegeneratorgroup_spike_time.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_time." << endl;
    }
    _dynamic_array_spikemonitor_1_i = dev_dynamic_array_spikemonitor_1_i;
    ofstream outfile__dynamic_array_spikemonitor_1_i;
    outfile__dynamic_array_spikemonitor_1_i.open("results/_dynamic_array_spikemonitor_1_i_7158638244621399231", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_1_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_1_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_1_i[0])), _dynamic_array_spikemonitor_1_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_i." << endl;
    }
    _dynamic_array_spikemonitor_1_t = dev_dynamic_array_spikemonitor_1_t;
    ofstream outfile__dynamic_array_spikemonitor_1_t;
    outfile__dynamic_array_spikemonitor_1_t.open("results/_dynamic_array_spikemonitor_1_t_6409207889459703585", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_1_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_1_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_1_t[0])), _dynamic_array_spikemonitor_1_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_1_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_t." << endl;
    }
    _dynamic_array_spikemonitor_2_i = dev_dynamic_array_spikemonitor_2_i;
    ofstream outfile__dynamic_array_spikemonitor_2_i;
    outfile__dynamic_array_spikemonitor_2_i.open("results/_dynamic_array_spikemonitor_2_i_-2319234593350803970", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_2_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_2_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_2_i[0])), _dynamic_array_spikemonitor_2_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_2_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_2_i." << endl;
    }
    _dynamic_array_spikemonitor_2_t = dev_dynamic_array_spikemonitor_2_t;
    ofstream outfile__dynamic_array_spikemonitor_2_t;
    outfile__dynamic_array_spikemonitor_2_t.open("results/_dynamic_array_spikemonitor_2_t_-7213119999462114063", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_2_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_2_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_2_t[0])), _dynamic_array_spikemonitor_2_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_2_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_2_t." << endl;
    }
    _dynamic_array_spikemonitor_i = dev_dynamic_array_spikemonitor_i;
    ofstream outfile__dynamic_array_spikemonitor_i;
    outfile__dynamic_array_spikemonitor_i.open("results/_dynamic_array_spikemonitor_i_7932276293891085154", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_i[0])), _dynamic_array_spikemonitor_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
    }
    _dynamic_array_spikemonitor_t = dev_dynamic_array_spikemonitor_t;
    ofstream outfile__dynamic_array_spikemonitor_t;
    outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t_-2690765867727077152", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_t[0])), _dynamic_array_spikemonitor_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_post;
    outfile__dynamic_array_synapses_1__synaptic_post.open("results/_dynamic_array_synapses_1__synaptic_post_3887552400782846830", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1__synaptic_post[0])), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
    outfile__dynamic_array_synapses_1__synaptic_pre.open("results/_dynamic_array_synapses_1__synaptic_pre_-3465909282819894897", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1__synaptic_pre[0])), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
    }
    _dynamic_array_synapses_1_Apost = dev_dynamic_array_synapses_1_Apost;
    ofstream outfile__dynamic_array_synapses_1_Apost;
    outfile__dynamic_array_synapses_1_Apost.open("results/_dynamic_array_synapses_1_Apost_-2802562643289696327", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_Apost.is_open())
    {
        outfile__dynamic_array_synapses_1_Apost.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_Apost[0])), _dynamic_array_synapses_1_Apost.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_Apost.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_Apost." << endl;
    }
    _dynamic_array_synapses_1_Apre = dev_dynamic_array_synapses_1_Apre;
    ofstream outfile__dynamic_array_synapses_1_Apre;
    outfile__dynamic_array_synapses_1_Apre.open("results/_dynamic_array_synapses_1_Apre_-4131744976984627475", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_Apre.is_open())
    {
        outfile__dynamic_array_synapses_1_Apre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_Apre[0])), _dynamic_array_synapses_1_Apre.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_Apre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_Apre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay;
    outfile__dynamic_array_synapses_1_delay.open("results/_dynamic_array_synapses_1_delay_-2231186514412628865", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay.is_open())
    {
        outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_delay[0])), _dynamic_array_synapses_1_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay_1;
    outfile__dynamic_array_synapses_1_delay_1.open("results/_dynamic_array_synapses_1_delay_1_-7861599511511956763", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay_1.is_open())
    {
        outfile__dynamic_array_synapses_1_delay_1.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_delay_1[0])), _dynamic_array_synapses_1_delay_1.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_delay_1.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay_1." << endl;
    }
    _dynamic_array_synapses_1_g_raw = dev_dynamic_array_synapses_1_g_raw;
    ofstream outfile__dynamic_array_synapses_1_g_raw;
    outfile__dynamic_array_synapses_1_g_raw.open("results/_dynamic_array_synapses_1_g_raw_-8690563007348565991", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_g_raw.is_open())
    {
        outfile__dynamic_array_synapses_1_g_raw.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_g_raw[0])), _dynamic_array_synapses_1_g_raw.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_g_raw.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_g_raw." << endl;
    }
    _dynamic_array_synapses_1_lastupdate = dev_dynamic_array_synapses_1_lastupdate;
    ofstream outfile__dynamic_array_synapses_1_lastupdate;
    outfile__dynamic_array_synapses_1_lastupdate.open("results/_dynamic_array_synapses_1_lastupdate_1417193071217998245", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_lastupdate.is_open())
    {
        outfile__dynamic_array_synapses_1_lastupdate.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_lastupdate[0])), _dynamic_array_synapses_1_lastupdate.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_lastupdate.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_lastupdate." << endl;
    }
    _dynamic_array_synapses_1_N_incoming = dev_dynamic_array_synapses_1_N_incoming;
    ofstream outfile__dynamic_array_synapses_1_N_incoming;
    outfile__dynamic_array_synapses_1_N_incoming.open("results/_dynamic_array_synapses_1_N_incoming_-4730020938361073333", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_N_incoming[0])), _dynamic_array_synapses_1_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
    }
    _dynamic_array_synapses_1_N_outgoing = dev_dynamic_array_synapses_1_N_outgoing;
    ofstream outfile__dynamic_array_synapses_1_N_outgoing;
    outfile__dynamic_array_synapses_1_N_outgoing.open("results/_dynamic_array_synapses_1_N_outgoing_5847266960107717791", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_N_outgoing[0])), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_post;
    outfile__dynamic_array_synapses_2__synaptic_post.open("results/_dynamic_array_synapses_2__synaptic_post_5774352615214944728", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_2__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_post[0])), _dynamic_array_synapses_2__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_pre;
    outfile__dynamic_array_synapses_2__synaptic_pre.open("results/_dynamic_array_synapses_2__synaptic_pre_-9012074162976892665", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_2__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_pre[0])), _dynamic_array_synapses_2__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_delay;
    outfile__dynamic_array_synapses_2_delay.open("results/_dynamic_array_synapses_2_delay_1607835271570489182", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_delay.is_open())
    {
        outfile__dynamic_array_synapses_2_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_delay[0])), _dynamic_array_synapses_2_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_2_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_delay." << endl;
    }
    _dynamic_array_synapses_2_N_incoming = dev_dynamic_array_synapses_2_N_incoming;
    ofstream outfile__dynamic_array_synapses_2_N_incoming;
    outfile__dynamic_array_synapses_2_N_incoming.open("results/_dynamic_array_synapses_2_N_incoming_8882135263890869691", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_2_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_incoming[0])), _dynamic_array_synapses_2_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_incoming." << endl;
    }
    _dynamic_array_synapses_2_N_outgoing = dev_dynamic_array_synapses_2_N_outgoing;
    ofstream outfile__dynamic_array_synapses_2_N_outgoing;
    outfile__dynamic_array_synapses_2_N_outgoing.open("results/_dynamic_array_synapses_2_N_outgoing_-7531657551927571654", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_2_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_outgoing[0])), _dynamic_array_synapses_2_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_post;
    outfile__dynamic_array_synapses__synaptic_post.open("results/_dynamic_array_synapses__synaptic_post_-5399118445813332823", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0])), _dynamic_array_synapses__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_pre;
    outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre_4910052875650969924", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0])), _dynamic_array_synapses__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay;
    outfile__dynamic_array_synapses_delay.open("results/_dynamic_array_synapses_delay_-7446265175536355663", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay.is_open())
    {
        outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_delay[0])), _dynamic_array_synapses_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
    }
    _dynamic_array_synapses_N_incoming = dev_dynamic_array_synapses_N_incoming;
    ofstream outfile__dynamic_array_synapses_N_incoming;
    outfile__dynamic_array_synapses_N_incoming.open("results/_dynamic_array_synapses_N_incoming_2716837380347339885", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_incoming[0])), _dynamic_array_synapses_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
    }
    _dynamic_array_synapses_N_outgoing = dev_dynamic_array_synapses_N_outgoing;
    ofstream outfile__dynamic_array_synapses_N_outgoing;
    outfile__dynamic_array_synapses_N_outgoing.open("results/_dynamic_array_synapses_N_outgoing_-946603928985537374", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_outgoing[0])), _dynamic_array_synapses_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
    }
    _dynamic_array_synapses_weight = dev_dynamic_array_synapses_weight;
    ofstream outfile__dynamic_array_synapses_weight;
    outfile__dynamic_array_synapses_weight.open("results/_dynamic_array_synapses_weight_5741023513072131750", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_weight.is_open())
    {
        outfile__dynamic_array_synapses_weight.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_weight[0])), _dynamic_array_synapses_weight.size()*sizeof(double));
        outfile__dynamic_array_synapses_weight.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_weight." << endl;
    }


    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open("results/last_run_info.txt", ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        std::cout << "Error writing last run info to file." << std::endl;
    }
}

__global__ void synapses_pre_destroy()
{
    using namespace brian;

    synapses_pre.destroy();
}
__global__ void synapses_1_post_destroy()
{
    using namespace brian;

    synapses_1_post.destroy();
}
__global__ void synapses_1_pre_destroy()
{
    using namespace brian;

    synapses_1_pre.destroy();
}
__global__ void synapses_2_pre_destroy()
{
    using namespace brian;

    synapses_2_pre.destroy();
}

void _dealloc_arrays()
{
    using namespace brian;


    CUDA_SAFE_CALL(
            curandDestroyGenerator(curand_generator)
            );

    synapses_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_pre_destroy");
    synapses_1_post_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_1_post_destroy");
    synapses_1_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_1_pre_destroy");
    synapses_2_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_2_pre_destroy");

    dev_dynamic_array_spikegeneratorgroup__timebins.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup__timebins);
    _dynamic_array_spikegeneratorgroup__timebins.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikegeneratorgroup__timebins);
    dev_dynamic_array_spikegeneratorgroup_neuron_index.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup_neuron_index);
    _dynamic_array_spikegeneratorgroup_neuron_index.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikegeneratorgroup_neuron_index);
    dev_dynamic_array_spikegeneratorgroup_spike_number.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup_spike_number);
    _dynamic_array_spikegeneratorgroup_spike_number.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikegeneratorgroup_spike_number);
    dev_dynamic_array_spikegeneratorgroup_spike_time.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikegeneratorgroup_spike_time);
    _dynamic_array_spikegeneratorgroup_spike_time.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikegeneratorgroup_spike_time);
    dev_dynamic_array_spikemonitor_1_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_1_i);
    _dynamic_array_spikemonitor_1_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_1_i);
    dev_dynamic_array_spikemonitor_1_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_1_t);
    _dynamic_array_spikemonitor_1_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_1_t);
    dev_dynamic_array_spikemonitor_2_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_2_i);
    _dynamic_array_spikemonitor_2_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_2_i);
    dev_dynamic_array_spikemonitor_2_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_2_t);
    _dynamic_array_spikemonitor_2_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_2_t);
    dev_dynamic_array_spikemonitor_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_i);
    _dynamic_array_spikemonitor_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_i);
    dev_dynamic_array_spikemonitor_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_t);
    _dynamic_array_spikemonitor_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_t);
    dev_dynamic_array_synapses_1__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_post);
    _dynamic_array_synapses_1__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1__synaptic_post);
    dev_dynamic_array_synapses_1__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_pre);
    _dynamic_array_synapses_1__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1__synaptic_pre);
    dev_dynamic_array_synapses_1_Apost.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_Apost);
    _dynamic_array_synapses_1_Apost.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_Apost);
    dev_dynamic_array_synapses_1_Apre.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_Apre);
    _dynamic_array_synapses_1_Apre.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_Apre);
    dev_dynamic_array_synapses_1_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_delay);
    _dynamic_array_synapses_1_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_delay);
    dev_dynamic_array_synapses_1_delay_1.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_delay_1);
    _dynamic_array_synapses_1_delay_1.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_delay_1);
    dev_dynamic_array_synapses_1_g_raw.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_g_raw);
    _dynamic_array_synapses_1_g_raw.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_g_raw);
    dev_dynamic_array_synapses_1_lastupdate.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_lastupdate);
    _dynamic_array_synapses_1_lastupdate.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_lastupdate);
    dev_dynamic_array_synapses_1_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_incoming);
    _dynamic_array_synapses_1_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1_N_incoming);
    dev_dynamic_array_synapses_1_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_outgoing);
    _dynamic_array_synapses_1_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1_N_outgoing);
    dev_dynamic_array_synapses_2__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2__synaptic_post);
    _dynamic_array_synapses_2__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2__synaptic_post);
    dev_dynamic_array_synapses_2__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2__synaptic_pre);
    _dynamic_array_synapses_2__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2__synaptic_pre);
    dev_dynamic_array_synapses_2_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_2_delay);
    _dynamic_array_synapses_2_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_2_delay);
    dev_dynamic_array_synapses_2_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2_N_incoming);
    _dynamic_array_synapses_2_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2_N_incoming);
    dev_dynamic_array_synapses_2_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2_N_outgoing);
    _dynamic_array_synapses_2_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2_N_outgoing);
    dev_dynamic_array_synapses__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_post);
    _dynamic_array_synapses__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_post);
    dev_dynamic_array_synapses__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_pre);
    _dynamic_array_synapses__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_pre);
    dev_dynamic_array_synapses_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_delay);
    _dynamic_array_synapses_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_delay);
    dev_dynamic_array_synapses_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_incoming);
    _dynamic_array_synapses_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_incoming);
    dev_dynamic_array_synapses_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_outgoing);
    _dynamic_array_synapses_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_outgoing);
    dev_dynamic_array_synapses_weight.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_weight);
    _dynamic_array_synapses_weight.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_weight);

    if(_array_defaultclock_dt!=0)
    {
        delete [] _array_defaultclock_dt;
        _array_defaultclock_dt = 0;
    }
    if(dev_array_defaultclock_dt!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_dt)
                );
        dev_array_defaultclock_dt = 0;
    }
    if(_array_defaultclock_t!=0)
    {
        delete [] _array_defaultclock_t;
        _array_defaultclock_t = 0;
    }
    if(dev_array_defaultclock_t!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_t)
                );
        dev_array_defaultclock_t = 0;
    }
    if(_array_defaultclock_timestep!=0)
    {
        delete [] _array_defaultclock_timestep;
        _array_defaultclock_timestep = 0;
    }
    if(dev_array_defaultclock_timestep!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_timestep)
                );
        dev_array_defaultclock_timestep = 0;
    }
    if(_array_neurongroup_1_g_eKC_eKC!=0)
    {
        delete [] _array_neurongroup_1_g_eKC_eKC;
        _array_neurongroup_1_g_eKC_eKC = 0;
    }
    if(dev_array_neurongroup_1_g_eKC_eKC!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_g_eKC_eKC)
                );
        dev_array_neurongroup_1_g_eKC_eKC = 0;
    }
    if(_array_neurongroup_1_g_iKC_eKC!=0)
    {
        delete [] _array_neurongroup_1_g_iKC_eKC;
        _array_neurongroup_1_g_iKC_eKC = 0;
    }
    if(dev_array_neurongroup_1_g_iKC_eKC!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_g_iKC_eKC)
                );
        dev_array_neurongroup_1_g_iKC_eKC = 0;
    }
    if(_array_neurongroup_1_h!=0)
    {
        delete [] _array_neurongroup_1_h;
        _array_neurongroup_1_h = 0;
    }
    if(dev_array_neurongroup_1_h!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_h)
                );
        dev_array_neurongroup_1_h = 0;
    }
    if(_array_neurongroup_1_i!=0)
    {
        delete [] _array_neurongroup_1_i;
        _array_neurongroup_1_i = 0;
    }
    if(dev_array_neurongroup_1_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_i)
                );
        dev_array_neurongroup_1_i = 0;
    }
    if(_array_neurongroup_1_lastspike!=0)
    {
        delete [] _array_neurongroup_1_lastspike;
        _array_neurongroup_1_lastspike = 0;
    }
    if(dev_array_neurongroup_1_lastspike!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_lastspike)
                );
        dev_array_neurongroup_1_lastspike = 0;
    }
    if(_array_neurongroup_1_m!=0)
    {
        delete [] _array_neurongroup_1_m;
        _array_neurongroup_1_m = 0;
    }
    if(dev_array_neurongroup_1_m!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_m)
                );
        dev_array_neurongroup_1_m = 0;
    }
    if(_array_neurongroup_1_n!=0)
    {
        delete [] _array_neurongroup_1_n;
        _array_neurongroup_1_n = 0;
    }
    if(dev_array_neurongroup_1_n!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_n)
                );
        dev_array_neurongroup_1_n = 0;
    }
    if(_array_neurongroup_1_not_refractory!=0)
    {
        delete [] _array_neurongroup_1_not_refractory;
        _array_neurongroup_1_not_refractory = 0;
    }
    if(dev_array_neurongroup_1_not_refractory!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_not_refractory)
                );
        dev_array_neurongroup_1_not_refractory = 0;
    }
    if(_array_neurongroup_1_V!=0)
    {
        delete [] _array_neurongroup_1_V;
        _array_neurongroup_1_V = 0;
    }
    if(dev_array_neurongroup_1_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_V)
                );
        dev_array_neurongroup_1_V = 0;
    }
    if(_array_neurongroup_g_PN_iKC!=0)
    {
        delete [] _array_neurongroup_g_PN_iKC;
        _array_neurongroup_g_PN_iKC = 0;
    }
    if(dev_array_neurongroup_g_PN_iKC!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_g_PN_iKC)
                );
        dev_array_neurongroup_g_PN_iKC = 0;
    }
    if(_array_neurongroup_h!=0)
    {
        delete [] _array_neurongroup_h;
        _array_neurongroup_h = 0;
    }
    if(dev_array_neurongroup_h!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_h)
                );
        dev_array_neurongroup_h = 0;
    }
    if(_array_neurongroup_i!=0)
    {
        delete [] _array_neurongroup_i;
        _array_neurongroup_i = 0;
    }
    if(dev_array_neurongroup_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_i)
                );
        dev_array_neurongroup_i = 0;
    }
    if(_array_neurongroup_lastspike!=0)
    {
        delete [] _array_neurongroup_lastspike;
        _array_neurongroup_lastspike = 0;
    }
    if(dev_array_neurongroup_lastspike!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_lastspike)
                );
        dev_array_neurongroup_lastspike = 0;
    }
    if(_array_neurongroup_m!=0)
    {
        delete [] _array_neurongroup_m;
        _array_neurongroup_m = 0;
    }
    if(dev_array_neurongroup_m!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_m)
                );
        dev_array_neurongroup_m = 0;
    }
    if(_array_neurongroup_n!=0)
    {
        delete [] _array_neurongroup_n;
        _array_neurongroup_n = 0;
    }
    if(dev_array_neurongroup_n!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_n)
                );
        dev_array_neurongroup_n = 0;
    }
    if(_array_neurongroup_not_refractory!=0)
    {
        delete [] _array_neurongroup_not_refractory;
        _array_neurongroup_not_refractory = 0;
    }
    if(dev_array_neurongroup_not_refractory!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_not_refractory)
                );
        dev_array_neurongroup_not_refractory = 0;
    }
    if(_array_neurongroup_V!=0)
    {
        delete [] _array_neurongroup_V;
        _array_neurongroup_V = 0;
    }
    if(dev_array_neurongroup_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_V)
                );
        dev_array_neurongroup_V = 0;
    }
    if(_array_spikegeneratorgroup__lastindex!=0)
    {
        delete [] _array_spikegeneratorgroup__lastindex;
        _array_spikegeneratorgroup__lastindex = 0;
    }
    if(dev_array_spikegeneratorgroup__lastindex!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup__lastindex)
                );
        dev_array_spikegeneratorgroup__lastindex = 0;
    }
    if(_array_spikegeneratorgroup__period_bins!=0)
    {
        delete [] _array_spikegeneratorgroup__period_bins;
        _array_spikegeneratorgroup__period_bins = 0;
    }
    if(dev_array_spikegeneratorgroup__period_bins!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup__period_bins)
                );
        dev_array_spikegeneratorgroup__period_bins = 0;
    }
    if(_array_spikegeneratorgroup_i!=0)
    {
        delete [] _array_spikegeneratorgroup_i;
        _array_spikegeneratorgroup_i = 0;
    }
    if(dev_array_spikegeneratorgroup_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup_i)
                );
        dev_array_spikegeneratorgroup_i = 0;
    }
    if(_array_spikegeneratorgroup_period!=0)
    {
        delete [] _array_spikegeneratorgroup_period;
        _array_spikegeneratorgroup_period = 0;
    }
    if(dev_array_spikegeneratorgroup_period!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup_period)
                );
        dev_array_spikegeneratorgroup_period = 0;
    }
    if(_array_spikemonitor_1__source_idx!=0)
    {
        delete [] _array_spikemonitor_1__source_idx;
        _array_spikemonitor_1__source_idx = 0;
    }
    if(dev_array_spikemonitor_1__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_1__source_idx)
                );
        dev_array_spikemonitor_1__source_idx = 0;
    }
    if(_array_spikemonitor_1_count!=0)
    {
        delete [] _array_spikemonitor_1_count;
        _array_spikemonitor_1_count = 0;
    }
    if(dev_array_spikemonitor_1_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_1_count)
                );
        dev_array_spikemonitor_1_count = 0;
    }
    if(_array_spikemonitor_1_N!=0)
    {
        delete [] _array_spikemonitor_1_N;
        _array_spikemonitor_1_N = 0;
    }
    if(dev_array_spikemonitor_1_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_1_N)
                );
        dev_array_spikemonitor_1_N = 0;
    }
    if(_array_spikemonitor_2__source_idx!=0)
    {
        delete [] _array_spikemonitor_2__source_idx;
        _array_spikemonitor_2__source_idx = 0;
    }
    if(dev_array_spikemonitor_2__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_2__source_idx)
                );
        dev_array_spikemonitor_2__source_idx = 0;
    }
    if(_array_spikemonitor_2_count!=0)
    {
        delete [] _array_spikemonitor_2_count;
        _array_spikemonitor_2_count = 0;
    }
    if(dev_array_spikemonitor_2_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_2_count)
                );
        dev_array_spikemonitor_2_count = 0;
    }
    if(_array_spikemonitor_2_N!=0)
    {
        delete [] _array_spikemonitor_2_N;
        _array_spikemonitor_2_N = 0;
    }
    if(dev_array_spikemonitor_2_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_2_N)
                );
        dev_array_spikemonitor_2_N = 0;
    }
    if(_array_spikemonitor__source_idx!=0)
    {
        delete [] _array_spikemonitor__source_idx;
        _array_spikemonitor__source_idx = 0;
    }
    if(dev_array_spikemonitor__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor__source_idx)
                );
        dev_array_spikemonitor__source_idx = 0;
    }
    if(_array_spikemonitor_count!=0)
    {
        delete [] _array_spikemonitor_count;
        _array_spikemonitor_count = 0;
    }
    if(dev_array_spikemonitor_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_count)
                );
        dev_array_spikemonitor_count = 0;
    }
    if(_array_spikemonitor_N!=0)
    {
        delete [] _array_spikemonitor_N;
        _array_spikemonitor_N = 0;
    }
    if(dev_array_spikemonitor_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_N)
                );
        dev_array_spikemonitor_N = 0;
    }
    if(_array_synapses_1_N!=0)
    {
        delete [] _array_synapses_1_N;
        _array_synapses_1_N = 0;
    }
    if(dev_array_synapses_1_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_1_N)
                );
        dev_array_synapses_1_N = 0;
    }
    if(_array_synapses_2_N!=0)
    {
        delete [] _array_synapses_2_N;
        _array_synapses_2_N = 0;
    }
    if(dev_array_synapses_2_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_2_N)
                );
        dev_array_synapses_2_N = 0;
    }
    if(_array_synapses_N!=0)
    {
        delete [] _array_synapses_N;
        _array_synapses_N = 0;
    }
    if(dev_array_synapses_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_N)
                );
        dev_array_synapses_N = 0;
    }


    // static arrays
    if(_static_array__dynamic_array_spikegeneratorgroup__timebins!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup__timebins;
        _static_array__dynamic_array_spikegeneratorgroup__timebins = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_neuron_index!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_neuron_index;
        _static_array__dynamic_array_spikegeneratorgroup_neuron_index = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_spike_number!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_spike_number;
        _static_array__dynamic_array_spikegeneratorgroup_spike_number = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_spike_time!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_spike_time;
        _static_array__dynamic_array_spikegeneratorgroup_spike_time = 0;
    }
    if(_timedarray_1_values!=0)
    {
        delete [] _timedarray_1_values;
        _timedarray_1_values = 0;
    }
    if(_timedarray_2_values!=0)
    {
        delete [] _timedarray_2_values;
        _timedarray_2_values = 0;
    }
    if(_timedarray_3_values!=0)
    {
        delete [] _timedarray_3_values;
        _timedarray_3_values = 0;
    }
    if(_timedarray_4_values!=0)
    {
        delete [] _timedarray_4_values;
        _timedarray_4_values = 0;
    }
    if(_timedarray_values!=0)
    {
        delete [] _timedarray_values;
        _timedarray_values = 0;
    }

}

