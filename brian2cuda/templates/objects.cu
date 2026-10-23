{% macro cu_file() %}

#include "objects.h"
#include "network.h"
{% if synapses %}
#include "synapses_classes.h"
{% endif %}
#include "brianlib/cuda_utils.h"
#include "brianlib/device_buffer.h"
#include "rand.h"
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cctype>
#include <vector>

size_t brian::used_device_memory = 0;

//////////////// clocks ///////////////////
// attributes will be set in run.cu
{% for clock in clocks | sort(attribute='name') %}
{% if clock.__class__.__name__ == "EventClock" %}
EventClock brian::{{clock.name}};
{% else %}
Clock brian::{{clock.name}};
{% endif %}
{% endfor %}

//////////////// networks /////////////////
{% for net in networks | sort(attribute='name') %}
Network brian::{{net.name}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
{% if not var in dynamic_array_specs %}
{{c_data_type(var.dtype)}} * brian::{{varname}};
{{c_data_type(var.dtype)}} * brian::dev{{varname}};
__device__ {{c_data_type(var.dtype)}} * brian::d{{varname}};
const int brian::_num_{{varname}} = {{var.size}};

{% endif %}
{% endfor %}

//////////////// eventspaces ///////////////
// we dynamically create multiple eventspaces in no_or_const_delay_mode
// for initiating the first spikespace, we need a host pointer
// for choosing the right spikespace, we need a global index variable
{% for var, varname in eventspace_arrays | dictsort(by='value') %}
{{c_data_type(var.dtype)}} * brian::{{varname}};
const int brian::_num_{{varname}} = {{var.size}};
std::vector<{{c_data_type(var.dtype)}}*> brian::dev{{varname}}(1);
int brian::current_idx{{varname}} = 0;
{% if varname in spikegenerator_eventspaces %}
int brian::previous_idx{{varname}};
{% endif %}
{% endfor %}

//////////////// dynamic arrays 1d /////////
{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
std::vector<{{c_data_type(var.dtype)}}> brian::{{varname}};
{% endfor %}

namespace brian {

//////////////// device storage ///////////
{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
DeviceBuffer dev{{varname}}(sizeof({{c_data_type(var.dtype)}}));
{% endfor %}
{% for varname in subgroups_with_spikemonitor %}
DeviceBuffer _dev_{{varname}}_eventspace(sizeof(int32_t));
{% endfor %}
{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
DeviceBuffer addresses_monitor_{{varname}}(sizeof({{c_data_type(var.dtype)}}*));
DeviceBuffer* {{varname}} = nullptr;
{% endfor %}
}  // namespace brian

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
{# arrays that are initialized from static data are already declared #}
{% if not (name in array_specs.values() or name in dynamic_array_specs.values() or name in dynamic_array_2d_specs.values())%}
{{dtype_spec}} * brian::{{name}};
{{dtype_spec}} * brian::dev{{name}};
__device__ {{dtype_spec}} * brian::d{{name}};
const int brian::_num_{{name}} = {{N}};
{% endif %}
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses | sort(attribute='name') %}
// {{S.name}}
int32_t {{S.name}}_source_start_index;
int32_t {{S.name}}_source_stop_index;
bool brian::{{S.name}}_multiple_pre_post = false;
{% for path in S._pathways | sort(attribute='name') %}
// {{path.name}}
__device__ int* brian::{{path.name}}_num_synapses_by_pre;
__device__ int* brian::{{path.name}}_num_synapses_by_bundle;
__device__ int* brian::{{path.name}}_unique_delays;
__device__ int* brian::{{path.name}}_synapses_offset_by_bundle;
__device__ int* brian::{{path.name}}_global_bundle_id_start_by_pre;
int brian::{{path.name}}_bundle_size_max = 0;
int brian::{{path.name}}_bundle_size_min = 0;
double brian::{{path.name}}_bundle_size_mean = 0;
double brian::{{path.name}}_bundle_size_std = 0;
int brian::{{path.name}}_max_size = 0;
__device__ int* brian::{{path.name}}_num_unique_delays_by_pre;
int brian::{{path.name}}_max_num_unique_delays = 0;
__device__ int32_t** brian::{{path.name}}_synapse_ids_by_pre;
__device__ int32_t* brian::{{path.name}}_synapse_ids;
__device__ int* brian::{{path.name}}_unique_delay_start_idcs;
__device__ int* brian::{{path.name}}_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::{{path.name}};
int brian::{{path.name}}_eventspace_idx = 0;
int brian::{{path.name}}_delay;
bool brian::{{path.name}}_scalar_delay;
{% endfor %}
{% endfor %}

int brian::num_parallel_blocks;
int brian::max_threads_per_block;
int brian::max_threads_per_sm;
int brian::max_shared_mem_size;
int brian::num_threads_per_warp;

{% if profiled_codeobjects is defined %}
// Profiling information for each code object
{% for codeobj in profiled_codeobjects | sort %}
std::chrono::nanoseconds brian::{{codeobj}}_profiling_info(0);
{#
{% if 'spatialstateupdater' in codeobj and 'prepare' not in codeobj %}
// Profiling information for each of the 5 kernels in spatialstateupdate
std::chrono::nanoseconds brian::{{codeobj}}_kernel_integration_profiling_info(0);
std::chrono::nanoseconds brian::{{codeobj}}_kernel_tridiagsolve_profiling_info(0);
std::chrono::nanoseconds brian::{{codeobj}}_kernel_coupling_profiling_info(0);
std::chrono::nanoseconds brian::{{codeobj}}_kernel_combine_profiling_info(0);
std::chrono::nanoseconds brian::{{codeobj}}_kernel_currents_profiling_info(0);
{% endif %}
#}
{% endfor %}
{% endif %}


{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

// typedefs need to be outside the include guards to
// be visible to all files including objects.h
typedef {{curand_float_type}} randomNumber_t;  // random number type

#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include <vector>
#include <stdint.h>
#include "brianlib/clocks.h"
{% if profiled_codeobjects %}
#include <chrono>
{% endif %}

class Network;
{% if synapses %}
class SynapticPathway;
{% endif %}

namespace brian {

class DeviceBuffer;  // forward declaration to avoid including device_buffer.h    
extern size_t used_device_memory;

//////////////// clocks ///////////////////
{% for clock in clocks | sort(attribute='name') %}
{% if clock.__class__.__name__ == "EventClock" %}
extern EventClock {{clock.name}};
{% else %}
extern Clock {{clock.name}};
{% endif %}
{% endfor %}

//////////////// networks /////////////////
{% for net in networks %}
extern Network {{net.name}};
{% endfor %}


//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
{% if not var in dynamic_array_specs %}
extern {{c_data_type(var.dtype)}} * {{varname}};
extern {{c_data_type(var.dtype)}} * dev{{varname}};
extern __device__ {{c_data_type(var.dtype)}} *d{{varname}};
extern const int _num_{{varname}};
{% endif %}
{% endfor %}

//////////////// eventspaces ///////////////
{% for var, varname in eventspace_arrays | dictsort(by='value') %}
extern {{c_data_type(var.dtype)}} * {{varname}};
extern std::vector<{{c_data_type(var.dtype)}}*> dev{{varname}};
extern const int _num_{{varname}};
extern int current_idx{{varname}};
{% if varname in spikegenerator_eventspaces %}
extern int previous_idx{{varname}};
{% endif %}
{% endfor %}

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
{# arrays that are initialized from static data are already declared #}
{% if not (name in array_specs.values() or name in dynamic_array_specs.values() or name in dynamic_array_2d_specs.values())%}
extern {{dtype_spec}} *{{name}};
extern {{dtype_spec}} *dev{{name}};
extern __device__ {{dtype_spec}} *d{{name}};
extern const int _num_{{name}};
{% endif %}
{% endfor %}

//////////////// dynamic arrays 1d ///////////
{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
extern std::vector<{{c_data_type(var.dtype)}}> {{varname}};
extern DeviceBuffer dev{{varname}};
{% endfor %}

//////////////// dynamic arrays 2d ///////////////
{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
extern DeviceBuffer* {{ varname }};
extern DeviceBuffer addresses_monitor_{{ varname }};
{% endfor %}

//////////////// subgroup eventspace buffers ///////////////
{% for varname in subgroups_with_spikemonitor %}
extern DeviceBuffer _dev_{{varname}}_eventspace;
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses | sort(attribute='name') %}
// {{S.name}}
extern bool {{S.name}}_multiple_pre_post;
{% for path in S._pathways | sort(attribute='name') %}
extern __device__ int* {{path.name}}_num_synapses_by_pre;
extern __device__ int* {{path.name}}_num_synapses_by_bundle;
extern __device__ int* {{path.name}}_unique_delays;
extern __device__ int* {{path.name}}_synapses_offset_by_bundle;
extern __device__ int* {{path.name}}_global_bundle_id_start_by_pre;
extern int {{path.name}}_bundle_size_max;
extern int {{path.name}}_bundle_size_min;
extern double {{path.name}}_bundle_size_mean;
extern double {{path.name}}_bundle_size_std;
extern int {{path.name}}_max_size;
extern __device__ int* {{path.name}}_num_unique_delays_by_pre;
extern int {{path.name}}_max_num_unique_delays;
extern __device__ int32_t** {{path.name}}_synapse_ids_by_pre;
extern __device__ int32_t* {{path.name}}_synapse_ids;
extern __device__ int* {{path.name}}_unique_delay_start_idcs;
extern __device__ int* {{path.name}}_unique_delays_offset_by_pre;
extern __device__ SynapticPathway {{path.name}};
extern int {{path.name}}_eventspace_idx;
extern int {{path.name}}_delay;
extern bool {{path.name}}_scalar_delay;
{% endfor %}
{% endfor %}

{% if profiled_codeobjects is defined %}
// Profiling information for each code object
{% for codeobj in profiled_codeobjects | sort %}
extern std::chrono::nanoseconds {{codeobj}}_profiling_info;
{#
{% if 'spatialstateupdater' in codeobj and 'prepare' not in codeobj %}
// Profiling information for each of the 5 kernels in spatialstateupdate
extern std::chrono::nanoseconds {{codeobj}}_kernel_integration_profiling_info;
extern std::chrono::nanoseconds {{codeobj}}_kernel_tridiagsolve_profiling_info;
extern std::chrono::nanoseconds {{codeobj}}_kernel_coupling_profiling_info;
extern std::chrono::nanoseconds {{codeobj}}_kernel_combine_profiling_info;
extern std::chrono::nanoseconds {{codeobj}}_kernel_currents_profiling_info;
{% endif %}
#}
{% endfor %}
{% endif %}

//CUDA
extern int num_parallel_blocks;
extern int max_threads_per_block;
extern int max_threads_per_sm;
extern int max_shared_mem_size;
extern int num_threads_per_warp;

//////////////// host helpers /////////////////
int filter_subgroup_eventspace(int32_t* src, int n, int32_t* dst, int32_t start, int32_t stop);

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


{% endmacro %}
