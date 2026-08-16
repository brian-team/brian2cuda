{% macro cu_file() %}

{% macro set_from_value(var_dtype, array_name) %}
{% if c_data_type(var_dtype) == 'double' %}
set_variable_from_value<double>(name, {{array_name}}, var_size, (double)atof(s_value.c_str()));
{% elif c_data_type(var_dtype) == 'float' %}
set_variable_from_value<float>(name, {{array_name}}, var_size, (float)atof(s_value.c_str()));
{% elif c_data_type(var_dtype) == 'int32_t' %}
set_variable_from_value<int32_t>(name, {{array_name}}, var_size, (int32_t)atoi(s_value.c_str()));
{% elif c_data_type(var_dtype) == 'int64_t' %}
set_variable_from_value<int64_t>(name, {{array_name}}, var_size, (int64_t)atol(s_value.c_str()));
{% elif c_data_type(var_dtype) == 'char' %}
set_variable_from_value(name, {{array_name}}, var_size, (char)atoi(s_value.c_str()));
{% endif %}
{%- endmacro %}

#include "objects.h"
#include "network.h"
{% if synapses %}
#include "synapses_classes.h"
{% endif %}
#include "brianlib/cuda_utils.h"
#include "rand.h"
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cctype>
#include <vector>

size_t brian::used_device_memory = 0;
std::string brian::results_dir = "results/";  // can be overwritten by --results_dir command line arg

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

void set_variable_from_value(std::string varname, char* var_pointer, size_t size, char value) {
    B2C_LOG_DEBUG("Setting '%s' to %s", varname.c_str(), (value == 1 ? "True" : "False"));
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_value(std::string varname, T* var_pointer, size_t size, T value) {
#if B2C_LOG_LEVEL <= B2C_LOG_LEVEL_DEBUG
    {
        std::ostringstream _b2c_msg;
        _b2c_msg << "Setting '" << varname << "' to " << value;
        B2C_LOG_DEBUG("%s", _b2c_msg.str().c_str());
    }
#endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_file(std::string varname, T* var_pointer, size_t data_size, std::string filename) {
    std::ifstream f;
    std::streampos size;
    B2C_LOG_DEBUG("Setting '%s' from file '%s'", varname.c_str(), filename.c_str());
    f.open(filename, std::ios::in | std::ios::binary | std::ios::ate);
    size = f.tellg();
    if (size != data_size) {
        B2C_LOG_ERROR("Error reading '%s': file size %ld does not match expected size %ld",
                      filename.c_str(), (long)size, (long)data_size);
        return;
    }
    f.seekg(0, std::ios::beg);
    if (f.is_open())
        f.read(reinterpret_cast<char *>(var_pointer), data_size);
    else
        B2C_LOG_ERROR("Could not read '%s'", filename.c_str());
    if (f.fail())
        B2C_LOG_ERROR("Error reading '%s'", filename.c_str());
}

//////////////// set arrays by name ///////
void brian::set_variable_by_name(std::string name, std::string s_value) {
	size_t var_size;
	size_t data_size;
	std::for_each(s_value.begin(), s_value.end(), [](char& c) // modify in-place
    {
        c = std::tolower(static_cast<unsigned char>(c));
    });
    if (s_value == "true")
        s_value = "1";
    else if (s_value == "false")
        s_value = "0";
	// non-dynamic arrays
	{% for var, varname in array_specs | dictsort(by='value') %}
    {% if not var in dynamic_array_specs and not var.read_only %}
    if (name == "{{var.owner.name}}.{{var.name}}") {
        var_size = {{var.size}};
        data_size = {{var.size}}*sizeof({{c_data_type(var.dtype)}});
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            {{ set_from_value(var.dtype, "brian::" + get_array_name(var)) }}
        } else {
            // set from file
            set_variable_from_file(name, brian::{{get_array_name(var)}}, data_size, s_value);
        }
        {% if get_array_name(var) not in variables_on_host_only %}
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev{{get_array_name(var)}},
                &brian::{{get_array_name(var)}}[0],
                sizeof(brian::{{get_array_name(var)}}[0])*brian::_num_{{get_array_name(var)}},
                cudaMemcpyHostToDevice
            )
        );
        {% endif %}
        return;
    }
    {% endif %}
    {% endfor %}
    // dynamic arrays (1d)
    {% for var, varname in dynamic_array_specs | dictsort(by='value') %}
    {% if not var.read_only %}
    {% set N = array_basename(varname) %}
    if (name == "{{var.owner.name}}.{{var.name}}") {
        var_size = brian::{{get_array_name(var, access_data=False)}}.size();
        data_size = var_size*sizeof({{c_data_type(var.dtype)}});
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            {{ set_from_value(var.dtype, "brian::" + get_array_name(var, False) + ".data()") }}
        } else {
            // set from file
            set_variable_from_file(name, brian::{{get_array_name(var, False)}}.data(), data_size, s_value);
        }
        {% if get_array_name(var) not in variables_on_host_only %}
        copy_host_to_dev_array_{{ N }}();
        {% endif %}
        return;
    }
    {% endif %}
    {% endfor %}
    {% for var, varname in timed_arrays | dictsort(by='value') %}
    if (name == "{{varname}}.values") {
        var_size = {{var.values.size}};
        data_size = var_size*sizeof({{c_data_type(var.values.dtype)}});
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            {{ set_from_value(var.values.dtype, "brian::" + varname + "_values") }}

        } else {
            // set from file
            set_variable_from_file(name, brian::{{varname}}_values, data_size, s_value);
        }
        {% if varname + "_values" not in variables_on_host_only %}
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev{{varname}}_values,
                &brian::{{varname}}_values[0],
                data_size,
                cudaMemcpyHostToDevice
            )
        );
        {% endif %}
        return;
    }
    {% endfor %}
    B2C_LOG_ERROR("Cannot set unknown variable '%s'.", name.c_str());
    exit(1);
}
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

{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
{% set N = array_basename(varname) %}
void copy_host_to_dev_array_{{ N }}() {
    dev{{ varname }}.copy_from_host(
        {{ varname }}.empty() ? nullptr : {{ varname }}.data(),
        {{ varname }}.size());
}
void copy_dev_to_host_array_{{ N }}() {
    {{ varname }}.resize(dev{{ varname }}.size());
    dev{{ varname }}.copy_to_host(
        {{ varname }}.empty() ? nullptr : {{ varname }}.data());
}
{% endfor %}

{% for var, varname in eventspace_arrays | dictsort(by='value') %}
void expand_eventspace{{ varname }}(int num_queues) {
    int num_eventspaces = static_cast<int>(dev{{ varname }}.size());
    if (num_queues <= num_eventspaces) return;
    std::rotate(
        dev{{ varname }}.begin(),
        dev{{ varname }}.begin() + current_idx{{ varname }},
        dev{{ varname }}.end());
    current_idx{{ varname }} = 0;
    for (int i = num_eventspaces; i < num_queues; i++) {
        {{c_data_type(var.dtype)}}* new_eventspace;
        CUDA_SAFE_CALL(cudaMalloc(
            (void**)&new_eventspace,
            sizeof({{c_data_type(var.dtype)}}) * _num_{{ varname }}));
        CUDA_SAFE_CALL(cudaMemcpy(
            new_eventspace, {{ varname }},
            sizeof({{c_data_type(var.dtype)}}) * _num_{{ varname }},
            cudaMemcpyHostToDevice));
        dev{{ varname }}.push_back(new_eventspace);
    }
}
{% endfor %}
void upload_monitor_row_addresses(DeviceBuffer& addresses, DeviceBuffer* rows, int n_rows)
{
    std::vector<void*> host_ptrs(n_rows);
    for (int i = 0; i < n_rows; i++)
        host_ptrs[i] = rows[i].data();
    addresses.copy_from_host(host_ptrs.data(), host_ptrs.size());
}
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


void _init_arrays()
{
    using namespace brian;

    const auto start_timer = std::chrono::high_resolution_clock::now();

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );

    {% if num_parallel_blocks %}
    num_parallel_blocks = {{num_parallel_blocks}};
    {% else %}
    num_parallel_blocks = props.multiProcessorCount * {{sm_multiplier}};
    {% endif %}
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
            curandCreateGenerator(&curand_generator, {{curand_generator_type}})
            );

    {% if curand_generator_ordering %}
    CUDA_SAFE_CALL(
        curandSetGeneratorOrdering(curand_generator, {{curand_generator_ordering}})
            );
    {% endif %}

    // this sets seed for host and device api RNG
    random_number_buffer.set_seed(seed);

    {% for S in synapses | sort(attribute='name') %}
    {% for path in S._pathways | sort(attribute='name') %}
    {% set src_name = dynamic_array_specs[path.synapse_sources] %}
    {% set tgt_name = dynamic_array_specs[path.synapse_targets] %}
    {{path.name}}_init<<<1,1>>>(
            dev{{ src_name }}.data_as<{{c_data_type(path.synapse_sources.dtype)}}>(),
            dev{{ tgt_name }}.data_as<{{c_data_type(path.synapse_targets.dtype)}}>(),
            0,  //was dt, maybe irrelevant?
            {{path.source.start}},
            {{path.source.stop}}
            );
    CUDA_CHECK_ERROR("{{path.name}}_init");
    {% endfor %}
    {% endfor %}

    // Arrays initialized to 0
    {% for var, varname in zero_arrays | sort(attribute='1') %}
        {% if varname in dynamic_array_specs.values() %}
            {% set N = array_basename(varname) %}
            {{varname}}.resize({{var.size}});
            for(int i=0; i<{{var.size}}; i++)
            {
                {{varname}}[i] = 0;
            }
            copy_host_to_dev_array_{{ N }}();
        {% elif not var in eventspace_arrays %}
            {{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
            for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}})
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev{{varname}}, {{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyHostToDevice)
                    );
        {% endif %}
    {% endfor %}

    // Arrays initialized to an "arange"
    {% for var, varname, start in arange_arrays | sort(attribute='1') %}
    {{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
    for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = {{start}} + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}})
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev{{varname}}, {{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyHostToDevice)
            );
    {% endfor %}

    // static arrays
    {% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
    {% if (name in dynamic_array_specs.values())  %}
    {{name}}.resize({{N}});
    dev{{name}}.resize({{N}});
    {% else %}
    {{name}} = new {{dtype_spec}}[{{N}}];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev{{name}}, sizeof({{dtype_spec}})*{{N}})
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d{{name}}, &dev{{name}}, sizeof({{dtype_spec}}*))
            );
    {% endif %}
    {% endfor %}

    {% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
    {{varname}} = new DeviceBuffer[_num__array_{{var.owner.name}}__indices];
    for (int i = 0; i < _num__array_{{var.owner.name}}__indices; i++)
        {{varname}}[i].set_elem_size(sizeof({{c_data_type(var.dtype)}}));
    {% endfor %}

    // eventspace_arrays
    {% for var, varname in eventspace_arrays | dictsort(by='value') %}
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev{{varname}}[0], sizeof({{c_data_type(var.dtype)}})*_num_{{varname}})
            );
    // initialize eventspace with -1
    {{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
    for (int i=0; i<{{var.size}}-1; i++)
    {
        {{varname}}[i] = -1;
    }
    // initialize eventspace counter with 0
    {{varname}}[{{var.size}} - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev{{varname}}[0],
            {{varname}},
            sizeof({{c_data_type(var.dtype)}}) * _num_{{varname}},
            cudaMemcpyHostToDevice
        )
    );
    {% endfor %}

    CUDA_CHECK_MEMORY();
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    double tot_memory_MB = (used_device_memory - used_device_memory_start) * to_MB;
    double time_passed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_timer).count();
    if (tot_memory_MB > 0)
        B2C_LOG_DEBUG("_init_arrays() took %g s and used %g MB of device memory.",
                      time_passed, tot_memory_MB);
    else
        B2C_LOG_DEBUG("_init_arrays() took %g s", time_passed);
}

void _load_arrays()
{
    using namespace brian;

    {% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
    std::ifstream f{{name}};
    f{{name}}.open("static_arrays/{{name}}", std::ios::in | std::ios::binary);
    if(f{{name}}.is_open())
    {
        {% if name in dynamic_array_specs.values() %}
        f{{name}}.read(reinterpret_cast<char*>({{name}}.data()), {{N}}*sizeof({{dtype_spec}}));
        {% else %}
        f{{name}}.read(reinterpret_cast<char*>({{name}}), {{N}}*sizeof({{dtype_spec}}));
        {% endif %}
    } else
    {
        B2C_LOG_ERROR("Error opening static array {{name}}.");
    }
    {% if not (name in dynamic_array_specs.values()) %}
    CUDA_SAFE_CALL(
            cudaMemcpy(dev{{name}}, {{name}}, sizeof({{dtype_spec}})*{{N}}, cudaMemcpyHostToDevice)
            );
    {% else %}
    {% set Nbase = array_basename(name) %}
    copy_host_to_dev_array_{{ Nbase }}();
    {% endif %}
    {% endfor %}
}

void _write_arrays()
{
    using namespace brian;

    {% for var, varname in array_specs | dictsort(by='value') %}
    {% if not (var in dynamic_array_specs
                or var in dynamic_array_2d_specs
                or var in static_array_specs
              ) %}
    {# Don't copy State-, Spike- & EventMonitor's N variables, which are modified on host only #}
    {% if varname not in variables_on_host_only %}
    CUDA_SAFE_CALL(
            cudaMemcpy({{varname}}, dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyDeviceToHost)
            );
    {% endif %}
    std::ofstream outfile_{{varname}};
    outfile_{{varname}}.open(results_dir + "{{get_array_filename(var) | replace('\\', '\\\\')}}", std::ios::binary | std::ios::out);
    if(outfile_{{varname}}.is_open())
    {
        outfile_{{varname}}.write(reinterpret_cast<char*>({{varname}}), {{var.size}}*sizeof({{c_data_type(var.dtype)}}));
        outfile_{{varname}}.close();
    } else
    {
        B2C_LOG_ERROR("Error writing output file for {{varname}}.");
    }
    {% endif %}
    {% endfor %}

    {% for var, varname in dynamic_array_specs | dictsort(by='value') %}
    {% set N = array_basename(varname) %}
    {% if varname not in variables_on_host_only %}
    copy_dev_to_host_array_{{ N }}();
    {% endif %}
    std::ofstream outfile_{{varname}};
    outfile_{{varname}}.open(results_dir + "{{get_array_filename(var) | replace('\\', '\\\\')}}", std::ios::binary | std::ios::out);
    if(outfile_{{varname}}.is_open())
    {
        outfile_{{varname}}.write(reinterpret_cast<char*>({{varname}}.data()), {{varname}}.size()*sizeof({{c_data_type(var.dtype)}}));
        outfile_{{varname}}.close();
    } else
    {
        B2C_LOG_ERROR("Error writing output file for {{varname}}.");
    }
    {% endfor %}

    {% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
        {% if var in profile_statemonitor_vars %}
        {# Record copying statemonitor variable from device to host for benchmarking #}
        std::chrono::nanoseconds before_copy_statemon;
        std::string profile_statemonitor_copy_to_host_varname = "{{var.owner.name}}_copy_to_host_{{profile_statemonitor_copy_to_host}}";
        std::chrono::nanoseconds copy_time_statemon;
        {% endif %}
        std::ofstream outfile_{{varname}};
        outfile_{{varname}}.open(results_dir + "{{get_array_filename(var) | replace('\\', '\\\\')}}", std::ios::binary | std::ios::out);
        if(outfile_{{varname}}.is_open())
        {
            {% if var in profile_statemonitor_vars %}
            before_copy_statemon = std::chrono::high_resolution_clock::now();
            {% endif %}
            std::vector<{{c_data_type(var.dtype)}}>* temp_array{{varname}} = new std::vector<{{c_data_type(var.dtype)}}>[_num__array_{{var.owner.name}}__indices];
            for (int n=0; n<_num__array_{{var.owner.name}}__indices; n++)
            {
                temp_array{{varname}}[n].resize({{varname}}[n].size());
                if (!{{varname}}[n].empty())
                {
                    {{varname}}[n].copy_to_host(temp_array{{varname}}[n].data());
                }
            }
            {% if var in profile_statemonitor_vars %}
            std::string profile_statemonitor_copy_to_host_varname = "{{varname}}_copy_to_host";
            copy_time_statemon += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - before_copy_statemon);
            {% endif %}
            for(int j = 0; j < temp_array{{varname}}[0].size(); j++)
            {
                for(int i = 0; i < _num__array_{{var.owner.name}}__indices; i++)
                {
                    outfile_{{varname}}.write(reinterpret_cast<char*>(&temp_array{{varname}}[i][j]), sizeof({{c_data_type(var.dtype)}}));
                }
            }
            outfile_{{varname}}.close();
        } else
        {
        B2C_LOG_ERROR("Error writing output file for {{varname}}.");
        }
    {% endfor %}

    {% if profiled_codeobjects is defined and profiled_codeobjects %}
    // Write profiling info to disk
    std::ofstream outfile_profiling_info;
    outfile_profiling_info.open(results_dir + "profiling_info.txt", std::ios::out);
    if(outfile_profiling_info.is_open())
    {
    {% for codeobj in profiled_codeobjects | sort %}
    {#
    {% if 'spatialstateupdater' in codeobj and 'prepare' not in codeobj %}
    outfile_profiling_info << "{{codeobj}}_kernel_integration\t" << std::chrono::duration<double>({{codeobj}}_kernel_integration_profiling_info).count() << std::endl;
    outfile_profiling_info << "{{codeobj}}_kernel_tridiagsolve\t" << std::chrono::duration<double>({{codeobj}}_kernel_tridiagsolve_profiling_info).count() << std::endl;
    outfile_profiling_info << "{{codeobj}}_kernel_coupling\t" << std::chrono::duration<double>({{codeobj}}_kernel_coupling_profiling_info).count() << std::endl;
    outfile_profiling_info << "{{codeobj}}_kernel_combine\t" << std::chrono::duration<double>({{codeobj}}_kernel_combine_profiling_info).count() << std::endl;
    outfile_profiling_info << "{{codeobj}}_kernel_currents\t" << std::chrono::duration<double>({{codeobj}}_kernel_currents_profiling_info).count() << std::endl;
    {% endif %}
    #}
    outfile_profiling_info << "{{codeobj}}\t" << std::chrono::duration<double>({{codeobj}}_profiling_info).count() << std::endl;
    {% if profile_statemonitor_copy_to_host %}
    outfile_profiling_info << profile_statemonitor_copy_to_host_varname << "\t" << copy_time_statemon << std::endl;
    {% endif %}
    {% endfor %}
    outfile_profiling_info.close();
    } else
    {
        B2C_LOG_ERROR("Error writing profiling info to file.");
    }
    {% endif %}
    // Write last run info to disk
    std::ofstream outfile_last_run_info;
    outfile_last_run_info.open(results_dir + "last_run_info.txt", std::ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        B2C_LOG_ERROR("Error writing last run info to file.");
    }
}

void _dealloc_arrays()
{
    using namespace brian;

    {% for rng_type in all_codeobj_with_host_rng.keys() %}
    {% for co in all_codeobj_with_host_rng[rng_type] | sort(attribute='name') %}
    CUDA_SAFE_CALL(
            cudaFree(dev_{{co.name}}_{{rng_type}}_allocator)
            );
    {% endfor %}{# rng_type #}
    {% endfor %}{# co #}

    CUDA_SAFE_CALL(
            curandDestroyGenerator(curand_generator)
            );

    {% for S in synapses | sort(attribute='name') %}
    {% for path in S._pathways | sort(attribute='name') %}
    {{path.name}}_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("{{path.name}}_destroy");
    {% endfor %}
    {% endfor %}

    {% for var, varname in dynamic_array_specs | dictsort(by='value') %}
    dev{{varname}}.clear();
    {{varname}}.clear();
    std::vector<{{c_data_type(var.dtype)}}>().swap({{varname}});
    {% endfor %}

    {% for var, varname in array_specs | dictsort(by='value') %}
    {% if not var in dynamic_array_specs %}
    if({{varname}}!=0)
    {
        delete [] {{varname}};
        {{varname}} = 0;
    }
    if(dev{{varname}}!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev{{varname}})
                );
        dev{{varname}} = 0;
    }
    {% endif %}
    {% endfor %}

    {% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
    if ({{varname}} != nullptr)
    {
        for(int i = 0; i < _num__array_{{var.owner.name}}__indices; i++)
            {{varname}}[i].clear();
        delete [] {{varname}};
        {{varname}} = nullptr;
    }
    addresses_monitor_{{varname}}.clear();
    {% endfor %}

    // static arrays
    {% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
    {% if not (name in dynamic_array_specs.values()) %}
    if({{name}}!=0)
    {
        delete [] {{name}};
        {{name}} = 0;
    }
    {% endif %}
    {% endfor %}

    {% for varname in subgroups_with_spikemonitor %}
    _dev_{{varname}}_eventspace.clear();
    {% endfor %}

}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

// typedefs need to be outside the include guards to
// be visible to all files including objects.h
typedef {{curand_float_type}} randomNumber_t;  // random number type

#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include <vector>
#include <string>
#include <stdint.h>
#include "brianlib/clocks.h"
#include "brianlib/device_buffer.h"
{% if profiled_codeobjects is defined %}
#include <chrono>
{% endif %}

class Network;
{% if synapses %}
class SynapticPathway;
{% endif %}

namespace brian {

extern size_t used_device_memory;
extern std::string results_dir;

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

extern void set_variable_by_name(std::string, std::string);


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
{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
{% set N = array_basename(varname) %}
void copy_host_to_dev_array_{{ N }}();
void copy_dev_to_host_array_{{ N }}();
{% endfor %}

{% for var, varname in eventspace_arrays | dictsort(by='value') %}
void expand_eventspace{{ varname }}(int num_queues);
{% endfor %}
int filter_subgroup_eventspace(int32_t* src, int n, int32_t* dst, int32_t start, int32_t stop);
void upload_monitor_row_addresses(DeviceBuffer& addresses, DeviceBuffer* rows, int n_rows);

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


{% endmacro %}
