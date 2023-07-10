{# USES_VARIABLES { N, count, _source_start, _source_stop, _source_idx} #}
{# WRITES_TO_READ_ONLY_VARIABLES { N, count } #}
{% extends 'common_group.cu' %}


{% block extra_headers %}
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
{% endblock %}


{% block extra_device_helper %}
{% if owner.source.__class__.__name__ == 'Subgroup' %}
struct is_in_subgroup
{
  __device__
  bool operator()(const int32_t &neuron)
  {
    return ({{_source_start}} <= neuron && neuron < {{_source_stop}});
  }
};
{% endif %}{# Subgroup #}
{% endblock extra_device_helper %}


{# We change _N depending on number of events and subgroups #}
{% block define_N %}
// The _N of this kernel (total number of threads) is defined by the number of events
int _N;
{% endblock %}


{# num_block changes depending on num_events #}
{% block static_kernel_dimensions %}
static int num_threads;
int num_events, num_blocks;
{% endblock %}


{% block modify_kernel_dimensions %}
{% if owner.source.__class__.__name__ == 'Subgroup' %}
// Initialize device vector for subgroup eventspace
THRUST_CHECK_ERROR(
    _dev_{{owner.source.name}}_eventspace.resize(_num_source_idx)
);
{% endif %}{# Subgroup #}
{% endblock %}


{% block kernel_call %}
{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}
{# Our eventspace is an unsorted array of neuron indices, followed by -1 values
   (for neurons for which the event was not triggered). For Subgroups, the same
   eventspace is used, which means that the neuron indices in the eventspace
   have to be checked for being part of the subgroup first, before recording
   them in the spikemonitor. Here, we copy all neuron indices which are within
   the subgroup into a new array. This way we can treat that array just like
   the eventspace for a full neurongroup (not having to check if the spiking
   neurons are part of the subgroup). #}

// Number of events in eventspace
int _num_events, _num_events_subgroup;
int32_t* _eventspace = dev{{_eventspace}}[current_idx{{_eventspace}}];
CUDA_SAFE_CALL(
        cudaMemcpy(
            &_num_events,
            &_eventspace[_num{{eventspace_variable.name}} - 1],
            sizeof(int32_t),
            cudaMemcpyDeviceToHost
            )
        );

{# TODO: Use isintance instead (needs to be made available in Jinja templates) #}
{% if owner.source.__class__.__name__ == 'Subgroup' %}
// Count the elements in eventspace that are in the subgroup
thrust::device_ptr<int32_t> _dev_eventspace(_eventspace);
THRUST_CHECK_ERROR(
    _N = thrust::count_if(
        _dev_eventspace,
        _dev_eventspace + _num_events,
        is_in_subgroup()
    )
);
// Copy all neuron IDs that are in this subgroup to the new device pointer
THRUST_CHECK_ERROR(
    thrust::copy_if(
        _dev_eventspace,
        _dev_eventspace + _num_events,
        _dev_{{owner.source.name}}_eventspace.begin(),
        is_in_subgroup()
    )
);
// Use same kernel as without subgroups on copied subgroup eventspace
_eventspace = thrust::raw_pointer_cast(&_dev_{{owner.source.name}}_eventspace[0]);
{% else %}{# not is_subgroup #}
// Get the number of events
_N = _num_events;
{% endif %}{# is_subgroup #}

{# If we don't record variables, we don't need to resize the monitor vectors #}
{% if record_variables %}
{# Get any item to read the size (we need to resize before HOST_CONSTANTS
   because the pointers underlying the device vectors change when resizing
   leads to memory reallocation) #}
{% set var = record_variables.values() | first %}
{% set dev_array_name = get_array_name(var, access_data=False, prefix='dev') %}
// Get current size of device vectors
int _monitor_size = {{dev_array_name}}.size();

// Increase device vectors based on number of events
{% for varname, var in record_variables.items() %}
{% set dev_array_name = get_array_name(var, access_data=False, prefix='dev') %}
THRUST_CHECK_ERROR(
        {{dev_array_name}}.resize(_monitor_size + _N)
        );
{% endfor %}
{% endif %}{# not record_variables #}

// Round up number of blocks according to number of events
num_blocks = (_N + num_threads - 1) / num_threads;

// Only call kernel if there are events in the current time step
if (_N > 0)
{
    _run_kernel_{{codeobj_name}}<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            {% if record_variables %}
            _monitor_size,
            {% endif %}
            _eventspace,
            {# We need to get a new raw_pointer_cast because the thrust vectors
               were resized, which changes the underlying data pointer if memory
               has to be reallocated #}
            {% for varname, var in record_variables.items() %}
            {% set dev_array_name = get_array_name(var, access_data=False, prefix='dev') %}
            thrust::raw_pointer_cast(&{{dev_array_name}}[0]),
            {% endfor %}
            ///// HOST_PARAMETERS /////
            %HOST_PARAMETERS%
        );

    CUDA_CHECK_ERROR("_run_kernel_{{codeobj_name}}");
}

// Increase total number of events in monitor
_array_{{owner.name}}_N[0] += _N;
{% endblock kernel_call %}


{# Add _eventspace manually as it can be the copied event space for subgroups #}
{% block extra_kernel_parameters %}
        {% if record_variables %}
        int _monitor_size,
        {% endif %}
        int32_t* _eventspace,
        {% for varname, var in record_variables.items() %}
        {% set _array_name = get_array_name(var) %}
        {{c_data_type(var.dtype)}}* _new{{_array_name}},
        {% endfor %}
{% endblock %}


{% block kernel_maincode %}
{# We pass as _eventspace the filtered eventspace, such that all neuron IDs are
   within the subgroup (if this is one). We take care of that with the
   thrust::copy_if above. #}

// Eventspace is filled from left with all neuron IDs that triggered an event, rest -1
int32_t spiking_neuron = _eventspace[_idx];
assert(spiking_neuron != -1);

{% if record_variables %}
int _monitor_idx = _vectorisation_idx + _monitor_size;
{% endif %}
_idx = spiking_neuron;
_vectorisation_idx = _idx;

// vector_code
{{vector_code|autoindent}}

{#
{% if owner.source.__class__.__name__ == 'Subgroup' %}
assert({{_source_idx}}[_source_start] <= spiking_neuron);
assert(spiking_neuron < {{_source_idx}}[_source_stop]);
{% endif %}
#}

// fill the monitors
{% for varname, var in record_variables.items() %}
_new{{get_array_name(var)}}[_monitor_idx] = _to_record_{{varname}};
{% endfor %}

{{count}}[_idx - _source_start]++;
{% endblock kernel_maincode %}


{% block extra_functions_cu %}
void _debugmsg_{{codeobj_name}}()
{
    using namespace brian;

    // HOST_CONSTANTS
    %HOST_CONSTANTS%

    printf("Number of spikes: %d\n", _array_{{owner.name}}_N[0]);
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
