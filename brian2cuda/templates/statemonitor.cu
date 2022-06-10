{# USES_VARIABLES { t, _indices, N } #}
{# WRITES_TO_READ_ONLY_VARIABLES { t, N } #}
{% extends 'common_group.cu' %}

{% block define_N %}
{% endblock %}

{# We are using block modify_kernel_dimensions for additional kernel preparation #}
{% block modify_kernel_dimensions %}
{% for varname, var in _recorded_variables | dictsort %}
{% set _recorded = get_array_name(var, access_data=False) %}
addresses_monitor_{{_recorded}}.clear();
{% endfor %}
for(int i = 0; i < _num__array_{{owner.name}}__indices; i++)
{
    {% for varname, var in _recorded_variables | dictsort %}
    {% set _recorded = get_array_name(var, access_data=False) %}
    {{_recorded}}[i].resize(_numt_host + num_iterations - current_iteration);
    addresses_monitor_{{_recorded}}.push_back(thrust::raw_pointer_cast(&{{_recorded}}[i][0]));
    {% endfor %}
}
{% endblock modify_kernel_dimensions %}

{% block host_maincode %}
// NOTE: We are using _N as the number of recorded indices here (the relevant size for
// parallelization). This is different from `StateMonitor.N` in Python, which refers to
// the number of recorded time steps (while `StateMonitor.n_indices` gives the number of
// recorded indices).
const int _N = _num_indices;

// We are using an extra variable because HOST_CONSTANTS uses the device vector, which
// is not used (TODO: Fix this in HOST_CONSTANTS instead of this hack here...)
const int _numt_host = _dynamic_array_{{owner.name}}_t.size();

// We push t only on host and don't make a device->host copy in write_arrays()
_dynamic_array_{{owner.name}}_t.push_back({{owner.clock.name}}.t[0]);

// Update size variables for Python side indexing to work
_array_{{owner.name}}_N[0] += 1;

int num_iterations = {{owner.clock.name}}.i_end;
int current_iteration = {{owner.clock.name}}.timestep[0];
static int start_offset = current_iteration - _numt_host;
{% endblock host_maincode %}


{% block extra_kernel_call %}
// If the StateMonitor is run outside the MagicNetwork, we need to resize it.
// Happens e.g. when StateMonitor.record_single_timestep() is called.
if(current_iteration >= num_iterations)
{
    for(int i = 0; i < _num__array_{{owner.name}}__indices; i++)
    {
        {% for varname, var in _recorded_variables | dictsort %}
        {% set _recorded =  get_array_name(var, access_data=False) %}
        {{_recorded}}[i].resize(_numt_host + 1);
        addresses_monitor_{{_recorded}}[i] = thrust::raw_pointer_cast(&{{_recorded}}[i][0]);
        {% endfor %}
    }
}

// TODO we get invalid launch configuration if this is 0, which happens e.g. for StateMonitor(..., variables=[])
if (_num__array_{{owner.name}}__indices > 0)
{
{% endblock extra_kernel_call %}


{% block extra_kernel_call_post %}
{# Close conditional from block extra_kernel_call #}
}
{% endblock %}


{# Need to set _idx here, after threads >= N returend, else this fails #}
{% block after_return_N %}
    _idx = {{_indices}}[_vectorisation_idx];
{% endblock %}


{% block extra_vector_code %}
    {% for varname, var in _recorded_variables | dictsort %}
    monitor_{{varname}}[_vectorisation_idx][current_iteration] = _to_record_{{varname}};
    {% endfor %}
{% endblock extra_vector_code %}


{% block extra_kernel_parameters %}
    int current_iteration,
    {% for varname, var in _recorded_variables | dictsort %}
    {{c_data_type(var.dtype)}}** monitor_{{varname}},
    {% endfor %}
{% endblock %}


{% block extra_host_parameters %}
    current_iteration - start_offset,
    {% for varname, var in _recorded_variables | dictsort %}
    {% set _recorded =  get_array_name(var, access_data=False) %}
    thrust::raw_pointer_cast(&addresses_monitor_{{_recorded}}[0]),
    {% endfor %}
{% endblock %}
