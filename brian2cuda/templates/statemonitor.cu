{# USES_VARIABLES { t, _indices, N } #}
{# WRITES_TO_READ_ONLY_VARIABLES { t, N } #}
{% extends 'common_group.cu' %}

{% block define_N %}
{% endblock %}

{# We are using block modify_kernel_dimensions for additional kernel preparation #}
{% block modify_kernel_dimensions %}
for(int i = 0; i < _num__array_{{owner.name}}__indices; i++)
{
    {% for varname, var in _recorded_variables | dictsort %}
    {% set _recorded = get_array_name(var, access_data=False) %}
    {{ _recorded }}[i].resize(_numt_host + num_iterations - current_iteration);
    {% endfor %}
}
{% for varname, var in _recorded_variables | dictsort %}
{% set _recorded = get_array_name(var, access_data=False) %}
upload_monitor_row_addresses(
    addresses_monitor_{{ _recorded }},
    {{ _recorded }},
    _num__array_{{ owner.name }}__indices);
{% endfor %}
{% endblock modify_kernel_dimensions %}

{% block host_maincode %}
// NOTE: We are using _N as the number of recorded indices here (the relevant size for
// parallelization). This is different from `StateMonitor.N` in Python, which refers to
// the number of recorded time steps (while `StateMonitor.n_indices` gives the number of
// recorded indices).
const int _N = _num_indices;

const int _numt_host = _dynamic_array_{{ owner.name }}_t.size();
_dynamic_array_{{ owner.name }}_t.push_back({{ owner.clock.name }}.t[0]);

// Update size variables for Python side indexing to work
_array_{{owner.name}}_N[0] += 1;

int num_iterations = {{owner.clock.name}}.i_end;
int current_iteration = {{owner.clock.name}}.timestep[0];
static int start_offset = current_iteration - _numt_host;
{% endblock host_maincode %}


{% block extra_kernel_call %}
// If the StateMonitor is run outside the Network, we need to resize it.
// Happens e.g. when StateMonitor.record_single_timestep() is called.
if(current_iteration >= num_iterations)
{
    for(int i = 0; i < _num__array_{{owner.name}}__indices; i++)
    {
        {% for varname, var in _recorded_variables | dictsort %}
        {% set _recorded = get_array_name(var, access_data=False) %}
        {{ _recorded }}[i].resize(_numt_host + 1);
        {% endfor %}
    }
    {% for varname, var in _recorded_variables | dictsort %}
    {% set _recorded = get_array_name(var, access_data=False) %}
    upload_monitor_row_addresses(
        addresses_monitor_{{ _recorded }},
        {{ _recorded }},
        _num__array_{{ owner.name }}__indices);
    {% endfor %}
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
    addresses_monitor_{{ _recorded }}.data_as<{{c_data_type(var.dtype)}}*>(),
    {% endfor %}
{% endblock %}
