{# USES_VARIABLES { t, _clock_t, _indices, N } #}
{# WRITES_TO_READ_ONLY_VARIABLES { t, N } #}
{% extends 'common_group.cu' %}

{% block define_N %}
{% endblock %}

{% block modify_kernel_dimensions %}
// TODO _clock_t = {{_clock_t}} (used in cpp_standalone, can we use it?)
// TODO RENAME block modify_kernel_dimensions

{% for varname, var in _recorded_variables | dictsort %}
{% set _recorded =  get_array_name(var, access_data=False) %}
addresses_monitor_{{_recorded}}.clear();
{% endfor %}
for(int i = 0; i < _num__array_{{owner.name}}__indices; i++)
{
    {% for varname, var in _recorded_variables | dictsort %}
    {% set _recorded =  get_array_name(var, access_data=False) %}
    {{_recorded}}[i].resize(_numt + num_iterations - current_iteration);
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
// TODO: this pushes a new value to the device each time step? Looks
// inefficient, can we keep the t values on the host instead? Do we need them
// on the device?
dev_dynamic_array_{{owner.name}}_t.push_back({{owner.clock.name}}.t[0]);
// Update size variables for Python side indexing to work
// (Note: Need to update device variable which will be copied to host in write_arrays())
// TODO: This is one cudaMemcpy per time step, this should be done only once in the last
// time step, fix when fixing the statemonitor (currently only works for <=1024 threads)
_array_{{owner.name}}_N[0] += 1;
CUDA_SAFE_CALL(
        cudaMemcpy(dev_array_{{owner.name}}_N, _array_{{owner.name}}_N, sizeof(int32_t),
                   cudaMemcpyHostToDevice)
        );

int num_iterations = {{owner.clock.name}}.i_end;
int current_iteration = {{owner.clock.name}}.timestep[0];
static int start_offset = current_iteration - _numt;
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
        {{_recorded}}[i].resize(_numt + 1);
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


{% block indices %}
    int _vectorisation_idx = bid * THREADS_PER_BLOCK + tid;
    int _idx = {{_indices}}[_vectorisation_idx];
{% endblock %}


{% block extra_vector_code %}
    {% for varname, var in _recorded_variables | dictsort %}
    {% set _recorded =  get_array_name(var, access_data=False) %}
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
