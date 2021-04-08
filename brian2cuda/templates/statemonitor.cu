{% extends 'common_group.cu' %}
{# USES_VARIABLES { t, N } #}
{# WRITES_TO_READ_ONLY_VARIABLES { t, N } #}

{% block define_N %}
{% endblock %}

{# remove this once we have properly defined num_threads, num_blocks here... #}
{% block occupancy %}
{% endblock %}
{% block update_occupancy %}
{% endblock %}
{% block kernel_info %}
{% endblock %}

{% block prepare_kernel_inner %}
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
// Print a warning when the monitor is not going to work (#50)
if (_num__array_{{owner.name}}__indices > 1024)
{
    printf("ERROR in {{owner.name}}: Too many neurons recorded. Due to a bug (brian-team/brian2cuda#50), "
            "currently only as many neurons can be recorded as threads can be called from a single block!\n");
}
{% endblock prepare_kernel_inner %}

{% block host_maincode %}
// TODO: this pushes a new value to the device each time step? Looks
// inefficient, can we keep the t values on the host instead? Do we need them
// on the device?
dev_dynamic_array_{{owner.name}}_t.push_back({{owner.clock.name}}.t[0]);

int num_iterations = {{owner.clock.name}}.i_end;
int current_iteration = {{owner.clock.name}}.timestep[0];
static int start_offset = current_iteration - _numt;
{% endblock host_maincode %}

{% block kernel_call %}
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

if (_num__array_{{owner.name}}__indices > 0)
// TODO we get invalid launch configuration if this is 0, which happens e.g. for StateMonitor(..., variables=[])
{
    kernel_{{codeobj_name}}<<<1, _num__array_{{owner.name}}__indices>>>(
        _num__array_{{owner.name}}__indices,
        dev_array_{{owner.name}}__indices,
        current_iteration - start_offset,
        {% for varname, var in _recorded_variables | dictsort %}
        {% set _recorded =  get_array_name(var, access_data=False) %}
        thrust::raw_pointer_cast(&addresses_monitor_{{_recorded}}[0]),
        {% endfor %}
        ///// HOST_PARAMETERS /////
        %HOST_PARAMETERS%
        );

    CUDA_CHECK_ERROR("kernel_{{codeobj_name}}");
}
{% endblock kernel_call %}

{% block kernel %}
__global__ void
{% if launch_bounds %}
__launch_bounds__(1024, {{sm_multiplier}})
{% endif %}
kernel_{{codeobj_name}}(
    int _num_indices,
    int32_t* indices,
    int current_iteration,
    {% for varname, var in _recorded_variables | dictsort %}
    {{c_data_type(var.dtype)}}** monitor_{{varname}},
    {% endfor %}
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    if(tid > _num_indices)
    {
        return;
    }
    int32_t _idx = indices[tid];

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// kernel_lines /////
    {{kernel_lines|autoindent}}

    ///// scalar_code /////
    {{scalar_code|autoindent}}

    // need different scope here since scalar_code and vector_code can
    // declare the same variables
    {
        ///// vector_code /////
        {{vector_code|autoindent}}

        {% for varname, var in _recorded_variables | dictsort %}
        {% set _recorded =  get_array_name(var, access_data=False) %}
        monitor_{{varname}}[tid][current_iteration] = _to_record_{{varname}};
        {% endfor %}
    }
}
{% endblock %}
