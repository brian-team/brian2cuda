{# USES_VARIABLES { N, rate, t, _spikespace, _clock_t, _clock_dt,
                    _num_source_neurons, _source_start, _source_stop } #}
{# WRITES_TO_READ_ONLY_VARIABLES { N } #}
{% extends 'common_group.cu' %}

{% block define_N %}
{% endblock %}

{% block host_maincode %}
int current_iteration = {{owner.clock.name}}.timestep[0];
static int start_offset = current_iteration;
{% endblock %}

{% block prepare_kernel_inner %}
int num_iterations = {{owner.clock.name}}.i_end;
int size_till_now = static_cast<int>(dev{{_dynamic_t}}.size());
int new_size = num_iterations + size_till_now - start_offset;
dev{{_dynamic_t}}.resize(new_size);
dev{{_dynamic_rate}}.resize(new_size);
// Update size variables for Python side indexing to work
_array_{{owner.name}}_N[0] = new_size;

num_threads = 1;
num_blocks = 1;
{% endblock %}

{% block kernel_call %}
_run_kernel_{{codeobj_name}}<<<num_blocks, num_threads>>>(
    current_iteration - start_offset,
    dev{{_dynamic_rate}}.data_as<{{c_data_type(variables['rate'].dtype)}}>(),
    dev{{_dynamic_t}}.data_as<{{c_data_type(variables['t'].dtype)}}>(),
    ///// HOST_PARAMETERS /////
    %HOST_PARAMETERS%);

CUDA_CHECK_ERROR("_run_kernel_{{codeobj_name}}");
{% endblock %}

{% block kernel %}
__global__ void
{% if launch_bounds %}
__launch_bounds__(1024, {{sm_multiplier}})
{% endif %}
_run_kernel_{{codeobj_name}}(
    int32_t current_iteration,
    {% set c_type = c_data_type(variables['rate'].dtype) %}
    {{c_type}}* ratemonitor_rate,
    {% set c_type = c_data_type(variables['t'].dtype) %}
    {{c_type}}* ratemonitor_t,
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    using namespace brian;

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// kernel_lines /////
    {{kernel_lines|autoindent}}

    int num_spikes = 0;

    if (_num_spikespace-1 != _num_source_neurons)  // we have a subgroup
    {
        // TODO shouldn't this be 'i < _num_spikespace -1'?
        for (int i=0; i < _num_spikespace; i++)
        {
            const int spiking_neuron = {{_spikespace}}[i];
            if (spiking_neuron != -1)
            {
                // check if spiking neuron is in this subgroup
                if (_source_start <= spiking_neuron && spiking_neuron < _source_stop)
                    num_spikes++;
            }
            else  // end of spiking neurons
            {
                break;
            }
        }
    }
    else  // we don't have a subgroup
    {
        num_spikes = {{_spikespace}}[_num_source_neurons];
    }

    // TODO: we should be able to use {{rate}} and {{t}} here instead of passing these
    //       additional pointers. But this results in thrust::system_error illegal memory access.
    //       Don't know why... {{rate}} and ratemonitor_rate should be the same...
    ratemonitor_rate[current_iteration] = 1.0*num_spikes/{{_clock_dt}}/_num_source_neurons;
    ratemonitor_t[current_iteration] = {{_clock_t}};
}
{% endblock %}
