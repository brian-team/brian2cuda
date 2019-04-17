{% extends 'common_group.cu' %}
{# USES_VARIABLES { N, count,
                    _source_start, _source_stop} #}

{% block extra_device_helper %}
    {% for varname, var in record_variables.items() %}
    // declare monitor cudaVectors
    __device__ cudaVector<{{c_data_type(var.dtype)}}>* monitor_{{varname}};
    {% endfor %}
{% endblock %}


{% block prepare_kernel_inner %}
_run_{{codeobj_name}}_init<<<1,1>>>();

CUDA_CHECK_ERROR("_run_{{codeobj_name}}_init");
num_blocks = 1;
num_threads = 1;
{% endblock prepare_kernel_inner %}


{% block kernel_call %}
kernel_{{codeobj_name}}<<<num_blocks, num_threads>>>(
        _num{{eventspace_variable.name}}-1,
        dev_array_{{owner.name}}_count,
        // HOST_PARAMETERS
        %HOST_PARAMETERS%);

CUDA_CHECK_ERROR("kernel_{{codeobj_name}}");
{% endblock %}


{% block kernel %}
__global__ void _run_{{codeobj_name}}_init()
{
    {% for varname, var in record_variables.items() %}
        monitor_{{varname}} = new cudaVector<{{c_data_type(var.dtype)}}>();
    {% endfor %}
}

__global__ void
{% if launch_bounds %}
__launch_bounds__(1024, {{sm_multiplier}})
{% endif %}
kernel_{{codeobj_name}}(
    int neurongroup_N,
    int32_t* count,
    // DEVICE_PARAMETERS
    %DEVICE_PARAMETERS%
    )
{
    using namespace brian;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    {# If there are no record_variables, we need to sum up the number of events at each kernel call #}
    {% if not record_variables %}
    // TODO: fix int types, num_events and  cudaVector::size() are int but {{N}} is size32_t
    int num_events = 0;
    {% endif %}

    // KERNEL_VARIABLES
    %KERNEL_VARIABLES%

    // scalar_code
    {{scalar_code|autoindent}}

    // using not parallel spikespace: filled from left with all spiking neuron IDs, -1 ends the list
    for(int i = 0; i < neurongroup_N; i++)
    {
        {% set _eventspace = get_array_name(eventspace_variable) %}
        int32_t spiking_neuron = {{_eventspace}}[i];
        if(spiking_neuron != -1)
        {
            if(_source_start <= spiking_neuron && spiking_neuron < _source_stop)
            {
                int _idx = spiking_neuron;
                int _vectorisation_idx = _idx;

                // vector_code
                {{vector_code|autoindent}}

                // push to monitors
                {% for varname, var in record_variables.items() %}
                monitor_{{varname}}->push(_to_record_{{varname}});
                {% endfor %}

                count[_idx -_source_start]++;

                {% if not record_variables %}
                num_events++;
                {% endif %}
            }
        }
        else
        {
            {% if not record_variables %}
            {{N}} += num_events;
            {% endif %}

            break;
        }
    }
}
{% endblock %}

{% block extra_functions_cu %}
__global__ void _run_debugmsg_{{codeobj_name}}_kernel(
    // DEVICE_PARAMETERS
    %DEVICE_PARAMETERS%
)
{
    using namespace brian;

    // KERNEL_VARIABLES
    %KERNEL_VARIABLES%

    printf("Number of spikes: %d\n", {{N}});
}

__global__ void _count_{{codeobj_name}}_kernel(
    int* dev_num_events,
    // DEVICE_PARAMETERS
    %DEVICE_PARAMETERS%
)
{
    using namespace brian;
    // TODO: fix int types, num_events and  cudaVector::size() are int but {{N}} is size32_t
    int num_events;

    // KERNEL_VARIABLES
    %KERNEL_VARIABLES%

    {# If there are any record_variables, get the size of one arbitrary monitor #}
    {% if record_variables %}
    {% set varname = record_variables.keys()[0] %}
    num_events = monitor_{{varname}}->size();
    {{N}} = num_events;
    {# If there are no record_variables, we add the count after each kernel call #}
    {% else %}
    num_events = {{N}};
    {% endif %}

    *dev_num_events = num_events;
}

__global__ void _copy_{{codeobj_name}}_kernel(
    {% for varname, var in record_variables.items() %}
    {{c_data_type(var.dtype)}}* dev_monitor_{{varname}},
    {% endfor %}
    int dummy  {# loop ends with comma... #}
)
{
    using namespace brian;
    int index = 0;

    // copy monitors
    {% for varname, var in record_variables.items() %}
    index = 0;
    for(int j = 0; j < monitor_{{varname}}->size(); j++)
    {
        dev_monitor_{{varname}}[index] = monitor_{{varname}}->at(j);
        index++;
    }
    {% endfor %}
}

void _copyToHost_{{codeobj_name}}()
{
    using namespace brian;

    const std::clock_t _start_time = std::clock();

    // TODO: Use the correct dev_eventmonitor_N instead of dev_num_events
    //   and the correct _array_eventmonitor_N instead of host_num_events.
    //       use: dev_array_{{owner.name}}_N and _array_{{owner.name}}_N
    //   dev_array_.. gets copied to _array_... in objects.cu::write_arrays()
    //   copying it here would result in copying it twice.
    //   monitor_... and dev_monitor... store the exact same values, but we
    //   need monitor_... as cudaVector for changing size from device funtions.
    //   Maybe use cudaVector as default for dynamic arrays, then we would not
    //   need monitor... at all. This would mean changing the copying in objects.cu
    //   for dynamic arrays (currently we just use thrust device to host vector).
    int host_num_events;
    int* dev_num_events;

    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_num_events, sizeof(int))
            );

    // CONSTANTS
    %CONSTANTS%

    _count_{{codeobj_name}}_kernel<<<1,1>>>(
        dev_num_events,
        // HOST_PARAMETERS
        %HOST_PARAMETERS%
        );

    CUDA_CHECK_ERROR("_count_{{codeobj_name}}_kernel");

    CUDA_SAFE_CALL(
            cudaMemcpy(&host_num_events, dev_num_events, sizeof(int), cudaMemcpyDeviceToHost)
            );

    // resize monitor device vectors
    {% for varname, var in record_variables.items() %}
    THRUST_CHECK_ERROR(
            dev_dynamic_array_{{owner.name}}_{{varname}}.resize(host_num_events)
            );
    {% endfor %}

    _copy_{{codeobj_name}}_kernel<<<1,1>>>(
        {% for varname, var in record_variables.items() %}
        thrust::raw_pointer_cast(&dev_dynamic_array_{{owner.name}}_{{varname}}[0]),
        {% endfor %}
        0  {# dummy, becaus loop ends with comma #}
        );

    CUDA_CHECK_ERROR("_copy_{{codeobj_name}}_kernel");
}

void _debugmsg_{{codeobj_name}}()
{
    using namespace brian;

    // CONSTANTS
    %CONSTANTS%

    // TODO: can't we acces the correct _array_eventmonitor_N[0]
    //   value here without any kernel call?
    //   Yes: use _array_{{owner.name}}_N
    _run_debugmsg_{{codeobj_name}}_kernel<<<1,1>>>(
            // HOST_PARAMETERS
            %HOST_PARAMETERS%
            );

    CUDA_CHECK_ERROR("_run_debugmsg_{{codeobj_name}}_kernel");
}
{% endblock %}

{% block extra_functions_h %}
void _copyToHost_{{codeobj_name}}();
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_copyToHost_{{codeobj_name}}();
_debugmsg_{{codeobj_name}}();
{% endmacro %}
