{% extends 'common_group.cu' %}
{# USES_VARIABLES { N, t, i, _clock_t, _spikespace, count,
                    _source_start, _source_stop} #}
                    
{% block extra_device_helper %}
	{% for varname, var in record_variables.items() %}
		__device__ cudaVector<{{c_data_type(var.dtype)}}>* monitor_{{varname}};
	{% endfor %}
{% endblock %}

{% block kernel_call %}
static bool first_run = true;
if(first_run)
{
	_run_{{codeobj_name}}_init<<<1,1>>>();
	first_run = false;
}
_run_{{codeobj_name}}_kernel<<<1, 1>>>(
		_num{{eventspace_variable.name}}-1,
		dev_array_{{owner.name}}_count,
		%HOST_PARAMETERS%);
{% endblock %}

{% block kernel %}
__global__ void _run_{{codeobj_name}}_init()
{
	{% for varname, var in record_variables.items() %}
		monitor_{{varname}} = new cudaVector<{{c_data_type(var.dtype)}}>();
	{% endfor %}
}

__global__ void _run_{{codeobj_name}}_kernel(
	unsigned int neurongroup_N,
	int32_t* count,
	%DEVICE_PARAMETERS%
	)
{
	using namespace brian;
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	__syncthreads();
	
	%KERNEL_VARIABLES%

	{{scalar_code|autoindent}}

	// using not parallel spikespace: filled from left with all spiking neuron IDs, -1 ends the list
	for(int i = 0; i < neurongroup_N; i++)
	{
		{% set _eventspace = get_array_name(eventspace_variable) %}
		int32_t spiking_neuron = {{_eventspace}}[i];
		if(spiking_neuron != -1)
		{
			if(spiking_neuron >= _source_start && spiking_neuron < _source_stop)
			{
				int _idx = spiking_neuron;
				int _vectorisation_idx = _idx;
				{{vector_code|autoindent}}
				{% for varname, var in record_variables.items() %}
					monitor_{{varname}}->push(_to_record_{{varname}});
				{% endfor %}
				count[_idx -_source_start]++;
			}
		}
		else
		{
			break;
		}
	}
}
{% endblock %}

{% block extra_functions_cu %}
__global__ void _run_debugmsg_{{codeobj_name}}_kernel()
{
	using namespace brian;
	unsigned int total_number = 0;
	// TODO: whay are we looping over all record_variables.items() in template?
	{% for varname, var in record_variables.items() %}
	total_number = monitor_{{varname}}->size();
	{% endfor %}
	printf("Number of spikes: %d\n", total_number);
}

__global__ void _count_{{codeobj_name}}_kernel(
	%DEVICE_PARAMETERS%,
	unsigned int* total
)
{
	using namespace brian;
	%KERNEL_VARIABLES%
	
	unsigned int total_number = 0;
	{% for varname, var in record_variables.items() %}
	total_number = monitor_{{varname}}->size();
	{% endfor %}
	*total = total_number;
	{{N}} = total_number;
}

__global__ void _copy_{{codeobj_name}}_kernel(
	{% for varname, var in record_variables.items() %}
		{{c_data_type(var.dtype)}}* dev_monitor_{{varname}},
	{% endfor %}
	unsigned int num_blocks // TODO: no need of num_blocks, but for loop end with komma, fix?
)
{
	using namespace brian;
	unsigned int index = 0;
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

	%CONSTANTS%

    {% set _eventspace = get_array_name(eventspace_variable) %}
    unsigned int* dev_total;
    cudaMalloc((void**)&dev_total, sizeof(unsigned int));
	_count_{{codeobj_name}}_kernel<<<1,1>>>(
		%HOST_PARAMETERS%,
		dev_total);
	unsigned int total;
	cudaMemcpy(&total, dev_total, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	{% for varname, var in record_variables.items() %}
		dev_dynamic_array_{{owner.name}}_{{varname}}.resize(total);
	{% endfor %}

	_copy_{{codeobj_name}}_kernel<<<1,1>>>(
		{% for varname, var in record_variables.items() %}
			thrust::raw_pointer_cast(&dev_dynamic_array_{{owner.name}}_{{varname}}[0]),
		{% endfor %}
		true
		);
}

void _debugmsg_{{codeobj_name}}()
{
	using namespace brian;

	%CONSTANTS%

    {% set _eventspace = get_array_name(eventspace_variable) %}
	_run_debugmsg_{{codeobj_name}}_kernel<<<1,1>>>();
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
