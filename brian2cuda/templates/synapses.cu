{% extends 'common_synapses.cu' %}

{% set _non_synaptic = [] %}
{% for var in variables %}
    {% if variable_indices[var] != '_idx' %}
        {# This is a trick to get around the scoping problem #}
        {% if _non_synaptic.append(1) %}{% endif %}
    {% endif %}
{% endfor %}

{% block kernel %}

__global__ void kernel_{{codeobj_name}}(
	unsigned int bid_offset,
	unsigned int THREADS_PER_BLOCK,
	%DEVICE_PARAMETERS%
	)
{
	{# USES_VARIABLES { N, _synaptic_pre, _spikespace} #}
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x + bid_offset;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%
	{% block additional_variables %}
	{% endblock %}

	cudaVector<int32_t>* synapses_queue;
	
	{{pathway.name}}.queue->peek(
		&synapses_queue);

	{{scalar_code|autoindent}}

	{
	if (!({{pathway.name}}.no_or_const_delay_mode))
	{
		int size = synapses_queue[bid].size();
		for(int j = tid; j < size; j+=THREADS_PER_BLOCK)
		{
			int32_t _idx = synapses_queue[bid].at(j);
	
			{{vector_code|autoindent}}
		}
	}
	else
	{
		if(bid != 0)
			return;
		//no or const delay mode
		for(int j = 0; j < _num_spikespace; j++)
		{
			int32_t spiking_neuron = {{_spikespace}}[j];
			if(spiking_neuron == -1)
			{
				break;
			}
			for(int i = tid; i < {{pathway.name}}_size_by_pre[spiking_neuron]; i+= THREADS_PER_BLOCK)
			{
				int32_t _idx = {{pathway.name}}_synapses_id_by_pre[spiking_neuron][i];
			
				{{vector_code|autoindent}}
			}
			__syncthreads();
		}
	}
	}
}

{% endblock %}

{% block kernel_call %}
	{% if serializing_mode == "syn" %}
	kernel_{{codeobj_name}}<<<num_parallel_blocks,max_threads_per_block>>>(
		0,
		max_threads_per_block,
		%HOST_PARAMETERS%
	);
	{% endif %}
	{% if serializing_mode == "post" %}
	kernel_{{codeobj_name}}<<<num_parallel_blocks,1>>>(
		0,
		1,
		%HOST_PARAMETERS%
	);
	{% endif %}
	{% if serializing_mode == "pre" %}
	for(int i = 0; i < num_parallel_blocks; i++)
	{
		kernel_{{codeobj_name}}<<<1,1>>>(
			i,
			1,
			%HOST_PARAMETERS%
		);
	}
	{% endif %}
{% endblock %}

{% block extra_maincode %}
{% endblock %}

{% block extra_functions_cu %}
void _debugmsg_{{codeobj_name}}()
{
	using namespace brian;
	std::cout << "Number of synapses: " << dev{{_dynamic__synaptic_pre}}.size() << endl;
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
