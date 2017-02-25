{% extends 'common_synapses.cu' %}
{# USES_VARIABLES { N, _synaptic_pre} #}

{% set _non_synaptic = [] %}
{% for var in variables %}
    {% if variable_indices[var] != '_idx' %}
        {# This is a trick to get around the scoping problem #}
        {% if _non_synaptic.append(1) %}{% endif %}
    {% endif %}
{% endfor %}

{% block kernel %}

__global__ void kernel_{{codeobj_name}}_scalar_delays(
	unsigned int bid_offset,
	unsigned int timestep,
	unsigned int THREADS_PER_BLOCK,
	int32_t* eventspace,
	unsigned int neurongroup_size,
	%DEVICE_PARAMETERS%
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x + bid_offset;
	if (tid==0 && bid==0)
		printf("DEBUG: timestep = %u\n", timestep);
	//TODO: do we need _idx here? if now, get also rid of scoping after scalar code
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;

	///// KERNEL_VARIABLES /////
	%KERNEL_VARIABLES%

	{% block additional_variables %}
	{% endblock %}

	///// scalar_code /////
	{{scalar_code|autoindent}}

	{
		if (timestep < 5 && tid==0 && bid==0)
			printf("DEBUG in {{pathway.name}}_codeobject: NO OR CONST DEALAY MODE\n");
	        // for the first delay timesteps the eventspace is not yet filled
		// note that max_delay is the number of eventspaces, max_delay-1 the delay in timesteps
	        if (timestep >= {{pathway.name}}.queue->max_delay - 1)
	        {
	                // loop through neurons in eventspace (indices of event neurons, rest -1)
	                for(int i = 0; i < neurongroup_size; i++)
	                {
	                        // spiking_neuron is index in NeuronGroup
	                        int32_t spiking_neuron = eventspace[i];

	                        if(spiking_neuron == -1) // end of spiking neurons
	                        {
	                                assert(i == eventspace[neurongroup_size]);
					if (tid==0 && bid==0)
						printf("DEBUG in timestep %u we left eventspace at index %i\n", timestep, i);
	                                return;
	                        }
	                        // apply effects if event neuron is in sources of current SynapticPathway
	                        if({{pathway.name}}.spikes_start <= spiking_neuron && spiking_neuron < {{pathway.name}}.spikes_stop)
	                        {
	                                unsigned int right_offset = (spiking_neuron - {{pathway.name}}.spikes_start) * {{pathway.name}}.queue->num_blocks + bid;
	                                int size = {{pathway.name}}_size_by_pre[right_offset];
	                                int32_t* propagating_synapses = {{pathway.name}}_synapses_id_by_pre[right_offset];
	                                for(int j = tid; j < size; j+=THREADS_PER_BLOCK)
	                                {
						// _idx is the synapse id
	                                        int32_t _idx = propagating_synapses[j];

						///// vector_code /////
	                                        {{vector_code|autoindent}}
	                                }
	                        }

	                        __syncthreads();
	                }
	        }
	        else if (tid==0 && bid==0)
	        {
	                printf("not applying effects in synapses.cu for timestep=%u\n", timestep);
	        }
	}
}

__global__ void kernel_{{codeobj_name}}_heterogeneous_delays(
	unsigned int bid_offset,
	unsigned int timestep,
	unsigned int THREADS_PER_BLOCK,
	int32_t* eventspace,
	unsigned int neurongroup_size,
	%DEVICE_PARAMETERS%
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x + bid_offset;
	if (tid==0 && bid==0)
		printf("DEBUG: timestep = %u\n", timestep);
	//TODO: do we need _idx here? if now, get also rid of scoping after scalar code
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;

	///// KERNEL_VARIABLES /////
	%KERNEL_VARIABLES%

	{# this calls the macro created by the additional_variables block above #}
	{{ self.additional_variables() }}

	///// scalar_code /////
	{{scalar_code|autoindent}}

	{
		if (timestep < 5)
			printf("DEBUG in {{pathway.name}}_codeobject: HETEROG DELAYS\n");
	        cudaVector<int32_t>* synapses_queue;
	        {{pathway.name}}.queue->peek(
	                &synapses_queue);

		int size = synapses_queue[bid].size();
		for(int j = tid; j < size; j+=THREADS_PER_BLOCK)
		{
			int32_t _idx = synapses_queue[bid].at(j);
	
			///// vector_code /////
			{{vector_code|autoindent}}
		}
	}
}

{% endblock %}

{% block kernel_call %}
{% set eventspace_variable = pathway.variables[pathway.eventspace_name] %}
{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}

{% if serializing_mode == "syn" %}
// serializing_mode == "syn"
unsigned int num_blocks = num_parallel_blocks;
unsigned int num_threads = max_threads_per_block;
unsigned int num_loops = 1;
if ({{owner.clock.name}}.timestep[0] == 0)
	printf("INFO {{owner.name}} Serializing in 'syn' mode!\n");
{% elif serializing_mode == "post" %}
// serializing_mode == "post"
unsigned int num_blocks = num_parallel_blocks;
unsigned int num_threads = 1;
unsigned int num_loops = 1;
if ({{pathway.name}}_scalar_delay && !{{owner.name}}_multiple_pre_post)
{
	num_threads = max_threads_per_block;
	if ({{owner.clock.name}}.timestep[0] == 0)
		printf("INFO {{owner.name}} Not serializing in 'post' mode (no multiple pre/post pairs)!\n");
}
else if ({{owner.clock.name}}.timestep[0] == 0)
{
	printf("INFO {{owner.name}} Serializing in 'post' mode!\n");
}
{% elif serializing_mode == "pre" %}
// serializing_mode == "pre"
unsigned int num_blocks = 1;
unsigned int num_threads = 1;
unsigned int num_loops = num_parallel_blocks;
if ({{owner.clock.name}}.timestep[0] == 0)
	printf("INFO {{owner.name}} Serializing in 'pre' mode!\n");
{% endif %}

if ({{pathway.name}}_scalar_delay)
{
	for(unsigned int bid_offset = 0; bid_offset < num_loops; bid_offset++)
	{
		kernel_{{codeobj_name}}_scalar_delays<<<num_blocks, num_threads>>>(
			bid_offset,
			{{owner.clock.name}}.timestep[0],
			num_threads,
			dev{{_eventspace}}[{{pathway.name}}_eventspace_idx],
			_num_{{_eventspace}}-1,
			%HOST_PARAMETERS%
		);
	//	cudaDeviceSynchronize();
	}
}
else  // heterogeneous delays
{
	for(unsigned int bid_offset = 0; bid_offset < num_loops; bid_offset++)
	{
		kernel_{{codeobj_name}}_heterogeneous_delays<<<num_blocks, num_threads>>>(
			bid_offset,
			{{owner.clock.name}}.timestep[0],
			num_threads,
			dev{{_eventspace}}[{{pathway.name}}_eventspace_idx],
			_num_{{_eventspace}}-1,
			%HOST_PARAMETERS%
		);
	//	cudaDeviceSynchronize();
	}
}
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
