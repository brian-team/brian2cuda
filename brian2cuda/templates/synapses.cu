{% extends 'common_synapses.cu' %}

{% set _non_synaptic = [] %}
{% for var in variables %}
    {% if variable_indices[var] != '_idx' %}
        {# This is a trick to get around the scoping problem #}
        {% if _non_synaptic.append(1) %}{% endif %}
    {% endif %}
{% endfor %}

{% block kernel %}

__global__ void
{% if launch_bounds %}
__launch_bounds__(1024, {{sm_multiplier}})
{% endif %}
kernel_{{codeobj_name}}(
	unsigned int bid_offset,
	unsigned int timestep,
	unsigned int THREADS_PER_BLOCK,
	int32_t* eventspace,
	unsigned int neurongroup_size,
	%DEVICE_PARAMETERS%
	)
{
	{# USES_VARIABLES { N, _synaptic_pre} #}
	using namespace brian;

	assert(THREADS_PER_BLOCK == blockDim.x);

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x + bid_offset;
	//TODO: do we need _idx here? if now, get also rid of scoping after scalar code
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%
	{% block additional_variables %}
	{% endblock %}

	{{scalar_code|autoindent}}

	{
	if ({{pathway.name}}.no_or_const_delay_mode)
	{
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

	                                        {{vector_code|autoindent}}
	                                }
	                        }

	                        __syncthreads();
	                }
	        }
	}
	else  // heterogeneous delay mode
	{
	        cudaVector<int32_t>* synapses_queue;
	        {{pathway.name}}.queue->peek(
	                &synapses_queue);

		int size = synapses_queue[bid].size();
		for(int j = tid; j < size; j+=THREADS_PER_BLOCK)
		{
			int32_t _idx = synapses_queue[bid].at(j);
	
			{{vector_code|autoindent}}
		}
	}
	}
}

{% endblock %}

{% block extra_maincode %}
static unsigned int num_loops;
{% endblock %}

{% block prepare_kernel_inner %}
{% if synaptic_effects == "synapse" %}
// Synaptic effects modify only synapse variables.
num_blocks = num_parallel_blocks;
num_threads = max_threads_per_block;
num_loops = 1;
{% elif synaptic_effects == "target" %}
// Synaptic effects modify target group variables but NO source group variables.
num_blocks = num_parallel_blocks;
num_threads = 1;
num_loops = 1;
if ({{pathway.name}}_scalar_delay && !{{owner.name}}_multiple_pre_post)
{
	num_threads = max_threads_per_block;
}
{% elif synaptic_effects == "source" %}
// Synaptic effects modify source group variables.
num_blocks = 1;
num_threads = 1;
num_loops = num_parallel_blocks;
{% endif %}
{% endblock prepare_kernel_inner %}

{% block kernel_call %}
{% set eventspace_variable = pathway.variables[pathway.eventspace_name] %}
{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}
for(unsigned int bid_offset = 0; bid_offset < num_loops; bid_offset++)
{
	kernel_{{codeobj_name}}<<<num_blocks, num_threads>>>(
		bid_offset,
		{{owner.clock.name}}.timestep[0],
		num_threads,
		dev{{_eventspace}}[{{pathway.name}}_eventspace_idx],
		_num_{{_eventspace}}-1,
		%HOST_PARAMETERS%
	);
}

cudaError_t status = cudaGetLastError();
if (status != cudaSuccess)
{
	printf("ERROR launching kernel_{{codeobj_name}} in %s:%d %s\n",
			__FILE__, __LINE__, cudaGetErrorString(status));
	_dealloc_arrays();
	exit(status);
}
{% endblock kernel_call %}

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
