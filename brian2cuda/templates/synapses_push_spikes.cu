////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cu_file() %}
{#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

#include "code_objects/{{codeobj_name}}.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"
#include <assert.h>

namespace {
	int _num_blocks(int num_objects)
    {
		static int needed_num_block = -1;
	    if(needed_num_block == -1)
		{
			needed_num_block = brian::num_parallel_blocks;
			while(needed_num_block * brian::max_threads_per_block < num_objects)
			{
				needed_num_block *= 2;
			}
		}
		return needed_num_block;
    }

	int _num_threads(int num_objects)
    {
		static int needed_num_threads = -1;
		if(needed_num_threads == -1)
		{
			int needed_num_block = _num_blocks(num_objects);
			needed_num_threads = min(brian::max_threads_per_block, (int)ceil(num_objects/(double)needed_num_block));
		}
		return needed_num_threads;
	}
}

// TODO: when push is done, uncomment
//#define MEM_PER_THREAD (sizeof(unsigned int))  // each thread copies one unique_delay_start_idx into shared memory
#define MEM_PER_THREAD (sizeof(unsigned int) + sizeof(int32_t))

__global__ void _run_{{codeobj_name}}_advance_kernel()
{
	using namespace brian;
	unsigned int tid = threadIdx.x;
	{% if no_or_const_delay_mode %}
	{{owner.name}}.which_spikespace = ({{owner.name}}.which_spikespace + 1) % {{owner.name}}.queue->max_delay;
	{% else %}
	{{owner.name}}.queue->advance(
		tid);
	{% endif %}
}

__global__ void _run_{{codeobj_name}}_push_kernel(
	unsigned int neurongroup_size,
	unsigned int _num_blocks,
	unsigned int _num_threads,
	int32_t* {{_eventspace}})
{
	// apperently this is not always true and that is why _num_threads is passed as function argument
	// if this assert never fails, we could remove the _num_threads form the argument list
	assert(blockDim.x == _num_threads);

	using namespace brian;

	// TODO: check if static shared memory is faster / makes any difference 
	extern __shared__ char shared_mem[];
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	// TODO: no delay mode is hard coded here!
	char no_delay_mode = false;

	// loop through spiking neurons in spikespace (indices of spiking neurons, rest -1)
	for(int i = 0; i < neurongroup_size; i++)
	{
		// spiking_neuron is index in NeuronGroup
		int32_t spiking_neuron = {{_eventspace}}[i];

		if(spiking_neuron == -1) // end of spiking neurons
		{
			assert(i == {{_eventspace}}[neurongroup_size]);
			return;
		}
		// push to spikequeue if spiking_neuron is in sources of current SynapticPathway
		if({{owner.name}}.spikes_start <= spiking_neuron && spiking_neuron < {{owner.name}}.spikes_stop)
		{
			__syncthreads();
			{{owner.name}}.queue->push(
				bid,
				tid,
				_num_threads,
				spiking_neuron - {{owner.name}}.spikes_start,
				shared_mem,
				no_delay_mode);
		}
	}
}

void _run_{{codeobj_name}}()
{
	using namespace brian;
    ///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////

	_run_{{codeobj_name}}_advance_kernel<<<1, num_parallel_blocks>>>();
	// TODO: since we are only copying arays of size = number of unique delays, this needs to be adjusted
	// here we are expecting MEM_PER_THREAD for each thread, but we only need it for max(num_unique_delays) threads
	unsigned int num_threads = max_shared_mem_size / MEM_PER_THREAD;

	if (num_threads > max_threads_per_block)
	{
		num_threads = max_threads_per_block;
	}
	
	{% if not no_or_const_delay_mode %}	
		// TODO: check performance decrease when spawning many unused threads! Maybe call kernel with max(num_synapses) threads?
		_run_{{codeobj_name}}_push_kernel<<<num_parallel_blocks, num_threads, num_threads*MEM_PER_THREAD>>>(
			//TODO: does this have to be _num{{eventspace_variable.name}}-1
			_num_spikespace - 1,
			num_parallel_blocks,
			num_threads,
			//TODO: does this have to be eventspace_variable instead of owner.variables['_eventspace'] ?
    			{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}
			dev{{_eventspace}});
	{% else %}
	//No pushing in no_or_const_delay_mode
	{% endif %}
}
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// HEADER FILE ///////////////////////////////////////////////////////////

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "objects.h"

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
