////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cu_file() %}
{#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <ctime>

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

	const std::clock_t _start_time = std::clock();

    ///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////

	_run_{{codeobj_name}}_advance_kernel<<<1, num_parallel_blocks>>>();

	{% if not no_or_const_delay_mode %}
	// We are copying next_delay_start_idx (size = num_unique_delays) into shared memory. Since num_unique_delays
	// varies for different combinations of pre neuron and bid, we allocate for max(num_unique_delays).
	// And +1 per block for copying size_before_resize into shared memory when we need to use the outer loop.
	unsigned int needed_shared_memory = ({{owner.name}}_max_unique_delay_size + 1) * sizeof(unsigned int);
	assert (needed_shared_memory <= max_shared_mem_size);

	// We don't need more then max(num_synapses) threads per block.
	unsigned int num_threads = {{owner.name}}_max_size;
	if (num_threads > max_threads_per_block)
	{
		num_threads = max_threads_per_block;
	}
	
	_run_{{codeobj_name}}_push_kernel<<<num_parallel_blocks, num_threads, needed_shared_memory>>>(
		_num{{eventspace_variable.name}}-1,
		num_parallel_blocks,
		num_threads,
		{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}
		dev{{_eventspace}});
	{% else %}
	//No pushing in no_or_const_delay_mode
	{% endif %}

	// Profiling
	const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
	{{codeobj_name}}_profiling_info += _run_time;
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
