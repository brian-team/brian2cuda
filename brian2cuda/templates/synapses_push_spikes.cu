////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cu_file() %}
{# USES_VARIABLES { _spikespace } #}

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
	unsigned int sourceN,
	unsigned int _num_blocks,
	unsigned int _num_threads,
	unsigned int block_size,
	int32_t* {{_spikespace}})
{
	// apperently this is not always true and that is why _num_threads is passed as function argument
	// if this assert never fails, we could remove the _num_threads form the argument list
	assert(blockDim.x == _num_threads);

	using namespace brian;

	// TODO: check if static shared memory is faster / makes any difference 
	extern __shared__ char shared_mem[];
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	// TODO: what is this for? it's not used, delete if not needed
	//REMINDER: spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
	block_size = ((sourceN + _num_blocks - 1) / _num_blocks);
	unsigned int start_index = {{owner.name}}.spikes_start - ({{owner.name}}.spikes_start % block_size);	//find start of last block
	
	// TODO: no delay mode is hard coded here!
	char no_delay_mode = false;

	// TODO: shouldn't this loop start at {{owner.name}}.spikes_start?
	// loop through spikespace until first -1 (end of spiking neurons)
	for(int i = 0; i < {{owner.name}}.spikes_stop; i++)
	{
		int32_t spiking_neuron = {{_spikespace}}[i];
		if(spiking_neuron != -1 && spiking_neuron >= {{owner.name}}.spikes_start && spiking_neuron < {{owner.name}}.spikes_stop)
		{
			__syncthreads();
			{{owner.name}}.queue->push(
				bid,
				tid,
				_num_threads,
				spiking_neuron,
				shared_mem,
				no_delay_mode);
		}
		if(spiking_neuron == -1)
		{
			assert(i == {{_spikespace}}[sourceN]);
			return;
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
			_num_spikespace - 1,
			num_parallel_blocks,
			num_threads,
			_num_threads(_num_spikespace - 1),
			{% set _spikespace = get_array_name(owner.variables['_spikespace'], access_data=False) %}
			dev{{_spikespace}});
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
