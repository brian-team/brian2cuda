#include "objects.h"

#include "code_objects/synapses_post_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

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

#define MEM_PER_THREAD (sizeof(int32_t) + sizeof(unsigned int))

__global__ void _run_synapses_post_push_spikes_advance_kernel()
{
	using namespace brian;
	unsigned int tid = threadIdx.x;
	synapses_post.queue->advance(
		tid);
}

__global__ void _run_synapses_post_push_spikes_push_kernel(
	unsigned int sourceN,
	unsigned int _num_blocks,
	unsigned int _num_threads,
	unsigned int block_size,
	int32_t* _ptr_array_neurongroup__spikespace)
{
	using namespace brian;

	extern __shared__ char shared_mem[];
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	//REMINDER: spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
	block_size = ((sourceN + _num_blocks - 1) / _num_blocks);
	unsigned int start_index = synapses_post.spikes_start - (synapses_post.spikes_start % block_size);	//find start of last block
	
	char no_delay_mode = false;
	for(int i = 0; i < synapses_post.spikes_stop; i++)
	{
		int32_t spiking_neuron = _ptr_array_neurongroup__spikespace[i];
		if(spiking_neuron != -1 && spiking_neuron >= synapses_post.spikes_start && spiking_neuron < synapses_post.spikes_stop)
		{
			__syncthreads();
			synapses_post.queue->push(
				bid,
				tid,
				_num_threads,
				spiking_neuron,
				shared_mem,
				no_delay_mode);
		}
		if(spiking_neuron == -1)
		{
			return;
		}
	}
}

void _run_synapses_post_push_spikes()
{
	using namespace brian;
    ///// CONSTANTS ///////////
	const int _num_spikespace = 101;
	///// POINTERS ////////////

	_run_synapses_post_push_spikes_advance_kernel<<<1, num_parallel_blocks>>>();
	unsigned int num_threads = max_shared_mem_size / MEM_PER_THREAD;
	num_threads = num_threads < max_threads_per_block? num_threads : max_threads_per_block; // get min of both
	
	
		_run_synapses_post_push_spikes_push_kernel<<<num_parallel_blocks, num_threads, num_threads*MEM_PER_THREAD>>>(
			_num_spikespace - 1,
			num_parallel_blocks,
			num_threads,
			_num_threads(_num_spikespace - 1),
			dev_array_neurongroup__spikespace);
}
