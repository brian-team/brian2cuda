#include<iostream>
#include<vector>
#include<algorithm>
#include<inttypes.h>

#include "cudaVector.h"
#include <assert.h>

#include <cstdio>

using namespace std;

//TODO: The data type for indices is currently fixed (int), all floating point
//      variables (delays, dt) are assumed to use the same data type
typedef int32_t DTYPE_int;

template <class scalar>
class CSpikeQueue
{
public:
	//these vectors should ALWAYS be the same size, since each index refers to a triple of (pre_id, syn_id, post_id)
	cudaVector<DTYPE_int>** synapses_queue;

	//our connectivity matrix with dimensions (num_blocks) * neuron_N
	//each element
	unsigned int* size_by_pre;
	unsigned int* unique_delay_size_by_pre;
	DTYPE_int** synapses_id_by_pre;
	unsigned int** delay_by_pre;
	unsigned int** unique_delay_by_pre;
	unsigned int** unique_delay_start_idx_by_pre;

	unsigned int current_offset;
	unsigned int max_delay;
	unsigned int num_blocks;
	unsigned int neuron_N;
	unsigned int syn_N;

	//Since we can't have a destructor, we need to call this function manually
	__device__ void destroy()
	{
		if(synapses_queue)
		{
			delete [] synapses_queue;
			synapses_queue = 0;
		}
	}
	
	/* this function also initiliases all variables, allocs arrays, etc.
	 * so we need to call it before using the queue
	 */
	__device__ void prepare(
		int tid,
		int num_threads,
		unsigned int _num_blocks,
		scalar _dt,
		unsigned int _neuron_N,
		unsigned int _syn_N,
		unsigned int _max_delay,
		unsigned int* _size_by_pre,
		unsigned int* _unique_delay_size_by_pre,
		DTYPE_int** _synapses_by_pre,
		unsigned int** _delay_by_pre,
		unsigned int** _unique_delay_by_pre,
		unsigned int** _unique_delay_start_idx_by_pre
		)
	{
		if(tid == 0)
		{
			// TODO add comments

			current_offset = 0;
			num_blocks = _num_blocks;
			neuron_N = _neuron_N;
			syn_N = _syn_N;
			max_delay = _max_delay;

			// TODO: do we need size_by_pre? is size_by_pre[right_offset] faster then synapses_by_pre[right_offset].size()?
			// if so, add unique_size_by_pre as well!
			size_by_pre = _size_by_pre;
			unique_delay_size_by_pre = _unique_delay_size_by_pre;
			synapses_id_by_pre = _synapses_by_pre;
			delay_by_pre = _delay_by_pre;
			unique_delay_by_pre = _unique_delay_by_pre;
			unique_delay_start_idx_by_pre = _unique_delay_start_idx_by_pre;

			synapses_queue = new cudaVector<DTYPE_int>*[max_delay];
			if(!synapses_queue)
			{
				printf("ERROR while allocating memory with size %ld in spikequeue.h/prepare()\n", sizeof(cudaVector<DTYPE_int>*)*max_delay);
			}
		}
		__syncthreads();

        for(int i = tid; i < max_delay; i+=num_threads)
        {
	    	synapses_queue[i] = new cudaVector<DTYPE_int>[num_blocks];
    		if(!synapses_queue[i])
	    	{
    			printf("ERROR while allocating memory with size %ld in spikequeue.h/prepare()\n", sizeof(cudaVector<DTYPE_int>)*num_blocks);
    		}
        }
	};

	__device__ void push(
		unsigned int bid,
		unsigned int tid,
		unsigned int num_threads,
		unsigned int _pre_id,
		char* _shared_mem,
		char no_delay_mode)
	{

		// call with max_num_threads

		// following arrays are in global device memory:
		//
		//  with size == number of synapses:
		// 	synapses_id_by_pre
		//
		//  with size == number of different delays:
		//	unique_delays
		//	delay_start_idx


		assert(blockDim.x == num_threads);

		// TODO: why not use _pre_id directly?
		unsigned int neuron_pre_id = _pre_id;
		unsigned int right_offset = neuron_pre_id * num_blocks + bid;
		// TODO: use size_by_pre and unique_size if we keep it
		unsigned int num_synapses = size_by_pre[right_offset];
		unsigned int num_unique_delays = unique_delay_size_by_pre[right_offset];
		// shared_mem is allocated in push_spikes
		unsigned int* shared_mem_unique_delay_start_idx_by_pre = (unsigned int*)_shared_mem;

		// previous code hat
		// if (neuron_pre_id >= neuron_N) return;
		// but why should that ever happen? assert to find out
		assert(neuron_pre_id < neuron_N);

		// Copy to shared memory. If more entries then threads, loop.
		shared_mem_unique_delay_start_idx_by_pre[tid] = unique_delay_start_idx_by_pre[right_offset][tid];
//		TODO:
//		for (int i = tid; i < num_unique_delays; i += num_threads)
//		{
//			shared_mem_unique_delay_start_idx_by_pre[i] = unique_delay_start_idx_by_pre[right_offset][i];
//		}
		__syncthreads();


		// ( tid <-> syn_ID ) correspondence
		// If num_threads < num_synapses, loop.
// 		TODO: for (int i = tid; i < num_synapses; i += num_threads)

		// find the starting index (current) in synapse_id_by_pre for the delay corresponding to the current thread
		unsigned int next_delay_start_idx_in_synapses_id = 0;
		unsigned int delay_start_idx_in_synapses_id, idx_in_unique_delays;
		for (int j; j < num_unique_delays; j++)
		{
			delay_start_idx_in_synapses_id = next_delay_start_idx_in_synapses_id;
			next_delay_start_idx_in_synapses_id = shared_mem_unique_delay_start_idx_by_pre[j];
			if (next_delay_start_idx_in_synapses_id > tid)
			{
				idx_in_unique_delays = j;
				break;
			}
		}

		// get the delay and the number of synapses with that delay
		// TODO: is it faster to once make a coalesced copy of unique_delay_by_pre to shared memory? try!
		unsigned int delay = unique_delay_by_pre[right_offset][idx_in_unique_delays];
		unsigned int delay_occurrence = next_delay_start_idx_in_synapses_id - delay_start_idx_in_synapses_id;

		// find spike queue corresponding to delay
		unsigned int delay_queue = (current_offset + delay) % max_delay;

		// uncoalseced memory access, TODO: use pointer for cudaVector.size
		// currently multiple consecutive threads read same global memory address,
		// then next consecutive threads read next global memory address
		unsigned int size_before_resize = synapses_queue[delay_queue][bid].size();

		// RESIZE QUEUES
		//if ( i < num_unique_delays )  // only one thread for each unique delay
		if ( tid == delay_start_idx_in_synapses_id )  // only one thread for each unique delay
		{
			synapses_queue[delay_queue][bid].resize(size_before_resize + delay_occurrence);
		}

		// make sure all queues are resized before actually pushing
		__syncthreads();

		// PUSH INTO QUEUES

		// ( tid <-> synID ) correspondence.
		// If num_threads < num_synapses, loop.
//		TODO: for (int i = tid; i < num_unique_delays; i += num_threads)
		unsigned int syn_id = synapses_id_by_pre[right_offset][tid];

		// find position in queue for tid
		unsigned int idx_in_queue = size_before_resize + (tid - delay_start_idx_in_synapses_id);
		// each thread updates one value in queue
		synapses_queue[delay_queue][bid].update(idx_in_queue, syn_id);

	} // end push()


//		/////////////////////////////////////////////////////////////////////////////
//		PREVIOUS VERSION FROM KONRAD
//
//		assert(blockDim.x == num_threads);
//
//		unsigned int neuron_pre_id = _pre_id;
//		unsigned int right_offset = neuron_pre_id*num_blocks + bid;
//		unsigned int num_connected_synapses = size_by_pre[right_offset];
//		//shared_mem is allocated in push_spikes
//		int32_t* shared_mem_synapses_id = (int32_t*)_shared_mem;
//		unsigned int* shared_mem_synapses_delay = (unsigned int*)((int32_t*)shared_mem_synapses_id + num_threads);
//
//		//ignore invalid pre_ids
//		if(neuron_pre_id >= neuron_N)
//		{
//			return;
//		}
//
//		for(int i = tid; i < num_connected_synapses; i += num_threads)
//		{
//			if(!no_delay_mode)
//			{
//				int32_t syn_id = synapses_id_by_pre[right_offset][i];
//				shared_mem_synapses_id[tid] = syn_id;
//				unsigned int delay = delay_by_pre[right_offset][i];
//				shared_mem_synapses_delay[tid] = delay;
//
//				for(int delay_id = tid; delay_id < max_delay; delay_id += num_threads)
//				{
//					for(int j = 0; j < num_threads && i + j < num_connected_synapses; j++)
//					{
//						int32_t queue_syn_id = shared_mem_synapses_id[j];
//						unsigned int queue_delay = shared_mem_synapses_delay[j];
//						if(queue_delay != delay_id)
//						{
//							continue;
//						}
//						unsigned int adjusted_delay = (current_offset + queue_delay)%max_delay;
//						unsigned int queue_id = bid;
//
//						synapses_queue[adjusted_delay][queue_id].push(queue_syn_id);
//					}
//				}
//				__syncthreads();
//			}
//			else
//			{
//				unsigned int queue_delay = max_delay - 1;
//				unsigned int adjusted_delay = (current_offset + queue_delay)%max_delay;
//				unsigned int queue_id = bid;
//
//				synapses_queue[adjusted_delay][queue_id].push(_pre_id);
//			}
//		}

	__device__ void advance(
		unsigned int tid)
	{
		if(tid >= num_blocks || current_offset >= max_delay)
		{
			return;
		}
		synapses_queue[current_offset][tid].reset();
		__syncthreads();
		if(tid == 0)
		{
			current_offset = (current_offset + 1)%max_delay;
		}
	}

	__device__  void peek(
		cudaVector<DTYPE_int>** _synapses_queue)
	{
		*(_synapses_queue) =  &(synapses_queue[current_offset][0]);
	}
};

