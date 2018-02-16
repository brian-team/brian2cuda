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
class CudaSpikeQueue
{
private:
	// critical path coding, taken from
	// https://stackoverflow.com/questions/18963293/cuda-atomics-change-flag/18968893#18968893
	volatile int* semaphore;  // controll data access when reallocating
	__device__ void acquire_semaphore(volatile int *lock){
		while (atomicCAS((int *)lock, 0, 1) != 0);
	}

	__device__ void release_semaphore(volatile int *lock){
		*lock = 0;
		__threadfence();
	}

public:
	//these vectors should ALWAYS be the same size, since each index refers to a triple of (pre_id, syn_id, post_id)
	cudaVector<DTYPE_int>** synapses_queue;

	//our connectivity matrix with dimensions (num_blocks) * neuron_N
	//each element
	//unsigned int* size_by_pre;
	unsigned int* size_by_bundle_id;
	unsigned int* unique_delay_size_by_pre;
	//DTYPE_int** synapses_id_by_pre;
	DTYPE_int** synapses_id_by_bundle_id;
	unsigned int** unique_delay_by_pre;
	//unsigned int** unique_delay_start_idx_by_pre;

	unsigned int current_offset;
	unsigned int num_queues;
	unsigned int max_num_delays_per_block;
	unsigned int num_blocks;
	unsigned int neuron_N; // number of neurons in source of SynapticPathway
	unsigned int syn_N;

	// When we have 0 synapses, prepare() is not called in synapses_initialise_queue.cu
	// and for destroy() to still work, synapses_queue needs to be a null pointer
	__device__ CudaSpikeQueue(): synapses_queue(0) {};

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
		unsigned int _num_queues,
		unsigned int _max_num_delays_per_block,
		//unsigned int* _size_by_pre,
		unsigned int* _size_by_bundle_id,
		unsigned int* _unique_delay_size_by_pre,
		//DTYPE_int** _synapses_by_pre,
		DTYPE_int** _synapses_by_bundle_id,
		unsigned int** _unique_delay_by_pre
		//unsigned int** _unique_delay_start_idx_by_pre
		)
	{
		if(tid == 0)
		{
			// TODO add comments

			semaphore = new int[_num_blocks];
			current_offset = 0;
			num_blocks = _num_blocks;
			neuron_N = _neuron_N;
			syn_N = _syn_N;
			num_queues = _num_queues;
			max_num_delays_per_block = _max_num_delays_per_block;

			// TODO: do we need size_by_pre? is size_by_pre[right_offset] faster then synapses_by_pre[right_offset].size()?
			// if so, add unique_size_by_pre as well!
			//size_by_pre = _size_by_pre;
			size_by_bundle_id = _size_by_bundle_id;
			unique_delay_size_by_pre = _unique_delay_size_by_pre;
			//synapses_id_by_pre = _synapses_by_pre;
			synapses_id_by_bundle_id = _synapses_by_bundle_id;
			unique_delay_by_pre = _unique_delay_by_pre;
			//unique_delay_start_idx_by_pre = _unique_delay_start_idx_by_pre;

			synapses_queue = new cudaVector<DTYPE_int>*[num_queues];
			if(!synapses_queue)
			{
				printf("ERROR while allocating memory with size %ld in spikequeue.h/prepare()\n", sizeof(cudaVector<DTYPE_int>*)*num_queues);
			}
		}
		for (int i = tid; i < _num_blocks; i++)
		{
			semaphore[i] = 0;
		}
		__syncthreads();

        for(int i = tid; i < num_queues; i+=num_threads)
        {
	    	synapses_queue[i] = new cudaVector<DTYPE_int>[num_blocks];
    		if(!synapses_queue[i])
	    	{
    			printf("ERROR while allocating memory with size %ld in spikequeue.h/prepare()\n", sizeof(cudaVector<DTYPE_int>)*num_blocks);
    		}
        }
	};

	__device__ void push(
		unsigned int post_neuron_bid,
		unsigned int tid,
		unsigned int num_threads,
		unsigned int spiking_neuron_id)
	{

		// following arrays are in global device memory:
		//
		// 	synapses_id_by_pre
		//  (size == number of synapses)
		//
		//	unique_delays
		//	delay_start_idx
		//  (size == number of unique delays)

		assert(blockDim.x == num_threads);

		unsigned int right_offset = spiking_neuron_id * num_blocks + post_neuron_bid;
		// num_unique_delays == num_bundles
		unsigned int num_unique_delays = unique_delay_size_by_pre[right_offset];

		// spiking_neuron_id should be in range [0,neuron_N]
		assert(spiking_neuron_id < neuron_N);


		// ( thread <-> synapse_bundle ) correspondence
		// If num_threads < num_unique_delays, loop.
		// bundle_idx is bundle number per block (not global bundle ID!)
		for (unsigned int i = 0; i < num_unique_delays; i += num_threads)
		{
			// start loop at 0 to make sure all threads are executing the same number of loops (for __syncthread())
			unsigned int bundle_idx = i + tid;

			unsigned int global_bundle_id, delay_queue;
			if (bundle_idx < num_unique_delays)
			{
				// we have per right_offset (total of num_blocks * source_N) a
				// local bundle index going from 0 to max_num_delays_per_block for that
				// right_offset
				global_bundle_id = max_num_delays_per_block * right_offset + bundle_idx;

				unsigned int delay = unique_delay_by_pre[right_offset][bundle_idx];
				// find the spike queue corresponding to this synapses delay
				delay_queue = (current_offset + delay) % num_queues;
			}

			// make sure only one block resizes (and therefore possibly
			// reallocates) and fills this CudaSpikeQueues' CudaVectors.
			__syncthreads();
			if (tid == 0)
				acquire_semaphore(semaphore + post_neuron_bid);
			__syncthreads();

			if (bundle_idx < num_unique_delays)
			{
				// begin critical section
				synapses_queue[delay_queue][post_neuron_bid].push(global_bundle_id);
				// end critical section
			}

			__syncthreads();
			if (tid == 0)
				release_semaphore(semaphore + post_neuron_bid);
			__syncthreads();

		} // end for

	} // end push()

	__device__ void advance(
		unsigned int tid)
	{
		assert(tid < num_blocks && current_offset < num_queues);
		synapses_queue[current_offset][tid].reset();
		__syncthreads(); //TODO no need for this?...
		if(tid == 0)
			current_offset = (current_offset + 1)%num_queues;
	}

	__device__  void peek(
		cudaVector<DTYPE_int>** _synapses_queue)
	{
		*(_synapses_queue) =  &(synapses_queue[current_offset][0]);
	}
};

