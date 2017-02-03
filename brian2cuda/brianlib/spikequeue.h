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
	unsigned int neuron_N; // number of neurons in source of SynapticPathway
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

		// following arrays are in global device memory:
		//
		// 	synapses_id_by_pre
		//  (size == number of synapses)
		//
		//	unique_delays
		//	delay_start_idx
		//  (size == number of unique delays)

		assert(blockDim.x == num_threads);

		// TODO: why not use _pre_id directly?
		unsigned int neuron_pre_id = _pre_id; // index in sources SynapticPathway ([0, neuron_N])
		unsigned int right_offset = neuron_pre_id * num_blocks + bid;
		// TODO: use size_by_pre and unique_size if we keep it
		unsigned int num_synapses = size_by_pre[right_offset];
		unsigned int num_unique_delays = unique_delay_size_by_pre[right_offset];
		// shared_mem is allocated in push_spikes
		unsigned int* shared_mem_unique_delay_start_idx_by_pre = (unsigned int*)_shared_mem;

		// neuron_pre_id should be in range [0,neuron_N]
		assert(neuron_pre_id < neuron_N);

		// Copy to shared memory. If more entries then threads, loop.
		// TODO: is it possible to know num_unique_delays beforehand to avoid allocation of unnecessary shared mem?
		// since right now there is sizeof(unsigned int) allocated for each thread, not for num_unique_delays
		for (unsigned int i = tid; i < num_unique_delays; i += num_threads)
		{
			shared_mem_unique_delay_start_idx_by_pre[i] = unique_delay_start_idx_by_pre[right_offset][i];
		}
		__syncthreads();


		// ( thread <-> synapse ) correspondence 
		// If num_threads < num_synapses, loop.
		// syn is synapse number (not ID!)
		for (unsigned int i = 0; i < num_synapses; i += num_threads)
		{
			// start loop at 0 to make sure all threads are executing the same number of loops (for __syncthread())
			unsigned int syn = i + tid;
			// declare variables which we will need after __syncthread() call
			unsigned int delay_queue, size_before_resize, delay_start_idx_in_synapses_id;


			// TODO: use global size_by_pre only in host memory and call push() with max num_synapses threads
			if (syn < num_synapses)
			{
				// find the starting index in synapse_id_by_pre for the delay corresponding
				// to the current synapse and the starting index for the next delay
				unsigned int next_delay_start_idx_in_synapses_id = 0;
				// delay_start_idx_in_synapses_id is declared outside if {...}
				unsigned int idx_in_unique_delays;
				for (unsigned int j = 1; j < num_unique_delays; j++)
				{
					delay_start_idx_in_synapses_id = next_delay_start_idx_in_synapses_id;
					next_delay_start_idx_in_synapses_id = shared_mem_unique_delay_start_idx_by_pre[j];
					if (next_delay_start_idx_in_synapses_id > syn)
					{
						idx_in_unique_delays = j-1;
						break;
					}
					if (j == num_unique_delays - 1) // end of loop
					{
						// this synapse has the highest delay for the current pre_neuron and post_neuron_block
						delay_start_idx_in_synapses_id = next_delay_start_idx_in_synapses_id;
						idx_in_unique_delays = j;
						// there is no next delay, for the calculation of delay_occurence we need
						next_delay_start_idx_in_synapses_id = num_synapses;
					}
				}

				// TODO: remove this if statement once we have no_or_const_delay_mode implementation and add
				// assert(num_unique_delays > 1)
				// otherwise aboves loop is not entered and results in wrong delay_start_idx values
				if (num_unique_delays == 1)
				{
					delay_start_idx_in_synapses_id = 0;
					next_delay_start_idx_in_synapses_id = num_synapses;
					idx_in_unique_delays = 0;
				}

				assert(delay_start_idx_in_synapses_id <= syn && syn < next_delay_start_idx_in_synapses_id);

				// get the delay of the current synapse and the number of synapses with that delay
				// TODO: is it faster to once make a coalesced copy of unique_delay_by_pre to shared memory? try!
				unsigned int delay = unique_delay_by_pre[right_offset][idx_in_unique_delays];
				unsigned int delay_occurrence = next_delay_start_idx_in_synapses_id - delay_start_idx_in_synapses_id;

				// find the spike queue corresponding to this synapses delay
				delay_queue = (current_offset + delay) % max_delay;

				// uncoalseced memory access, TODO: use pointers to consecutive memory locations for cudaVector::m_size
				// currently multiple consecutive threads read same global memory address,
				// then next consecutive threads read next global memory address
				// TODO check memory broadcasting mechanism
				// 		maybe copy size_before_resize into shared memory when copying the unique delay start idx
				size_before_resize = synapses_queue[delay_queue][bid].size();

				// RESIZE QUEUES
				// TODO: if we use pointers for cudaVector::m_size, consecutive threads should to the resize
				// in order to get coalesced memory access, e.g. by letting the threads that copy the start_idx
				// to shared memory then perform aboves code until resize and then let all threads do it again
				// for their respective syn number
				// -> we get coalesced memory access but have to do more shared mem reads and numerics
				if (syn == delay_start_idx_in_synapses_id)  // only one thread for each unique delay
				{
					synapses_queue[delay_queue][bid].resize(size_before_resize + delay_occurrence);
				}
			} // end if

			// make sure all queues are resized before actually pushing
			__syncthreads();

			if (syn < num_synapses)
			{
				unsigned int size_before_push = synapses_queue[delay_queue][bid].size();

				// PUSH INTO QUEUES
				unsigned int syn_id = synapses_id_by_pre[right_offset][syn];
				// find position in queue for syn
				unsigned int idx_in_queue = size_before_resize + (syn - delay_start_idx_in_synapses_id);
				// each thread updates one value in queue
				synapses_queue[delay_queue][bid].at(idx_in_queue) = syn_id;
			} // end if

		} // end for

	} // end push()

	__device__ void advance(
		unsigned int tid)
	{
		assert(tid < num_blocks && current_offset < max_delay);
		synapses_queue[current_offset][tid].reset();
		__syncthreads();
		if(tid == 0)
			current_offset = (current_offset + 1)%max_delay;
	}

	__device__  void peek(
		cudaVector<DTYPE_int>** _synapses_queue)
	{
		*(_synapses_queue) =  &(synapses_queue[current_offset][0]);
	}
};

