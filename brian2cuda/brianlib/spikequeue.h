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
    unsigned int* num_synapses_by_pre;
    unsigned int* num_synapses_by_bundle;
    unsigned int* num_unique_delays_by_pre;
    unsigned int* delay_by_bundle;
    unsigned int* global_bundle_id_start_by_pre;
    unsigned int* synapses_offset_by_bundle;
    DTYPE_int* synapse_ids;
    DTYPE_int** synapse_ids_by_pre;
    unsigned int** unique_delays_by_pre;
    unsigned int** unique_delay_start_idcs_by_pre;

    unsigned int current_offset;
    unsigned int num_queues;
    //unsigned int max_num_delays_per_block;
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
        unsigned int* _num_synapses_by_pre,
        unsigned int* _num_synapses_by_bundle,
        unsigned int* _num_unique_delays_by_pre,
        unsigned int* _delay_by_bundle,
        unsigned int* _global_bundle_id_start_by_pre,
        unsigned int* _synapses_offset_by_bundle,
        DTYPE_int* _synapse_ids,
        DTYPE_int** _synapse_ids_by_pre,
        unsigned int** _unique_delays_by_pre,
        unsigned int** _unique_delay_start_idcs_by_pre
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

            // TODO: do we need num_synapses_by_pre? is num_synapses_by_pre[pre_post_block_id] faster then synapses_by_pre[pre_post_block_id].size()?
            // if so, add unique_num_synapses_by_pre as well!
            num_synapses_by_pre = _num_synapses_by_pre;
            num_synapses_by_bundle = _num_synapses_by_bundle;
            num_unique_delays_by_pre = _num_unique_delays_by_pre;
            delay_by_bundle = _delay_by_bundle;
            global_bundle_id_start_by_pre = _global_bundle_id_start_by_pre;
            synapses_offset_by_bundle = _synapses_offset_by_bundle;
            synapse_ids = _synapse_ids;
            synapse_ids_by_pre = _synapse_ids_by_pre;
            unique_delays_by_pre = _unique_delays_by_pre;
            unique_delay_start_idcs_by_pre = _unique_delay_start_idcs_by_pre;

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

    __device__ void push_synapses(
        char* _shared_mem,
        unsigned int post_neuron_bid,
        unsigned int tid,
        unsigned int num_threads,
        unsigned int spiking_neuron_id)
    {

        // following arrays are in global device memory:
        //
        //  synapse_ids_by_pre
        //  (size == number of synapses)
        //
        //  unique_delays
        //  delay_start_idx
        //  (size == number of unique delays)

        assert(blockDim.x == num_threads);

        unsigned int pre_post_block_id = spiking_neuron_id * num_blocks + post_neuron_bid;
        unsigned int num_synapses = num_synapses_by_pre[pre_post_block_id];
        unsigned int num_unique_delays = num_unique_delays_by_pre[pre_post_block_id];
        // shared_mem is allocated in push_spikes
        unsigned int* shared_mem_unique_delay_start_idcs_by_pre = (unsigned int*)_shared_mem;
        // shared memory for inter thread communication needs to be volatile
        volatile unsigned int* shared_mem_size_before_resize = shared_mem_unique_delay_start_idcs_by_pre + num_unique_delays;
        volatile unsigned int* shared_mem_last_cycle_size_before_resize = shared_mem_size_before_resize + num_unique_delays;

        // spiking_neuron_id should be in range [0,neuron_N]
        assert(spiking_neuron_id < neuron_N);

        // Copy to shared memory. If more entries then threads, loop.
        for (unsigned int i = tid; i < num_unique_delays; i += num_threads)
        {
            shared_mem_unique_delay_start_idcs_by_pre[i] = unique_delay_start_idcs_by_pre[pre_post_block_id][i];
        }
        __syncthreads();

        // ( thread <-> synapse ) correspondence
        // If num_threads < num_synapses, loop.
        // syn is synapse number (not ID!)
        unsigned int delay_previous_loop_cycle, size_before_resize;
        for (unsigned int i = 0; i < num_synapses; i += num_threads)
        {
            ///////////////////////////////////////////////////////////////////////////////////////
            // Example values and code paths for each thread for given delays, num_threads=3, num_synapses=12:
            //
            // syn (range(0,num_synapses), not ID!) 0  1  2 | 3  4  5 | 6  7  8 | 9 10 11
            // delay                                0  0  0 | 0  0  0 | 0  1  1 | 1  2  2
            //
            // tid                                  0  1  2 | 0  1  2 | 0  1  2 | 0  1  2
            // i                                    0  0  0 | 3  3  3 | 6  6  6 | 9  9  9
            // loop cycle (i/num_threads)           0  0  0 | 1  1  1 | 2  2  2 | 3  3  3
            // delay_start_idx_in_synapses_id       0  0  0 | 0  0  0 | 0  7  7 | 7 10 10
            // next_delay_start_idx_in_synapses_id  7  7  7 | 7  7  7 | 7 10 10 |10 12 12
            // delay_occurrence                     7  7  7 | 7  7  7 | 7  3  3 | 3  2  2
            //
            // Different code paths (see below):
            //                                      ^  ^  ^                ^  ^   ^  ^  ^
            //                                                                    a
            //                                      *                      *         *
            //                                                             b
            ///////////////////////////////////////////////////////////////////////////////////////

            // start loop at 0 to make sure all threads are executing the same number of loops (for __syncthread())
            unsigned int syn = i + tid;
            // declare variables which we will need after __syncthread() call
            unsigned int delay_queue, delay_start_idx_in_synapses_id, delay, delay_occurrence, idx_in_unique_delays;
            unsigned int next_delay_start_idx_in_synapses_id = 0;

            if (syn < num_synapses)
            {
                // find the starting index in synapse_id_by_pre for the delay corresponding
                // to the current synapse and the starting index for the next delay
                for (unsigned int j = 1; j < num_unique_delays; j++)
                {
                    delay_start_idx_in_synapses_id = next_delay_start_idx_in_synapses_id;
                    next_delay_start_idx_in_synapses_id = shared_mem_unique_delay_start_idcs_by_pre[j];
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
                        // there is no next delay, for the calculation of delay_occurrence we need
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
                // TODO: is it faster to once make a coalesced copy of unique_delays_by_pre to shared memory? try!
                delay = unique_delays_by_pre[pre_post_block_id][idx_in_unique_delays];
                delay_occurrence = next_delay_start_idx_in_synapses_id - delay_start_idx_in_synapses_id;

                // find the spike queue corresponding to this synapses delay
                delay_queue = (current_offset + delay) % num_queues;
            }

            // make sure only one block resizes (and therefore possibly
            // reallocates) and fills this CudaSpikeQueues' CudaVectors.
            __syncthreads();
            if (tid == 0)
                acquire_semaphore(semaphore + post_neuron_bid);
            __syncthreads();

            // begin critical section

            // Check if there will be at least one resize this loop cycle.
            // We always resize in the first cycle (i == 0).
            // We have NO resize this loop cycle only if delay_start_idx was in
            // the last loop cycle (< i) and next_delay_start_idx is in the
            // next cycle loop cycle (>= i + num_threads).
            if (i == 0 ||
                    !(delay_start_idx_in_synapses_id < i &&
                        i + num_threads <= next_delay_start_idx_in_synapses_id))
            {
                if (syn < num_synapses)  // for __synchtheads()
                {
                    // RESIZE QUEUES
                    // TODO: if we use pointers for cudaVector::m_size, consecutive threads should to the resize
                    // in order to get coalesced memory access, e.g. by letting the threads that copy the start_idx
                    // to shared memory then perform aboves code until resize and then let all threads do it again
                    // for their respective syn number
                    // -> we get coalesced memory access but have to do more shared mem reads and numerics
                    if (syn == delay_start_idx_in_synapses_id)  // only one thread for each unique delay
                    {
                        // only the first thread for each delay does the resizing, in example marked as (*)
                        // and copies its offset from atomic resizing into shared memory for broadcasting to
                        // the other threads with same delay
                        shared_mem_size_before_resize[idx_in_unique_delays] = synapses_queue[delay_queue][post_neuron_bid].increaseSizeBy(
                                delay_occurrence);
                    }
                }
                // make sure size_before_resize is written to shared mem before read by other threads
                __syncthreads();
            }

            if (syn < num_synapses)
            {
                // uncoalseced memory access, TODO: use pointers to consecutive memory locations for cudaVector::m_size
                // currently multiple consecutive threads read same global memory address,
                // then next consecutive threads read next global memory address
                // TODO check memory broadcasting mechanism
                //      maybe copy size_before_resize into shared memory when copying the unique delay start idx
                if (i == 0 || delay != delay_previous_loop_cycle)
                {
                    // only update size_before_resize if we are pushing into a new delay_queue (for the same tid)
                    // or if we are in the 1. loop cycle (i/num_threads==0)
                    // in example marked as (^)
                    if (delay_start_idx_in_synapses_id < i)
                    {
                        // if in the previous loop cycle we were not done with that delay, then the delay_queue is
                        // already resized and we need to take the size_before_resize we saved to shared memory
                        // in example marked as (a)
                        // TODO just access the shared mem from the last delay of last cycle?
                        size_before_resize = shared_mem_last_cycle_size_before_resize[0];
                    }
                    else
                    {
                        // the size_before_resize for this delay has not been broadcasted yet
                        // in example marked all (^), except of (a)
                        size_before_resize = shared_mem_size_before_resize[idx_in_unique_delays];
                    }
                }
                delay_previous_loop_cycle = delay;

                // PUSH INTO QUEUES
                unsigned int syn_id = synapse_ids_by_pre[pre_post_block_id][syn];
                // find position in queue for syn
                unsigned int idx_in_queue = size_before_resize + (syn - delay_start_idx_in_synapses_id);
                // each thread updates one value in queue
                synapses_queue[delay_queue][post_neuron_bid].at(idx_in_queue) = syn_id;

            } // end if
            //end critical section

            __syncthreads();
            if (tid == 0)
                release_semaphore(semaphore + post_neuron_bid);
            __syncthreads();

            if (syn < num_synapses)
            {

                // TODO: we could do this block right after resizing and save
                // the __syncthread below, but would spend more time in the
                // critical section, possibly blocking entire blocks
                // --> needs benchmarking
                if (syn == delay_start_idx_in_synapses_id && (num_threads - tid) < delay_occurrence && tid != 0)
                {
                    // If pushing into this delay queue will not be finished within this loop cycle,
                    // then in the next loop cycle the queue will already be resized and we won't
                    // have access to size_before_resize. --> save it to shared memory
                    // If tid==0, all threads in this loop cycle will push into the same delay queue
                    // and the size_before_resize will be unchanged next loop cycle if the delay didn't change.
                    // in example marked as (b)
                    shared_mem_last_cycle_size_before_resize[0] = size_before_resize;

                    if (num_synapses - i > num_threads)  // true if there is another loop cycle needed
                    {
                        // make sure shared_mem_last_cycle_size_before_resize is written for the next loop cycle
                        __syncthreads();
                    }
                }
            } // end if
        } // end for
    } // end push_synapses()

    __device__ void push_bundles(
        unsigned int post_neuron_bid,
        unsigned int tid,
        unsigned int num_threads,
        unsigned int spiking_neuron_id)
    {

        // following arrays are in global device memory:
        //
        //  synapse_ids_by_pre
        //  (size == number of synapses)
        //
        //  unique_delays
        //  delay_start_idx
        //  (size == number of unique delays)

        assert(blockDim.x == num_threads);

        unsigned int pre_post_block_id = spiking_neuron_id * num_blocks + post_neuron_bid;
        unsigned int global_bundle_id_start_idx = global_bundle_id_start_by_pre[pre_post_block_id];
        // num_unique_delays == num_bundles
        unsigned int num_unique_delays = global_bundle_id_start_by_pre[pre_post_block_id + 1]
                                         - global_bundle_id_start_idx;

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
                // we have per pre_post_block_id (total of num_blocks * source_N) a
            //// local bundle index going from 0 to max_num_delays_per_block for that
                // local bundle index going from 0 to num_delays for that
                // pre_post_block_id
                //global_bundle_id = max_num_delays_per_block * pre_post_block_id + bundle_idx;
                global_bundle_id = global_bundle_id_start_idx + bundle_idx;

                unsigned int delay = delay_by_bundle[global_bundle_id];
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

    } // end push_bundles()

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

