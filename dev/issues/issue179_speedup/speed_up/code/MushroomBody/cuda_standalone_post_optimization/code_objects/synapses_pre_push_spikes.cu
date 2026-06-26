#include "objects.h"

#include "code_objects/synapses_pre_push_spikes.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include "brianlib/cuda_utils.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <ctime>


__global__ void _run_synapses_pre_push_spikes_advance_kernel()
{
    using namespace brian;
    int tid = threadIdx.x;
    synapses_pre.queue->advance(
        tid);
}

__global__ void
_run_synapses_pre_push_spikes_push_kernel(
    int num_parallel_blocks,
    int _num_blocks,
    int _num_threads,
    int32_t* _ptr_array_spikegeneratorgroup__spikespace)
{
    // apperently this is not always true and that is why _num_threads is passed as function argument
    // if this assert never fails, we could remove the _num_threads form the argument list
    assert(blockDim.x == _num_threads);

    using namespace brian;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int post_neuron_bid = bid % num_parallel_blocks;
    int pre_neuron_idx = bid / num_parallel_blocks;

    int32_t spiking_neuron = _ptr_array_spikegeneratorgroup__spikespace[pre_neuron_idx];
    assert(spiking_neuron != -1);

    // push to spikequeue if spiking_neuron is in sources of current SynapticPathway
    if(synapses_pre.spikes_start <= spiking_neuron && spiking_neuron < synapses_pre.spikes_stop)
    {
        synapses_pre.queue->push_bundles(
            post_neuron_bid,
            tid,
            _num_threads,
            spiking_neuron - synapses_pre.spikes_start);
    }
}

void _run_synapses_pre_push_spikes()
{
    using namespace brian;


    ///// HOST_CONSTANTS /////
    const int _num_spikespace = 101;

    if (synapses_pre_scalar_delay)
    {
        int num_eventspaces = dev_array_spikegeneratorgroup__spikespace.size();
        synapses_pre_eventspace_idx = (current_idx_array_spikegeneratorgroup__spikespace - synapses_pre_delay + num_eventspaces) % num_eventspaces;

        //////////////////////////////////////////////
        //// No pushing in no_or_const_delay_mode ////
        //////////////////////////////////////////////
    }
    else if (synapses_pre_max_size > 0)
    {

        // get the number of spiking neurons
        int32_t num_spiking_neurons;
        CUDA_SAFE_CALL(
                cudaMemcpy(&num_spiking_neurons,
                    dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace] + _num_spikespace - 1,
                    sizeof(int32_t), cudaMemcpyDeviceToHost)
                );

        // advance spike queues
        _run_synapses_pre_push_spikes_advance_kernel<<<1, num_parallel_blocks>>>();

        CUDA_CHECK_ERROR("_run_synapses_pre_push_spikes_advance_kernel");

        static int num_threads, num_blocks;
        static size_t needed_shared_memory;
        static bool first_run = true;
        if (first_run)
        {

            needed_shared_memory = 0;

            // We don't need more then max(num_synapses) threads per block.
            num_threads = synapses_pre_max_size;
            if (num_threads > max_threads_per_block)
            {
                num_threads = max_threads_per_block;
            }

            // calculate theoretical occupancy
            int max_active_blocks;
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                        _run_synapses_pre_push_spikes_push_kernel, num_threads,
                        needed_shared_memory)
                    );

            float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                              (float)(max_threads_per_sm / num_threads_per_warp);

            // check if we have enough ressources to call kernel with given
            // number of blocks and threads
            struct cudaFuncAttributes funcAttrib;
            CUDA_SAFE_CALL(
                    cudaFuncGetAttributes(&funcAttrib, _run_synapses_pre_push_spikes_push_kernel)
                    );
            if (num_threads > funcAttrib.maxThreadsPerBlock)
            {
                // use the max num_threads before launch failure
                num_threads = funcAttrib.maxThreadsPerBlock;
                printf("WARNING Not enough ressources available to call "
                       "_run_synapses_pre_push_spikes_push_kernel "
                       "with maximum possible threads per block (%u). "
                       "Reducing num_threads to %u. (Kernel needs %i "
                       "registers per block, %i bytes of "
                       "statically-allocated shared memory per block, %i "
                       "bytes of local memory per thread and a total of %i "
                       "bytes of user-allocated constant memory)\n",
                       max_threads_per_block, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes);
            }
            else
            {
                printf("INFO _run_synapses_pre_push_spikes_push_kernel\n"
                       "\t%u blocks per spiking neuron\n"
                       "\t%u threads\n"
                       "\t%i registers per block\n"
                       "\t%i bytes statically-allocated shared memory per block\n"
                       "\t%i bytes local memory per thread\n"
                       "\t%i bytes user-allocated constant memory\n"
                       "\t%.3f theoretical occupancy\n",
                       num_parallel_blocks, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes, occupancy);
            }
            first_run = false;
        }


        if (num_spiking_neurons > 0)
        {
            num_blocks = num_parallel_blocks * num_spiking_neurons;

            _run_synapses_pre_push_spikes_push_kernel<<<num_blocks, num_threads, needed_shared_memory>>>(
                    num_parallel_blocks,
                    num_blocks,
                    num_threads,
                    dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace]);

            CUDA_CHECK_ERROR("_run_synapses_pre_push_spikes_push_kernel");
        }
    }

}
