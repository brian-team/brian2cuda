////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cu_file() %}
{#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include "brianlib/cuda_utils.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <ctime>


__global__ void _run_{{codeobj_name}}_advance_kernel()
{
    using namespace brian;
    int tid = threadIdx.x;
    {{owner.name}}.queue->advance(
        tid);
}

__global__ void
{% if launch_bounds or syn_launch_bounds %}
__launch_bounds__(1024, {{sm_multiplier}})
{% endif %}
_run_{{codeobj_name}}_push_kernel(
    int num_parallel_blocks,
    int _num_blocks,
    int _num_threads,
    int32_t* {{_eventspace}})
{
    // apperently this is not always true and that is why _num_threads is passed as function argument
    // if this assert never fails, we could remove the _num_threads form the argument list
    assert(blockDim.x == _num_threads);

    using namespace brian;

    {% if not bundle_mode %}
    // TODO: check if static shared memory is faster / makes any difference
    extern __shared__ char shared_mem[];
    {% endif %}
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int post_neuron_bid = bid % num_parallel_blocks;
    int pre_neuron_idx = bid / num_parallel_blocks;

    int32_t spiking_neuron = {{_eventspace}}[pre_neuron_idx];
    assert(spiking_neuron != -1);

    // push to spikequeue if spiking_neuron is in sources of current SynapticPathway
    if({{owner.name}}.spikes_start <= spiking_neuron && spiking_neuron < {{owner.name}}.spikes_stop)
    {
        {% if bundle_mode %}
        {{owner.name}}.queue->push_bundles(
        {% else %}
        {{owner.name}}.queue->push_synapses(
            shared_mem,
        {% endif %}
            post_neuron_bid,
            tid,
            _num_threads,
            spiking_neuron - {{owner.name}}.spikes_start);
    }
}

void _run_{{codeobj_name}}()
{
    using namespace brian;

    {% if profiled %}
    const std::clock_t _start_time = std::clock();
    {% endif %}

    ///// HOST_CONSTANTS /////
    %HOST_CONSTANTS%

    {% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}
    if ({{owner.name}}_scalar_delay)
    {
        int num_eventspaces = dev{{_eventspace}}.size();
        {{owner.name}}_eventspace_idx = (current_idx{{_eventspace}} - {{owner.name}}_delay + num_eventspaces) % num_eventspaces;

        //////////////////////////////////////////////
        //// No pushing in no_or_const_delay_mode ////
        //////////////////////////////////////////////
    }
    else if ({{owner.name}}_max_size > 0)
    {

        // get the number of spiking neurons
        int32_t num_spiking_neurons;
        CUDA_SAFE_CALL(
                cudaMemcpy(&num_spiking_neurons,
                    dev{{_eventspace}}[current_idx{{_eventspace}}] + _num_{{owner.event}}space - 1,
                    sizeof(int32_t), cudaMemcpyDeviceToHost)
                );

        // advance spike queues
        _run_{{codeobj_name}}_advance_kernel<<<1, num_parallel_blocks>>>();

        CUDA_CHECK_ERROR("_run_{{codeobj_name}}_advance_kernel");

        static int num_threads, num_blocks;
        static size_t needed_shared_memory;
        static bool first_run = true;
        if (first_run)
        {

            {% if not bundle_mode %}
            /* We are copying next_delay_start_idx and the atomic offset (both
             * size = num_unique_delays) into shared memory. Since
             * num_unique_delays varies for different combinations of pre
             * neuron and bid, we allocate for max(num_unique_delays). And +1
             * per block for copying size_before_resize into shared memory when
             * we need to use the outer loop.
             */
            needed_shared_memory = (2 * {{owner.name}}_max_num_unique_delays + 1) * sizeof(int);
            assert (needed_shared_memory <= max_shared_mem_size);
            {% else %}{# bundle_mode #}
            needed_shared_memory = 0;
            {% endif %}{# not bundle_mode #}

            // We don't need more then max(num_synapses) threads per block.
            num_threads = {{owner.name}}_max_size;
            if (num_threads > max_threads_per_block)
            {
                num_threads = max_threads_per_block;
            }

            // calculate theoretical occupancy
            int max_active_blocks;
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                        _run_{{codeobj_name}}_push_kernel, num_threads,
                        needed_shared_memory)
                    );

            float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                              (float)(max_threads_per_sm / num_threads_per_warp);

            // check if we have enough ressources to call kernel with given
            // number of blocks and threads
            struct cudaFuncAttributes funcAttrib;
            CUDA_SAFE_CALL(
                    cudaFuncGetAttributes(&funcAttrib, _run_{{codeobj_name}}_push_kernel)
                    );
            if (num_threads > funcAttrib.maxThreadsPerBlock)
            {
                // use the max num_threads before launch failure
                num_threads = funcAttrib.maxThreadsPerBlock;
                printf("WARNING Not enough ressources available to call "
                       "_run_{{codeobj_name}}_push_kernel "
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
                printf("INFO _run_{{codeobj_name}}_push_kernel\n"
                       "\t%u blocks per spiking neuron\n"
                       "\t%u threads\n"
                       "\t%i registers per block\n"
                       "\t%i bytes statically-allocated shared memory per block\n"
                       "\t%i bytes local memory per thread\n"
                       "\t%i bytes user-allocated constant memory\n"
                       {% if calc_occupancy %}
                       "\t%.3f theoretical occupancy\n",
                       {% else %}
                       "",
                       {% endif %}
                       num_parallel_blocks, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes{% if calc_occupancy %}, occupancy{% endif %});
            }
            first_run = false;
        }


        if (num_spiking_neurons > 0)
        {
            num_blocks = num_parallel_blocks * num_spiking_neurons;

            _run_{{codeobj_name}}_push_kernel<<<num_blocks, num_threads, needed_shared_memory>>>(
                    num_parallel_blocks,
                    num_blocks,
                    num_threads,
                    dev{{_eventspace}}[current_idx{{_eventspace}}]);

            CUDA_CHECK_ERROR("_run_{{codeobj_name}}_push_kernel");
        }
    }

    {% if profiled %}
    CUDA_SAFE_CALL(
            cudaDeviceSynchronize()
            );
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    {{codeobj_name}}_profiling_info += _run_time;
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
