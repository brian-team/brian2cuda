{% macro cu_file() %}
#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>
{% block extra_headers %}
{% endblock %}

////// SUPPORT CODE ///////
namespace {
    {% block random_functions %}
    // Implement dummy functions such that the host compiled code of binomial
    // functions works. Hacky, hacky ...
    double host_rand(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    double host_randn(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    {% endblock random_functions %}

    {% block extra_device_helper %}
    {% endblock %}

    {{support_code_lines|autoindent}}
}

{{hashdefine_lines|autoindent}}

{% block kernel %}
__global__ void
{% if launch_bounds %}
__launch_bounds__(1024, {{sm_multiplier}})
{% endif %}
kernel_{{codeobj_name}}(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    {# USES_VARIABLES { N } #}
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// kernel_lines /////
    {{kernel_lines|autoindent}}

    assert(THREADS_PER_BLOCK == blockDim.x);

    {% block additional_variables %}
    {% endblock %}

    {% block num_thread_check %}
    if(_idx >= _N)
    {
        return;
    }
    {% endblock %}

    {% block kernel_maincode %}

    ///// scalar_code /////
    {{scalar_code|autoindent}}

    {
        ///// vector_code /////
        {{vector_code|autoindent}}

        {% block extra_vector_code %}
        {% endblock %}
    }
    {% endblock kernel_maincode %}
}
{% endblock kernel %}

void _run_{{codeobj_name}}()
{
    {# USES_VARIABLES { N } #}
    using namespace brian;

    {% block profiling_start %}
    {% if profiled %}
    const std::clock_t _start_time = std::clock();
    {% endif %}
    {% endblock %}

    {% block define_N %}
    {# N is a constant in most cases (NeuronGroup, etc.), but a scalar array for
       synapses, we therefore have to take care to get its value in the right
       way. #}
    const int _N = {{constant_or_scalar('N', variables['N'])}};
    {% endblock %}

    ///// HOST_CONSTANTS ///////////
    %HOST_CONSTANTS%

    {% block host_maincode %}
    {% endblock %}

    {% block prepare_kernel %}
    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
        {% block prepare_kernel_inner %}
        // get number of blocks and threads
        {% if calc_occupancy %}
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    kernel_{{codeobj_name}}, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;
        {% else %}
        num_blocks = num_parallel_blocks;
        while(num_blocks * max_threads_per_block < _N)
        {
            num_blocks *= 2;
        }
        num_threads = min(max_threads_per_block, (int)ceil(_N/(double)num_blocks));
        {% endif %}

        {% block modify_kernel_dimensions %}
        {% endblock %}

        {% endblock prepare_kernel_inner %}

        {% block occupancy %}
        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_{{codeobj_name}}, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);
        {% endblock occupancy %}


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_{{codeobj_name}})
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_{{codeobj_name}} "
                   "with maximum possible threads per block (%u). "
                   "Reducing num_threads to %u. (Kernel needs %i "
                   "registers per block, %i bytes of "
                   "statically-allocated shared memory per block, %i "
                   "bytes of local memory per thread and a total of %i "
                   "bytes of user-allocated constant memory)\n",
                   max_threads_per_block, num_threads, funcAttrib.numRegs,
                   funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                   funcAttrib.constSizeBytes);

            {% block update_occupancy %}
            // calculate theoretical occupancy for new num_threads
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                        kernel_{{codeobj_name}}, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
            {% endblock update_occupancy %}
        }
        {% block extra_info_msg %}
        {% endblock %}
        {% block kernel_info %}
        else
        {
            printf("INFO kernel_{{codeobj_name}}\n"
                   "\t%u blocks\n"
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
                   num_blocks, num_threads, funcAttrib.numRegs,
                   funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                   funcAttrib.constSizeBytes{% if calc_occupancy %}, occupancy{% endif %});
        }
        {% endblock %}
        first_run = false;
    }
    {% endblock prepare_kernel %}

    {% block extra_kernel_call %}
    {% endblock %}

    {% block kernel_call %}
    kernel_{{codeobj_name}}<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            %HOST_PARAMETERS%
        );

    CUDA_CHECK_ERROR("kernel_{{codeobj_name}}");
    {% endblock kernel_call %}

    {% block extra_kernel_call_post %}
    {% endblock %}

    {% block profiling_stop %}
    {% if profiled %}
    CUDA_SAFE_CALL(
            cudaDeviceSynchronize()
            );
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    {{codeobj_name}}_profiling_info += _run_time;
    {% endif %}
    {% endblock %}
}

{% block extra_functions_cu %}
{% endblock %}

{% endmacro %}


{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "objects.h"

void _run_{{codeobj_name}}();

{% block extra_functions_h %}
{% endblock %}

#endif
{% endmacro %}
