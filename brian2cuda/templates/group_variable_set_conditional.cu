{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}
{% macro cu_file() %}
#include "code_objects/{{codeobj_name}}.h"
#include "rand.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include "brianlib/cuda_utils.h"
#include<math.h>
#include<stdint.h>
#include<iostream>
#include<fstream>

{% block extra_headers %}
{% endblock %}

{% for name in user_headers %}
#include {{name}}
{%endfor %}

////// SUPPORT CODE ///////
namespace {
    // Implement dummy functions such that the host compiled code of binomial
    // functions works. Hacky, hacky ...
    double _host_rand(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    double _host_randn(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_poisson` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }

    {{support_code_lines|autoindent}}
}

__global__ void kernel_{{codeobj_name}}(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// kernel_lines /////
    {{kernel_lines|autoindent}}

    if(_idx >= _N)
    {
        return;
    }

    ///// scalar_code['condition'] /////
    {{scalar_code['condition']|autoindent}}

    ///// scalar_code['statement'] /////
    {{scalar_code['statement']|autoindent}}

    ///// vector_code['condition'] /////
    {{vector_code['condition']|autoindent}}

    if (_cond)
    {
        ///// vector_code['statement'] /////
        {{vector_code['statement']|autoindent}}
    }
}

////// HASH DEFINES ///////
{{hashdefine_lines|autoindent}}

void _run_{{codeobj_name}}()
{
    using namespace brian;

    {# N is a constant in most cases (NeuronGroup, etc.), but a scalar array for
       synapses, we therefore have to take care to get its value in the right
       way. #}
    const int _N = {{constant_or_scalar('N', variables['N'])}};

    ///// HOST_CONSTANTS ///////////
    %HOST_CONSTANTS%

    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
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

        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_{{codeobj_name}}, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);

        {% else %}
        num_blocks = num_parallel_blocks;
        while(num_blocks * max_threads_per_block < _N)
        {
            num_blocks *= 2;
        }
        num_threads = min(max_threads_per_block, (int)ceil(_N/(double)num_blocks));
        {% endif %}

        // check if we have enough ressources to call kernel with given number
        // of blocks and threads
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
        }
        else
        {
            printf("INFO kernel_{{codeobj_name}}\n"
                   "\t%u blocks\n"
                   "\t%u threads\n"
                   "\t%i registers per block\n"
                   "\t%i bytes statically-allocated shared memory per block\n"
                   "\t%i bytes local memory per thread\n"
                   "\t%i bytes user-allocated constant memory\n",
                   num_blocks, num_threads, funcAttrib.numRegs,
                   funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                   funcAttrib.constSizeBytes);
        }
        first_run = false;
    }

    kernel_{{codeobj_name}}<<<num_blocks, num_threads>>>(
        _N,
        num_threads,
        ///// HOST_PARAMETERS /////
        %HOST_PARAMETERS%
    );

    CUDA_CHECK_ERROR("kernel_{{codeobj_name}}");
    {% for var in variables.values() %}
    {# We want to copy only those variables that were potentially modified in aboves kernel call. #}
    {% if var is not callable and var.array and not var.constant and not var.dynamic %}
    {% set varname = get_array_name(var, access_data=False) %}
    CUDA_SAFE_CALL(
            cudaMemcpy({{varname}}, dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyDeviceToHost)
            );
    {% endif %}
    {% endfor %}
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



