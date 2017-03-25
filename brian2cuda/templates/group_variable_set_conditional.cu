{% macro cu_file() %}
#include "code_objects/{{codeobj_name}}.h"
#include<math.h>
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

{% block extra_headers %}
{% endblock %}

////// SUPPORT CODE ///////
namespace {
	{{support_code_lines|autoindent}}
}

__global__ void kernel_{{codeobj_name}}(
	unsigned int _N,
	unsigned int THREADS_PER_BLOCK,
	///// DEVICE_PARAMETERS /////
	%DEVICE_PARAMETERS%
	)
{
	{# USES_VARIABLES { N } #}
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;

	///// KERNEL_VARIABLES /////
	%KERNEL_VARIABLES%

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
    {# USES_VARIABLES { N } #}
    {# ALLOWS_SCALAR_WRITE #}
	using namespace brian;

	{# N is a constant in most cases (NeuronGroup, etc.), but a scalar array for
           synapses, we therefore have to take care to get its value in the right
           way. #}
	const int _N = {{constant_or_scalar('N', variables['N'])}};

	///// CONSTANTS ///////////
	%CONSTANTS%

	static int num_threads, num_blocks;
	static bool first_run = true;
	if (first_run)
	{
		// get number of blocks and threads
		{% if calc_occupancy %}
		int min_num_threads; // The minimum grid size needed to achieve the
							 // maximum occupancy for a full device launch

		cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
				kernel_{{codeobj_name}}, 0, 0);  // last args: dynamicSMemSize, blockSizeLimit

		// Round up according to array size
		num_blocks = (_N + num_threads - 1) / num_threads;

		// calculate theoretical occupancy
		int max_active_blocks;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
				kernel_{{codeobj_name}}, num_threads, 0);

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

		// check if we have enough ressources to call kernel with given number of blocks and threads
		struct cudaFuncAttributes funcAttrib;
		cudaFuncGetAttributes(&funcAttrib, kernel_{{codeobj_name}});
		if (num_threads > funcAttrib.maxThreadsPerBlock)
		{
			// use the max num_threads before launch failure
			num_threads = funcAttrib.maxThreadsPerBlock;
			printf("WARNING Not enough ressources available to call kernel_{{codeobj_name}} with "
					"maximum possible threads per block (%u). Reducing num_threads to "
					"%u. (Kernel needs %i registers per block, %i bytes of statically-allocated "
					"shared memory per block, %i bytes of local memory per thread and "
					"a total of %i bytes of user-allocated constant memory)\n",
					max_threads_per_block, num_threads, funcAttrib.numRegs, funcAttrib.sharedSizeBytes,
					funcAttrib.localSizeBytes, funcAttrib.constSizeBytes);
		}
		else
		{
			printf("INFO calling kernel_{{codeobj_name}} with %u blocks and %u threads. "
					"Kernel needs %i registers per block, %i bytes of statically-allocated "
					"shared memory per block, %i bytes of local memory per thread and "
					"a total of %i bytes of user-allocated constant memory\n",
					num_blocks, num_threads, funcAttrib.numRegs, funcAttrib.sharedSizeBytes,
					funcAttrib.localSizeBytes, funcAttrib.constSizeBytes);
		}
		first_run = false;
	}

	kernel_{{codeobj_name}}<<<num_blocks, num_threads>>>(
		_N,
		num_threads,
		///// HOST_PARAMETERS /////
		%HOST_PARAMETERS%
	);

	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		printf("ERROR launching kernel_{{codeobj_name}} in %s:%d %s\n",
				__FILE__, __LINE__, cudaGetErrorString(status));
		_dealloc_arrays();
		exit(status);
	}

	{% for var in variables.itervalues() %}
	{# We want to copy only those variables that were potentially modified in aboves kernel call. #}
	{% if var is not callable and var.array and not var.constant and not var.dynamic %}
	{% set varname = get_array_name(var, access_data=False) %}
	cudaMemcpy({{varname}}, dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyDeviceToHost);
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



