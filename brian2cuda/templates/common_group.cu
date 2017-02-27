{% macro cu_file() %}
#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>
{% block extra_headers %}
{% endblock %}

////// SUPPORT CODE ///////
namespace {
	int num_blocks(int num_objects)
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

	int num_threads(int num_objects)
    {
		static int needed_num_threads = -1;
		if(needed_num_threads == -1)
		{
			int needed_num_block = num_blocks(num_objects);
			needed_num_threads = min(brian::max_threads_per_block, (int)ceil(num_objects/(double)needed_num_block));
		}
		return needed_num_threads;
	}
	{% block extra_device_helper %}
	{% endblock %}
	{{support_code_lines|autoindent}}
}

{{hashdefine_lines|autoindent}}

{% block kernel %}
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

	{% block additional_variables %}
	{% endblock %}

	{% block num_thread_check %}
	if(_idx >= _N)
	{
		return;
	}
	{% endblock %}

	{% block maincode %}
	{% block maincode_inner %}
	
	///// scalar_code /////
	{{scalar_code|autoindent}}
	
	{
		///// vector_code /////
		{{vector_code|autoindent}}
	}
	{% endblock maincode_inner %}
	{% endblock maincode %}
}
{% endblock kernel %}

void _run_{{codeobj_name}}()
{	
	{# USES_VARIABLES { N } #}
	using namespace brian;
	
	const std::clock_t _start_time = std::clock();

	///// CONSTANTS ///////////
	%CONSTANTS%

	{% block extra_maincode %}
	{% endblock %}

	{% block kernel_call %}
	{# N is a constant in most cases (NeuronGroup, etc.), but a scalar array for
           synapses, we therefore have to take care to get its value in the right
           way. #}
	const int _N = {{constant_or_scalar('N', variables['N'])}};

	kernel_{{codeobj_name}}<<<num_blocks(_N),num_threads(_N)>>>(
			_N,
			num_threads(_N),
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
	{% endblock kernel_call %}

	// Profiling
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    {{codeobj_name}}_profiling_info += _run_time;
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
