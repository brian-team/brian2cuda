{% macro cu_file() %}
#include "code_objects/{{codeobj_name}}.h"
#include<cmath>
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<stdint.h>
#include<iostream>
#include<fstream>
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
	unsigned int THREADS_PER_BLOCK,
	%DEVICE_PARAMETERS%
	)
{
	{# USES_VARIABLES { N } #}
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%
	{% block additional_variables %}
	{% endblock %}

	{% block num_thread_check %}
	if(_idx >= N)
	{
		return;
	}
	{% endblock %}

	{% block maincode %}
	{% block maincode_inner %}
	
	{{scalar_code|autoindent}}
	
	{
		{{vector_code|autoindent}}
	}
	{% endblock %}
	{% endblock %}
}
{% endblock %}

void _run_{{codeobj_name}}()
{	
	{# USES_VARIABLES { N } #}
	using namespace brian;
	
	///// CONSTANTS ///////////
	%CONSTANTS%

	{% block extra_maincode %}
	{% endblock %}

	{% block kernel_call %}
	kernel_{{codeobj_name}}<<<num_blocks(N),num_threads(N)>>>(
			num_threads(N),
			%HOST_PARAMETERS%
		);
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
