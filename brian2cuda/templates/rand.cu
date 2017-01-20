{% macro cu_file() %}

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "network.h"
#include<iostream>
#include<fstream>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void _run_random_number_generation()
{
	using namespace brian;

	// Get the number of needed random numbers.
	// curandGenerateNormal requires an even number for pseudorandom generators
	{% for co in codeobj_with_rand %}
	static unsigned int num_rand_{{co.name}} = ({{co.owner._N}} % 2 == 0) ? {{co.owner._N}} : {{co.owner._N}} + 1;
	{% endfor %}
	{% for co in codeobj_with_randn %}
	static unsigned int num_randn_{{co.name}} = ({{co.owner._N}} % 2 == 0) ? {{co.owner._N}} : {{co.owner._N}} + 1;
	{% endfor %}

	// Allocate device memory
	static bool first_run = true;
	if (first_run)
	{
		{% for co in codeobj_with_rand | sort(attribute='name') %}
		cudaMalloc((void**)&dev_{{co.name}}_rand, sizeof(float)*num_rand_{{co.name}} * {{co.rand_calls}});
		cudaMemcpyToSymbol(_array_{{co.name}}_rand, &dev_{{co.name}}_rand, sizeof(float*));
		{% endfor %}
		{% for co in codeobj_with_randn | sort(attribute='name') %}
		cudaMalloc((void**)&dev_{{co.name}}_randn, sizeof(float)*num_randn_{{co.name}} * {{co.randn_calls}});
		cudaMemcpyToSymbol(_array_{{co.name}}_randn, &dev_{{co.name}}_randn, sizeof(float*));
		{% endfor %}
		first_run = false;

	}

	// Generate random numbers
	{% for co in codeobj_with_rand %}
	curandGenerateUniform(random_float_generator, dev_{{co.name}}_rand, num_rand_{{co.name}} * {{co.rand_calls}});
	{% endfor %}
	{% for co in codeobj_with_randn %}
	curandGenerateNormal(random_float_generator, dev_{{co.name}}_randn, num_randn_{{co.name}} * {{co.randn_calls}}, 0, 1);
	{% endfor %}
}
{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_RAND_H
#define _BRIAN_RAND_H

#include <curand.h>

void _run_random_number_generation();

#endif


{% endmacro %}
