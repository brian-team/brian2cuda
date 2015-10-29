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

	float mean = 0.0;
	float std_deviation = 1.0;
	
	unsigned int needed_random_numbers;

	{% for co in codeobj_with_rand %}
	//curand calls always need a even number for some reason
	needed_random_numbers = {{co.owner._N}} % 2 == 0?{{co.owner._N}}:{{co.owner._N}}+1;
	curandGenerateUniform(random_float_generator, dev_{{co.name}}_random_uniform_floats, needed_random_numbers * {{co.rand_calls}});
	{% endfor %}
	{% for co in codeobj_with_randn %}
	//curand calls always need a even number for some reason
	needed_random_numbers = {{co.owner._N}} % 2 == 0?{{co.owner._N}}:{{co.owner._N}}+1;
	curandGenerateNormal(random_float_generator, dev_{{co.name}}_random_normal_floats, needed_random_numbers * {{co.randn_calls}}, mean, std_deviation);
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
