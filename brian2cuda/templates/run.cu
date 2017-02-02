{% macro cu_file() %}
#include<stdlib.h>
#include "objects.h"
#include<ctime>

{% for codeobj in code_objects | sort(attribute='name') %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in additional_headers %}
#include "{{name}}"
{% endfor %}

void _sync_clocks()
{
	using namespace brian;

	{% for clock in clocks | sort(attribute='name') %}
	cudaMemcpy(dev{{array_specs[clock.variables['timestep']]}}, {{array_specs[clock.variables['timestep']]}}, sizeof(uint64_t)*_num_{{array_specs[clock.variables['timestep']]}}, cudaMemcpyHostToDevice);
	cudaMemcpy(dev{{array_specs[clock.variables['dt']]}}, {{array_specs[clock.variables['dt']]}}, sizeof(double)*_num_{{array_specs[clock.variables['dt']]}}, cudaMemcpyHostToDevice);
    	cudaMemcpy(dev{{array_specs[clock.variables['t']]}}, {{array_specs[clock.variables['t']]}}, sizeof(double)*_num_{{array_specs[clock.variables['t']]}}, cudaMemcpyHostToDevice);
    	{% endfor %}
}

void brian_start()
{
	const std::clock_t _start_time = std::clock();

	_init_arrays();
	_load_arrays();
	srand((unsigned int)time(NULL));

	// Initialize clocks (link timestep and dt to the respective arrays)
    	{% for clock in clocks | sort(attribute='name') %}
    	brian::{{clock.name}}.timestep = brian::{{array_specs[clock.variables['timestep']]}};
    	brian::{{clock.name}}.dt = brian::{{array_specs[clock.variables['dt']]}};
    	brian::{{clock.name}}.t = brian::{{array_specs[clock.variables['t']]}};
    	{% endfor %}

	// Profiling
	const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
	brian::brian_start_profiling_info += _run_time;
}

void brian_end()
{
	const std::clock_t _start_time = std::clock();

	_write_arrays();
	_dealloc_arrays();

	// Profiling
	const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
	brian::brian_end_profiling_info += _run_time;
}

{% for name, lines in run_funcs.items() | sort(attribute='name') %}
void {{name}}()
{
	using namespace brian;

    {{lines|autoindent}}
}

{% endfor %}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

void _sync_clocks();
void brian_start();
void brian_end();

{% for name, lines in run_funcs.items() | sort(attribute='name') %}
void {{name}}();
{% endfor %}

{% endmacro %}
