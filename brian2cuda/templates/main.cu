#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
#include "run.h"
#include "brianlib/common_math.h"
#include "rand.h"

{% for codeobj in code_objects %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in additional_headers %}
#include "{{name}}"
{% endfor %}

#include <iostream>
#include <fstream>

{{report_func|autoindent}}

int main(int argc, char **argv)
{	
	const std::clock_t _start_time = std::clock();

	const std::clock_t _start_time2 = std::clock();

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	size_t limit = 128 * 1024 * 1024;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);
	cudaDeviceSynchronize();
	
	const double _run_time2 = (double)(std::clock() -_start_time2)/CLOCKS_PER_SEC;
	printf("INFO: setting cudaDevice stuff took %f seconds\n", _run_time2);

	brian_start();

	const std::clock_t _start_time3 = std::clock();
	{
		using namespace brian;

        {{main_lines|autoindent}}
	}

	const double _run_time3 = (double)(std::clock() -_start_time3)/CLOCKS_PER_SEC;
	printf("INFO: main_lines took %f seconds\n", _run_time3);

	brian_end();

	// Profiling
	const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
	printf("INFO: main function took %f seconds\n", _run_time);

	return 0;
}
