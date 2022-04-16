#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
#include "run.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "rand.h"

{% for codeobj in code_objects %}
#include "code_objects/{{codeobj.name}}.h"
{% for block in codeobj.before_after_blocks %}
#include "code_objects/{{block}}_{{codeobj.name}}.h"
{% endfor %}
{% endfor %}

{% for name in user_headers %}
#include {{name}}
{% endfor %}

#include <iostream>
#include <fstream>
#include <string>
#include "cuda_profiler_api.h"

{{report_func|autoindent}}

int main(int argc, char **argv)
{
    {{'\n'.join(code_lines['before_start'])|autoindent}}

    // seed variable set in Python through brian2.seed() calls can use this
    // variable (see device.py CUDAStandaloneDevice.generate_main_source())
    unsigned long long seed;

    //const std::clock_t _start_time = std::clock();

    CUDA_SAFE_CALL(
            cudaSetDevice({{gpu_id}})
            );

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, {{gpu_id}})
            );
    size_t limit = {{gpu_heap_size}} * 1024 * 1024;
    CUDA_SAFE_CALL(
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit)
            );
    CUDA_SAFE_CALL(
            cudaDeviceSynchronize()
            );

    //const double _run_time2 = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    //printf("INFO: setting cudaDevice stuff took %f seconds\n", _run_time2);

    brian_start();

    {{'\n'.join(code_lines['after_start'])|autoindent}}

    //const std::clock_t _start_time3 = std::clock();
    {
        using namespace brian;

        {{main_lines|autoindent}}
    }

    //const double _run_time3 = (double)(std::clock() -_start_time3)/CLOCKS_PER_SEC;
    //printf("INFO: main_lines took %f seconds\n", _run_time3);

    {{'\n'.join(code_lines['before_end'])|autoindent}}
    brian_end();
    {{'\n'.join(code_lines['after_end'])|autoindent}}

    // Profiling
    //const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    //printf("INFO: main function took %f seconds\n", _run_time);

    return 0;
}
