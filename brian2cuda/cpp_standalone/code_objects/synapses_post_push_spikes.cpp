#include "objects.h"
#include "code_objects/synapses_post_push_spikes.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>

void _run_synapses_post_push_spikes()
{
    using namespace brian;

    const double _start_time = omp_get_wtime();

    ///// CONSTANTS ///////////
    const int _num_spikespace = 101;
    ///// POINTERS ////////////
        
    int32_t* __restrict  _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;


    //// MAIN CODE ////////////
    // we do advance at the beginning rather than at the end because it saves us making
    // a copy of the current spiking synapses
    #pragma omp parallel
    {
        synapses_post.advance();
        synapses_post.push(_ptr_array_neurongroup__spikespace, _ptr_array_neurongroup__spikespace[_num_spikespace-1]);
    }

    // Profiling
    const double _run_time = omp_get_wtime() -_start_time;
    synapses_post_push_spikes_profiling_info += _run_time;
}
