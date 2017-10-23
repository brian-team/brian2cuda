#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
#include "run.h"
#include "brianlib/common_math.h"
#include "rand.h"

#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"


#include <iostream>
#include <fstream>




int main(int argc, char **argv)
{	
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	size_t limit = 128 * 1024 * 1024;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);
	cudaDeviceSynchronize();
	
	brian_start();

	{
		using namespace brian;

                
                        for(int i=0; i<_num__array_neurongroup__spikespace; i++)
                        {
                            _array_neurongroup__spikespace[i] = -1;
                        }
                        
        cudaMemcpy(dev_array_neurongroup__spikespace, &_array_neurongroup__spikespace[0], sizeof(_array_neurongroup__spikespace[0])*_num__array_neurongroup__spikespace, cudaMemcpyHostToDevice);
        
                        _array_defaultclock_dt[0] = 0.0001;
                        cudaMemcpy(dev_array_defaultclock_dt, &_array_defaultclock_dt[0], sizeof(_array_defaultclock_dt[0]), cudaMemcpyHostToDevice);
                        
        
                        _array_defaultclock_timestep[0] = 0L;
                        cudaMemcpy(dev_array_defaultclock_timestep, &_array_defaultclock_timestep[0], sizeof(_array_defaultclock_timestep[0]), cudaMemcpyHostToDevice);
                        
        
                        _array_defaultclock_t[0] = 0.0;
                        cudaMemcpy(dev_array_defaultclock_t, &_array_defaultclock_t[0], sizeof(_array_defaultclock_t[0]), cudaMemcpyHostToDevice);
                        
        magicnetwork.clear();
        magicnetwork.add(&defaultclock, _sync_clocks);
        magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_codeobject);
        magicnetwork.add(&defaultclock, _run_random_number_generation);
        magicnetwork.run(0.001, NULL, 10.0);
        _copyToHost_spikemonitor_codeobject();
        _debugmsg_spikemonitor_codeobject();

	}

	brian_end();

	return 0;
}