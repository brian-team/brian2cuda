#include<stdlib.h>
#include "objects.h"
#include<ctime>

#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"


void _sync_clocks()
{
	using namespace brian;

	cudaMemcpy(dev_array_defaultclock_timestep, _array_defaultclock_timestep, sizeof(uint64_t)*_num__array_defaultclock_timestep, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_array_defaultclock_dt, _array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyHostToDevice);
    	cudaMemcpy(dev_array_defaultclock_t, _array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyHostToDevice);
}

void brian_start()
{
	_init_arrays();
	_load_arrays();
	srand((unsigned int)time(NULL));

	// Initialize clocks (link timestep and dt to the respective arrays)
    	brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    	brian::defaultclock.dt = brian::_array_defaultclock_dt;
    	brian::defaultclock.t = brian::_array_defaultclock_t;
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


