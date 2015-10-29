#include<stdlib.h>
#include "objects.h"
#include<ctime>

#include "code_objects/neurongroup_resetter_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include "code_objects/spikegeneratorgroup_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/statemonitor_1_codeobject.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include "code_objects/synapses_1_pre_initialise_queue.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/synapses_1_synapses_create_codeobject.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_post_codeobject.h"
#include "code_objects/synapses_post_initialise_queue.h"
#include "code_objects/synapses_post_push_spikes.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_initialise_queue.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_synapses_create_codeobject.h"


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


