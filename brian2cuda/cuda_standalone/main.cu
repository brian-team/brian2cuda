#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
#include "run.h"
#include "brianlib/common_math.h"
#include "rand.h"

#include "code_objects/neurongroup_resetter_codeobject.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_post_codeobject.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/spikegeneratorgroup_codeobject.h"
#include "code_objects/synapses_post_initialise_queue.h"
#include "code_objects/statemonitor_1_codeobject.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_1_synapses_create_codeobject.h"
#include "code_objects/synapses_post_push_spikes.h"
#include "code_objects/synapses_1_pre_initialise_queue.h"
#include "code_objects/synapses_synapses_create_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/synapses_pre_initialise_queue.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"


#include <iostream>
#include <fstream>


        void report_progress(const double elapsed, const double completed, const double duration)
        {
            if (completed == 0.0)
            {
                std::cout << "Starting simulation for duration " << duration << " s";
            } else
            {
                std::cout << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << elapsed << " s";
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    std::cout << ", estimated " << remaining << " s remaining.";
                }
            }

            std::cout << std::endl << std::flush;
        }
        


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

                
                        for(int i=0; i<_num__array_spikegeneratorgroup__spikespace; i++)
                        {
                            _array_spikegeneratorgroup__spikespace[i] = -1;
                        }
                        
        cudaMemcpy(dev_array_spikegeneratorgroup__spikespace, &_array_spikegeneratorgroup__spikespace[0], sizeof(_array_spikegeneratorgroup__spikespace[0])*_num__array_spikegeneratorgroup__spikespace, cudaMemcpyHostToDevice);
        
                        for(int i=0; i<_num__array_neurongroup__spikespace; i++)
                        {
                            _array_neurongroup__spikespace[i] = -1;
                        }
                        
        cudaMemcpy(dev_array_neurongroup__spikespace, &_array_neurongroup__spikespace[0], sizeof(_array_neurongroup__spikespace[0])*_num__array_neurongroup__spikespace, cudaMemcpyHostToDevice);
        
                        _array_defaultclock_dt[0] = 0.0001;
                        cudaMemcpy(&dev_array_defaultclock_dt[0], &_array_defaultclock_dt[0], sizeof(_array_defaultclock_dt[0]), cudaMemcpyHostToDevice);
                        
        
                        for(int i=0; i<_num__array_neurongroup_lastspike; i++)
                        {
                            _array_neurongroup_lastspike[i] = - INFINITY;
                        }
                        
        cudaMemcpy(dev_array_neurongroup_lastspike, &_array_neurongroup_lastspike[0], sizeof(_array_neurongroup_lastspike[0])*_num__array_neurongroup_lastspike, cudaMemcpyHostToDevice);
        
                        for(int i=0; i<_num__array_neurongroup_not_refractory; i++)
                        {
                            _array_neurongroup_not_refractory[i] = true;
                        }
                        
        cudaMemcpy(dev_array_neurongroup_not_refractory, &_array_neurongroup_not_refractory[0], sizeof(_array_neurongroup_not_refractory[0])*_num__array_neurongroup_not_refractory, cudaMemcpyHostToDevice);
        
                            _dynamic_array_spikegeneratorgroup_spike_number.resize(1000);
                            dev_dynamic_array_spikegeneratorgroup_spike_number.resize(1000);
                        
        
                        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup_spike_number; i++)
                        {
                            _dynamic_array_spikegeneratorgroup_spike_number[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_number[i];
                        }
                        
        
                        cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_spike_number[0]), &_dynamic_array_spikegeneratorgroup_spike_number[0], sizeof(_dynamic_array_spikegeneratorgroup_spike_number[0])*_dynamic_array_spikegeneratorgroup_spike_number.size(), cudaMemcpyHostToDevice);
                        
        
                            _dynamic_array_spikegeneratorgroup_neuron_index.resize(1000);
                            dev_dynamic_array_spikegeneratorgroup_neuron_index.resize(1000);
                        
        
                        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup_neuron_index; i++)
                        {
                            _dynamic_array_spikegeneratorgroup_neuron_index[i] = _static_array__dynamic_array_spikegeneratorgroup_neuron_index[i];
                        }
                        
        
                        cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_neuron_index[0]), &_dynamic_array_spikegeneratorgroup_neuron_index[0], sizeof(_dynamic_array_spikegeneratorgroup_neuron_index[0])*_dynamic_array_spikegeneratorgroup_neuron_index.size(), cudaMemcpyHostToDevice);
                        
        
                            _dynamic_array_spikegeneratorgroup_spike_time.resize(1000);
                            dev_dynamic_array_spikegeneratorgroup_spike_time.resize(1000);
                        
        
                        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup_spike_time; i++)
                        {
                            _dynamic_array_spikegeneratorgroup_spike_time[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_time[i];
                        }
                        
        
                        cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_spike_time[0]), &_dynamic_array_spikegeneratorgroup_spike_time[0], sizeof(_dynamic_array_spikegeneratorgroup_spike_time[0])*_dynamic_array_spikegeneratorgroup_spike_time.size(), cudaMemcpyHostToDevice);
                        
        
                        _array_spikegeneratorgroup__lastindex[0] = 1;
                        cudaMemcpy(&dev_array_spikegeneratorgroup__lastindex[0], &_array_spikegeneratorgroup__lastindex[0], sizeof(_array_spikegeneratorgroup__lastindex[0]), cudaMemcpyHostToDevice);
                        
        
                        _array_spikegeneratorgroup_period[0] = 1e+100;
                        cudaMemcpy(&dev_array_spikegeneratorgroup_period[0], &_array_spikegeneratorgroup_period[0], sizeof(_array_spikegeneratorgroup_period[0]), cudaMemcpyHostToDevice);
                        
        
                        for(int i=0; i<_num__static_array__array_neurongroup_v; i++)
                        {
                            _array_neurongroup_v[i] = _static_array__array_neurongroup_v[i];
                        }
                        
        
                        cudaMemcpy(dev_array_neurongroup_v, &_array_neurongroup_v[0], sizeof(_array_neurongroup_v[0])*_num__array_neurongroup_v, cudaMemcpyHostToDevice);
                        
        
                        for(int i=0; i<_num__array_neurongroup_g; i++)
                        {
                            _array_neurongroup_g[i] = 0.0;
                        }
                        
        cudaMemcpy(dev_array_neurongroup_g, &_array_neurongroup_g[0], sizeof(_array_neurongroup_g[0])*_num__array_neurongroup_g, cudaMemcpyHostToDevice);
        _run_synapses_synapses_create_codeobject();
        
                        for(int i=0; i<_num__static_array__dynamic_array_synapses_w; i++)
                        {
                            _dynamic_array_synapses_w[i] = _static_array__dynamic_array_synapses_w[i];
                        }
                        
        
                        cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_w[0]), &_dynamic_array_synapses_w[0], sizeof(_dynamic_array_synapses_w[0])*_dynamic_array_synapses_w.size(), cudaMemcpyHostToDevice);
                        
        _run_synapses_1_synapses_create_codeobject();
        
                        for(int i=0; i<_dynamic_array_synapses_1_w.size(); i++)
                        {
                            _dynamic_array_synapses_1_w[i] = 16.200000000000003;
                        }
                        
        cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_w[0]), &_dynamic_array_synapses_1_w[0], sizeof(_dynamic_array_synapses_1_w[0])*_dynamic_array_synapses_1_w.size(), cudaMemcpyHostToDevice);
        
                        for(int i=0; i<_num__static_array__array_statemonitor__indices; i++)
                        {
                            _array_statemonitor__indices[i] = _static_array__array_statemonitor__indices[i];
                        }
                        
        
                        cudaMemcpy(dev_array_statemonitor__indices, &_array_statemonitor__indices[0], sizeof(_array_statemonitor__indices[0])*_num__array_statemonitor__indices, cudaMemcpyHostToDevice);
                        
        
                        for(int i=0; i<_num__static_array__array_statemonitor_1__indices; i++)
                        {
                            _array_statemonitor_1__indices[i] = _static_array__array_statemonitor_1__indices[i];
                        }
                        
        
                        cudaMemcpy(dev_array_statemonitor_1__indices, &_array_statemonitor_1__indices[0], sizeof(_array_statemonitor_1__indices[0])*_num__array_statemonitor_1__indices, cudaMemcpyHostToDevice);
                        
        
                        _array_defaultclock_timestep[0] = 0L;
                        cudaMemcpy(&dev_array_defaultclock_timestep[0], &_array_defaultclock_timestep[0], sizeof(_array_defaultclock_timestep[0]), cudaMemcpyHostToDevice);
                        
        
                        _array_defaultclock_t[0] = 0.0;
                        cudaMemcpy(&dev_array_defaultclock_t[0], &_array_defaultclock_t[0], sizeof(_array_defaultclock_t[0]), cudaMemcpyHostToDevice);
                        
        _run_synapses_group_variable_set_conditional_codeobject();
        _run_synapses_1_group_variable_set_conditional_codeobject();
        _run_synapses_1_pre_initialise_queue();
        _run_synapses_pre_initialise_queue();
        _run_synapses_post_initialise_queue();
        
                        _array_spikegeneratorgroup__lastindex[0] = 0;
                        cudaMemcpy(&dev_array_spikegeneratorgroup__lastindex[0], &_array_spikegeneratorgroup__lastindex[0], sizeof(_array_spikegeneratorgroup__lastindex[0]), cudaMemcpyHostToDevice);
                        
        magicnetwork.clear();
        magicnetwork.add(&defaultclock, _sync_clocks);
        magicnetwork.add(&defaultclock, _run_statemonitor_codeobject);
        magicnetwork.add(&defaultclock, _run_statemonitor_1_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_1_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_1_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_post_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_spikegeneratorgroup_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_resetter_codeobject);
        magicnetwork.add(&defaultclock, _run_random_number_generation);
        magicnetwork.run(0.2, report_progress, 10.0);
        _debugmsg_synapses_post_codeobject();
        
        _copyToHost_spikemonitor_codeobject();
        _debugmsg_spikemonitor_codeobject();
        
        _debugmsg_synapses_pre_codeobject();
        
        _debugmsg_synapses_1_pre_codeobject();

	}

	brian_end();

	return 0;
}