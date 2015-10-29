#include "objects.h"
#include "code_objects/spikegeneratorgroup_codeobject.h"
#include<cmath>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

////// SUPPORT CODE ///////
namespace {
	int num_blocks(int num_objects)
    {
		static int needed_num_block = -1;
	    if(needed_num_block == -1)
		{
			needed_num_block = brian::num_parallel_blocks;
			while(needed_num_block * brian::max_threads_per_block < num_objects)
			{
				needed_num_block *= 2;
			}
		}
		return needed_num_block;
    }

	int num_threads(int num_objects)
    {
		static int needed_num_threads = -1;
		if(needed_num_threads == -1)
		{
			int needed_num_block = num_blocks(num_objects);
			needed_num_threads = min(brian::max_threads_per_block, (int)ceil(num_objects/(double)needed_num_block));
		}
		return needed_num_threads;
	}
int mem_per_thread(){
	return sizeof(int32_t);
}

__device__ unsigned int _last_element_checked = 0;
 	

}




__global__ void kernel_spikegeneratorgroup_codeobject(
	unsigned int THREADS_PER_BLOCK,
	int32_t* par__array_spikegeneratorgroup_spike_number,
	int par_num_spike_number,
	int32_t* par__array_spikegeneratorgroup__spikespace,
	int32_t* par__array_spikegeneratorgroup__lastindex,
	double* par__array_spikegeneratorgroup_spike_time,
	int par_num_spike_time,
	double* par__array_spikegeneratorgroup_period,
	int32_t* par__array_spikegeneratorgroup_neuron_index,
	int par_num_neuron_index,
	double* par__array_defaultclock_t,
	double* par__array_defaultclock_dt
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	int32_t* _ptr_array_spikegeneratorgroup_spike_number = par__array_spikegeneratorgroup_spike_number;
	const int _numspike_number = par_num_spike_number;
	int32_t* _ptr_array_spikegeneratorgroup__spikespace = par__array_spikegeneratorgroup__spikespace;
	const int _num_spikespace = 101;
	int32_t* _ptr_array_spikegeneratorgroup__lastindex = par__array_spikegeneratorgroup__lastindex;
	const int _num_lastindex = 1;
	double* _ptr_array_spikegeneratorgroup_spike_time = par__array_spikegeneratorgroup_spike_time;
	const int _numspike_time = par_num_spike_time;
	double* _ptr_array_spikegeneratorgroup_period = par__array_spikegeneratorgroup_period;
	const int _numperiod = 1;
	int32_t* _ptr_array_spikegeneratorgroup_neuron_index = par__array_spikegeneratorgroup_neuron_index;
	const int _numneuron_index = par_num_neuron_index;
	double* _ptr_array_defaultclock_t = par__array_defaultclock_t;
	const int _numt = 1;
	double* _ptr_array_defaultclock_dt = par__array_defaultclock_dt;
	const int _numdt = 1;

	if(_idx >= 100)
	{
		return;
	}



    double _the_period = _ptr_array_spikegeneratorgroup_period[0];
    double padding_before = fmod(_ptr_array_defaultclock_t[0], _the_period);
    double padding_after  = fmod(_ptr_array_defaultclock_t[0]+_ptr_array_defaultclock_dt[0], _the_period);
    double epsilon        = 1e-3*_ptr_array_defaultclock_dt[0];

    // We need some precomputed values that will be used during looping
    bool not_first_spike = (_ptr_array_spikegeneratorgroup__lastindex[0] > 0);
    bool not_end_period  = (fabs(padding_after) > epsilon);
    bool test;
	
	//// MAIN CODE ////////////
	// scalar code
	
	for(int i = tid; i < 100; i+= THREADS_PER_BLOCK)
	{
		_ptr_array_spikegeneratorgroup__spikespace[i] = -1;
	}

	if(tid == 0)
	{
		//init number of spikes with 0
		_ptr_array_spikegeneratorgroup__spikespace[100] = 0;
	}
	__syncthreads();

	for(int spike_idx = _ptr_array_spikegeneratorgroup__lastindex[0] + tid; spike_idx < _numspike_time; spike_idx += THREADS_PER_BLOCK)
	{
		if (not_end_period)
		{
	        test = (_ptr_array_spikegeneratorgroup_spike_time[spike_idx] > padding_after) || (fabs(_ptr_array_spikegeneratorgroup_spike_time[spike_idx] - padding_after) < epsilon);
	    }
	    else
	    {
	        // If we are in the last timestep before the end of the period, we remove the first part of the
	        // test, because padding will be 0
	        test = (fabs(_ptr_array_spikegeneratorgroup_spike_time[spike_idx] - padding_after) < epsilon);
	    }
	    if (test)
	    {
	        break;
	    }
	    int32_t neuron_id = _ptr_array_spikegeneratorgroup_neuron_index[spike_idx];
    	int32_t spikespace_index = atomicAdd(&_ptr_array_spikegeneratorgroup__spikespace[100], 1);
		atomicAdd(&_ptr_array_spikegeneratorgroup__lastindex[0], 1);
    	_ptr_array_spikegeneratorgroup__spikespace[spikespace_index] = neuron_id;
		__syncthreads();
	}
}

void _run_spikegeneratorgroup_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	int32_t* const _array_spikegeneratorgroup_spike_number = thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_spike_number[0]);
		const int _numspike_number = dev_dynamic_array_spikegeneratorgroup_spike_number.size();
		const int _num_spikespace = 101;
		const int _num_lastindex = 1;
		double* const _array_spikegeneratorgroup_spike_time = thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_spike_time[0]);
		const int _numspike_time = dev_dynamic_array_spikegeneratorgroup_spike_time.size();
		const int _numperiod = 1;
		int32_t* const _array_spikegeneratorgroup_neuron_index = thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_neuron_index[0]);
		const int _numneuron_index = dev_dynamic_array_spikegeneratorgroup_neuron_index.size();
		const int _numt = 1;
		const int _numdt = 1;


kernel_spikegeneratorgroup_codeobject<<<1,1>>>(
		1,
		_array_spikegeneratorgroup_spike_number,
			_numspike_number,
			dev_array_spikegeneratorgroup__spikespace,
			dev_array_spikegeneratorgroup__lastindex,
			_array_spikegeneratorgroup_spike_time,
			_numspike_time,
			dev_array_spikegeneratorgroup_period,
			_array_spikegeneratorgroup_neuron_index,
			_numneuron_index,
			dev_array_defaultclock_t,
			dev_array_defaultclock_dt
	);
}


