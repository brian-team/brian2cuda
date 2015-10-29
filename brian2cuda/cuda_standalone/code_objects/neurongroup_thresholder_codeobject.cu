#include "objects.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
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
	return sizeof(bool);
}
 	

}




__global__ void kernel_neurongroup_thresholder_codeobject(
	unsigned int THREADS_PER_BLOCK,
	int32_t* par__array_neurongroup__spikespace,
	double* par__array_defaultclock_t,
	double* par__array_neurongroup_v,
	double* par__array_neurongroup_lastspike,
	char* par__array_neurongroup_not_refractory
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	int32_t* _ptr_array_neurongroup__spikespace = par__array_neurongroup__spikespace;
	const int _num_spikespace = 101;
	double* _ptr_array_defaultclock_t = par__array_defaultclock_t;
	const int _numt = 1;
	double* _ptr_array_neurongroup_v = par__array_neurongroup_v;
	const int _numv = 100;
	double* _ptr_array_neurongroup_lastspike = par__array_neurongroup_lastspike;
	const int _numlastspike = 100;
	char* _ptr_array_neurongroup_not_refractory = par__array_neurongroup_not_refractory;
	const int _numnot_refractory = 100;

	if(_idx >= 100)
	{
		return;
	}

	// not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	// scalar code
 	


	_ptr_array_neurongroup__spikespace[_idx] = -1;

	if(tid == 0 && bid == 0)
	{
		//init number of spikes with 0
		_ptr_array_neurongroup__spikespace[100] = 0;
	}
	__syncthreads();

 	
 const char not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
 const double v = _ptr_array_neurongroup_v[_idx];
 const double _cond = (v > (-0.05)) && not_refractory;

	if(_cond) {
		int32_t spike_index = atomicAdd(&_ptr_array_neurongroup__spikespace[100], 1);
		_ptr_array_neurongroup__spikespace[spike_index] = _idx;
		// We have to use the pointer names directly here: The condition
		// might contain references to not_refractory or lastspike and in
		// that case the names will refer to a single entry.
		_ptr_array_neurongroup_not_refractory[_idx] = false;
		_ptr_array_neurongroup_lastspike[_idx] = _ptr_array_defaultclock_t[0];
	}
}

void _run_neurongroup_thresholder_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	const int _num_spikespace = 101;
		const int _numt = 1;
		const int _numv = 100;
		const int _numlastspike = 100;
		const int _numnot_refractory = 100;


kernel_neurongroup_thresholder_codeobject<<<num_blocks(100), num_threads(100)>>>(
		num_threads(100),
		dev_array_neurongroup__spikespace,
			dev_array_defaultclock_t,
			dev_array_neurongroup_v,
			dev_array_neurongroup_lastspike,
			dev_array_neurongroup_not_refractory
	);
}


