#include "objects.h"
#include "code_objects/statemonitor_1_codeobject.h"
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
 	

}




__global__ void _run_statemonitor_1_codeobject_kernel(
	int _num_indices,
	int32_t* indices,
	int current_iteration,
		double** monitor_v,
	double* par__array_neurongroup_v,
	double* par__array_statemonitor_1_t,
	int par_num_t
	)
{
	unsigned int tid = threadIdx.x;
	if(tid > _num_indices)
	{
		return;
	}
	int32_t _idx = indices[tid];
	
	double* _ptr_array_neurongroup_v = par__array_neurongroup_v;
	const int _num_source_v = 100;
	double* _ptr_array_statemonitor_1_t = par__array_statemonitor_1_t;
	const int _numt = par_num_t;
	
 	

 	
 const double _source_v = _ptr_array_neurongroup_v[_idx];
 const double _to_record_v = _source_v;


		monitor_v[tid][current_iteration] = _to_record_v;
}

void _run_statemonitor_1_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	const int _num_source_v = 100;
		double* const _array_statemonitor_1_t = thrust::raw_pointer_cast(&dev_dynamic_array_statemonitor_1_t[0]);
		const int _numt = dev_dynamic_array_statemonitor_1_t.size();


int current_iteration = defaultclock.timestep[0];
static unsigned int start_offset = current_iteration - dev_dynamic_array_statemonitor_1_t.size();
dev_dynamic_array_statemonitor_1_t.push_back(defaultclock.t[0]);
static bool first_run = true;
if(first_run)
{
	int num_iterations = defaultclock.i_end;
	
		addresses_monitor__dynamic_array_statemonitor_1_v.clear();			
	for(int i = 0; i < _num__array_statemonitor_1__indices; i++)
	{
			_dynamic_array_statemonitor_1_v[i].resize(_dynamic_array_statemonitor_1_v[i].size() + num_iterations - current_iteration);
			addresses_monitor__dynamic_array_statemonitor_1_v.push_back(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_1_v[i][0]));
	}
	first_run = false;
}

_run_statemonitor_1_codeobject_kernel<<<1, _num__array_statemonitor_1__indices>>>(
	_num__array_statemonitor_1__indices,
	dev_array_statemonitor_1__indices,
	current_iteration - start_offset,
		thrust::raw_pointer_cast(&addresses_monitor__dynamic_array_statemonitor_1_v[0]),
	dev_array_neurongroup_v,
			_array_statemonitor_1_t,
			_numt
	);
}


