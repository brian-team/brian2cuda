#include "objects.h"
#include "code_objects/statemonitor_codeobject.h"
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




__global__ void _run_statemonitor_codeobject_kernel(
	int _num_indices,
	int32_t* indices,
	int current_iteration,
		double** monitor_w,
	double* par__array_synapses_w,
	int par_num__source_w,
	double* par__array_statemonitor_t,
	int par_num_t
	)
{
	unsigned int tid = threadIdx.x;
	if(tid > _num_indices)
	{
		return;
	}
	int32_t _idx = indices[tid];
	
	double* _ptr_array_synapses_w = par__array_synapses_w;
	const int _num_source_w = par_num__source_w;
	double* _ptr_array_statemonitor_t = par__array_statemonitor_t;
	const int _numt = par_num_t;
	
 	

 	
 const double _source_w = _ptr_array_synapses_w[_idx];
 const double _to_record_w = _source_w;


		monitor_w[tid][current_iteration] = _to_record_w;
}

void _run_statemonitor_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	double* const _array_synapses_w = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_w[0]);
		const int _num_source_w = dev_dynamic_array_synapses_w.size();
		double* const _array_statemonitor_t = thrust::raw_pointer_cast(&dev_dynamic_array_statemonitor_t[0]);
		const int _numt = dev_dynamic_array_statemonitor_t.size();


int current_iteration = defaultclock.timestep[0];
static unsigned int start_offset = current_iteration - dev_dynamic_array_statemonitor_t.size();
dev_dynamic_array_statemonitor_t.push_back(defaultclock.t[0]);
static bool first_run = true;
if(first_run)
{
	int num_iterations = defaultclock.i_end;
	
		addresses_monitor__dynamic_array_statemonitor_w.clear();			
	for(int i = 0; i < _num__array_statemonitor__indices; i++)
	{
			_dynamic_array_statemonitor_w[i].resize(_dynamic_array_statemonitor_w[i].size() + num_iterations - current_iteration);
			addresses_monitor__dynamic_array_statemonitor_w.push_back(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_w[i][0]));
	}
	first_run = false;
}

_run_statemonitor_codeobject_kernel<<<1, _num__array_statemonitor__indices>>>(
	_num__array_statemonitor__indices,
	dev_array_statemonitor__indices,
	current_iteration - start_offset,
		thrust::raw_pointer_cast(&addresses_monitor__dynamic_array_statemonitor_w[0]),
	_array_synapses_w,
			_num_source_w,
			_array_statemonitor_t,
			_numt
	);
}


