#include "objects.h"
#include "code_objects/spikemonitor_codeobject.h"
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
		__device__ cudaVector<int32_t>** monitor_i;
		__device__ cudaVector<double>** monitor_t;
 	

}




__global__ void _run_spikemonitor_codeobject_init(
	unsigned int num_blocks)
{
		monitor_i = new cudaVector<int32_t>*[num_blocks];
		for(int i = 0; i < num_blocks; i++)
		{
			monitor_i[i] = new cudaVector<int32_t>();
		}
		monitor_t = new cudaVector<double>*[num_blocks];
		for(int i = 0; i < num_blocks; i++)
		{
			monitor_t[i] = new cudaVector<double>();
		}
}

__global__ void _run_spikemonitor_codeobject_kernel(
	unsigned int neurongroup_N,
	unsigned int num_blocks,
	unsigned int block_size,
	int32_t* count,
	double* par__array_defaultclock_t,
	int32_t* par__array_spikemonitor_count,
	int32_t* par__array_neurongroup__spikespace,
	int32_t* par__array_spikemonitor_i,
	int par_num_i,
	int32_t* par__array_spikemonitor_N,
	double* par__array_spikemonitor_t,
	int par_num_t,
	int32_t* par__array_spikemonitor__source_idx,
	int32_t* par__array_neurongroup_i
	)
{
	using namespace brian;
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	__syncthreads();
	
	double* _ptr_array_defaultclock_t = par__array_defaultclock_t;
	const int _num_source_t = 1;
	int32_t* _ptr_array_spikemonitor_count = par__array_spikemonitor_count;
	const int _numcount = 100;
	const int _num_clock_t = 1;
	int32_t* _ptr_array_neurongroup__spikespace = par__array_neurongroup__spikespace;
	const int _num_spikespace = 101;
	int32_t* _ptr_array_spikemonitor_i = par__array_spikemonitor_i;
	const int _numi = par_num_i;
	int32_t* _ptr_array_spikemonitor_N = par__array_spikemonitor_N;
	const int _numN = 1;
	double* _ptr_array_spikemonitor_t = par__array_spikemonitor_t;
	const int _numt = par_num_t;
	int32_t* _ptr_array_spikemonitor__source_idx = par__array_spikemonitor__source_idx;
	const int _num_source_idx = 100;
	int32_t* _ptr_array_neurongroup_i = par__array_neurongroup_i;
	const int _num_source_i = 100;

 	


	//REMINDER: spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
	for(int i = bid*block_size; i < neurongroup_N && i < (bid + 1)*block_size; i++)
	{
		int32_t spiking_neuron = _ptr_array_neurongroup__spikespace[i];
		if(spiking_neuron != -1)
		{
			if(spiking_neuron >= 0 && spiking_neuron < 100)
			{
				int _idx = spiking_neuron;
				int _vectorisation_idx = _idx;
    				
    const double _source_t = _ptr_array_defaultclock_t[0];
    const int32_t _source_i = _ptr_array_neurongroup_i[_idx];
    const int32_t _to_record_i = _source_i;
    const double _to_record_t = _source_t;

					monitor_i[bid]->push(_to_record_i);
					monitor_t[bid]->push(_to_record_t);
				count[_idx -0]++;
			}
		}
	}
}

void _run_spikemonitor_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	const int _num_source_t = 1;
		const int _numcount = 100;
		const int _num_clock_t = 1;
		const int _num_spikespace = 101;
		int32_t* const _array_spikemonitor_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]);
		const int _numi = dev_dynamic_array_spikemonitor_i.size();
		const int _numN = 1;
		double* const _array_spikemonitor_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]);
		const int _numt = dev_dynamic_array_spikemonitor_t.size();
		const int _num_source_idx = 100;
		const int _num_source_i = 100;


static bool first_run = true;
if(first_run)
{
	_run_spikemonitor_codeobject_init<<<1,1>>>(
		num_blocks(_num_spikespace-1));
	first_run = false;
}
_run_spikemonitor_codeobject_kernel<<<num_blocks(_num_spikespace-1), 1>>>(
		_num_spikespace-1,
		num_blocks(_num_spikespace-1),
		num_threads(_num_spikespace-1),
		dev_array_spikemonitor_count,
		dev_array_defaultclock_t,
			dev_array_spikemonitor_count,
			dev_array_neurongroup__spikespace,
			_array_spikemonitor_i,
			_numi,
			dev_array_spikemonitor_N,
			_array_spikemonitor_t,
			_numt,
			dev_array_spikemonitor__source_idx,
			dev_array_neurongroup_i);
}

__global__ void _run_debugmsg_spikemonitor_codeobject_kernel(
	unsigned int num_blocks
)
{
	using namespace brian;
	unsigned int total_number = 0;
	total_number = 0;
	for(int i = 0; i < num_blocks; i++)
	{
		total_number += monitor_i[i]->size();
	}
	total_number = 0;
	for(int i = 0; i < num_blocks; i++)
	{
		total_number += monitor_t[i]->size();
	}
	printf("Number of spikes: %d\n", total_number);
}

__global__ void _count_spikemonitor_codeobject_kernel(
	double* par__array_defaultclock_t,
	int32_t* par__array_spikemonitor_count,
	int32_t* par__array_neurongroup__spikespace,
	int32_t* par__array_spikemonitor_i,
	int par_num_i,
	int32_t* par__array_spikemonitor_N,
	double* par__array_spikemonitor_t,
	int par_num_t,
	int32_t* par__array_spikemonitor__source_idx,
	int32_t* par__array_neurongroup_i,
	unsigned int num_blocks,
	unsigned int* total
)
{
	using namespace brian;
	double* _ptr_array_defaultclock_t = par__array_defaultclock_t;
	const int _num_source_t = 1;
	int32_t* _ptr_array_spikemonitor_count = par__array_spikemonitor_count;
	const int _numcount = 100;
	const int _num_clock_t = 1;
	int32_t* _ptr_array_neurongroup__spikespace = par__array_neurongroup__spikespace;
	const int _num_spikespace = 101;
	int32_t* _ptr_array_spikemonitor_i = par__array_spikemonitor_i;
	const int _numi = par_num_i;
	int32_t* _ptr_array_spikemonitor_N = par__array_spikemonitor_N;
	const int _numN = 1;
	double* _ptr_array_spikemonitor_t = par__array_spikemonitor_t;
	const int _numt = par_num_t;
	int32_t* _ptr_array_spikemonitor__source_idx = par__array_spikemonitor__source_idx;
	const int _num_source_idx = 100;
	int32_t* _ptr_array_neurongroup_i = par__array_neurongroup_i;
	const int _num_source_i = 100;
	
	unsigned int total_number = 0;
	total_number = 0;
	for(int i = 0; i < num_blocks; i++)
	{
		total_number += monitor_i[i]->size();
	}
	total_number = 0;
	for(int i = 0; i < num_blocks; i++)
	{
		total_number += monitor_t[i]->size();
	}
	*total = total_number;
	_ptr_array_spikemonitor_N[0] = total_number;
}

__global__ void _copy_spikemonitor_codeobject_kernel(
		int32_t* dev_monitor_i,
		double* dev_monitor_t,
	unsigned int num_blocks
)
{
	using namespace brian;
	unsigned int index = 0;
	index = 0;
	for(int i = 0; i < num_blocks; i++)
	{
		for(int j = 0; j < monitor_i[i]->size(); j++)
		{
			dev_monitor_i[index] = monitor_i[i]->at(j);
			index++;
		}
	}
	index = 0;
	for(int i = 0; i < num_blocks; i++)
	{
		for(int j = 0; j < monitor_t[i]->size(); j++)
		{
			dev_monitor_t[index] = monitor_t[i]->at(j);
			index++;
		}
	}
}

void _copyToHost_spikemonitor_codeobject()
{
	using namespace brian;

	const int _num_source_t = 1;
		const int _numcount = 100;
		const int _num_clock_t = 1;
		const int _num_spikespace = 101;
		int32_t* const _array_spikemonitor_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]);
		const int _numi = dev_dynamic_array_spikemonitor_i.size();
		const int _numN = 1;
		double* const _array_spikemonitor_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]);
		const int _numt = dev_dynamic_array_spikemonitor_t.size();
		const int _num_source_idx = 100;
		const int _num_source_i = 100;

    unsigned int* dev_total;
    cudaMalloc((void**)&dev_total, sizeof(unsigned int));
	_count_spikemonitor_codeobject_kernel<<<1,1>>>(
		dev_array_defaultclock_t,
			dev_array_spikemonitor_count,
			dev_array_neurongroup__spikespace,
			_array_spikemonitor_i,
			_numi,
			dev_array_spikemonitor_N,
			_array_spikemonitor_t,
			_numt,
			dev_array_spikemonitor__source_idx,
			dev_array_neurongroup_i,
		num_blocks(_num_spikespace-1),
		dev_total);
	unsigned int total;
	cudaMemcpy(&total, dev_total, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		dev_dynamic_array_spikemonitor_i.resize(total);
		dev_dynamic_array_spikemonitor_t.resize(total);
	_copy_spikemonitor_codeobject_kernel<<<1,1>>>(
			thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]),
			thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]),
		num_blocks(_num_spikespace-1));
}

void _debugmsg_spikemonitor_codeobject()
{
	using namespace brian;

	const int _num_source_t = 1;
		const int _numcount = 100;
		const int _num_clock_t = 1;
		const int _num_spikespace = 101;
		int32_t* const _array_spikemonitor_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]);
		const int _numi = dev_dynamic_array_spikemonitor_i.size();
		const int _numN = 1;
		double* const _array_spikemonitor_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]);
		const int _numt = dev_dynamic_array_spikemonitor_t.size();
		const int _num_source_idx = 100;
		const int _num_source_i = 100;

	_run_debugmsg_spikemonitor_codeobject_kernel<<<1,1>>>(num_blocks(_num_spikespace-1));
}

