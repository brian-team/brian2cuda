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
	int32_t* par__array_neurongroup_i,
	int32_t* par__array_neurongroup__spikespace,
	double* par__array_defaultclock_t,
	int* glob_time
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	int32_t* _ptr_array_neurongroup_i = par__array_neurongroup_i;
	const int _numi = 10000;
	int32_t* _ptr_array_neurongroup__spikespace = par__array_neurongroup__spikespace;
	const int _num_spikespace = 10001;
	double* _ptr_array_defaultclock_t = par__array_defaultclock_t;
	const int _numt = 1;

	if(_idx >= 10000)
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
		_ptr_array_neurongroup__spikespace[10000] = 0;
	}
	__syncthreads();

 	
 const int32_t i = _ptr_array_neurongroup_i[_idx];
 const double _cond = true;//(i / 2) == ((i + 1) / 2);

 	__syncthreads();
 	clock_t start = clock();
	if(_cond) {
		int32_t spike_index = atomicAdd(&_ptr_array_neurongroup__spikespace[10000], 1);
		_ptr_array_neurongroup__spikespace[spike_index] = _idx;
	}
 	clock_t end = clock();
	glob_time[tid] = (int)(end - start);

}

void _run_neurongroup_thresholder_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	const int _numi = 10000;
		const int _num_spikespace = 10001;
		const int _numt = 1;

	int* h_glob_time = new int[num_threads(10000)];
	int* d_glob_time;

//	for (int i=0; i != num_threads(10000); ++i){
//		std::cout << "BEFORE TID " << i << " took " << h_block_time[i] << " clocks." << std::endl;
//	}

//	cudaMemcpy(d_block_time, h_block_time, num_threads(10000)*sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_glob_time, h_glob_time, num_blocks(10000)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc( (void**) &d_glob_time, num_threads(10000)*sizeof(int));

kernel_neurongroup_thresholder_codeobject<<<num_blocks(10000), num_threads(10000)>>>(
		num_threads(10000),
		dev_array_neurongroup_i,
			dev_array_neurongroup__spikespace,
			dev_array_defaultclock_t,
			d_glob_time
	);
	cudaMemcpy(h_glob_time, d_glob_time, num_threads(10000)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_glob_time);

	int glob_sum = 0;
	for (int i=0; i != num_threads(10000); ++i){
		std::cout << "TID " << i << " took " << h_glob_time[i] << " clocks." << std::endl;
		glob_sum += h_glob_time[i];
	}

	double glob_time = (double)(glob_sum)/980;
	
	std::cout << "Average time per global atomicAdd:\n" << glob_time/num_threads(10000) << "ms\n\n" << std::endl;
	std::cout << "Total time in global atomicAdd:\n" << glob_time << "ms\n\n" << std::endl;
}


