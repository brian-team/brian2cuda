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
	int* block_time,
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


	// use one shared spike counter per block --> parallel atomicAdd on shared memory
	// and afterwards only one atomicAdd on global memory per block
	__shared__ int32_t spike_counter_block;

	// not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	// scalar code
 	


	_ptr_array_neurongroup__spikespace[_idx] = -1;

	if(tid == 0)
	{
		//init spike counter per block with 0
		spike_counter_block = 0;

		if(bid == 0)
		{
			//init global spike counter with 0
			_ptr_array_neurongroup__spikespace[10000] = 0;
		}
	}
	__syncthreads();

 	
 const int32_t i = _ptr_array_neurongroup_i[_idx];
 const double _cond = true;//(i / 2) == ((i + 1) / 2);


	int32_t spike_index;

 	__syncthreads();
 	clock_t block_start = clock();
	if(_cond) {
		spike_index = atomicAdd(&spike_counter_block, 1);
	}
 	clock_t block_end = clock();
	block_time[tid] = (int)(block_end - block_start);


	if(_cond) {
		_ptr_array_neurongroup__spikespace[bid * THREADS_PER_BLOCK + spike_index] = _idx;
	}

	__syncthreads();

 	clock_t glob_start = clock();
	if (tid == 0) {
		atomicAdd(&_ptr_array_neurongroup__spikespace[10000], spike_counter_block);
	}
 	clock_t glob_end = clock();
	glob_time[bid] = (int)(glob_end - glob_start);

}

void _run_neurongroup_thresholder_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	const int _numi = 10000;
		const int _num_spikespace = 10001;
		const int _numt = 1;

	int* h_block_time = new int[num_threads(10000)];
	int* h_glob_time = new int[num_blocks(10000)];
	int* d_block_time;
	int* d_glob_time;

//	for (int i=0; i != num_threads(10000); ++i){
//		std::cout << "BEFORE TID " << i << " took " << h_block_time[i] << " clocks." << std::endl;
//	}

//	cudaMemcpy(d_block_time, h_block_time, num_threads(10000)*sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_glob_time, h_glob_time, num_blocks(10000)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc( (void**) &d_block_time, num_threads(10000)*sizeof(int));
	cudaMalloc( (void**) &d_glob_time, num_blocks(10000)*sizeof(int));

kernel_neurongroup_thresholder_codeobject<<<num_blocks(10000), num_threads(10000)>>>(
		num_threads(10000),
		dev_array_neurongroup_i,
			dev_array_neurongroup__spikespace,
			dev_array_defaultclock_t,
			d_block_time,
			d_glob_time
	);

	cudaMemcpy(h_block_time, d_block_time, num_threads(10000)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_glob_time, d_glob_time, num_blocks(10000)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_block_time);
	cudaFree(d_glob_time);

	int block_sum = 0;
	int glob_sum = 0;
	for (int i=0; i != num_threads(10000); ++i){
		std::cout << "TID " << i << " took " << h_block_time[i] << " clocks." << std::endl;
		block_sum += h_block_time[i];
	}
	for (int i=0; i != num_blocks(10000); ++i){
		glob_sum += h_glob_time[i];
	}

	double block_time = (double)(block_sum)/980;
	double glob_time = (double)(glob_sum)/980;
	
	std::cout << "Average time per shared atomicAdd:\n" << block_time/num_threads(10000) << "us\n\n" << std::endl;
	std::cout << "Total time in shared atomicAdd:\n" << block_time << "us\n\n" << std::endl;
	std::cout << "Average time per global atomicAdd:\n" << glob_time/num_blocks(10000) << "us\n\n" << std::endl;
	std::cout << "Total time in global atomicAdd:\n" << glob_time << "us\n\n" << std::endl;
	std::cout << "Total time in all atomicAdd:\n" << (block_time + glob_time) << "us\n\n" << std::endl;

}


