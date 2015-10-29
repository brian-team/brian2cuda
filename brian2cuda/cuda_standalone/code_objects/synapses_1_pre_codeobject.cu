#include "objects.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include<cmath>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>
#include <stdint.h>
#include "synapses_classes.h"

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





__global__ void kernel_synapses_1_pre_codeobject(
	unsigned int bid_offset,
	unsigned int THREADS_PER_BLOCK,
	double* par__array_synapses_1_lastupdate,
	int par_num_lastupdate,
	int32_t* par__array_spikegeneratorgroup__spikespace,
	double* par__array_neurongroup_g,
	int32_t* par__array_synapses_1__synaptic_pre,
	int par_num__synaptic_pre,
	double* par__array_defaultclock_t,
	double* par__array_synapses_1_w,
	int par_num_w,
	int32_t* par__array_synapses_1__synaptic_post,
	int par_num__postsynaptic_idx,
	int32_t* par__array_synapses_1_N
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x + bid_offset;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	double* _ptr_array_synapses_1_lastupdate = par__array_synapses_1_lastupdate;
	const int _numlastupdate = par_num_lastupdate;
	int32_t* _ptr_array_spikegeneratorgroup__spikespace = par__array_spikegeneratorgroup__spikespace;
	const int _num_spikespace = 101;
	double* _ptr_array_neurongroup_g = par__array_neurongroup_g;
	const int _numg = 100;
	int32_t* _ptr_array_synapses_1__synaptic_pre = par__array_synapses_1__synaptic_pre;
	const int _num_synaptic_pre = par_num__synaptic_pre;
	double* _ptr_array_defaultclock_t = par__array_defaultclock_t;
	const int _numt = 1;
	double* _ptr_array_synapses_1_w = par__array_synapses_1_w;
	const int _numw = par_num_w;
	int32_t* _ptr_array_synapses_1__synaptic_post = par__array_synapses_1__synaptic_post;
	const int _num_postsynaptic_idx = par_num__postsynaptic_idx;
	int32_t* _ptr_array_synapses_1_N = par__array_synapses_1_N;
	const int _numN = 1;

	cudaVector<int32_t>* synapses_queue;
	
	synapses_1_pre.queue->peek(
		&synapses_queue);

 	


	{
	if (!(synapses_1_pre.no_or_const_delay_mode))
	{
		int size = synapses_queue[bid].size();
		for(int j = tid; j < size; j+=THREADS_PER_BLOCK)
		{
			int32_t _idx = synapses_queue[bid].at(j);
	
   			
   const int32_t _postsynaptic_idx = _ptr_array_synapses_1__synaptic_post[_idx];
   const double w = _ptr_array_synapses_1_w[_idx];
   const double t = _ptr_array_defaultclock_t[0];
   double g = _ptr_array_neurongroup_g[_postsynaptic_idx];
   double lastupdate;
   g += w * 0.001;
   lastupdate = t;
   _ptr_array_synapses_1_lastupdate[_idx] = lastupdate;
   _ptr_array_neurongroup_g[_postsynaptic_idx] = g;

		}
	}
	else
	{
		if(bid != 0)
			return;
		//no or const delay mode
		for(int j = 0; j < _num_spikespace; j++)
		{
			int32_t spiking_neuron = _ptr_array_spikegeneratorgroup__spikespace[j];
			if(spiking_neuron == -1)
			{
				break;
			}
			for(int i = tid; i < synapses_1_pre_size_by_pre[spiking_neuron]; i+= THREADS_PER_BLOCK)
			{
				int32_t _idx = synapses_1_pre_synapses_id_by_pre[spiking_neuron][i];
			
    				
    const int32_t _postsynaptic_idx = _ptr_array_synapses_1__synaptic_post[_idx];
    const double w = _ptr_array_synapses_1_w[_idx];
    const double t = _ptr_array_defaultclock_t[0];
    double g = _ptr_array_neurongroup_g[_postsynaptic_idx];
    double lastupdate;
    g += w * 0.001;
    lastupdate = t;
    _ptr_array_synapses_1_lastupdate[_idx] = lastupdate;
    _ptr_array_neurongroup_g[_postsynaptic_idx] = g;

			}
			__syncthreads();
		}
	}
	}
}


void _run_synapses_1_pre_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	double* const _array_synapses_1_lastupdate = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_lastupdate[0]);
		const int _numlastupdate = dev_dynamic_array_synapses_1_lastupdate.size();
		const int _num_spikespace = 101;
		const int _numg = 100;
		int32_t* const _array_synapses_1__synaptic_pre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]);
		const int _num_synaptic_pre = dev_dynamic_array_synapses_1__synaptic_pre.size();
		const int _numt = 1;
		double* const _array_synapses_1_w = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_w[0]);
		const int _numw = dev_dynamic_array_synapses_1_w.size();
		int32_t* const _array_synapses_1__synaptic_post = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]);
		const int _num_postsynaptic_idx = dev_dynamic_array_synapses_1__synaptic_post.size();
		const int _numN = 1;


	kernel_synapses_1_pre_codeobject<<<num_parallel_blocks,1>>>(
		0,
		1,
		_array_synapses_1_lastupdate,
			_numlastupdate,
			dev_array_spikegeneratorgroup__spikespace,
			dev_array_neurongroup_g,
			_array_synapses_1__synaptic_pre,
			_num_synaptic_pre,
			dev_array_defaultclock_t,
			_array_synapses_1_w,
			_numw,
			_array_synapses_1__synaptic_post,
			_num_postsynaptic_idx,
			dev_array_synapses_1_N
	);
}

void _debugmsg_synapses_1_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << dev_dynamic_array_synapses_1__synaptic_pre.size() << endl;
}

