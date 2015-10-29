#include "objects.h"
#include "code_objects/synapses_pre_codeobject.h"
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





__global__ void kernel_synapses_pre_codeobject(
	unsigned int bid_offset,
	unsigned int THREADS_PER_BLOCK,
	double* par__array_synapses_Apre,
	int par_num_Apre,
	double* par__array_synapses_lastupdate,
	int par_num_lastupdate,
	double* par__array_synapses_Apost,
	int par_num_Apost,
	int32_t* par__array_neurongroup__spikespace,
	double* par__array_neurongroup_g,
	int32_t* par__array_synapses_N,
	double* par__array_defaultclock_t,
	double* par__array_synapses_w,
	int par_num_w,
	int32_t* par__array_synapses__synaptic_post,
	int par_num__postsynaptic_idx,
	int32_t* par__array_synapses__synaptic_pre,
	int par_num__synaptic_pre
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x + bid_offset;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	double* _ptr_array_synapses_Apre = par__array_synapses_Apre;
	const int _numApre = par_num_Apre;
	double* _ptr_array_synapses_lastupdate = par__array_synapses_lastupdate;
	const int _numlastupdate = par_num_lastupdate;
	double* _ptr_array_synapses_Apost = par__array_synapses_Apost;
	const int _numApost = par_num_Apost;
	int32_t* _ptr_array_neurongroup__spikespace = par__array_neurongroup__spikespace;
	const int _num_spikespace = 101;
	double* _ptr_array_neurongroup_g = par__array_neurongroup_g;
	const int _numg = 100;
	int32_t* _ptr_array_synapses_N = par__array_synapses_N;
	const int _numN = 1;
	double* _ptr_array_defaultclock_t = par__array_defaultclock_t;
	const int _numt = 1;
	double* _ptr_array_synapses_w = par__array_synapses_w;
	const int _numw = par_num_w;
	int32_t* _ptr_array_synapses__synaptic_post = par__array_synapses__synaptic_post;
	const int _num_postsynaptic_idx = par_num__postsynaptic_idx;
	int32_t* _ptr_array_synapses__synaptic_pre = par__array_synapses__synaptic_pre;
	const int _num_synaptic_pre = par_num__synaptic_pre;

	cudaVector<int32_t>* synapses_queue;
	
	synapses_pre.queue->peek(
		&synapses_queue);

 	


	{
	if (!(synapses_pre.no_or_const_delay_mode))
	{
		int size = synapses_queue[bid].size();
		for(int j = tid; j < size; j+=THREADS_PER_BLOCK)
		{
			int32_t _idx = synapses_queue[bid].at(j);
	
   			
   const int32_t _postsynaptic_idx = _ptr_array_synapses__synaptic_post[_idx];
   double Apre = _ptr_array_synapses_Apre[_idx];
   double lastupdate = _ptr_array_synapses_lastupdate[_idx];
   double Apost = _ptr_array_synapses_Apost[_idx];
   double g = _ptr_array_neurongroup_g[_postsynaptic_idx];
   const double t = _ptr_array_defaultclock_t[0];
   double w = _ptr_array_synapses_w[_idx];
   Apre *= exp((lastupdate - t) / 0.02);
   Apost *= exp((lastupdate - t) / 0.02);
   g += w * 0.001;
   Apre += 0.032400000000000005;
   w += Apost;
   lastupdate = t;
   _ptr_array_synapses_Apre[_idx] = Apre;
   _ptr_array_synapses_lastupdate[_idx] = lastupdate;
   _ptr_array_synapses_Apost[_idx] = Apost;
   _ptr_array_synapses_w[_idx] = w;
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
			int32_t spiking_neuron = _ptr_array_neurongroup__spikespace[j];
			if(spiking_neuron == -1)
			{
				break;
			}
			for(int i = tid; i < synapses_pre_size_by_pre[spiking_neuron]; i+= THREADS_PER_BLOCK)
			{
				int32_t _idx = synapses_pre_synapses_id_by_pre[spiking_neuron][i];
			
    				
    const int32_t _postsynaptic_idx = _ptr_array_synapses__synaptic_post[_idx];
    double Apre = _ptr_array_synapses_Apre[_idx];
    double lastupdate = _ptr_array_synapses_lastupdate[_idx];
    double Apost = _ptr_array_synapses_Apost[_idx];
    double g = _ptr_array_neurongroup_g[_postsynaptic_idx];
    const double t = _ptr_array_defaultclock_t[0];
    double w = _ptr_array_synapses_w[_idx];
    Apre *= exp((lastupdate - t) / 0.02);
    Apost *= exp((lastupdate - t) / 0.02);
    g += w * 0.001;
    Apre += 0.032400000000000005;
    w += Apost;
    lastupdate = t;
    _ptr_array_synapses_Apre[_idx] = Apre;
    _ptr_array_synapses_lastupdate[_idx] = lastupdate;
    _ptr_array_synapses_Apost[_idx] = Apost;
    _ptr_array_synapses_w[_idx] = w;
    _ptr_array_neurongroup_g[_postsynaptic_idx] = g;

			}
			__syncthreads();
		}
	}
	}
}


void _run_synapses_pre_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	double* const _array_synapses_Apre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_Apre[0]);
		const int _numApre = dev_dynamic_array_synapses_Apre.size();
		double* const _array_synapses_lastupdate = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_lastupdate[0]);
		const int _numlastupdate = dev_dynamic_array_synapses_lastupdate.size();
		double* const _array_synapses_Apost = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_Apost[0]);
		const int _numApost = dev_dynamic_array_synapses_Apost.size();
		const int _num_spikespace = 101;
		const int _numg = 100;
		const int _numN = 1;
		const int _numt = 1;
		double* const _array_synapses_w = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_w[0]);
		const int _numw = dev_dynamic_array_synapses_w.size();
		int32_t* const _array_synapses__synaptic_post = thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]);
		const int _num_postsynaptic_idx = dev_dynamic_array_synapses__synaptic_post.size();
		int32_t* const _array_synapses__synaptic_pre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]);
		const int _num_synaptic_pre = dev_dynamic_array_synapses__synaptic_pre.size();


	kernel_synapses_pre_codeobject<<<num_parallel_blocks,1>>>(
		0,
		1,
		_array_synapses_Apre,
			_numApre,
			_array_synapses_lastupdate,
			_numlastupdate,
			_array_synapses_Apost,
			_numApost,
			dev_array_neurongroup__spikespace,
			dev_array_neurongroup_g,
			dev_array_synapses_N,
			dev_array_defaultclock_t,
			_array_synapses_w,
			_numw,
			_array_synapses__synaptic_post,
			_num_postsynaptic_idx,
			_array_synapses__synaptic_pre,
			_num_synaptic_pre
	);
}

void _debugmsg_synapses_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << dev_dynamic_array_synapses__synaptic_pre.size() << endl;
}

