#include "objects.h"
#include "code_objects/synapses_post_codeobject.h"
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





__global__ void kernel_synapses_post_codeobject(
	unsigned int bid_offset,
	unsigned int THREADS_PER_BLOCK,
	double* par__array_synapses_Apre,
	int par_num_Apre,
	double* par__array_synapses_lastupdate,
	int par_num_lastupdate,
	double* par__array_synapses_Apost,
	int par_num_Apost,
	int32_t* par__array_neurongroup__spikespace,
	int32_t* par__array_synapses_N,
	double* par__array_defaultclock_t,
	double* par__array_synapses_w,
	int par_num_w,
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
	int32_t* _ptr_array_synapses_N = par__array_synapses_N;
	const int _numN = 1;
	double* _ptr_array_defaultclock_t = par__array_defaultclock_t;
	const int _numt = 1;
	double* _ptr_array_synapses_w = par__array_synapses_w;
	const int _numw = par_num_w;
	int32_t* _ptr_array_synapses__synaptic_pre = par__array_synapses__synaptic_pre;
	const int _num_synaptic_pre = par_num__synaptic_pre;

	cudaVector<int32_t>* synapses_queue;
	
	synapses_post.queue->peek(
		&synapses_queue);

 	


	{
	if (!(synapses_post.no_or_const_delay_mode))
	{
		int size = synapses_queue[bid].size();
		for(int j = tid; j < size; j+=THREADS_PER_BLOCK)
		{
			int32_t _idx = synapses_queue[bid].at(j);
	
   			
   double Apre = _ptr_array_synapses_Apre[_idx];
   double lastupdate = _ptr_array_synapses_lastupdate[_idx];
   double Apost = _ptr_array_synapses_Apost[_idx];
   const double t = _ptr_array_defaultclock_t[0];
   double w = _ptr_array_synapses_w[_idx];
   Apre *= exp((lastupdate - t) / 0.02);
   Apost *= exp((lastupdate - t) / 0.02);
   Apost += (-0.03402000000000001);
   w += Apre;
   lastupdate = t;
   _ptr_array_synapses_Apre[_idx] = Apre;
   _ptr_array_synapses_lastupdate[_idx] = lastupdate;
   _ptr_array_synapses_Apost[_idx] = Apost;
   _ptr_array_synapses_w[_idx] = w;

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
			for(int i = tid; i < synapses_post_size_by_pre[spiking_neuron]; i+= THREADS_PER_BLOCK)
			{
				int32_t _idx = synapses_post_synapses_id_by_pre[spiking_neuron][i];
			
    				
    double Apre = _ptr_array_synapses_Apre[_idx];
    double lastupdate = _ptr_array_synapses_lastupdate[_idx];
    double Apost = _ptr_array_synapses_Apost[_idx];
    const double t = _ptr_array_defaultclock_t[0];
    double w = _ptr_array_synapses_w[_idx];
    Apre *= exp((lastupdate - t) / 0.02);
    Apost *= exp((lastupdate - t) / 0.02);
    Apost += (-0.03402000000000001);
    w += Apre;
    lastupdate = t;
    _ptr_array_synapses_Apre[_idx] = Apre;
    _ptr_array_synapses_lastupdate[_idx] = lastupdate;
    _ptr_array_synapses_Apost[_idx] = Apost;
    _ptr_array_synapses_w[_idx] = w;

			}
			__syncthreads();
		}
	}
	}
}


void _run_synapses_post_codeobject()
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
		const int _numN = 1;
		const int _numt = 1;
		double* const _array_synapses_w = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_w[0]);
		const int _numw = dev_dynamic_array_synapses_w.size();
		int32_t* const _array_synapses__synaptic_pre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]);
		const int _num_synaptic_pre = dev_dynamic_array_synapses__synaptic_pre.size();


	kernel_synapses_post_codeobject<<<num_parallel_blocks,max_threads_per_block>>>(
		0,
		max_threads_per_block,
		_array_synapses_Apre,
			_numApre,
			_array_synapses_lastupdate,
			_numlastupdate,
			_array_synapses_Apost,
			_numApost,
			dev_array_neurongroup__spikespace,
			dev_array_synapses_N,
			dev_array_defaultclock_t,
			_array_synapses_w,
			_numw,
			_array_synapses__synaptic_pre,
			_num_synaptic_pre
	);
}

void _debugmsg_synapses_post_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << dev_dynamic_array_synapses__synaptic_pre.size() << endl;
}

