#include "objects.h"
#include "code_objects/synapses_synapses_create_codeobject.h"
#include<cmath>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>
#include <stdint.h>
#include "synapses_classes.h"

#include<iostream>

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
 	
 #define _rand(vectorisation_idx) (_array_synapses_synapses_create_codeobject_rand[vectorisation_idx])

}





void _run_synapses_synapses_create_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	int32_t* const _array_synapses_N_incoming = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_N_incoming[0]);
		const int _numN_incoming = dev_dynamic_array_synapses_N_incoming.size();
		const int _numN = 1;
		const int _num_all_post = 100;
		int32_t* const _array_synapses__synaptic_post = thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]);
		const int _num_synaptic_post = dev_dynamic_array_synapses__synaptic_post.size();
		int32_t* const _array_synapses_N_outgoing = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_N_outgoing[0]);
		const int _numN_outgoing = dev_dynamic_array_synapses_N_outgoing.size();
		const int _num_all_pre = 100;
		int32_t* const _array_synapses__synaptic_pre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]);
		const int _num_synaptic_pre = dev_dynamic_array_synapses__synaptic_pre.size();


	srand(time(0));
 	


	//these two vectors just cache everything on the CPU-side
	//data is copied to GPU at the end
	thrust::host_vector<int32_t> temp_synaptic_post;
	thrust::host_vector<int32_t> temp_synaptic_pre;

 	
 int32_t * __restrict _ptr_array_synapses_N_incoming = _array_synapses_N_incoming;
 int32_t * __restrict _ptr_array_synapses_N = _array_synapses_N;
 int32_t * __restrict _ptr_array_neurongroup_i = _array_neurongroup_i;
 int32_t * __restrict _ptr_array_synapses__synaptic_post = _array_synapses__synaptic_post;
 int32_t * __restrict _ptr_array_synapses_N_outgoing = _array_synapses_N_outgoing;
 int32_t * __restrict _ptr_array_synapses__synaptic_pre = _array_synapses__synaptic_pre;


	unsigned int syn_id = 0;
	for(int _i = 0; _i < _num_all_pre; _i++)
	{
		for(int _j = 0; _j < _num_all_post; _j++)
		{
		    const int _vectorisation_idx = _i*_num_all_post  + _j;
   			
   const int32_t _all_post = _ptr_array_neurongroup_i[_j];
   const int32_t _all_pre = _ptr_array_neurongroup_i[_i];
   const int32_t _pre_idx = _all_pre;
   const int32_t _post_idx = _all_post;
   const char _cond = true;
   const int32_t _n = 1;
   const double _p = 1.0;

			// Add to buffer
			if(_cond)
			{
				if (_p != 1.0)
				{
					float r = rand()/(float)RAND_MAX;
					if (r >= _p)
					{
						continue;
					}
				}
				for (int _repetition = 0; _repetition < _n; _repetition++)
				{
					temp_synaptic_pre.push_back(_pre_idx);
					temp_synaptic_post.push_back(_post_idx);
					syn_id++;
				}
			}
		}
	}

	dev_dynamic_array_synapses__synaptic_pre = temp_synaptic_pre;
	dev_dynamic_array_synapses__synaptic_post = temp_synaptic_post;
	
	// now we need to resize all registered variables
	const int32_t newsize = dev_dynamic_array_synapses__synaptic_pre.size();
	dev_dynamic_array_synapses__synaptic_post.resize(newsize);
	_dynamic_array_synapses__synaptic_post.resize(newsize);
	dev_dynamic_array_synapses__synaptic_pre.resize(newsize);
	_dynamic_array_synapses__synaptic_pre.resize(newsize);
	dev_dynamic_array_synapses_Apost.resize(newsize);
	_dynamic_array_synapses_Apost.resize(newsize);
	dev_dynamic_array_synapses_Apre.resize(newsize);
	_dynamic_array_synapses_Apre.resize(newsize);
	dev_dynamic_array_synapses_pre_delay.resize(newsize);
	_dynamic_array_synapses_pre_delay.resize(newsize);
	dev_dynamic_array_synapses_post_delay.resize(newsize);
	_dynamic_array_synapses_post_delay.resize(newsize);
	dev_dynamic_array_synapses_lastupdate.resize(newsize);
	_dynamic_array_synapses_lastupdate.resize(newsize);
	dev_dynamic_array_synapses_w.resize(newsize);
	_dynamic_array_synapses_w.resize(newsize);
	// Also update the total number of synapses
        _ptr_array_synapses_N[0] = newsize;

}


