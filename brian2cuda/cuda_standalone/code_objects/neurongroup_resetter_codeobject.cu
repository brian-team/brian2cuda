#include "objects.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
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




__global__ void kernel_neurongroup_resetter_codeobject(
	unsigned int THREADS_PER_BLOCK,
	int32_t* par__array_neurongroup__spikespace,
	double* par__array_neurongroup_v
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	int32_t* _ptr_array_neurongroup__spikespace = par__array_neurongroup__spikespace;
	const int _num_spikespace = 101;
	double* _ptr_array_neurongroup_v = par__array_neurongroup_v;
	const int _numv = 100;

	if(_idx >= 100)
	{
		return;
	}


	const int32_t *_spikes = _ptr_array_neurongroup__spikespace;
	const int32_t _num_spikes = _ptr_array_neurongroup__spikespace[100];

	//// MAIN CODE ////////////	
	// scalar code
 	

    
	//get spiking neuron_id
	_idx = _spikes[_idx];
	if(_idx != -1)
	{
  		
  double v;
  v = (-0.06);
  _ptr_array_neurongroup_v[_idx] = v;

	}
}

void _run_neurongroup_resetter_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	const int _num_spikespace = 101;
		const int _numv = 100;


	kernel_neurongroup_resetter_codeobject<<<num_blocks(100),num_threads(100)>>>(
			num_threads(100),
			dev_array_neurongroup__spikespace,
			dev_array_neurongroup_v
		);
}


