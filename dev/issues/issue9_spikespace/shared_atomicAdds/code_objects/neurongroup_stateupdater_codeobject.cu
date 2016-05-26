#include "objects.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
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




__global__ void kernel_neurongroup_stateupdater_codeobject(
	unsigned int THREADS_PER_BLOCK,
	int dummy
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	

	if(_idx >= 10000)
	{
		return;
	}

	
 	

	
	{
  		

	}
}

void _run_neurongroup_stateupdater_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	


	kernel_neurongroup_stateupdater_codeobject<<<num_blocks(10000),num_threads(10000)>>>(
			num_threads(10000),
			0
		);
}


