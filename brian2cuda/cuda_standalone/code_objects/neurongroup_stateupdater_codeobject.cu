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
	double* par__array_neurongroup_g,
	double* par__array_defaultclock_t,
	double* par__array_neurongroup_v,
	double* par__array_defaultclock_dt,
	double* par__array_neurongroup_lastspike,
	char* par__array_neurongroup_not_refractory
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	double* _ptr_array_neurongroup_g = par__array_neurongroup_g;
	const int _numg = 100;
	double* _ptr_array_defaultclock_t = par__array_defaultclock_t;
	const int _numt = 1;
	double* _ptr_array_neurongroup_v = par__array_neurongroup_v;
	const int _numv = 100;
	double* _ptr_array_defaultclock_dt = par__array_defaultclock_dt;
	const int _numdt = 1;
	double* _ptr_array_neurongroup_lastspike = par__array_neurongroup_lastspike;
	const int _numlastspike = 100;
	char* _ptr_array_neurongroup_not_refractory = par__array_neurongroup_not_refractory;
	const int _numnot_refractory = 100;

	if(_idx >= 100)
	{
		return;
	}

	
 	

	
	{
  		
  const double dt = _ptr_array_defaultclock_dt[0];
  const double lastspike = _ptr_array_neurongroup_lastspike[_idx];
  const double t = _ptr_array_defaultclock_t[0];
  double g = _ptr_array_neurongroup_g[_idx];
  double v = _ptr_array_neurongroup_v[_idx];
  char not_refractory;
  not_refractory = (t - lastspike) > 0.005;
  const double _g = (((- dt) * g) / 0.005) + g;
  const double _v = ((dt * (((-0.049) + g) - v)) / 0.02) + v;
  g = _g;
  v = _v;
  _ptr_array_neurongroup_not_refractory[_idx] = not_refractory;
  _ptr_array_neurongroup_g[_idx] = g;
  _ptr_array_neurongroup_v[_idx] = v;

	}
}

void _run_neurongroup_stateupdater_codeobject()
{	
	using namespace brian;
	
	///// CONSTANTS ///////////
	const int _numg = 100;
		const int _numt = 1;
		const int _numv = 100;
		const int _numdt = 1;
		const int _numlastspike = 100;
		const int _numnot_refractory = 100;


	kernel_neurongroup_stateupdater_codeobject<<<num_blocks(100),num_threads(100)>>>(
			num_threads(100),
			dev_array_neurongroup_g,
			dev_array_defaultclock_t,
			dev_array_neurongroup_v,
			dev_array_defaultclock_dt,
			dev_array_neurongroup_lastspike,
			dev_array_neurongroup_not_refractory
		);
}


