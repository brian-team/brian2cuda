#include "objects.h"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	int num_blocks(int objects){
		return ceil(objects / (double)brian::max_threads_per_block);
	}
	int num_threads(int objects){
		return brian::max_threads_per_block;
	}
 	

}

__global__ void _kernel_synapses_1_group_variable_set_conditional_codeobject(
	unsigned int _N,
	unsigned int THREADS_PER_BLOCK,
	double* par__array_synapses_1_lastupdate,
	int par_num_lastupdate,
	double* par__array_defaultclock_t,
	int32_t* par__array_synapses_1_N
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;

	double* _ptr_array_synapses_1_lastupdate = par__array_synapses_1_lastupdate;
	const int _numlastupdate = par_num_lastupdate;
	double* _ptr_array_defaultclock_t = par__array_defaultclock_t;
	const int _numt = 1;
	int32_t* _ptr_array_synapses_1_N = par__array_synapses_1_N;
	const int _numN = 1;

	if(_idx < 0 || _idx >= _N)
	{
		return;
	}

 	

 	


 	
 const char _cond = true;

	if (_cond)
	{
                
        const double t = _ptr_array_defaultclock_t[0];
        double lastupdate;
        lastupdate = t;
        _ptr_array_synapses_1_lastupdate[_idx] = lastupdate;

    }
}

////// HASH DEFINES ///////



void _run_synapses_1_group_variable_set_conditional_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _N = _array_synapses_1_N[0];
	double* const _array_synapses_1_lastupdate = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_lastupdate[0]);
		const int _numlastupdate = dev_dynamic_array_synapses_1_lastupdate.size();
		const int _numt = 1;
		const int _numN = 1;

	_kernel_synapses_1_group_variable_set_conditional_codeobject<<<num_blocks(_N),num_threads(_N)>>>(
		_N,
		num_threads(_N),
		_array_synapses_1_lastupdate,
			_numlastupdate,
			dev_array_defaultclock_t,
			dev_array_synapses_1_N
	);
}


