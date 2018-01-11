////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cu_file() %}
{#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <ctime>


__global__ void _run_{{codeobj_name}}_advance_kernel()
{
	using namespace brian;
	unsigned int tid = threadIdx.x;
	{{owner.name}}.queue->advance(
		tid);
}

__global__ void
{% if launch_bounds or syn_launch_bounds %}
__launch_bounds__(1024, {{sm_multiplier}})
{% endif %}
_run_{{codeobj_name}}_push_kernel(
    unsigned int num_parallel_blocks,
	unsigned int _num_blocks,
	unsigned int _num_threads,
	int32_t* {{_eventspace}})
{
	// apperently this is not always true and that is why _num_threads is passed as function argument
	// if this assert never fails, we could remove the _num_threads form the argument list
	assert(blockDim.x == _num_threads);

	using namespace brian;

	int bid = blockIdx.x;
	int tid = threadIdx.x;

    int post_neuron_bid = bid % num_parallel_blocks;
    int pre_neuron_idx = bid / num_parallel_blocks;

    int32_t spiking_neuron = {{_eventspace}}[pre_neuron_idx];
    assert(spiking_neuron != -1);

    // push to spikequeue if spiking_neuron is in sources of current SynapticPathway
    if({{owner.name}}.spikes_start <= spiking_neuron && spiking_neuron < {{owner.name}}.spikes_stop)
    {
        {{owner.name}}.queue->push(
            post_neuron_bid,
            tid,
            _num_threads,
            spiking_neuron - {{owner.name}}.spikes_start);
    }
}

void _run_{{codeobj_name}}()
{
	using namespace brian;

	{% if profile and profile == 'blocking'%}
	{{codeobj_name}}_timer_start = std::clock();
	{% elif profile %}
	cudaEventRecord({{codeobj_name}}_timer_start);
	{% endif %}

	///// CONSTANTS /////
	%CONSTANTS%

	{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}
	if ({{owner.name}}_scalar_delay)
	{
		//TODO: check int / unsigned int of all vars here and in objects pathway
		unsigned int num_eventspaces = dev{{_eventspace}}.size();
		{{owner.name}}_eventspace_idx = (current_idx{{_eventspace}} - {{owner.name}}_delay + num_eventspaces) % num_eventspaces;

		//////////////////////////////////////////////
		//// No pushing in no_or_const_delay_mode ////
		//////////////////////////////////////////////
	}
	else if ({{owner.name}}_max_size > 0)
	{

		// get the number of spiking neurons
		int32_t num_spiking_neurons;
		cudaMemcpy(&num_spiking_neurons,
				dev{{_eventspace}}[current_idx{{_eventspace}}] + _num_{{owner.event}}space - 1,
				sizeof(int32_t), cudaMemcpyDeviceToHost);

		// advance spike queues
		_run_{{codeobj_name}}_advance_kernel<<<1, num_parallel_blocks>>>();

		cudaError_t status = cudaGetLastError();
		if (status != cudaSuccess)
		{
			printf("ERROR launching _run_{{codeobj_name}}_advance_kernel in %s:%d %s\n",
					__FILE__, __LINE__, cudaGetErrorString(status));
			_dealloc_arrays();
			exit(status);
		}

	    static int num_threads, num_blocks;
	    static bool first_run = true;
	    if (first_run)
	    {

		    // We don't need more then max(num_synapses) threads per block.
		    num_threads = {{owner.name}}_max_size;
		    if (num_threads > max_threads_per_block)
		    {
		    	num_threads = max_threads_per_block;
		    }

	    	// calculate theoretical occupancy
	    	int max_active_blocks;
	    	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
					_run_{{codeobj_name}}_push_kernel, num_threads, 0);

	    	float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
	    	                  (float)(max_threads_per_sm / num_threads_per_warp);

            // check if we have enough ressources to call kernel with given
            // number of blocks and threads
	    	struct cudaFuncAttributes funcAttrib;
	    	cudaFuncGetAttributes(&funcAttrib, _run_{{codeobj_name}}_push_kernel);
	    	if (num_threads > funcAttrib.maxThreadsPerBlock)
	    	{
	    		// use the max num_threads before launch failure
	    		num_threads = funcAttrib.maxThreadsPerBlock;
	    		printf("WARNING Not enough ressources available to call "
                       "_run_{{codeobj_name}}_push_kernel "
                       "with maximum possible threads per block (%u). "
                       "Reducing num_threads to %u. (Kernel needs %i "
                       "registers per block, %i bytes of "
                       "statically-allocated shared memory per block, %i "
                       "bytes of local memory per thread and a total of %i "
                       "bytes of user-allocated constant memory)\n",
                       max_threads_per_block, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes);
	    	}
	    	else
	    	{
                printf("INFO _run_{{codeobj_name}}_push_kernel\n"
                       "\t%u blocks per spiking neuron\n"
                       "\t%u threads\n"
                       "\t%i registers per block\n"
                       "\t%i bytes statically-allocated shared memory per block\n"
                       "\t%i bytes local memory per thread\n"
                       "\t%i bytes user-allocated constant memory\n"
                       {% if calc_occupancy %}
                       "\t%.3f theoretical occupancy\n",
                       {% else %}
                       "",
                       {% endif %}
                       num_parallel_blocks, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes{% if calc_occupancy %}, occupancy{% endif %});
	    	}
	    	first_run = false;
	    }


		if (num_spiking_neurons > 0)
		{
			num_blocks = num_parallel_blocks * num_spiking_neurons;

			_run_{{codeobj_name}}_push_kernel<<<num_blocks, num_threads>>>(
					num_parallel_blocks,
					num_blocks,
					num_threads,
					dev{{_eventspace}}[current_idx{{_eventspace}}]);

			status = cudaGetLastError();
			if (status != cudaSuccess)
			{
				printf("ERROR launching _run_{{codeobj_name}}_push_kernel in %s:%d %s\n",
						__FILE__, __LINE__, cudaGetErrorString(status));
				_dealloc_arrays();
				exit(status);
			}
		}
	}

	{% if profile and profile == 'blocking'%}
	cudaDeviceSynchronize();
	{{codeobj_name}}_timer_stop = std::clock();
	{% elif profile %}
	cudaEventRecord({{codeobj_name}}_timer_stop);
	{% endif %}
}
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// HEADER FILE ///////////////////////////////////////////////////////////

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "objects.h"

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
