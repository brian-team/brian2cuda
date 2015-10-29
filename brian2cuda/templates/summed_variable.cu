{% extends 'common_group.cu' %}

{% block extra_device_helper %}

#define MEM_PER_THREAD (sizeof(double))

	//atomic add is currently not supported natively for double values
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
{% endblock %}

{% block kernel %}
{# USES_VARIABLES { _synaptic_post, _synaptic_pre, N_post, N_pre } #}
__global__ void kernel_{{codeobj_name}}(
	unsigned int num_blocks_per_neuron,
	unsigned int num_threads,
	unsigned int syn_N,
	%DEVICE_PARAMETERS%
	)
{
    {% set _target_var_array = get_array_name(_target_var) %}
    using namespace brian;

	extern __shared__ char shared_mem[];
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int neuron_id = bid/num_blocks_per_neuron;
	unsigned int num_block_for_neuron = bid % num_blocks_per_neuron;
	unsigned int _idx = num_block_for_neuron*num_threads + tid;
	double* shared_double_mem = (double*) shared_mem;
	%KERNEL_VARIABLES%

	//// MAIN CODE ////////////
	if(_idx < 0 || _idx >=  syn_N)
	{
		return;
	}
	double _local_sum = 0.0;
	shared_double_mem[tid] = 0.0;
	if(tid == 0 && num_block_for_neuron == 0)
	{
	    {{_target_var_array}}[neuron_id] = 0.0;
	}
	
	// vector code
    int _vectorisation_idx = bid;
    int post_id = {{_synaptic_post}}[_idx];
    
	if(post_id == neuron_id)
    {
        {{vector_code|autoindent}}
		shared_double_mem[tid] += _synaptic_var;
	}
	
	__syncthreads();
	if(tid != 0)
	{
		return;
	}
	for(int _idx = 0; _idx < num_threads; _idx++)
	{
		_local_sum += shared_double_mem[_idx];
	}
	atomicAddDouble(&{{_target_var_array}}[neuron_id], _local_sum);
}	
{% endblock %}

{% block kernel_call %}
	unsigned int syn_N = _num_postsynaptic_idx;
	int _num_blocks = num_blocks(syn_N) * N_post;
	unsigned int _num_threads = num_threads(syn_N);
	kernel_{{codeobj_name}}<<<_num_blocks, _num_threads, _num_threads*MEM_PER_THREAD>>>(
			_num_blocks,
			_num_threads,
			syn_N,
			%HOST_PARAMETERS%
		);
{% endblock %}
