{% extends 'common_group.cu' %}

{# USES_VARIABLES { N } #}
{% set _target_var_array = get_array_name(_target_var) %}
{% set _index_array = get_array_name(_index_var) %}


{% block extra_device_helper %}
#define MEM_PER_THREAD (sizeof(double))
{% endblock %}


{% block kernel %}
__global__ void kernel_{{codeobj_name}}(
    int num_blocks_per_neuron,
    int num_threads,
    int syn_N,
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    using namespace brian;

    extern __shared__ char shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int neuron_id = bid/num_blocks_per_neuron;
    int num_block_for_neuron = bid % num_blocks_per_neuron;
    int _idx = num_block_for_neuron*num_threads + tid;
    double* shared_double_mem = (double*) shared_mem;

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// kernel_lines /////
    {{kernel_lines|autoindent}}

    //// MAIN CODE ////////////
    if(_idx < 0 || _idx >=  syn_N)
    {
        return;
    }
    double _local_sum = 0.0;
    shared_double_mem[tid] = 0.0;

    // Let one thread per block range set all summed target variables to 0
    if(tid == 0 && num_block_for_neuron == 0)
    {
        {{_target_var_array}}[neuron_id] = 0.0;
    }

    // Get target ID for each synapse
    int target_id = {{_index_array}}[_idx];
    // For each target ID that equals the neuron ID of this block range, set
    // shared memory array to _synaptic_var (all others stay 0)
    if(target_id == neuron_id)
    {
        int _vectorisation_idx = _idx;
        ///// vector code /////
        {{vector_code|autoindent}}
        shared_double_mem[tid] = _synaptic_var;
    }

    __syncthreads();

    // Let only the first thread of each block sum all its _synaptic_vars
    if(tid != 0)
    {
        return;
    }

    for(int _idx = 0; _idx < num_threads; _idx++)
    {
        _local_sum += shared_double_mem[_idx];
    }

    // atomicAdd the block sums to the target neuron
    // We can only get atomic conflicts if there are multiple synapses with
    // same target variable
    _brian_atomicAdd(&{{_target_var_array}}[neuron_id], _local_sum);
}
{% endblock %}


{% block extra_maincode %}
{# This enables summed variables for connections to a synapse,
   copied from cpp template #}
const int _target_size = {{constant_or_scalar(_target_size_name, variables[_target_size_name])}};
static int num_blocks_per_target;
{% endblock %}


{% block modify_kernel_dimensions %}
num_blocks_per_target = num_blocks;
num_blocks *= _target_size;
{% endblock %}


{% block kernel_call %}
    kernel_{{codeobj_name}}<<<num_blocks, num_threads, num_threads*MEM_PER_THREAD>>>(
            num_blocks_per_target,
            num_threads,
            _N,
            ///// HOST_PARAMETERS /////
            %HOST_PARAMETERS%
        );

    CUDA_CHECK_ERROR("kernel_{{codeobj_name}}");
{% endblock %}
