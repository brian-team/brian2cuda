{% extends 'common_group.cu' %}

{# USES_VARIABLES { N } #}
{% set _target_var_array = get_array_name(_target_var) %}
{% set _index_array = get_array_name(_index_var) %}


{% block extra_device_helper %}
#define MEM_PER_THREAD (sizeof(double))
{% endblock %}


{% block kernel %}
__global__ void kernel_{{codeobj_name}}(
    int _num_blocks_per_target,
    int num_threads,
    int num_blocks,
    int syn_N,
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    using namespace brian;

    extern __shared__ char shared_mem[];
    int _tidx = threadIdx.x;
    int _bidx = blockIdx.x;
    // Group ID of summed target variable that this block works on
    int _target_id_this_block = _bidx/_num_blocks_per_target + {{_target_start}};
    // Block index working on this target (in [0, _target_size])
    int _bidx_this_target = _bidx % _num_blocks_per_target;
    // _idx is the synapse index and for each target, it goes from 0 to num_synapses
    int _idx = _bidx_this_target * num_threads + _tidx;
    int _target_size = num_blocks / _num_blocks_per_target;
    double* shared_double_mem = (double*) shared_mem;

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// kernel_lines /////
    {{kernel_lines|autoindent}}

    //// MAIN CODE ////////////
    if(_idx >=  syn_N)
    {
        return;
    }
    double _local_sum = 0.0;
    shared_double_mem[_tidx] = 0.0;

    // Let one thread per target set the summed target variable to 0
    // XXX TODO: We can't synchronize between blocks, therefore the target variable
    //           might not be 0 while other blocks for the same target might
    //           already do their atomicAdd instructions...
    if(_tidx == 0 && _bidx_this_target == 0)
    {
        {{_target_var_array}}[_target_id_this_block] = 0;
    }

    // Get target ID for each synapse
    int this_syn_target_id = {{_index_array}}[_idx];

    const int _vectorisation_idx = -1;
    ///// scalar_code /////
    {{scalar_code|autoindent}}

    // For each target IDX that equals the target ID of this synapse, set
    // shared memory array to _synaptic_var (all others stay 0)
    if(this_syn_target_id == _target_id_this_block)
    {
        const int _vectorisation_idx = _idx;
        ///// vector code /////
        {{vector_code|autoindent}}
        shared_double_mem[_tidx] = _synaptic_var;
    }

    __syncthreads();

    // Let only the first thread of each block sum all its _synaptic_vars
    if(_tidx != 0)
    {
        return;
    }

    for(int _idx = 0; _idx < num_threads; _idx++)
    {
        _local_sum += shared_double_mem[_idx];
    }

    // atomicAdd the block sums to the target variable
    // (potentially as many conflicts as there are blocks per target)
    _brian_atomicAdd(&{{_target_var_array}}[_target_id_this_block], _local_sum);
}
{% endblock %}


{% block host_maincode %}
{# This enables summed variables for connections to a synapse,
   copied from cpp template #}
const int _target_size = {{constant_or_scalar(_target_size_name, variables[_target_size_name])}};
static int _num_blocks_per_target;
{% endblock %}


{% block modify_kernel_dimensions %}
_num_blocks_per_target = num_blocks;
num_blocks *= _target_size;
{% endblock %}


{% block kernel_call %}
    kernel_{{codeobj_name}}<<<num_blocks, num_threads, num_threads*MEM_PER_THREAD>>>(
            _num_blocks_per_target,
            num_threads,
            num_blocks,
            _N,
            ///// HOST_PARAMETERS /////
            %HOST_PARAMETERS%
        );

    CUDA_CHECK_ERROR("kernel_{{codeobj_name}}");
{% endblock %}
