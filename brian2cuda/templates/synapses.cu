{% extends 'common_synapses.cu' %}

{% set _non_synaptic = [] %}
{% for var in variables %}
    {% if variable_indices[var] != '_idx' %}
        {# This is a trick to get around the scoping problem #}
        {% if _non_synaptic.append(1) %}{% endif %}
    {% endif %}
{% endfor %}

{% block kernel %}

__global__ void
{% if launch_bounds or syn_launch_bounds %}
__launch_bounds__(1024, {{sm_multiplier}})
{% endif %}
kernel_{{codeobj_name}}(
    {# TODO: we only need _N if we have random numbers per synapse, add a if test here #}
    unsigned int _N,
    unsigned int bid_offset,
    unsigned int timestep,
    unsigned int THREADS_PER_BLOCK,
    {% if bundle_mode %}
    unsigned int threads_per_bundle,
    {% endif %}
    int32_t* eventspace,
    unsigned int neurongroup_size,
    %DEVICE_PARAMETERS%
    )
{
    {# USES_VARIABLES { N, _synaptic_pre} #}
    using namespace brian;

    assert(THREADS_PER_BLOCK == blockDim.x);

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x + bid_offset;
    //TODO: do we need _idx here? if no, get also rid of scoping after scalar code
    // scalar_code can depend on _idx (e.g. if the state update depends on a
    // subexpression that is the same for all synapses, ?)
    unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
    unsigned int _vectorisation_idx = _idx;
    %KERNEL_VARIABLES%
    {% block additional_variables %}
    {% endblock %}

    {{scalar_code|autoindent}}

    {  // _idx is defined in outer and inner scope (for `scalar_code`)
        if ({{pathway.name}}.no_or_const_delay_mode)
        {
            // for the first delay timesteps the eventspace is not yet filled
            // note that num_queues is the number of eventspaces, num_queues-1 the delay in timesteps
            if (timestep >= {{pathway.name}}.queue->num_queues - 1)
            {
                // loop through neurons in eventspace (indices of event neurons, rest -1)
                for(int i = 0; i < neurongroup_size; i++)
                {
                    // spiking_neuron is index in NeuronGroup
                    int32_t spiking_neuron = eventspace[i];

                    if(spiking_neuron == -1) // end of spiking neurons
                    {
                        assert(i == eventspace[neurongroup_size]);
                        return;
                    }
                    // apply effects if event neuron is in sources of current SynapticPathway
                    if({{pathway.name}}.spikes_start <= spiking_neuron && spiking_neuron < {{pathway.name}}.spikes_stop)
                    {
                        unsigned int pre_post_block_id = (spiking_neuron - {{pathway.name}}.spikes_start) * {{pathway.name}}.queue->num_blocks + bid;
                        int num_synapses = {{pathway.name}}_num_synapses_by_pre[pre_post_block_id];
                        int32_t* propagating_synapses = {{pathway.name}}_synapse_ids_by_pre[pre_post_block_id];
                        for(int j = tid; j < num_synapses; j+=THREADS_PER_BLOCK)
                        {
                            // _idx is the synapse id
                            int32_t _idx = propagating_synapses[j];
                            _vectorisation_idx = j;

                            {{vector_code|autoindent}}
                        }
                    }

                    __syncthreads();
                }
            }
        }
        else  // heterogeneous delay mode
        {
            cudaVector<int32_t>* synapses_queue;
            {{pathway.name}}.queue->peek(&synapses_queue);

            int queue_size = synapses_queue[bid].size();

            {% if bundle_mode %}
            // use a fixed number of threads per bundle, i runs through all those threads of all bundles
            // for threads_per_bundle == 1, we have one thread per bundle (parallel)
            for (int i = tid; i < queue_size*threads_per_bundle; i+=THREADS_PER_BLOCK)
            {
                // bundle_idx runs through all bundles
                unsigned int bundle_idx = i / threads_per_bundle;
                // syn_in_bundle_idx runs through all threads in a single bundle
                unsigned int syn_in_bundle_idx = i % threads_per_bundle;

                unsigned int bundle_id = synapses_queue[bid].at(bundle_idx);
                unsigned int bundle_size = {{pathway.name}}_num_synapses_by_bundle[bundle_id];
                unsigned int synapses_offset = {{pathway.name}}_synapses_offset_by_bundle[bundle_id];
                int32_t* synapse_ids = {{pathway.name}}_synapse_ids;
                int32_t* synapse_bundle = synapse_ids + synapses_offset;

                // loop through synapses of this bundle with all available threads_per_bundle
                // if threads_per_bundle == 1, this is serial
                for (int j = syn_in_bundle_idx; j < bundle_size; j+=threads_per_bundle)
                {
                    int32_t _idx = synapse_bundle[j];

            {% else %}{# no bundle_mode #}

                    // use one thread per synapse
                    for(int j = tid; j < queue_size; j+=THREADS_PER_BLOCK)
                    {
                        int32_t _idx = synapses_queue[bid].at(j);
                        {

            {% endif %}{# bundle_mode #}

                            {{vector_code|autoindent}}
                        }
                    }
                }
            }
        }

{% endblock %}

{% block extra_maincode %}
static unsigned int num_threads_per_bundle;
static unsigned int num_loops;
{% endblock %}

{% block prepare_kernel_inner %}
{#######################################################################}
{% if synaptic_effects == "synapse" %}
// Synaptic effects modify only synapse variables.
num_blocks = num_parallel_blocks;
num_threads = max_threads_per_block;
// TODO: effect of mean instead of max?
{% if bundle_mode %}
num_threads_per_bundle = {{pathway.name}}_max_bundle_size;
{% endif %}
num_loops = 1;

{% elif synaptic_effects == "target" %}
// Synaptic effects modify target group variables but NO source group variables.
num_blocks = num_parallel_blocks;
num_loops = 1;
num_threads = 1;
if (!{{owner.name}}_multiple_pre_post){
    if ({{pathway.name}}_scalar_delay)
        num_threads = max_threads_per_block;
    {% if bundle_mode %}
    else  // heterogeneous delays
        num_threads = {{pathway.name}}_max_bundle_size;
    {% endif %}
}
if (num_threads > max_threads_per_block)
    num_threads = max_threads_per_block;
{% if bundle_mode %}
// num_threads_per_bundle only used for heterogeneous delays
num_threads_per_bundle = num_threads;
{% endif %}

{% elif synaptic_effects == "source" %}
// Synaptic effects modify source group variables.
num_blocks = 1;
num_threads = 1;
{% if bundle_mode %}
num_threads_per_bundle = 1;
{% endif %}
num_loops = num_parallel_blocks;

{% else %}
printf("ERROR: got unknown 'synaptic_effects' mode ({{synaptic_effects}})\n");
_dealloc_arrays();
exit(1);
{% endif %}
{#######################################################################}
{% endblock prepare_kernel_inner %}

{% block extra_info_msg %}
else if ({{pathway.name}}_max_size <= 0)
{
    printf("INFO there are no synapses in the {{pathway.name}} pathway. Skipping synapses_push and synapses kernels.\n");
}
{% endblock %}

{% block kernel_call %}
{% set eventspace_variable = pathway.variables[pathway.eventspace_name] %}
{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}
if ({{pathway.name}}_max_size > 0)
{
    // only call kernel if we have synapses (otherwise we skipped the push kernel)
    for(unsigned int bid_offset = 0; bid_offset < num_loops; bid_offset++)
    {
        kernel_{{codeobj_name}}<<<num_blocks, num_threads>>>(
            _N,
            bid_offset,
            {{owner.clock.name}}.timestep[0],
            num_threads,
            {% if bundle_mode %}
            num_threads_per_bundle,
            {% endif %}
            dev{{_eventspace}}[{{pathway.name}}_eventspace_idx],
            _num_{{_eventspace}}-1,
            %HOST_PARAMETERS%
        );
    }

    CUDA_CHECK_ERROR("kernel_{{codeobj_name}}");
}
{% endblock kernel_call %}

{% block extra_functions_cu %}
void _debugmsg_{{codeobj_name}}()
{
    using namespace brian;
    std::cout << "Number of synapses: " << dev{{_dynamic__synaptic_pre}}.size() << endl;
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
