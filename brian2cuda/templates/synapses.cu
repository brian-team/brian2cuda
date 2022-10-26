{# USES_VARIABLES { N, _synaptic_pre} #}
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
_run_kernel_{{codeobj_name}}(
    int _N,
    int bid_offset,
    int timestep,
    {% if bundle_mode %}
    int threads_per_bundle,
    {% endif %}
    int32_t* eventspace,
    {% if uses_atomics or synaptic_effects == "synapse" %}
    int num_spiking_neurons,
    {% else %}
    int neurongroup_size,
    {% endif %}
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x + bid_offset;
    //TODO: do we need _idx here? if no, get also rid of scoping after scalar code
    // scalar_code can depend on _idx (e.g. if the state update depends on a
    // subexpression that is the same for all synapses, ?)
    int _threads_per_block = blockDim.x;
    int _idx = bid * _threads_per_block + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// kernel_lines /////
    {{kernel_lines|autoindent}}

    {% block additional_variables %}
    {% endblock %}

    ///// scalar_code /////
    {{scalar_code|autoindent}}

    {  // _idx is defined in outer and inner scope (for `scalar_code`)
        if ({{pathway.name}}.no_or_const_delay_mode)
        {
            // TODO: pass as kernel parameter instead?
            int num_parallel_blocks = {{pathway.name}}.queue->num_blocks;
            int32_t spikes_start = {{pathway.name}}.spikes_start;
            int32_t spikes_stop = {{pathway.name}}.spikes_stop;

            // for the first delay timesteps the eventspace is not yet filled
            if (timestep >= {{pathway.name}}.queue->num_delays)
            {
                // `spiking_neuron_idx` runs through the eventspace
                // `post_block_idx` runs through the post neuron blocks of the connectivity matrix
                {% if uses_atomics or synaptic_effects == "synapse" %}
                int spiking_neuron_idx = bid / num_parallel_blocks;
                int post_block_idx = bid % num_parallel_blocks;
                {% else %}
                int post_block_idx = bid;
                // loop through neurons in eventspace (indices of event neurons, rest -1)
                for(int spiking_neuron_idx = 0;
                        spiking_neuron_idx < neurongroup_size;
                        spiking_neuron_idx++)
                {% endif %}
                {

                    // spiking_neuron is index in NeuronGroup
                    int32_t spiking_neuron = eventspace[spiking_neuron_idx];

                    {% if uses_atomics or synaptic_effects == "synapse" %}
                    assert(spiking_neuron != -1);
                    {% else %}
                    if(spiking_neuron == -1) // end of spiking neurons
                    {
                        assert(spiking_neuron_idx == eventspace[neurongroup_size]);
                        return;
                    }
                    {% endif %}

                    // apply effects if event neuron is in sources of current SynapticPathway
                    if(spikes_start <= spiking_neuron && spiking_neuron < spikes_stop)
                    {
                        int pre_post_block_id = (spiking_neuron - spikes_start) * num_parallel_blocks + post_block_idx;
                        int num_synapses = {{pathway.name}}_num_synapses_by_pre[pre_post_block_id];
                        int32_t* propagating_synapses = {{pathway.name}}_synapse_ids_by_pre[pre_post_block_id];
                        for(int j = tid; j < num_synapses; j+=_threads_per_block)
                        {
                            // _idx is the synapse id
                            int32_t _idx = propagating_synapses[j];
                            _vectorisation_idx = j;

                            ///// vector_code /////
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
            for (int i = tid; i < queue_size*threads_per_bundle; i+=_threads_per_block)
            {
                // bundle_idx runs through all bundles
                int bundle_idx = i / threads_per_bundle;
                // syn_in_bundle_idx runs through all threads in a single bundle
                int syn_in_bundle_idx = i % threads_per_bundle;

                int bundle_id = synapses_queue[bid].at(bundle_idx);
                int bundle_size = {{pathway.name}}_num_synapses_by_bundle[bundle_id];
                int synapses_offset = {{pathway.name}}_synapses_offset_by_bundle[bundle_id];
                int32_t* synapse_ids = {{pathway.name}}_synapse_ids;
                int32_t* synapse_bundle = synapse_ids + synapses_offset;

                // loop through synapses of this bundle with all available threads_per_bundle
                // if threads_per_bundle == 1, this is serial
                for (int j = syn_in_bundle_idx; j < bundle_size; j+=threads_per_bundle)
                {
                    int32_t _idx = synapse_bundle[j];

            {% else %}{# no bundle_mode #}

                    // use one thread per synapse
                    for(int j = tid; j < queue_size; j+=_threads_per_block)
                    {
                        int32_t _idx = synapses_queue[bid].at(j);
                        {

            {% endif %}{# bundle_mode #}

                            ///// vector_code /////
                            {{vector_code|autoindent}}
                        }
                    }
                }
            }
        }

{% endblock %}

{% block host_maincode %}
static int num_threads_per_bundle;
static int num_loops;
{% endblock %}

{% block extra_device_helper %}
int getThreadsPerBundle(){
    {# Allow using std functions (ceil, floor...) in
       prefs.device.cuda_standalone.threads_per_synapse_bundle #}
    using namespace std;
    using namespace brian;
    int threads_per_bundle = static_cast<int>({{threads_per_synapse_bundle}});
    {% if bundle_threads_warp_multiple %}
    int multiple = threads_per_bundle / num_threads_per_warp;
    {% if bundle_threads_warp_multiple == 'up' %}
    int remainder = threads_per_bundle % num_threads_per_warp;
    if (remainder != 0){
        // if remainder is 0, just use thread_per_bundle as is
        // round up to next multiple of warp size
        threads_per_bundle = (multiple + 1) * num_threads_per_warp;
    }
    {% elif bundle_threads_warp_multiple == 'down' %}
    // ignore remainder, round down to next muptiple of warp size
    threads_per_bundle = multiple * num_threads_per_warp;
    {% endif %}
    {% endif %}

    if (threads_per_bundle < 1){
        threads_per_bundle = 1;
    }
    return threads_per_bundle;
}
{% endblock extra_device_helper %}


{% block prepare_kernel_inner %}
{#######################################################################}
{% if uses_atomics or synaptic_effects == "synapse" %}
{% if uses_atomics %}
// We are using atomics, we can fully parallelise.
{% else %}{# synaptic_effects == "synapse" #}
// Synaptic effects modify only synapse variables.
{% endif %}
num_blocks = num_parallel_blocks;
num_threads = max_threads_per_block;
{% if bundle_mode %}
//num_threads_per_bundle = {{pathway.name}}_bundle_size_max;
num_threads_per_bundle = getThreadsPerBundle();
printf("INFO _run_kernel_{{codeobj_name}}: Using %d threads per bundle\n", num_threads_per_bundle);
{% endif %}
num_loops = 1;

{% elif synaptic_effects == "target" %}{# not uses_atomics #}
// Synaptic effects modify target group variables but NO source group variables.
num_blocks = num_parallel_blocks;
num_loops = 1;
num_threads = 1;
if (!{{owner.name}}_multiple_pre_post){
    if ({{pathway.name}}_scalar_delay)
        num_threads = max_threads_per_block;
    {% if bundle_mode %}
    else  // heterogeneous delays
        // Since we can only parallelize within each bundle, we use as many threads as
        // the maximum bundle size
        num_threads = {{pathway.name}}_bundle_size_max;
    {% endif %}
}
else {
    printf("WARNING: Detected multiple synapses for same (pre, post) neuron "
           "pair in Synapses object ``{{owner.name}}`` and no atomic operations are used. "
           "Falling back to serialised effect application for SynapticPathway "
           "``{{pathway.name}}``. This will be slow. You can avoid serialisation "
           "by separating this Synapses object into multiple Synapses objects "
           "with at most one connection between the same (pre, post) neuron pair.\n");
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
// only call kernel if we have synapses (otherwise we skipped the push kernel)
if ({{pathway.name}}_max_size > 0)
{
    {% if uses_atomics or synaptic_effects == "synapse" %}
    int32_t num_spiking_neurons;
    // we only need the number of spiking neurons if we parallelise effect
    // application over spiking neurons in homogeneous delay mode
    if ({{pathway.name}}_scalar_delay)
    {
        if (defaultclock.timestep[0] >= {{pathway.name}}_delay)
        {
            cudaMemcpy(&num_spiking_neurons,
                    &dev{{_eventspace}}[{{pathway.name}}_eventspace_idx][_num_{{_eventspace}} - 1],
                    sizeof(int32_t), cudaMemcpyDeviceToHost);
            num_blocks = num_parallel_blocks * num_spiking_neurons;
            //TODO collect info abt mean, std of num spiking neurons per time
            //step and print INFO at end of simulation
        }
    }
    // only call kernel if neurons spiked (else num_blocks is zero)
    if (num_blocks != 0) {
    {% endif %}
        for(int bid_offset = 0; bid_offset < num_loops; bid_offset++)
        {
            _run_kernel_{{codeobj_name}}<<<num_blocks, num_threads>>>(
                _N,
                bid_offset,
                {{owner.clock.name}}.timestep[0],
                {% if bundle_mode %}
                num_threads_per_bundle,
                {% endif %}
                dev{{_eventspace}}[{{pathway.name}}_eventspace_idx],
                {% if uses_atomics or synaptic_effects == "synapse" %}
                num_spiking_neurons,
                {% else %}
                _num_{{_eventspace}}-1,
                {% endif %}
                ///// HOST_PARAMETERS /////
                %HOST_PARAMETERS%
            );
        }
    {% if uses_atomics or synaptic_effects == "synapse" %}
    }
    {% endif %}

    CUDA_CHECK_ERROR("_run_kernel_{{codeobj_name}}");
}
{% endblock kernel_call %}

{% block extra_functions_cu %}
void _debugmsg_{{codeobj_name}}()
{
    using namespace brian;
    std::cout << "Number of synapses: " << {{constant_or_scalar('N', variables['N'])}} << endl;
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
