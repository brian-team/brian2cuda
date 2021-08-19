{# TEMPLATE INFO

    Notes: This codeobject is run every time step and fills the spikespace of
           the spikegenerator neurongroup with the neuron IDs of the neurons that
           spike in this time step. If there are multiple circular spikespaces
           due to synaptic delays, the correct spikespaces are filled.

           The algorithm here is based on the corresponding cpp_standalone template
           (spikegenerator.cpp), adapted for CUDA.

           This tempalte is very similar to the threshold.cu template since it
           has to fill the spikespace in a similar fashion. Therefore, it also
           needs a separate kernel to reset the spikespace before filling it
           again (we need a global synchronization after reset, therefore I use
           two separate kernels). Additionally, this template needs to keep
           track of the `lastindex` variable that references the index in the
           spikes of the spikegenerator. This also needs to be set once per
           time step with global synchronization. I added it to the reset
           kernel.

    Variables:
     - {{_timebins}} and {{neuron_index}} are `times` (in multiples of dt) and `index`
       of SpikeGeneratorGroup and are storted first by times, then by index
     - {{_period_bins}} gives generator period (in multiples of dt), after which all
       spikes defined by `times` and `index` are repeated.
     - {{_lastindex}} is the last index / spike in the `times` and `index` array that
       has already been generated.

    TODOs:
     - We do know all spike times before the simulation. We could just
       precompute all spikespaces once in the beginning for one period and reuse
       them here. This could save quite some compute when using many periods of
       input.
    - For additional optimizations, see #193.
#}

{# USES_VARIABLES {_spikespace, neuron_index, _timebins, _period_bins, _lastindex,
                   t_in_timesteps, N}
#}
{% extends 'common_group.cu' %}
{% block kernel_maincode %}
    // The period in multiples of dt
    const int32_t _the_period = {{_period_bins}};
    // The spike times in multiples of dt
    int32_t _timebin = {{t_in_timesteps}};

    if (_the_period > 0)
        _timebin %= _the_period;

    // We can have at most one spike per neuron per time step, which is the number of
    // threads we call this kernel with. Hence, no need for any loops.

    // _spike_idx runs through the spikes in the spike generator
    int _spike_idx = _idx + {{_lastindex}};

    // TODO: Solve this smarter. Currently, we will call the reset kernel and this
    // kernel at each time step even if the spikegenerator has emitted all its spikes!
    // Instead, we should know on the host when this happened and not call any kernels.
    // See also #193
    if (_spike_idx >= _num_timebins)
        return;

    // If the spike time of this spike comes after the current time bin, do nothing
    if ({{_timebins}}[_spike_idx] > _timebin)
    {
        return;
    }

    // Else add the spiking neuron to the spikespace
    int32_t _neuron_id = {{neuron_index}}[_spike_idx];
    int32_t _spikespace_index = atomicAdd(&{{_spikespace}}[N], 1);
    {{_spikespace}}[_spikespace_index] = _neuron_id;

{% endblock %}


{% block extra_device_helper %}
    // Function to reset spikespace and set lastindex
    __global__ void
    {% if launch_bounds %}
    __launch_bounds__(1024, {{sm_multiplier}})
    {% endif %}
    _reset_{{codeobj_name}}(
        //int32_t* _spikespace,
        int32_t* _previous_spikespace,
        ///// KERNEL_PARAMETERS /////
        %KERNEL_PARAMETERS%
        )
    {
        using namespace brian;

        int _idx = blockIdx.x * blockDim.x + threadIdx.x;

        const int N = {{owner.N}};

        // We need kernel_lines for time variables
        ///// kernel_lines /////
        {{kernel_lines|autoindent}}

        if (_idx >= N) {
            return;
        }

        if (_idx == 0)
        {
            // The period in multiples of dt
            const int32_t _the_period = {{_period_bins}};
            // The spike times in multiples of dt
            int32_t _timebin          = {{t_in_timesteps}};
            // index of the last spiking neuron in this spikespace
            int32_t _lastindex = {{_lastindex}};

            // Update the lastindex variable with the number of spikes from the
            // spikespace from the previous time step
            _lastindex += _previous_spikespace[N];

            // Now reset the _lastindex if the priod has passed
            if (_the_period > 0) {
                _timebin %= _the_period;
                // If there is a periodicity in the SpikeGenerator, we need to reset the
                // lastindex when the period has passed
                if (_lastindex > 0 && {{_timebins}}[_lastindex - 1] >= _timebin)
                    _lastindex = 0;
            }
            {{_lastindex}} = _lastindex;

            // Reset spikespace counter for this time step
            {{_spikespace}}[N] = 0;
        }

        // Reset the entire spikespace
        {{_spikespace}}[_idx] = -1;
    }
{% endblock %}


{% block extra_kernel_call %}
    {% set _spikespace_name = get_array_name(variables['_spikespace'], access_data=False) %}
    // Note: If we have no delays, there is only one spikespace and
    //       current_idx equals previous_idx.
    _reset_{{codeobj_name}}<<<num_blocks, num_threads>>>(
            dev{{_spikespace_name}}[previous_idx{{_spikespace_name}}],
            ///// HOST_PARAMETERS /////
            %HOST_PARAMETERS%
        );

    CUDA_CHECK_ERROR("_reset_{{codeobj_name}}");
{% endblock extra_kernel_call %}
