{% extends 'common_group.cu' %}
{#
    Notes: This codeobject is run every time step and fills the spikespace of
           the spikegenerator neurongroup with the neuron IDs of the neurons that
           spike in this time step. If there are multiple circular spikespaces
           due to synaptic delays, the correct spikespaces are filled.

           I found the variable naming in the cpp_standalone template not very
           intuitive. I changed names and referenced the corresponding name in
           the cpp_standalone template. The implemented algorithm is the same
           though.

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
     - {{spike_time}} and {{neuron_index}} are `times` and `index` of
         SpikeGeneratorGroup and are storted first by times, then by index
     - {{period}} gives generator period in ms, after which all spikes defined
         by `times` and `index` are repeated.
     - {{_lastindex} starts always with 0,
     - `test`: True if a spike time comes after the current time step
         (>= implementation with epsilon precision)

    TODOs:
     - We do know all spike times before the simulation. We could just
       precompute all spikespaces once in the beginning for one period and reuse
       them here. This could save quite some compute when using many periods of
       input.
     - The template ended up to be a bit messy with the
       _get_time_within_period() function calls. We could just compute everything
       on the host and pass the computed values that we need to the kernels
       (instead of letting each thread compute the same).
#}

{# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time, period, _lastindex } #}


{% block maincode %}
    double _epsilon = 1e-3 * {{dt}};
    {# In spikegenerator.cpp, this is `padding_after` #}
    double _time_within_period = _get_time_within_period({{t}}, {{dt}}, {{period}});
    {# In spikegenerator.cpp, this is `(! not_end_period)` #}
    const bool _last_in_period = _check_last_in_period(_time_within_period, _epsilon, {{period}});
    bool test;

    // Reset the spikespace
    for (int i = tid; i < N; i+= THREADS_PER_BLOCK)
    {
        {{_spikespace}}[i] = -1;
    }

    if (tid == 0)
    {
        // Reset number of spikes to 0
        {{_spikespace}}[N] = 0;
    }
    __syncthreads();

    for (int spike_idx = {{_lastindex}} + tid; spike_idx < _numspike_time; spike_idx += THREADS_PER_BLOCK)
    {
        if (! _last_in_period)
        {
            test = ({{spike_time}}[spike_idx] > _time_within_period) || (fabs({{spike_time}}[spike_idx] - _time_within_period) < _epsilon);
        }
        else
        {
            // If we are in the last timestep before the end of the period, we remove the first part of the
            // test, because padding will be 0
            test = (fabs({{spike_time}}[spike_idx] - _time_within_period) < _epsilon);
        }
        if (test)
        {
            break;
        }
        int32_t neuron_id = {{neuron_index}}[spike_idx];
        int32_t spikespace_index = atomicAdd(&{{_spikespace}}[N], 1);
        {{_spikespace}}[spikespace_index] = neuron_id;
    }

{% endblock %}


{% block extra_device_helper %}
    {# Since we use these functions in both, the codeobject and the reset
       kernel, I defined extra functions for them. #}
    __host__ __device__ inline double
    _get_time_within_period(double _t, double _dt, double _the_period)
    {
        // In cpp_standalone template, this is called `padding_after`
        return fmod(_t + _dt, _the_period);
    }


    __host__ __device__ inline const bool
    _check_last_in_period(
            double _time_within_period,
            double _epsilon,
            double _the_period)
    {
        // Whether the current time step IS NOT the last of the period
        // Naming taken from cpp_standalone template
        const bool not_end_period = (fabs(_time_within_period) > _epsilon) \
                                    && (fabs(_time_within_period) < (_the_period - _epsilon));

        // Return, whether the current time step IS the last of the period
        return (! not_end_period);
    }


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

        // We need kernel_lines for time variables
        ///// kernel_lines /////
        {{kernel_lines|autoindent}}

        if (_idx >= N) {
            return;
        }

        if (_idx == 0)
        {
            // TODO: Should we compute _first_in_period on the host instead?
            // If the previous time step was last in the period, this time step
            // is the first in the new period.
            double _epsilon = 1e-3 * {{dt}};
            double _previous_t = {{t}} - {{dt}};
            double _prev_time_within_period = _get_time_within_period(_previous_t, {{dt}}, {{period}});
            const bool _first_in_period = _check_last_in_period(_prev_time_within_period, _epsilon, {{period}});

            // If there is a periodicity in the SpikeGenerator, we need to reset the lastindex
            // when all spikes have been played and the start of a new period.
            if (_first_in_period) {
                // Reset the spike index within a period
                {{_lastindex}} = 0;
            }
            else {
                // Update the last spike index with the number of spikes in the
                // previous time step
                {{_lastindex}} += _previous_spikespace[N];
            }

            // Reset spikespace counter for this time step
            {{_spikespace}}[N] = 0;
        }

        // Reset spikespace
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
