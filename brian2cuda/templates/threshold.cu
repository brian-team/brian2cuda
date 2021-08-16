{# USES_VARIABLES { N } #}
{% extends 'common_group.cu' %}

{# not_refractory and lastspike are added as needed_variables in the
   Thresholder class, we cannot use the USES_VARIABLE mechanism
   conditionally
   Same goes for "eventspace" (e.g. spikespace) which depends on the type of
   event.
#}

{# Get the names of the array that stores these events (e.g. the spikespace array) #}
{# _ptr_array variable used in the kernels #}
{% set _eventspace = get_array_name(eventspace_variable) %}
{# _array variable used on the host #}
{% set _eventspace_name = get_array_name(eventspace_variable, access_data=False) %}

{% block kernel_maincode %}

    ///// scalar_code /////
    {{scalar_code|autoindent}}

    {% if not extra_threshold_kernel %}
    // reset eventspace
    {{_eventspace}}[_idx] = -1;
    {% endif %}

    {// there might be the same variable defined in scalar and vector code
    ///// vector_code /////
    {{vector_code|autoindent}}

    if (_cond)
    {
        int32_t spike_index = atomicAdd(&{{_eventspace}}[_N], 1);
        {{_eventspace}}[spike_index] = _idx;
        {% if _uses_refractory %}
        // We have to use the pointer names directly here: The condition
        // might contain references to not_refractory or lastspike and in
        // that case the names will refer to a single entry.
        {{not_refractory}}[_idx] = false;
        {{lastspike}}[_idx] = {{t}};
        {% endif %}
    }
    }
{% endblock %}

{% block extra_device_helper %}
    {% if extra_threshold_kernel %}
        __global__ void
        {% if launch_bounds %}
        __launch_bounds__(1024, {{sm_multiplier}})
        {% endif %}
        _reset_{{codeobj_name}}(
            int32_t* eventspace
            )
        {
            using namespace brian;

            int _idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (_idx >= N) {
                return;
            }

            if (_idx == 0) {
                // reset eventspace counter
                eventspace[N] = 0;
            }

            // reset eventspace
            eventspace[_idx] = -1;
        }
    {% endif %}
{% endblock %}

{% block extra_kernel_call %}
    {% if extra_threshold_kernel %}
        _reset_{{codeobj_name}}<<<num_blocks, num_threads>>>(
                dev{{_eventspace_name}}[current_idx{{_eventspace_name}}]
            );

        CUDA_CHECK_ERROR("_reset_{{codeobj_name}}");
    {% endif %}
{% endblock extra_kernel_call %}

{% block host_maincode %}
    {% if not extra_threshold_kernel %}
        // reset eventspace counter to 0
        CUDA_SAFE_CALL(
                cudaMemset(&(dev{{_eventspace_name}}[current_idx{{_eventspace_name}}][_N]), 0, sizeof(int32_t))
                );
{% endif %}
{% endblock host_maincode %}
