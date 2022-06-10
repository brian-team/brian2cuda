{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}
{% extends 'common_group.cu' %}


{% block kernel_maincode %}
    ///// block kernel_maincode /////

    ///// scalar_code['condition'] /////
    {{scalar_code['condition']|autoindent}}

    ///// scalar_code['statement'] /////
    {{scalar_code['statement']|autoindent}}

    ///// vector_code['condition'] /////
    {{vector_code['condition']|autoindent}}

    if (_cond)
    {
        ///// vector_code['statement'] /////
        {{vector_code['statement']|autoindent}}
    }

    ///// endblock kernel_maincode /////
{% endblock kernel_maincode %}


{% block extra_kernel_call_post %}
    {# We need to copy modifed variables back to host in case they are used in
       codeobjects that run on the host, which are synapse connect calls (e.g. in the
       connect condition) and before run synapses push spikes, which initialized
       synaptic variables.
    #}
    {% for var, varname in written_variables.items() %}
    {% if var.dynamic %}
    {{varname}} = dev{{varname}};
    {% else %}
    CUDA_SAFE_CALL(
        cudaMemcpy(
            {{varname}},
            dev{{varname}},
            sizeof({{c_data_type(var.dtype)}})*_num_{{varname}},
            cudaMemcpyDeviceToHost
        )
    );
    {% endif %}
    {% endfor %}
{% endblock %}


{% block profiling_start %}
{% endblock %}


{% block profiling_stop %}
{% endblock %}
