{# USES_VARIABLES { _group_idx } #}
{% extends 'common_group.cu' %}

{% block extra_headers %}
#include "rand.h"
{% endblock %}

{% block kernel_maincode %}
    ///// block kernel_maincode /////

    ///// scalar code /////
    {{scalar_code|autoindent}}

    _idx = {{_group_idx}}[_vectorisation_idx];
    _vectorisation_idx = _idx;

    ///// vector code /////
    {{vector_code|autoindent}}

    ///// endblock kernel_maincode /////
{% endblock %}

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

{# _num_group_idx is defined in HOST_CONSTANTS, so we can't set _N before #}
{% block define_N %}
const int _N = _num_group_idx;
{% endblock %}

{% block profiling_start %}
{% endblock %}

{% block profiling_stop %}
{% endblock %}
