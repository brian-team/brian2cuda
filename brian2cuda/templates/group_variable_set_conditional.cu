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
    {% for var in variables.values() %}
    {# We want to copy only those variables that were potentially modified in aboves kernel call. #}
    {% if var is not callable and var.array and not var.constant and not var.dynamic %}
    {% set varname = get_array_name(var, access_data=False) %}
    CUDA_SAFE_CALL(
            cudaMemcpy({{varname}}, dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyDeviceToHost)
            );
    {% endif %}
    {% endfor %}
{% endblock %}


{% block profiling_start %}
{% endblock %}


{% block profiling_stop %}
{% endblock %}
