{% extends 'common_group.cu' %}

{% block extra_headers %}
#include "rand.h"
{% endblock %}

{% block kernel_maincode %}
    {# USES_VARIABLES { _group_idx } #}
    ///// block kernel_maincode /////

    ///// scalar code /////
    {{scalar_code|autoindent}}

    _idx = {{_group_idx}}[_vectorisation_idx];
    _vectorisation_idx = _idx;

    ///// vector code /////
    {{vector_code|autoindent}}

    ///// endblock kernel_maincode /////
{% endblock %}

{# _num_group_idx is defined in HOST_CONSTANTS, so we can't set _N before #}
{% block define_N %}
{% endblock %}

{% block host_maincode %}
const int _N = _num_group_idx;
{% endblock %}

{% block profiling_start %}
{% endblock %}

{% block profiling_stop %}
{% endblock %}
