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

{% block define_N %}
const int _N = _num_group_idx;
{% endblock %}

{% block profiling_start %}
{% endblock %}

{% block profiling_stop %}
{% endblock %}
