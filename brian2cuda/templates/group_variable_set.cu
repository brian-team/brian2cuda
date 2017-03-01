{% extends 'common_group.cu' %}

{% block maincode %}
	{# USES_VARIABLES { _group_idx } #}
	///// block maincode /////

	///// scalar code /////
    {{scalar_code|autoindent}}

	_idx = {{_group_idx}}[_vectorisation_idx];
	_vectorisation_idx = _idx;

	///// vector code /////
    {{vector_code|autoindent}}

	///// endblock maincode /////
{% endblock %}

{% block extra_maincode %}
const int _N = _num_group_idx;
{% endblock %}
