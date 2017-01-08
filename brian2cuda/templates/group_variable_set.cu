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

{% block kernel_call %}
kernel_{{codeobj_name}}<<<num_blocks(_num_group_idx),num_threads(_num_group_idx)>>>(
		_num_group_idx,
		num_threads(_num_group_idx),
		///// HOST_PARAMETERS /////
		%HOST_PARAMETERS%
	);
{% endblock kernel_call %}
