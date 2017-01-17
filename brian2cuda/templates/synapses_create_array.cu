{% extends 'common_group.cu' %}

{% block kernel %}
{% endblock %}

{% block kernel_call %}
{% endblock %}

{% block extra_maincode %}

{# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets
	            N_incoming, N_outgoing, N,
	            N_pre, N_post, _source_offset, _target_offset } #}

{# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post,
                                   N_incoming, N_outgoing, N}
#}

{# Get N_post and N_pre in the correct way, regardless of whether they are
constants or scalar arrays#}
const int _N_pre = {{constant_or_scalar('N_pre', variables['N_pre'])}};
const int _N_post = {{constant_or_scalar('N_post', variables['N_post'])}};
{{_dynamic_N_incoming}}.resize(_N_post + _target_offset);
{{_dynamic_N_outgoing}}.resize(_N_pre + _source_offset);

///// pointers_lines /////
{{pointers_lines|autoindent}}

for (int _idx=0; _idx<_numsources; _idx++) {
    {# After this code has been executed, the arrays _real_sources and
       _real_variables contain the final indices. Having any code here it all is
       only necessary for supporting subgroups #}
    {{vector_code|autoindent}}

	{{_dynamic__synaptic_pre}}.push_back(_real_sources);
	{{_dynamic__synaptic_post}}.push_back(_real_targets);
    {{_dynamic_N_outgoing}}[_real_sources]++;
    {{_dynamic_N_incoming}}[_real_targets]++;
}

// now we need to resize all registered variables
const int32_t newsize = {{_dynamic__synaptic_pre}}.size();
{% for variable in owner._registered_variables | sort(attribute='name') %}
{% set varname = get_array_name(variable, access_data=False) %}
dev{{varname}}.resize(newsize);
{# //TODO: do we actually need to resize varname? #}
{{varname}}.resize(newsize);
{% endfor %}

// update the total number of synapses
{{N}} = newsize;

// copy changed host data to device
dev{{_dynamic_N_incoming}} = {{_dynamic_N_incoming}};
dev{{_dynamic_N_outgoing}} = {{_dynamic_N_outgoing}};
dev{{_dynamic__synaptic_pre}} = {{_dynamic__synaptic_pre}};
dev{{_dynamic__synaptic_post}} = {{_dynamic__synaptic_post}};
cudaMemcpy(dev{{get_array_name(variables['N'], access_data=False)}},
		{{get_array_name(variables['N'], access_data=False)}},
		sizeof({{c_data_type(variables['N'].dtype)}}),
		cudaMemcpyHostToDevice);

{% endblock extra_maincode %}
