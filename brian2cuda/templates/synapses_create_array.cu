{% extends 'common_group.cu' %}

{% block kernel %}
{% endblock %}

{% block kernel_call %}
{% endblock %}

{% block extra_maincode %}
{# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets
                 N_incoming, N_outgoing }
#}

const int _old_num_synapses = dev{{_dynamic__synaptic_pre}}.size();
const int _new_num_synapses = _old_num_synapses + _numsources;

//these two vectors just cache everything on the CPU-side
//data is copied to GPU at the end
thrust::host_vector<int32_t> temp_synaptic_post;
thrust::host_vector<int32_t> temp_synaptic_pre;

for (int _idx=0; _idx<_numsources; _idx++) {
    {# After this code has been executed, the arrays _real_sources and
       _real_variables contain the final indices. Having any code here it all is
       only necessary for supporting subgroups #}
   	int32_t* _ptr_array_{{owner.name}}_sources = _array_{{owner.name}}_sources;
   	int32_t* _ptr_array_{{owner.name}}_targets = _array_{{owner.name}}_targets;
   	
   	
    {{vector_code|autoindent}}

	temp_synaptic_pre.push_back(_real_sources);
	temp_synaptic_post.push_back(_real_targets);
    // Update the number of total outgoing/incoming synapses per source/target neuron
    _array_{{owner.name}}_N_outgoing[_real_sources]++;
    _array_{{owner.name}}_N_incoming[_real_targets]++;
}

dev{{_dynamic__synaptic_pre}} = temp_synaptic_pre;
dev{{_dynamic__synaptic_post}} = temp_synaptic_post;

// now we need to resize all registered variables
const int newsize = dev{{_dynamic__synaptic_pre}}.size();
{% for variable in owner._registered_variables | sort(attribute='name') %}
{% set varname = get_array_name(variable, access_data=False) %}
dev{{varname}}.resize(newsize);
{% endfor %}
// Also update the total number of synapses
{{owner.name}}._N_value = newsize;
{% endblock %}
