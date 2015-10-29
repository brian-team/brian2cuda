{% extends 'common_synapses.cu' %}

{% block extra_headers %}
{{ super() }}
#include<iostream>
{% endblock %}

{% block kernel %}
{% endblock %}

{% block kernel_call %}
{% endblock %}

{% block extra_maincode %}
	{# USES_VARIABLES { _synaptic_pre, _synaptic_post, rand,
	                    N_incoming, N_outgoing, N } #}

	srand(time(0));
	{{scalar_code|autoindent}}

	//these two vectors just cache everything on the CPU-side
	//data is copied to GPU at the end
	thrust::host_vector<int32_t> temp_synaptic_post;
	thrust::host_vector<int32_t> temp_synaptic_pre;

	{{pointers_lines|autoindent}}

	unsigned int syn_id = 0;
	for(int _i = 0; _i < _num_all_pre; _i++)
	{
		for(int _j = 0; _j < _num_all_post; _j++)
		{
			{% block maincode_inner %}
		    const int _vectorisation_idx = _i*_num_all_post  + _j;
			{{vector_code|autoindent}}
			// Add to buffer
			if(_cond)
			{
				if (_p != 1.0)
				{
					float r = rand()/(float)RAND_MAX;
					if (r >= _p)
					{
						continue;
					}
				}
				for (int _repetition = 0; _repetition < _n; _repetition++)
				{
					temp_synaptic_pre.push_back(_pre_idx);
					temp_synaptic_post.push_back(_post_idx);
					syn_id++;
				}
			}
			{% endblock %}
		}
	}

	dev{{_dynamic__synaptic_pre}} = temp_synaptic_pre;
	dev{{_dynamic__synaptic_post}} = temp_synaptic_post;
	
	// now we need to resize all registered variables
	const int32_t newsize = dev{{_dynamic__synaptic_pre}}.size();
	{% for variable in owner._registered_variables | sort(attribute='name') %}
	{% set varname = get_array_name(variable, access_data=False) %}
	dev{{varname}}.resize(newsize);
	{{varname}}.resize(newsize);
	{% endfor %}
	// Also update the total number of synapses
        {{N}} = newsize;
{% endblock %}
