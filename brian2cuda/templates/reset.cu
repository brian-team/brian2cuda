{% extends 'common_group.cu' %}
{% block maincode %}
	{# USES_VARIABLES { _spikespace, N } #}

	const int32_t *_spikes = {{_spikespace}};
	const int32_t _num_spikes = {{_spikespace}}[N];

	//// MAIN CODE ////////////	
	// scalar code
	{{scalar_code|autoindent}}
    
	//get spiking neuron_id
	_idx = _spikes[_idx];
	if(_idx != -1)
	{
		{{vector_code|autoindent}}
	}
{% endblock %}
