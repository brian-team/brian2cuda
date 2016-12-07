{% extends 'common_group.cu' %}
{% block maincode %}
	{# USES_VARIABLES { N } #}

        {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
        {% set _eventspace = get_array_name(eventspace_variable) %}

	const int32_t *_events = {{_eventspace}};
	const int32_t _num_events = {{_eventspace}}[N];

	//// MAIN CODE ////////////	
	// scalar code
	{{scalar_code|autoindent}}
    
	//get events (e.g. spiking) neuron_id
	_idx = _events[_idx];
	if(_idx != -1)
	{
		{{vector_code|autoindent}}
	}
{% endblock %}
