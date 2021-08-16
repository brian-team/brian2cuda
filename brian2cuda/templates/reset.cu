{# USES_VARIABLES { N } #}
{% extends 'common_group.cu' %}
{% block kernel_maincode %}

    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

    const int32_t *_events = {{_eventspace}};
    const int32_t _num_events = {{_eventspace}}[N];

    // TODO: call kernel only with as many threads as events
    if (_idx >= _num_events)
    {
        return;
    }

    //// MAIN CODE ////////////
    // scalar code
    {{scalar_code|autoindent}}

    //get events (e.g. spiking) neuron_id
    int neuron_id = _events[_idx];
    assert(neuron_id >= 0);
    _idx = neuron_id;

    {{vector_code|autoindent}}
{% endblock %}
