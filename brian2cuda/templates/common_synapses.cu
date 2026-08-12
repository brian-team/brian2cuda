{# USES_VARIABLES { N, no_delay_mode } #}
{% extends 'common_group.cu' %}
{% block extra_headers %}
#include "synapses_classes.h"
#include "brianlib/spikequeue.h"
{% endblock %}
