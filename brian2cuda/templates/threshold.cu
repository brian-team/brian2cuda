{% extends 'common_group.cu' %}

{# USES_VARIABLES { N } #}
{# not_refractory and lastspike are added as needed_variables in the
   Thresholder class, we cannot use the USES_VARIABLE mechanism
   conditionally
   Same goes for "eventspace" (e.g. spikespace) which depends on the type of
   event.
#}

{% block maincode %}
	{#  Get the name of the array that stores these events (e.g. the spikespace array) #}
	{% set _eventspace = get_array_name(eventspace_variable) %}

	{% if threadfence %}
	if (tid==0 && bid==0)
	{
		// reset eventspace counter to 0
		{{_eventspace}}[_N] = 0;
	}
	__threadfence();
	{% endif %}

	///// scalar_code /////
	{{scalar_code|autoindent}}

	// reset eventspace
	{{_eventspace}}[_idx] = -1;

	///// vector_code /////
	{{vector_code|autoindent}}

	if (_cond)
	{
		int32_t spike_index = atomicAdd(&{{_eventspace}}[_N], 1);
		{{_eventspace}}[spike_index] = _idx;
		{% if _uses_refractory %}
		// We have to use the pointer names directly here: The condition
		// might contain references to not_refractory or lastspike and in
		// that case the names will refer to a single entry.
		{{not_refractory}}[_idx] = false;
		{# we can't use {{t}} directly, since it returns ...[0] (in Device.code_object()) but our t is not a pointer #}
		{{lastspike}}[_idx] = {{get_array_name(variables['t'])}};
		{% endif %}
	}
{% endblock %}

{% block extra_maincode %}
{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}
{% if not threadfence %}
// reset eventspace counter to 0
cudaMemset(&(dev{{_eventspace}}[current_idx{{_eventspace}}][_N]), 0, sizeof(int32_t));
{% endif %}
{% endblock extra_maincode %}
