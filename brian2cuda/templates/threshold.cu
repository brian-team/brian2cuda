{% extends 'common_group.cu' %}

{% block extra_device_helper %}
int mem_per_thread(){
	return sizeof(bool);
}
{% endblock %}


{% block maincode %}
	{# USES_VARIABLES { N } #}

	{# not_refractory and lastspike are added as needed_variables in the
	   Thresholder class, we cannot use the USES_VARIABLE mechanism
	   conditionally
	   Same goes for "eventspace" (e.g. spikespace) which depends on the type of
	   event.
	#}

	{#  Get the name of the array that stores these events (e.g. the spikespace array) #}
	{% set _eventspace = get_array_name(eventspace_variable) %}

	// TODO this is a little hacky and results in unnecessary instructions, maybe find a better way?
	// we can't return before __syncthreads()
	// and after __syncthreads() we need access to declarations in scalar_code
	// so we just make sure _idx is not too high
	bool return_thread = false;
	if (_idx >= _N)
	{
		_idx = _N-1;
		return_thread = true;
	}

	//// MAIN CODE ////////////
	// scalar code
	{{scalar_code|autoindent}}

	{{_eventspace}}[_idx] = -1;

	if(tid == 0 && bid == 0)
	{
		//init number of spikes with 0
		{{_eventspace}}[N] = 0;
	}

	__syncthreads();

	// after __syncthreads() we can return
	if (return_thread)
	{
		return;
	}

	{{vector_code|autoindent}}
	if (_cond)
	{
		int32_t spike_index = atomicAdd(&{{_eventspace}}[N], 1);
		{{_eventspace}}[spike_index] = _idx;
		{% if _uses_refractory %}
		// We have to use the pointer names directly here: The condition
		// might contain references to not_refractory or lastspike and in
		// that case the names will refer to a single entry.
		{{not_refractory}}[_idx] = false;
		{{lastspike}}[_idx] = {{t}};
		{% endif %}
	}
{% endblock %}

{# we can't return threads when using __syncthreads() #}
{% block num_thread_check %}
{% endblock %}
