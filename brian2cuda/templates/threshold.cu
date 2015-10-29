{% extends 'common_group.cu' %}

{% block extra_device_helper %}
int mem_per_thread(){
	return sizeof(bool);
}
{% endblock %}


{% block maincode %}
	{# USES_VARIABLES { t, _spikespace, N } #}
	// not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	// scalar code
	{{scalar_code|autoindent}}

	{{_spikespace}}[_idx] = -1;

	if(tid == 0 && bid == 0)
	{
		//init number of spikes with 0
		{{_spikespace}}[N] = 0;
	}
	__syncthreads();

	{{vector_code|autoindent}}
	if(_cond) {
		int32_t spike_index = atomicAdd(&{{_spikespace}}[N], 1);
		{{_spikespace}}[spike_index] = _idx;
		{% if _uses_refractory %}
		// We have to use the pointer names directly here: The condition
		// might contain references to not_refractory or lastspike and in
		// that case the names will refer to a single entry.
		{{not_refractory}}[_idx] = false;
		{{lastspike}}[_idx] = {{t}};
		{% endif %}
	}
{% endblock %}

{% block kernel_call %}
kernel_{{codeobj_name}}<<<num_blocks(N), num_threads(N)>>>(
		num_threads(N),
		%HOST_PARAMETERS%
	);
{% endblock %}
