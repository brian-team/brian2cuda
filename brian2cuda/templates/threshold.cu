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
		{{lastspike}}[_idx] = {{t}};
		{% endif %}
	}
{% endblock %}


{% block kernel_call %}
{# N is a constant in most cases (NeuronGroup, etc.), but a scalar array for
       synapses, we therefore have to take care to get its value in the right
       way. #}
const int _N = {{constant_or_scalar('N', variables['N'])}};

cudaError_t status = cudaGetLastError();
if (status != cudaSuccess)
{
	printf("ERROR BEFORE resetting eventspace counter in %s:%d %s\n",
			__FILE__, __LINE__, cudaGetErrorString(status));
	_dealloc_arrays();
	exit(status);
}
{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}
cudaMemset(&(dev{{_eventspace}}[current_idx{{_eventspace}}][_N]), 0, sizeof(int32_t));

status = cudaGetLastError();
if (status != cudaSuccess)
{
	printf("ERROR while resetting eventspace counter in %s:%d %s\n",
			__FILE__, __LINE__, cudaGetErrorString(status));
	_dealloc_arrays();
	exit(status);
}

kernel_{{codeobj_name}}<<<num_blocks(_N),num_threads(_N)>>>(
		_N,
		num_threads(_N),
		///// HOST_PARAMETERS /////
		%HOST_PARAMETERS%
	);

status = cudaGetLastError();
if (status != cudaSuccess)
{
	printf("ERROR launching kernel_{{codeobj_name}} in %s:%d %s\n",
			__FILE__, __LINE__, cudaGetErrorString(status));
	_dealloc_arrays();
	exit(status);
}
{% endblock kernel_call %}
