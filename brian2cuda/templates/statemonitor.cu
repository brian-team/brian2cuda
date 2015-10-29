{% extends 'common_group.cu' %}

{% block extra_maincode %}
{# USES_VARIABLES {t} #}

int current_iteration = {{owner.clock.name}}.timestep[0];
static unsigned int start_offset = current_iteration - dev_dynamic_array_{{owner.name}}_t.size();
dev_dynamic_array_{{owner.name}}_t.push_back({{owner.clock.name}}.t[0]);
static bool first_run = true;
if(first_run)
{
	int num_iterations = {{owner.clock.name}}.i_end;
	
	{% for varname, var in _recorded_variables | dictsort %}
		{% set _recorded =  get_array_name(var, access_data=False) %}
		addresses_monitor_{{_recorded}}.clear();			
	{% endfor %}
	for(int i = 0; i < _num__array_{{owner.name}}__indices; i++)
	{
		{% for varname, var in _recorded_variables | dictsort %}
			{% set _recorded =  get_array_name(var, access_data=False) %}
			{{_recorded}}[i].resize({{_recorded}}[i].size() + num_iterations - current_iteration);
			addresses_monitor_{{_recorded}}.push_back(thrust::raw_pointer_cast(&{{_recorded}}[i][0]));
		{% endfor %}
	}
	first_run = false;
}
{% endblock %}

{% block kernel_call %}
_run_{{codeobj_name}}_kernel<<<1, _num__array_{{owner.name}}__indices>>>(
	_num__array_{{owner.name}}__indices,
	dev_array_{{owner.name}}__indices,
	current_iteration - start_offset,
	{% for varname, var in _recorded_variables | dictsort %}
		{% set _recorded =  get_array_name(var, access_data=False) %}
		thrust::raw_pointer_cast(&addresses_monitor_{{_recorded}}[0]),
	{% endfor %}
	%HOST_PARAMETERS%
	);
{% endblock %}

{% block kernel %}
__global__ void _run_{{codeobj_name}}_kernel(
	int _num_indices,
	int32_t* indices,
	int current_iteration,
	{% for varname, var in _recorded_variables | dictsort %}
		{{c_data_type(var.dtype)}}** monitor_{{varname}},
	{% endfor %}
	%DEVICE_PARAMETERS%
	)
{
	unsigned int tid = threadIdx.x;
	if(tid > _num_indices)
	{
		return;
	}
	int32_t _idx = indices[tid];
	
	%KERNEL_VARIABLES%
	
	{{scalar_code|autoindent}}
	{{vector_code|autoindent}}

	{% for varname, var in _recorded_variables | dictsort %}
		{% set _recorded =  get_array_name(var, access_data=False) %}
		monitor_{{varname}}[tid][current_iteration] = _to_record_{{varname}};
	{% endfor %}
}
{% endblock %}
