{% extends 'common_group.cu' %}

{% block extra_headers %}
{{ super() }}
#include<map>
{% endblock %}

{% block kernel %}
{% endblock %}

{% block kernel_call %}
{% endblock %}

{% block prepare_kernel %}
{% endblock %}

{% block occupancy %}
{% endblock occupancy %}

{% block update_occupancy %}
{% endblock update_occupancy %}

{% block kernel_info %}
{% endblock %}

{% block define_N %}
{% endblock %}

{% block profiling_start %}
std::clock_t start_timer = std::clock();

CUDA_CHECK_MEMORY();
size_t used_device_memory_start = used_device_memory;
{% endblock %}

{% block profiling_stop %}
CUDA_CHECK_MEMORY();
const double to_MB = 1.0 / (1024.0 * 1024.0);
double tot_memory_MB = (used_device_memory - used_device_memory_start) * to_MB;
double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
std::cout << "INFO: {{owner.name}} creation took " <<  time_passed << "s";
if (tot_memory_MB > 0)
    std::cout << " and used " << tot_memory_MB << "MB of device memory.";
std::cout << std::endl;
{% endblock %}

{% block host_maincode %}
{# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets
                    N_incoming, N_outgoing, N, N_pre, N_post, _source_offset,
                    _target_offset } #}

{# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post, N_incoming,
                                   N_outgoing, N} #}

{# Get N_post and N_pre in the correct way, regardless of whether they are
constants or scalar arrays#}
const int _N_pre = {{constant_or_scalar('N_pre', variables['N_pre'])}};
const int _N_post = {{constant_or_scalar('N_post', variables['N_post'])}};
{{_dynamic_N_incoming}}.resize(_N_post + _target_offset);
{{_dynamic_N_outgoing}}.resize(_N_pre + _source_offset);

///// pointers_lines /////
{{pointers_lines|autoindent}}

for (int _idx=0; _idx<_numsources; _idx++) {
    {# After this code has been executed, the arrays _real_sources and
       _real_variables contain the final indices. Having any code here it all is
       only necessary for supporting subgroups #}
    {{vector_code|autoindent}}

    {{_dynamic__synaptic_pre}}.push_back(_real_sources);
    {{_dynamic__synaptic_post}}.push_back(_real_targets);
    {{_dynamic_N_outgoing}}[_real_sources]++;
    {{_dynamic_N_incoming}}[_real_targets]++;
}

// now we need to resize all registered variables
const int32_t newsize = {{_dynamic__synaptic_pre}}.size();
{% for variable in owner._registered_variables | sort(attribute='name') %}
    {% set varname = get_array_name(variable, access_data=False) %}
    {% if variable.name == 'delay' and no_or_const_delay_mode %}
        THRUST_CHECK_ERROR(
                dev{{varname}}.resize(1)
                );
        {# //TODO: do we actually need to resize varname? #}
        {{varname}}.resize(1);
    {% else %}
        {% if not multisynaptic_index or not variable == multisynaptic_idx_var %}
        THRUST_CHECK_ERROR(
                dev{{varname}}.resize(newsize)
                );
        {% endif %}
        {# //TODO: do we actually need to resize varname? #}
        {{varname}}.resize(newsize);
    {% endif %}
{% endfor %}
CUDA_CHECK_MEMORY();

// update the total number of synapses
{{N}} = newsize;

// Check for occurrence of multiple source-target pairs in synapses ("synapse number")
std::map<std::pair<int32_t, int32_t>, int32_t> source_target_count;
for (int _i=0; _i<newsize; _i++)
{
    // Note that source_target_count will create a new entry initialized
    // with 0 when the key does not exist yet
    const std::pair<int32_t, int32_t> source_target = std::pair<int32_t, int32_t>({{_dynamic__synaptic_pre}}[_i], {{_dynamic__synaptic_post}}[_i]);
    {% if multisynaptic_index %}
    // Save the "synapse number"
    {% set dynamic_multisynaptic_idx = get_array_name(multisynaptic_idx_var, access_data=False) %}
    {{dynamic_multisynaptic_idx}}[_i] = source_target_count[source_target];
    {% endif %}
    source_target_count[source_target]++;
    //printf("source target count = %i\n", source_target_count[source_target]);
    if (source_target_count[source_target] > 1)
    {
        {{owner.name}}_multiple_pre_post = true;
        {% if not multisynaptic_index %}
        break;
        {% endif %}
    }
}
// Check
// copy changed host data to device
dev{{_dynamic_N_incoming}} = {{_dynamic_N_incoming}};
dev{{_dynamic_N_outgoing}} = {{_dynamic_N_outgoing}};
dev{{_dynamic__synaptic_pre}} = {{_dynamic__synaptic_pre}};
dev{{_dynamic__synaptic_post}} = {{_dynamic__synaptic_post}};
{% if multisynaptic_index %}
dev{{dynamic_multisynaptic_idx}} = {{dynamic_multisynaptic_idx}};
{% endif %}
CUDA_SAFE_CALL(
        cudaMemcpy(dev{{get_array_name(variables['N'], access_data=False)}},
            {{get_array_name(variables['N'], access_data=False)}},
            sizeof({{c_data_type(variables['N'].dtype)}}),
            cudaMemcpyHostToDevice)
        );

{% endblock host_maincode %}
