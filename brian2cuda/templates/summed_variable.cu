{% extends 'common_group.cu' %}


{% block extra_headers %}
#include <thrust/fill.h>
{% endblock %}


{% block extra_kernel_call %}
{# Get the thrust vector variable name with `access_data=False` #}
{% set _target_var_array_host = get_array_name(_target_var, access_data=False) %}
{% set _target_var_dtype = c_data_type(_target_var.dtype) %}
{# `constant_or_scalar` enables summed variables for connections to a synapse #}
const int _target_size = {{constant_or_scalar(_target_size_name, variables[_target_size_name])}};
// thrust functions interpret normal pointers as host pointers
// -> wrap pointer in thrust::device_ptr
thrust::device_ptr<{{_target_var_dtype}}> _target_var_thrust_ptr(dev{{_target_var_array_host}});
// Reset summed variables to zero
THRUST_CHECK_ERROR(
        thrust::fill_n(_target_var_thrust_ptr + {{_target_start}},
                       _target_size,
                       0)
        );
{% endblock extra_kernel_call %}


{% block extra_vector_code %}
{# USES_VARIABLES { N } #}
{% set _target_var_array = get_array_name(_target_var) %}
{% set _index_array = get_array_name(_index_var) %}
int _target_id = {{_index_array}}[_idx];
_brian_atomicAdd(&{{_target_var_array}}[_target_id], _synaptic_var);
{% endblock extra_vector_code %}
