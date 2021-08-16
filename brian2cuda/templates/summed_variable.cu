{# USES_VARIABLES { N } #}
{% extends 'common_group.cu' %}


{% block extra_kernel_call %}
{# Get the device pointer from the thrust device vector for usage in host code #}
{% set _target_var_dev_ptr = get_array_name(_target_var, access_data=True, prefix='dev') %}
{% set _target_var_dtype = c_data_type(_target_var.dtype) %}
{# `constant_or_scalar` enables summed variables for connections to a synapse #}
const int _target_size = {{constant_or_scalar(_target_size_name, variables[_target_size_name])}};

// Reset summed variables to zero
{# Note: cudaMemset sets bytes to value, setting a byte to 0 sets all bits to
         0 and all 0 bits in a floating point number represent zero value.
         cudaMemset does not work for setting non-zero floating point values. #}
CUDA_SAFE_CALL(
        cudaMemset({{_target_var_dev_ptr}} + {{_target_start}},
                   0,
                   _target_size * sizeof({{_target_var_dtype}}))
        );
{% endblock extra_kernel_call %}


{% block extra_vector_code %}
{# For the vector_code, get the _ptr array #}
{% set _target_var_ptr = get_array_name(_target_var) %}
{% set _index_array = get_array_name(_index_var) %}
int _target_id = {{_index_array}}[_idx];
_brian_atomicAdd(&{{_target_var_ptr}}[_target_id], _synaptic_var);
{% endblock extra_vector_code %}
