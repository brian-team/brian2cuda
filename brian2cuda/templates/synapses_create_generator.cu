{% extends 'common_synapses.cu' %}

{% block extra_headers %}
{{ super() }}
#include<iostream>
#include<curand.h>
#include<brianlib/curand_buffer.h>
#include "brianlib/cuda_utils.h"
#include<map>
{% endblock extra_headers %}

{% block random_functions %}
// NOTE: _ptr_array_%CODEOBJ_NAME%_rand is NOT an array
// but an instance of CurandBuffer, which overloads the operator[], which then just
// returns the next random number in the buffer, ignoring the argument passed to operator[]
// NOTE: Put buffers into anonymous namespace such that _host_rand/n and rand/n
// in main code have access to it.
// NOTE: _host_rand/n is used in the host compiled implementation of binomial
// functions. Here, it just returns the next element from the CurandBuffer.
CurandBuffer<randomNumber_t> _ptr_array_%CODEOBJ_NAME%_rand(&brian::curand_generator, RAND);
randomNumber_t _host_rand(const int _vectorisation_idx)
{
    return _ptr_array_%CODEOBJ_NAME%_rand[_vectorisation_idx];
}

CurandBuffer<randomNumber_t> _ptr_array_%CODEOBJ_NAME%_randn(&brian::curand_generator, RANDN);
randomNumber_t _host_randn(const int _vectorisation_idx)
{
    return _ptr_array_%CODEOBJ_NAME%_randn[_vectorisation_idx];
}
// This is the C++ Standalone implementation of the poisson function, which we use
int32_t _host_poisson(double lam, int32_t _idx) {
  if (lam >= 10)
    return _poisson_ptrs(lam, _idx);
  else if (lam == 0)
    return 0;
  else
    return _poisson_mult(lam, _idx);
}
double _loggam(double x) {
  double x0, x2, xp, gl, gl0;
  int32_t k, n;

  static double a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                         7.936507936507937e-04, -5.952380952380952e-04,
                         8.417508417508418e-04, -1.917526917526918e-03,
                         6.410256410256410e-03, -2.955065359477124e-02,
                         1.796443723688307e-01, -1.39243221690590e+00};
  x0 = x;
  n = 0;
  if ((x == 1.0) || (x == 2.0))
    return 0.0;
  else if (x <= 7.0) {
    n = (int32_t)(7 - x);
    x0 = x + n;
  }
  x2 = 1.0 / (x0 * x0);
  xp = 2 * M_PI;
  gl0 = a[9];
  for (k=8; k>=0; k--) {
    gl0 *= x2;
    gl0 += a[k];
  }
  gl = gl0 / x0 + 0.5 * log(xp) + (x0 - 0.5) * log(x0) - x0;
  if (x <= 7.0) {
    for (k=1; k<=n; k++) {
      gl -= log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}

int32_t _poisson_mult(double lam, int _vectorisation_idx) {
  int32_t X;
  double prod, U, enlam;

  enlam = exp(-lam);
  X = 0;
  prod = 1.0;
  while (1) {
    U = _rand(_vectorisation_idx);
    prod *= U;
    if (prod > enlam)
      X += 1;
    else
      return X;
  }
}

int32_t _poisson_ptrs(double lam, int _vectorisation_idx) {
  int32_t k;
  double U, V, slam, loglam, a, b, invalpha, vr, us;

  slam = sqrt(lam);
  loglam = log(lam);
  b = 0.931 + 2.53 * slam;
  a = -0.059 + 0.02483 * b;
  invalpha = 1.1239 + 1.1328 / (b - 3.4);
  vr = 0.9277 - 3.6224 / (b - 2);

  while (1) {
    U = _rand(_vectorisation_idx) - 0.5;
    V = _rand(_vectorisation_idx);
    us = 0.5 - abs(U);
    k = (int32_t)floor((2 * a / us + b) * U + lam + 0.43);
    if ((us >= 0.07) && (V <= vr))
      return k;
    if ((k < 0) || ((us < 0.013) && (V > us)))
      continue;
    if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
        (-lam + k * loglam - _loggam(k + 1)))
      return k;
  }
}
{% endblock random_functions %}


{% block kernel %}
{% endblock %}

{% block kernel_call %}
{% endblock %}

{% block prepare_kernel %}
{% endblock %}

{% block occupancy %}
{% endblock %}

{% block update_occupancy %}
{% endblock %}

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
    std::cout << " and used " << tot_memory_MB << "MB of memory.";
std::cout << std::endl;
{% endblock %}

{% block host_maincode %}
    {# // TODO: get rid of the variables we don't actually use
       USES_VARIABLES { _synaptic_pre, _synaptic_post, rand,
                        N_incoming, N_outgoing, N,
                        N_pre, N_post, _source_offset, _target_offset } #}

    {# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post,
                                       N_incoming, N_outgoing, N}
    #}

    ///// pointers_lines /////
    {{pointers_lines|autoindent}}

    {# Get N_post and N_pre in the correct way, regardless of whether they are
    constants or scalar arrays#}
    const int _N_pre = {{constant_or_scalar('N_pre', variables['N_pre'])}};
    const int _N_post = {{constant_or_scalar('N_post', variables['N_post'])}};
    {{_dynamic_N_incoming}}.resize(_N_post + _target_offset);
    {{_dynamic_N_outgoing}}.resize(_N_pre + _source_offset);

    int _raw_pre_idx, _raw_post_idx;
    const int _vectorisation_idx = -1;
    ///// scalar_code['setup_iterator'] /////
    {{scalar_code['setup_iterator']|autoindent}}
    ///// scalar_code['create_j'] /////
    {{scalar_code['create_j']|autoindent}}
    ///// scalar_code['create_cond'] /////
    {{scalar_code['create_cond']|autoindent}}
    ///// scalar_code['update_post'] /////
    {{scalar_code['update_post']|autoindent}}

    for(int _i = 0; _i < _N_pre; _i++)
    {

        bool __cond, _cond;
        _raw_pre_idx = _i + _source_offset;
        {% if not postsynaptic_condition %}
        {
            ///// vector_code['create_cond'] /////
            {{vector_code['create_cond']|autoindent}}
            __cond = _cond;
        }
        _cond = __cond;
        if(!_cond) continue;
        {% endif %}
        // Some explanation of this hackery. The problem is that we have multiple code blocks.
        // Each code block is generated independently of the others, and they declare variables
        // at the beginning if necessary (including declaring them as const if their values don't
        // change). However, if two code blocks follow each other in the same C++ scope then
        // that causes a redeclaration error. So we solve it by putting each block inside a
        // pair of braces to create a new scope specific to each code block. However, that brings
        // up another problem: we need the values from these code blocks. I don't have a general
        // solution to this problem, but in the case of this particular template, we know which
        // values we need from them so we simply create outer scoped variables to copy the value
        // into. Later on we have a slightly more complicated problem because the original name
        // _j has to be used, so we create two variables __j, _j at the outer scope, copy
        // _j to __j in the inner scope (using the inner scope version of _j), and then
        // __j to _j in the outer scope (to the outer scope version of _j). This outer scope
        // version of _j will then be used in subsequent blocks.
        long _uiter_low;
        long _uiter_high;
        long _uiter_step;
        {% if iterator_func=='sample' %}
        double _uiter_p;
        {% endif %}
        {
            ///// vector_code['setup_iterator'] /////
            {{vector_code['setup_iterator']|autoindent}}
            _uiter_low = _iter_low;
            _uiter_high = _iter_high;
            _uiter_step = _iter_step;
            {% if iterator_func=='sample' %}
            _uiter_p = _iter_p;
            {% endif %}
        }
        {% if iterator_func=='range' %}
        for(int {{iteration_variable}}=_uiter_low; {{iteration_variable}}<_uiter_high; {{iteration_variable}}+=_uiter_step)
        {
        {% elif iterator_func=='sample' %}
        if(_uiter_p==0) continue;
        const bool _jump_algo = _uiter_p<0.25;
        double _log1p;
        if(_jump_algo)
            _log1p = log(1-_uiter_p);
        else
            _log1p = 1.0; // will be ignored
        const double _pconst = 1.0/log(1-_uiter_p);
        for(int {{iteration_variable}}=_uiter_low; {{iteration_variable}}<_uiter_high; {{iteration_variable}}++)
        {
            if(_jump_algo) {
                const double _r = _rand(_vectorisation_idx);
                if(_r==0.0) break;
                const int _jump = floor(log(_r)*_pconst)*_uiter_step;
                {{iteration_variable}} += _jump;
                if({{iteration_variable}}>=_uiter_high) continue;
            } else {
                if(_rand(_vectorisation_idx)>=_uiter_p) continue;
            }
        {% endif %}
            long __j, _j, _pre_idx, __pre_idx;
            {
            ///// vector_code['create_j'] /////
                {{vector_code['create_j']|autoindent}}
                __j = _j; // pick up the locally scoped _j and store in __j
                __pre_idx = _pre_idx;
            }
            _j = __j; // make the previously locally scoped _j available
            _pre_idx = __pre_idx;
            _raw_post_idx = _j + _target_offset;
            {% if postsynaptic_condition %}
            {
                {% if postsynaptic_variable_used %}
                {# The condition could index outside of array range #}
                if(_j<0 || _j>=_N_post)
                {
                    {% if skip_if_invalid %}
                    continue;
                    {% else %}
                    cout << "Error: tried to create synapse to neuron j=" << _j << " outside range 0 to " <<
                            _N_post-1 << endl;
                    exit(1);
                    {% endif %}
                }
                {% endif %}
                ///// vector_code['create_cond'] /////
                {{vector_code['create_cond']|autoindent}}
                __cond = _cond;
            }
            _cond = __cond;
            {% endif %}

            {% if if_expression!='True' %}
            if(!_cond) continue;
            {% endif %}

            {% if not postsynaptic_variable_used %}
            {# Otherwise, we already checked before #}
            if(_j<0 || _j>=_N_post)
            {
                {% if skip_if_invalid %}
                continue;
                {% else %}
                cout << "Error: tried to create synapse to neuron j=" << _j << " outside range 0 to " <<
                        _N_post-1 << endl;
                exit(1);
                {% endif %}
            }
            {% endif %}

            ///// vector_code['update_post'] /////
            {{vector_code['update_post']|autoindent}}

            for (int _repetition=0; _repetition<_n; _repetition++) {
                {{_dynamic_N_outgoing}}[_pre_idx] += 1;
                {{_dynamic_N_incoming}}[_post_idx] += 1;
                {{_dynamic__synaptic_pre}}.push_back(_pre_idx);
                {{_dynamic__synaptic_post}}.push_back(_post_idx);
            }
        }
    }

    // now we need to resize all registered variables
    const int32_t newsize = {{_dynamic__synaptic_pre}}.size();
    {% for variable in owner._registered_variables | sort(attribute='name') %}
        {% set varname = get_array_name(variable, access_data=False) %}
        {% if variable.name == 'delay' and no_or_const_delay_mode %}
            assert(dev{{varname}}.size() <= 1);
            THRUST_CHECK_ERROR(
                    dev{{varname}}.resize(1)
                    );
            {# //TODO: do we actually need to resize varname? #}
            {{varname}}.resize(1);
        {% elif variable.name == '_synaptic_pre' and no_pre_references %}
        // prefs['devices.cuda_standalone.no_pre_references'] was set,
        // skipping synaptic_pre resize
        {% elif variable.name == '_synaptic_post' and no_post_references %}
        // prefs['devices.cuda_standalone.no_post_references'] was set,
        // skipping synaptic_post resize
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

{% block extra_kernel_call_post %}
// free memory in CurandBuffers
_ptr_array_%CODEOBJ_NAME%_rand.free_memory();
_ptr_array_%CODEOBJ_NAME%_randn.free_memory();
{% endblock extra_kernel_call_post %}
