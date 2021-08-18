'''
CUDA implementation of `BinomialFunction`
'''
import numpy as np

from brian2.core.functions import DEFAULT_FUNCTIONS
from brian2.units.fundamentalunits import check_units
from brian2.utils.stringtools import replace
from brian2.input.binomial import (BinomialFunction, _pre_calc_constants,
                                   _pre_calc_constants_approximated)
from brian2 import prefs


def _generate_cuda_code(n, p, use_normal, name):
    # CUDA implementation
    # Inversion transform sampling
    if not '_binomial' in name:
        # TODO: mark issue
        raise NameError("Currently the `name` parameter of `BinomialFunction` "
                        "needs to have '_binomial' in it, got "
                        f"'{name}'")
    float_suffix = ''
    float_dtype = 'float'
    if prefs['core.default_float_dtype'] == np.float64:
        float_suffix = '_double'
        float_dtype = 'double'
    # TODO: we should load the state once in scalar code and pass as function
    # argument, needs modification of CUDAGenerator._add_user_function to
    # modify scalar code
    if use_normal:
        loc, scale = _pre_calc_constants_approximated(n, p)
        cuda_code = '''
        __host__ __device__
        %DTYPE% %NAME%(const int vectorisation_idx)
        {
        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
            return curand_normal%SUFFIX%(brian::d_curand_states + vectorisation_idx) * %SCALE% + %LOC%;
        #else
            return _host_randn(vectorisation_idx) * %SCALE% + %LOC%;
        #endif
        }
        '''
        cuda_code = replace(cuda_code, {'%SCALE%': f'{scale:.15f}',
                                        '%LOC%': f'{loc:.15f}',
                                        '%NAME%': name,
                                        '%DTYPE%': float_dtype,
                                        '%SUFFIX%': float_suffix})
        dependencies = {'_randn': DEFAULT_FUNCTIONS['randn']}
    else:
        reverse, q, P, qn, bound = _pre_calc_constants(n, p)
        # The following code is an almost exact copy of numpy's
        # rk_binomial_inversion function
        # (numpy/random/mtrand/distributions.c)
        cuda_code = '''
        __host__ __device__
        long %NAME%(const int vectorisation_idx)
        {
        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
            curandState localState = brian::d_curand_states[vectorisation_idx];
            %DTYPE% U = curand_uniform%SUFFIX%(&localState);
        #else
            %DTYPE% U = _host_rand(vectorisation_idx);
        #endif
            long X = 0;
            %DTYPE% px = %QN%;
            while (U > px)
            {
                X++;
                if (X > %BOUND%)
                {
                    X = 0;
                    px = %QN%;
        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
                    U = curand_uniform%SUFFIX%(&localState);
        #else
                    U = _host_rand(vectorisation_idx);
        #endif
                } else
                {
                    U -= px;
                    px = ((%N%-X+1) * %P% * px)/(X*%Q%);
                }
            }
        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
            // copy the locally changed CuRAND state back to global memory
            brian::d_curand_states[vectorisation_idx] = localState;
        #endif
            return %RETURN_VALUE%;
        }
        '''
        cuda_code = replace(cuda_code, {'%N%': '%d' % n,
                                        '%P%': f'{P:.15f}',
                                        '%Q%': f'{q:.15f}',
                                        '%QN%': f'{qn:.15f}',
                                        '%BOUND%': f'{bound:.15f}',
                                        '%RETURN_VALUE%': '%d-X' % n if reverse else 'X',
                                        '%NAME%': name,
                                        '%DTYPE%': float_dtype,
                                        '%SUFFIX%': float_suffix})
        dependencies = {'_rand': DEFAULT_FUNCTIONS['rand']}

    return {'support_code': cuda_code}, dependencies


BinomialFunction.implementations['cuda'] = _generate_cuda_code
