'''
CUDA implementation of `TimedArray`
'''


from brian2.input.timedarray import TimedArray, _find_K
from brian2.utils.stringtools import replace


def _generate_cuda_code_1d(values, dt, name):
    def cuda_impl(owner):
        K = _find_K(owner.clock.dt_, dt)
        code = '''
        __host__ __device__
        static inline double %NAME%(const double t)
        {
            using namespace brian;

            const double epsilon = %DT% / %K%;
            int i = (int)((t/epsilon + 0.5)/%K%);
            if(i < 0)
               i = 0;
            if(i >= %NUM_VALUES%)
                i = %NUM_VALUES%-1;
        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
            return d%NAME%_values[i];
        #else
            return %NAME%_values[i];
        #endif
        }
        '''.replace('%NAME%', name).replace('%DT%', '%.18f' % dt).replace(
            '%K%', str(K)).replace('%NUM_VALUES%', str(len(values)))

        return code

    return cuda_impl


def _generate_cuda_code_2d(values, dt, name):
    def cuda_impl(owner):
        K = _find_K(owner.clock.dt_, dt)
        code = '''
        __host__ __device__
        static inline double %NAME%(const double t, const int i)
        {
            using namespace brian;

            const double epsilon = %DT% / %K%;
            if (i < 0 || i >= %COLS%)
                return NAN;
            int timestep = (int)((t/epsilon + 0.5)/%K%);
            if(timestep < 0)
               timestep = 0;
            else if(timestep >= %ROWS%)
                timestep = %ROWS%-1;
        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
            return d%NAME%_values[timestep*%COLS% + i];
        #else
            return %NAME%_values[timestep*%COLS% + i];
        #endif

        }
        '''
        code = replace(code, {'%NAME%': name,
                              '%DT%': '%.18f' % dt,
                              '%K%': str(K),
                              '%COLS%': str(values.shape[1]),
                              '%ROWS%': str(values.shape[0])})
        return code
    return cuda_impl


TimedArray.implementations['cuda'] = (_generate_cuda_code_1d,
                                      _generate_cuda_code_2d)
