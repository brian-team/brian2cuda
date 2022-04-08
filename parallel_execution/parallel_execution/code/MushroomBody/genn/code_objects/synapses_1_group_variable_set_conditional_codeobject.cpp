#include "objects.h"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>

////// SUPPORT CODE ///////
namespace synapses_1_group_variable_set_conditional_codeobject {
        
    double _rand(const int _vectorisation_idx) {
        return rk_double(brian::_mersenne_twister_states[0]);
    }
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int,int> { typedef int type; };
    template < > struct _higher_type<int,long> { typedef long type; };
    template < > struct _higher_type<int,long long> { typedef long long type; };
    template < > struct _higher_type<int,float> { typedef float type; };
    template < > struct _higher_type<int,double> { typedef double type; };
    template < > struct _higher_type<int,long double> { typedef long double type; };
    template < > struct _higher_type<long,int> { typedef long type; };
    template < > struct _higher_type<long,long> { typedef long type; };
    template < > struct _higher_type<long,long long> { typedef long long type; };
    template < > struct _higher_type<long,float> { typedef float type; };
    template < > struct _higher_type<long,double> { typedef double type; };
    template < > struct _higher_type<long,long double> { typedef long double type; };
    template < > struct _higher_type<long long,int> { typedef long long type; };
    template < > struct _higher_type<long long,long> { typedef long long type; };
    template < > struct _higher_type<long long,long long> { typedef long long type; };
    template < > struct _higher_type<long long,float> { typedef float type; };
    template < > struct _higher_type<long long,double> { typedef double type; };
    template < > struct _higher_type<long long,long double> { typedef long double type; };
    template < > struct _higher_type<float,int> { typedef float type; };
    template < > struct _higher_type<float,long> { typedef float type; };
    template < > struct _higher_type<float,long long> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<float,long double> { typedef long double type; };
    template < > struct _higher_type<double,int> { typedef double type; };
    template < > struct _higher_type<double,long> { typedef double type; };
    template < > struct _higher_type<double,long long> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < > struct _higher_type<double,long double> { typedef long double type; };
    template < > struct _higher_type<long double,int> { typedef long double type; };
    template < > struct _higher_type<long double,long> { typedef long double type; };
    template < > struct _higher_type<long double,long long> { typedef long double type; };
    template < > struct _higher_type<long double,float> { typedef long double type; };
    template < > struct _higher_type<long double,double> { typedef long double type; };
    template < > struct _higher_type<long double,long double> { typedef long double type; };
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {{
        return x-y*floor(1.0*x/y);
    }}
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif

}

////// HASH DEFINES ///////



void _run_synapses_1_group_variable_set_conditional_codeobject()
{
    using namespace brian;
using namespace synapses_1_group_variable_set_conditional_codeobject;


    ///// CONSTANTS ///////////
    const int _numN = 1;
double* const _array_synapses_1_g_raw = &_dynamic_array_synapses_1_g_raw[0];
const int _numg_raw = _dynamic_array_synapses_1_g_raw.size();
    ///// POINTERS ////////////
        
    int32_t*   _ptr_array_synapses_1_N = _array_synapses_1_N;
    double* __restrict  _ptr_array_synapses_1_g_raw = _array_synapses_1_g_raw;


//// MAIN CODE ////////////
// scalar code
const size_t _vectorisation_idx = -1;





const int _N = _array_synapses_1_N[0];


for(int _idx=0; _idx<_N; _idx++)
{
    // vector code
    const size_t _vectorisation_idx = _idx;
        
    const char _cond = true;

    if (_cond)
    {
                
        double g_raw;
        g_raw = 1.0f*(1.0f*(_rand(_vectorisation_idx) * 3.7500000000000005e-09)/10)/1.0;
        _ptr_array_synapses_1_g_raw[_idx] = g_raw;

    }
}

}


