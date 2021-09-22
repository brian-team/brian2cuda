
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>
#include "brianlib/stdint_compat.h"
#include "synapses_classes.h"

////// SUPPORT CODE ///////
namespace synapses_max_row_length_generator {
 	
 double _rand(const int _vectorisation_idx) {
     return rk_double(brian::_mersenne_twister_states[omp_get_thread_num()]);
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



void _run_synapses_max_row_length()
{
    using namespace brian;
    using namespace  synapses_max_row_length_generator;

    ///// CONSTANTS ///////////
    const int _num_synaptic_post = _dynamic_array_synapses__synaptic_post.size();
int32_t* const _array_synapses_N_incoming = &_dynamic_array_synapses_N_incoming[0];
const int _num_synaptic_pre = _dynamic_array_synapses__synaptic_pre.size();
const int _numN = 1;
int32_t* const _array_synapses_N_outgoing = &_dynamic_array_synapses_N_outgoing[0];
int32_t* const _array_synapses__synaptic_pre = &_dynamic_array_synapses__synaptic_pre[0];
int32_t* const _array_synapses__synaptic_post = &_dynamic_array_synapses__synaptic_post[0];
const int _numN_outgoing = _dynamic_array_synapses_N_outgoing.size();
const int _numN_incoming = _dynamic_array_synapses_N_incoming.size();

    ///// POINTERS ////////////
        
    int32_t*   _ptr_array_synapses_N = _array_synapses_N;
    int32_t* __restrict  _ptr_array_synapses__synaptic_post = _array_synapses__synaptic_post;
    int32_t* __restrict  _ptr_array_synapses__synaptic_pre = _array_synapses__synaptic_pre;
    int32_t* __restrict  _ptr_array_synapses_N_incoming = _array_synapses_N_incoming;
    int32_t* __restrict  _ptr_array_synapses_N_outgoing = _array_synapses_N_outgoing;

	
    const size_t _N_pre = 100;
    const size_t _N_post = 2500;
    _dynamic_array_synapses_N_incoming.resize(_N_post + 0);
    _dynamic_array_synapses_N_outgoing.resize(_N_pre + 0);
    size_t _raw_pre_idx, _raw_post_idx;
    // scalar code
    const int _vectorisation_idx = -1;
        

        

        

        

    for(size_t _i=0; _i<_N_pre; _i++)
    {
        bool __cond, _cond;
        _raw_pre_idx = _i + 0;
        {
                        
            const char _cond = true;

            __cond = _cond;
        }
        _cond = __cond;
        if(!_cond) continue;
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
        double _uiter_p;
        {
                        
            const int32_t _iter_low = 0;
            const int32_t _iter_high = 2500;
            const int32_t _iter_step = 1;
            const double _iter_p = 0.15;

            _uiter_low = _iter_low;
            _uiter_high = _iter_high;
            _uiter_step = _iter_step;
            _uiter_p = _iter_p;
        }
        if(_uiter_p==0) continue;
        const bool _jump_algo = _uiter_p<0.25;
        double _log1p;
        if(_jump_algo)
            _log1p = log(1-_uiter_p);
        else
            _log1p = 1.0; // will be ignored
        const double _pconst = 1.0/log(1-_uiter_p);
        for(long _k=_uiter_low; _k<_uiter_high; _k++)
        {
            if(_jump_algo) {
                const double _r = _rand(_vectorisation_idx);
                if(_r==0.0) break;
                const int _jump = floor(log(_r)*_pconst)*_uiter_step;
                _k += _jump;
                if(_k>=_uiter_high) continue;
            } else {
                if(_rand(_vectorisation_idx)>=_uiter_p) continue;
            }
            long __j, _j, _pre_idx, __pre_idx;
            {
                                
                const int32_t _pre_idx = _raw_pre_idx;
                const int32_t _j = _k;

                __j = _j; // pick up the locally scoped _j and store in __j
                __pre_idx = _pre_idx;
            }
            _j = __j; // make the previously locally scoped _j available
            _pre_idx = __pre_idx;
            _raw_post_idx = _j + 0;

            if(_j<0 || _j>=_N_post)
            {
                cout << "Error: tried to create synapse to neuron j=" << _j << " outside range 0 to " <<
                        _N_post-1 << endl;
                exit(1);
            }
                        
            const int32_t _post_idx = _raw_post_idx;
            const int32_t _n = 1;
 
	    
	    for (size_t _repetition=0; _repetition<_n; _repetition++) {
	        _dynamic_array_synapses_N_outgoing[_pre_idx] += 1;
	        _dynamic_array_synapses_N_incoming[_post_idx] += 1;
	    }
	  }
        }
    
}