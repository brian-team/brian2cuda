#include "objects.h"
#include "code_objects/spikegeneratorgroup_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>
#include "magicnetwork_model_CODE/definitions.h"

////// SUPPORT CODE ///////
namespace spikegeneratorgroup_codeobject {
        
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



void _run_spikegeneratorgroup_codeobject()
{
    using namespace brian;
using namespace spikegeneratorgroup_codeobject;


    ///// CONSTANTS ///////////
    int32_t* const _array_spikegeneratorgroup_neuron_index = &_dynamic_array_spikegeneratorgroup_neuron_index[0];
const int _numneuron_index = _dynamic_array_spikegeneratorgroup_neuron_index.size();
const int _num_period_bins = 1;
const int _num_lastindex = 1;
int32_t* const _array_spikegeneratorgroup__timebins = &_dynamic_array_spikegeneratorgroup__timebins[0];
const int _num_timebins = _dynamic_array_spikegeneratorgroup__timebins.size();
const int _num_spikespace = 101;
int32_t* const _array_spikegeneratorgroup_spike_number = &_dynamic_array_spikegeneratorgroup_spike_number[0];
const int _numspike_number = _dynamic_array_spikegeneratorgroup_spike_number.size();
    ///// POINTERS ////////////
        
    int32_t* __restrict  _ptr_array_spikegeneratorgroup_neuron_index = _array_spikegeneratorgroup_neuron_index;
    int32_t*   _ptr_array_spikegeneratorgroup__period_bins = _array_spikegeneratorgroup__period_bins;
    int32_t*   _ptr_array_spikegeneratorgroup__lastindex = _array_spikegeneratorgroup__lastindex;
    int32_t* __restrict  _ptr_array_spikegeneratorgroup__timebins = _array_spikegeneratorgroup__timebins;
    int32_t* __restrict  _ptr_array_spikegeneratorgroup__spikespace = _array_spikegeneratorgroup__spikespace;
    int32_t* __restrict  _ptr_array_spikegeneratorgroup_spike_number = _array_spikegeneratorgroup_spike_number;



        const int32_t _the_period = _ptr_array_spikegeneratorgroup__period_bins[0];
        int32_t _timebin          = iT;

        if (_the_period > 0) {
            _timebin %= _the_period;
            // If there is a periodicity in the SpikeGenerator, we need to reset the
            // lastindex when the period has passed
            if (_ptr_array_spikegeneratorgroup__lastindex[0] > 0 && _ptr_array_spikegeneratorgroup__timebins[_ptr_array_spikegeneratorgroup__lastindex[0] - 1] >= _timebin)
                _ptr_array_spikegeneratorgroup__lastindex[0] = 0;
        }

        int32_t _cpp_numspikes = 0;

        for(int _idx=_ptr_array_spikegeneratorgroup__lastindex[0]; _idx < _num_timebins; _idx++)
        {
            if (_ptr_array_spikegeneratorgroup__timebins[_idx] > _timebin)
                break;

            spike_spikegeneratorgroup[_cpp_numspikes++] = _ptr_array_spikegeneratorgroup_neuron_index[_idx];
        }

        spikeCount_spikegeneratorgroup = _cpp_numspikes;

        _ptr_array_spikegeneratorgroup__lastindex[0] += _cpp_numspikes;


}


