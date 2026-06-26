#include "objects.h"
#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>

#include <stdint.h>
#include "synapses_classes.h"

#include<iostream>
#include<curand.h>
#include<brianlib/curand_buffer.h>
#include "brianlib/cuda_utils.h"
#include<map>


////// SUPPORT CODE ///////
namespace {
    double _host_rand(const int _vectorisation_idx);
    double _host_randn(const int _vectorisation_idx);
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx);

    ///// block extra_device_helper /////

    ///// support_code_lines /////
        
    #define _rand(vectorisation_idx) (_ptr_array_synapses_2_synapses_create_generator_codeobject_rand[vectorisation_idx])
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int,int> { typedef int type; };
    template < > struct _higher_type<int,long> { typedef long type; };
    template < > struct _higher_type<int,long long> { typedef long long type; };
    template < > struct _higher_type<int,float> { typedef float type; };
    template < > struct _higher_type<int,double> { typedef double type; };
    template < > struct _higher_type<long,int> { typedef long type; };
    template < > struct _higher_type<long,long> { typedef long type; };
    template < > struct _higher_type<long,long long> { typedef long long type; };
    template < > struct _higher_type<long,float> { typedef float type; };
    template < > struct _higher_type<long,double> { typedef double type; };
    template < > struct _higher_type<long long,int> { typedef long long type; };
    template < > struct _higher_type<long long,long> { typedef long long type; };
    template < > struct _higher_type<long long,long long> { typedef long long type; };
    template < > struct _higher_type<long long,float> { typedef float type; };
    template < > struct _higher_type<long long,double> { typedef double type; };
    template < > struct _higher_type<float,int> { typedef float type; };
    template < > struct _higher_type<float,long> { typedef float type; };
    template < > struct _higher_type<float,long long> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<double,int> { typedef double type; };
    template < > struct _higher_type<double,long> { typedef double type; };
    template < > struct _higher_type<double,long long> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < typename T1, typename T2 >
    __host__ __device__ static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {{
        return x-y*floor(1.0*x/y);
    }}
    template < typename T1, typename T2 >
    __host__ __device__ static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif
                    inline __device__ int _brian_atomicAdd(int* address, int val)
                    {
            // hardware implementation
            return atomicAdd(address, val);
                    }
                    inline __device__ float _brian_atomicAdd(float* address, float val)
                    {
            // hardware implementation
            return atomicAdd(address, val);
                    }
                    inline __device__ double _brian_atomicAdd(double* address, double val)
                    {
                            #if (__CUDA_ARCH__ >= 600)
            // hardware implementation
            return atomicAdd(address, val);
                            #else
            // software implementation
            unsigned long long int* address_as_int = (unsigned long long int*)address;
            unsigned long long int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __double_as_longlong(val +
                                       __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __longlong_as_double(old);
                            #endif
                    }
                    inline __device__ int _brian_atomicMul(int* address, int val)
                    {
                        // software implementation
                        int old = *address, assumed;
                        do {
                            assumed = old;
                            old = atomicCAS(address, assumed, val * assumed);
                        } while (assumed != old);
                        return old;
                    }
                    inline __device__ float _brian_atomicMul(float* address, float val)
                    {
            // software implementation
            int* address_as_int = (int*)address;
            int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __float_as_int(val *
                                       __int_as_float(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __int_as_float(old);
                    }
                    inline __device__ double _brian_atomicMul(double* address, double val)
                    {
            // software implementation
            unsigned long long int* address_as_int = (unsigned long long int*)address;
            unsigned long long int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __double_as_longlong(val *
                                       __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __longlong_as_double(old);
                    }
                    inline __device__ int _brian_atomicDiv(int* address, int val)
                    {
                        // software implementation
                        int old = *address, assumed;
                        do {
                            assumed = old;
                            old = atomicCAS(address, assumed, val / assumed);
                        } while (assumed != old);
                        return old;
                    }
                    inline __device__ float _brian_atomicDiv(float* address, float val)
                    {
            // software implementation
            int* address_as_int = (int*)address;
            int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __float_as_int(val /
                                       __int_as_float(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __int_as_float(old);
                    }
                    inline __device__ double _brian_atomicDiv(double* address, double val)
                    {
            // software implementation
            unsigned long long int* address_as_int = (unsigned long long int*)address;
            unsigned long long int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __double_as_longlong(val /
                                       __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __longlong_as_double(old);
                    }


// NOTE: _ptr_array_synapses_2_synapses_create_generator_codeobject_rand is NOT an array
// but an instance of CurandBuffer, which overloads the operator[], which then just
// returns the next random number in the buffer, ignoring the argument passed to operator[]
// NOTE: Put buffers into anonymous namespace such that _host_rand/n and rand/n
// in main code have access to it.
// NOTE: _host_rand/n is used in the host compiled implementation of binomial
// functions. Here, it just returns the next element from the CurandBuffer.
CurandBuffer<randomNumber_t> _ptr_array_synapses_2_synapses_create_generator_codeobject_rand(&brian::curand_generator, RAND);
randomNumber_t _host_rand(const int _vectorisation_idx)
{
    return _ptr_array_synapses_2_synapses_create_generator_codeobject_rand[_vectorisation_idx];
}

CurandBuffer<randomNumber_t> _ptr_array_synapses_2_synapses_create_generator_codeobject_randn(&brian::curand_generator, RANDN);
randomNumber_t _host_randn(const int _vectorisation_idx)
{
    return _ptr_array_synapses_2_synapses_create_generator_codeobject_randn[_vectorisation_idx];
}

// This is the C++ Standalone implementation of the poisson function, which we use
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

int32_t _host_poisson(double lam, int32_t _idx) {
  if (lam >= 10)
    return _poisson_ptrs(lam, _idx);
  else if (lam == 0)
    return 0;
  else
    return _poisson_mult(lam, _idx);
}
}

////// hashdefine_lines ///////




void _run_synapses_2_synapses_create_generator_codeobject()
{
    using namespace brian;

std::clock_t start_timer = std::clock();

CUDA_CHECK_MEMORY();
size_t used_device_memory_start = used_device_memory;


    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
		int32_t* const _array_synapses_2_N_incoming = thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_incoming[0]);
		const int _numN_incoming = _dynamic_array_synapses_2_N_incoming.size();
		int32_t* const _array_synapses_2_N_outgoing = thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_outgoing[0]);
		const int _numN_outgoing = _dynamic_array_synapses_2_N_outgoing.size();
		int32_t* const _array_synapses_2__synaptic_post = thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_post[0]);
		const int _num_synaptic_post = _dynamic_array_synapses_2__synaptic_post.size();
		int32_t* const _array_synapses_2__synaptic_pre = thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_pre[0]);
		const int _num_synaptic_pre = _dynamic_array_synapses_2__synaptic_pre.size();


    ///// pointers_lines /////
        
    int32_t* __restrict  _ptr_array_synapses_2__synaptic_post = _array_synapses_2__synaptic_post;
    int32_t* __restrict  _ptr_array_synapses_2_N_incoming = _array_synapses_2_N_incoming;
    int32_t* __restrict  _ptr_array_synapses_2__synaptic_pre = _array_synapses_2__synaptic_pre;
    int32_t* __restrict  _ptr_array_synapses_2_N_outgoing = _array_synapses_2_N_outgoing;
    int32_t*   _ptr_array_synapses_2_N = _array_synapses_2_N;


    const int _N_pre = 100;
    const int _N_post = 100;
    _dynamic_array_synapses_2_N_incoming.resize(_N_post + 0);
    _dynamic_array_synapses_2_N_outgoing.resize(_N_pre + 0);

    int _raw_pre_idx, _raw_post_idx;
    const int _vectorisation_idx = -1;
    ///// scalar_code['setup_iterator'] /////
        

    ///// scalar_code['create_j'] /////
        

    ///// scalar_code['create_cond'] /////
        

    ///// scalar_code['update_post'] /////
        


    for(int _i = 0; _i < _N_pre; _i++)
    {

        bool __cond, _cond;
        _raw_pre_idx = _i + 0;
        {
            ///// vector_code['create_cond'] /////
                        
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
        {
            ///// vector_code['setup_iterator'] /////
                        
            const int32_t _iter_low = 0;
            const int32_t _iter_high = 100;
            const int32_t _iter_step = 1;

            _uiter_low = _iter_low;
            _uiter_high = _iter_high;
            _uiter_step = _iter_step;
        }
        for(int _k=_uiter_low; _k<_uiter_high; _k+=_uiter_step)
        {
            long __j, _j, _pre_idx, __pre_idx;
            {
            ///// vector_code['create_j'] /////
                                
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

            ///// vector_code['update_post'] /////
                        
            const int32_t _post_idx = _raw_post_idx;
            const int32_t _n = 1;


            for (int _repetition=0; _repetition<_n; _repetition++) {
                _dynamic_array_synapses_2_N_outgoing[_pre_idx] += 1;
                _dynamic_array_synapses_2_N_incoming[_post_idx] += 1;
                _dynamic_array_synapses_2__synaptic_pre.push_back(_pre_idx);
                _dynamic_array_synapses_2__synaptic_post.push_back(_post_idx);
            }
        }
    }

    // now we need to resize all registered variables
    const int32_t newsize = _dynamic_array_synapses_2__synaptic_pre.size();
            THRUST_CHECK_ERROR(
                    dev_dynamic_array_synapses_2__synaptic_post.resize(newsize)
                    );
            _dynamic_array_synapses_2__synaptic_post.resize(newsize);
            THRUST_CHECK_ERROR(
                    dev_dynamic_array_synapses_2__synaptic_pre.resize(newsize)
                    );
            _dynamic_array_synapses_2__synaptic_pre.resize(newsize);

    // update the total number of synapses
    _ptr_array_synapses_2_N[0] = newsize;

    // Check for occurrence of multiple source-target pairs in synapses ("synapse number")
    std::map<std::pair<int32_t, int32_t>, int32_t> source_target_count;
    for (int _i=0; _i<newsize; _i++)
    {
        // Note that source_target_count will create a new entry initialized
        // with 0 when the key does not exist yet
        const std::pair<int32_t, int32_t> source_target = std::pair<int32_t, int32_t>(_dynamic_array_synapses_2__synaptic_pre[_i], _dynamic_array_synapses_2__synaptic_post[_i]);
        source_target_count[source_target]++;
        //printf("source target count = %i\n", source_target_count[source_target]);
        if (source_target_count[source_target] > 1)
        {
            synapses_2_multiple_pre_post = true;
            break;
        }
    }

    // copy changed host data to device
    dev_dynamic_array_synapses_2_N_incoming = _dynamic_array_synapses_2_N_incoming;
    dev_dynamic_array_synapses_2_N_outgoing = _dynamic_array_synapses_2_N_outgoing;
    dev_dynamic_array_synapses_2__synaptic_pre = _dynamic_array_synapses_2__synaptic_pre;
    dev_dynamic_array_synapses_2__synaptic_post = _dynamic_array_synapses_2__synaptic_post;
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_synapses_2_N,
                _array_synapses_2_N,
                sizeof(int32_t),
                cudaMemcpyHostToDevice)
            );




// free memory in CurandBuffers
_ptr_array_synapses_2_synapses_create_generator_codeobject_rand.free_memory();
_ptr_array_synapses_2_synapses_create_generator_codeobject_randn.free_memory();

CUDA_CHECK_MEMORY();
const double to_MB = 1.0 / (1024.0 * 1024.0);
double tot_memory_MB = (used_device_memory - used_device_memory_start) * to_MB;
double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
std::cout << "INFO: synapses_2 creation took " <<  time_passed << "s";
if (tot_memory_MB > 0)
    std::cout << " and used " << tot_memory_MB << "MB of memory.";
std::cout << std::endl;
}


