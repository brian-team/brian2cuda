#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>



////// SUPPORT CODE ///////
namespace {
    double _host_rand(const int _vectorisation_idx);
    double _host_randn(const int _vectorisation_idx);
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx);

    ///// block extra_device_helper /////

    ///// support_code_lines /////
        
    template <typename T>
    __host__ __device__
    double _brian_exp(T value)
    {
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
        return exp((double)value);
    #else
        return exp(value);
    #endif
    }
    inline __host__ __device__
    float _brian_exp(float value)
    {
        return exp(value);
    }
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


    // Implement dummy functions such that the host compiled code of binomial
    // functions works. Hacky, hacky ...
    double _host_rand(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    double _host_randn(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_poisson` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
}

////// hashdefine_lines ///////



__global__ void
_run_kernel_neurongroup_1_stateupdater_codeobject(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    double* _ptr_array_neurongroup_1_V,
	const double _value_array_defaultclock_dt,
	double* _ptr_array_neurongroup_1_g_eKC_eKC,
	double* _ptr_array_neurongroup_1_g_iKC_eKC,
	double* _ptr_array_neurongroup_1_h,
	double* _ptr_array_neurongroup_1_m,
	double* _ptr_array_neurongroup_1_n,
	char* _ptr_array_neurongroup_1_not_refractory
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numV = 100;
	const int _numg_eKC_eKC = 100;
	const int _numg_iKC_eKC = 100;
	const int _numh = 100;
	const int _numm = 100;
	const int _numn = 100;
	const int _numnot_refractory = 100;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_dt = &_value_array_defaultclock_dt;


    assert(THREADS_PER_BLOCK == blockDim.x);


    if(_idx >= _N)
    {
        return;
    }


    ///// scalar_code /////
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double _lio_1 = 1.0f*((- 0.06356) * 2.67e-08)/3e-10;
    const double _lio_2 = 1.0f*((- 0.095) * 1.4299999999999999e-06)/3e-10;
    const double _lio_3 = 1.0f*(0.05 * 7.15e-06)/3e-10;
    const double _lio_4 = 1.0f*0.0/3e-10;
    const double _lio_5 = 1.0f*(- 0.092)/3e-10;
    const double _lio_6 = 0.0 - (1.0f*2.67e-08/3e-10);
    const double _lio_7 = 1.0f*((- 1.0) * 1.4299999999999999e-06)/3e-10;
    const double _lio_8 = 1.0f*7.15e-06/3e-10;
    const double _lio_9 = 1.0f*1.0/3e-10;
    const double _lio_10 = _brian_exp(1.0f*(- dt)/0.005);
    const double _lio_11 = _brian_exp(1.0f*(- dt)/0.01);
    const double _lio_12 = 1.0f*(0.329137207652868 * _brian_exp(1.0f*(0.0555555555555556 * (- 0.063))/0.001))/0.001;
    const double _lio_13 = 1.0f*(- 0.0555555555555556)/0.001;
    const double _lio_14 = 2980.95798704173 * (0.001 * _brian_exp(1.0f*(0.2 * (- 0.063))/0.001));
    const double _lio_15 = 1.0f*(- 0.2)/0.001;
    const double _lio_16 = ((- 1.0) * (_brian_pow(0.001, 1.0))) * 0.001;
    const double _lio_17 = 25.7903399171931 * (((_brian_pow(0.001, 1.0)) * 0.001) * _brian_exp(1.0f*(0.25 * (- 0.063))/0.001));
    const double _lio_18 = 1.0f*(- 0.25)/0.001;
    const double _lio_19 = 0.32 * (- 0.063);
    const double _lio_20 = 4.16 * 0.001;
    const double _lio_21 = 0.0 - ((_brian_pow(0.001, 1.0)) * 0.001);
    const double _lio_22 = 0.000335462627902512 * (((_brian_pow(0.001, 1.0)) * 0.001) * _brian_exp(1.0f*((- 0.2) * (- 0.063))/0.001));
    const double _lio_23 = 1.0f*0.2/0.001;
    const double _lio_24 = 0.28 * (- 0.063);
    const double _lio_25 = 11.2 * 0.001;
    const double _lio_26 = ((- 1.0) * 0.001) * 0.001;
    const double _lio_27 = 20.0855369231877 * ((0.001 * 0.001) * _brian_exp(1.0f*(0.2 * (- 0.063))/0.001));
    const double _lio_28 = 0.032 * (- 0.063);
    const double _lio_29 = 0.48 * 0.001;
    const double _lio_30 = 1.0f*(0.642012708343871 * _brian_exp(1.0f*(0.025 * (- 0.063))/0.001))/0.001;
    const double _lio_31 = 1.0f*(- 0.025)/0.001;


    {
        ///// vector_code /////
                
        double m = _ptr_array_neurongroup_1_m[_idx];
        double g_eKC_eKC = _ptr_array_neurongroup_1_g_eKC_eKC[_idx];
        char not_refractory = _ptr_array_neurongroup_1_not_refractory[_idx];
        double n = _ptr_array_neurongroup_1_n[_idx];
        double h = _ptr_array_neurongroup_1_h[_idx];
        double V = _ptr_array_neurongroup_1_V[_idx];
        const double dt = _ptr_array_defaultclock_dt[0];
        double g_iKC_eKC = _ptr_array_neurongroup_1_g_iKC_eKC[_idx];
        if(!not_refractory)
            not_refractory = false || (! (V > 0.0));
        else 
            not_refractory = true || (! (V > 0.0));
        const double _BA_V = 1.0f*(_lio_1 + ((((_lio_2 * (_brian_pow(n, 4.0))) + (_lio_3 * (h * (_brian_pow(m, 3.0))))) + (_lio_4 * g_iKC_eKC)) + (_lio_5 * g_eKC_eKC)))/((_lio_6 + (_lio_7 * (_brian_pow(n, 4.0)))) - (((_lio_8 * (h * (_brian_pow(m, 3.0)))) + (_lio_9 * g_eKC_eKC)) + (_lio_9 * g_iKC_eKC)));
        const double _V = (- _BA_V) + ((V + _BA_V) * _brian_exp(dt * ((_lio_6 + (_lio_7 * (_brian_pow(n, 4.0)))) - (((_lio_8 * (h * (_brian_pow(m, 3.0)))) + (_lio_9 * g_eKC_eKC)) + (_lio_9 * g_iKC_eKC)))));
        const double _g_eKC_eKC = _lio_10 * g_eKC_eKC;
        const double _g_iKC_eKC = _lio_11 * g_iKC_eKC;
        const double _BA_h = 1.0f*(_lio_12 * _brian_exp(_lio_13 * V))/((1.0f*(- 4.0)/(0.001 + (_lio_14 * _brian_exp(_lio_15 * V)))) - (_lio_12 * _brian_exp(_lio_13 * V)));
        const double _h = (- _BA_h) + ((_BA_h + h) * _brian_exp(dt * ((1.0f*(- 4.0)/(0.001 + (_lio_14 * _brian_exp(_lio_15 * V)))) - (_lio_12 * _brian_exp(_lio_13 * V)))));
        const double _BA_m = 1.0f*(((1.0f*((- 0.32) * V)/(_lio_16 + (_lio_17 * _brian_exp(_lio_18 * V)))) + (1.0f*_lio_19/(_lio_16 + (_lio_17 * _brian_exp(_lio_18 * V))))) + (1.0f*_lio_20/(_lio_16 + (_lio_17 * _brian_exp(_lio_18 * V)))))/(((((1.0f*((- 0.28) * V)/(_lio_21 + (_lio_22 * _brian_exp(_lio_23 * V)))) + (1.0f*(0.32 * V)/(_lio_16 + (_lio_17 * _brian_exp(_lio_18 * V))))) + (1.0f*_lio_24/(_lio_21 + (_lio_22 * _brian_exp(_lio_23 * V))))) + (1.0f*_lio_25/(_lio_21 + (_lio_22 * _brian_exp(_lio_23 * V))))) - ((1.0f*_lio_19/(_lio_16 + (_lio_17 * _brian_exp(_lio_18 * V)))) + (1.0f*_lio_20/(_lio_16 + (_lio_17 * _brian_exp(_lio_18 * V))))));
        const double _m = (- _BA_m) + ((_BA_m + m) * _brian_exp(dt * (((((1.0f*((- 0.28) * V)/(_lio_21 + (_lio_22 * _brian_exp(_lio_23 * V)))) + (1.0f*(0.32 * V)/(_lio_16 + (_lio_17 * _brian_exp(_lio_18 * V))))) + (1.0f*_lio_24/(_lio_21 + (_lio_22 * _brian_exp(_lio_23 * V))))) + (1.0f*_lio_25/(_lio_21 + (_lio_22 * _brian_exp(_lio_23 * V))))) - ((1.0f*_lio_19/(_lio_16 + (_lio_17 * _brian_exp(_lio_18 * V)))) + (1.0f*_lio_20/(_lio_16 + (_lio_17 * _brian_exp(_lio_18 * V))))))));
        const double _BA_n = 1.0f*(((1.0f*((- 0.032) * V)/(_lio_26 + (_lio_27 * _brian_exp(_lio_15 * V)))) + (1.0f*_lio_28/(_lio_26 + (_lio_27 * _brian_exp(_lio_15 * V))))) + (1.0f*_lio_29/(_lio_26 + (_lio_27 * _brian_exp(_lio_15 * V)))))/((1.0f*(0.032 * V)/(_lio_26 + (_lio_27 * _brian_exp(_lio_15 * V)))) - (((1.0f*_lio_28/(_lio_26 + (_lio_27 * _brian_exp(_lio_15 * V)))) + (1.0f*_lio_29/(_lio_26 + (_lio_27 * _brian_exp(_lio_15 * V))))) + (_lio_30 * _brian_exp(_lio_31 * V))));
        const double _n = (- _BA_n) + ((_BA_n + n) * _brian_exp(dt * ((1.0f*(0.032 * V)/(_lio_26 + (_lio_27 * _brian_exp(_lio_15 * V)))) - (((1.0f*_lio_28/(_lio_26 + (_lio_27 * _brian_exp(_lio_15 * V)))) + (1.0f*_lio_29/(_lio_26 + (_lio_27 * _brian_exp(_lio_15 * V))))) + (_lio_30 * _brian_exp(_lio_31 * V))))));
        V = _V;
        g_eKC_eKC = _g_eKC_eKC;
        g_iKC_eKC = _g_iKC_eKC;
        h = _h;
        m = _m;
        n = _n;
        _ptr_array_neurongroup_1_g_eKC_eKC[_idx] = g_eKC_eKC;
        _ptr_array_neurongroup_1_m[_idx] = m;
        _ptr_array_neurongroup_1_not_refractory[_idx] = not_refractory;
        _ptr_array_neurongroup_1_n[_idx] = n;
        _ptr_array_neurongroup_1_h[_idx] = h;
        _ptr_array_neurongroup_1_V[_idx] = V;
        _ptr_array_neurongroup_1_g_iKC_eKC[_idx] = g_iKC_eKC;


    }
}


void _run_neurongroup_1_stateupdater_codeobject()
{
    using namespace brian;


    const int _N = 100;

    ///// HOST_CONSTANTS ///////////
    const int _numV = 100;
		const int _numg_eKC_eKC = 100;
		const int _numg_iKC_eKC = 100;
		const int _numh = 100;
		const int _numm = 100;
		const int _numn = 100;
		const int _numnot_refractory = 100;


    static int num_threads, num_blocks;
    static size_t needed_shared_memory = 0;
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    _run_kernel_neurongroup_1_stateupdater_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;





        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    _run_kernel_neurongroup_1_stateupdater_codeobject, num_threads, needed_shared_memory)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, _run_kernel_neurongroup_1_stateupdater_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "_run_kernel_neurongroup_1_stateupdater_codeobject "
                   "with maximum possible threads per block (%u). "
                   "Reducing num_threads to %u. (Kernel needs %i "
                   "registers per block, %i bytes of "
                   "statically-allocated shared memory per block, %i "
                   "bytes of local memory per thread and a total of %i "
                   "bytes of user-allocated constant memory)\n",
                   max_threads_per_block, num_threads, funcAttrib.numRegs,
                   funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                   funcAttrib.constSizeBytes);

            // calculate theoretical occupancy for new num_threads
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                        _run_kernel_neurongroup_1_stateupdater_codeobject, num_threads, needed_shared_memory)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }


        else
        {
            printf("INFO _run_kernel_neurongroup_1_stateupdater_codeobject\n"
                   "\t%u blocks\n"
                   "\t%u threads\n"
                   "\t%i registers per block\n"
                   "\t%i bytes statically-allocated shared memory per block\n"
                   "\t%i bytes local memory per thread\n"
                   "\t%i bytes user-allocated constant memory\n"
                   "\t%.3f theoretical occupancy\n",
                   num_blocks, num_threads, funcAttrib.numRegs,
                   funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                   funcAttrib.constSizeBytes, occupancy);
        }
        first_run = false;
    }


    _run_kernel_neurongroup_1_stateupdater_codeobject<<<num_blocks, num_threads,0, neurongroup_stream1>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            dev_array_neurongroup_1_V,
			_array_defaultclock_dt[0],
			dev_array_neurongroup_1_g_eKC_eKC,
			dev_array_neurongroup_1_g_iKC_eKC,
			dev_array_neurongroup_1_h,
			dev_array_neurongroup_1_m,
			dev_array_neurongroup_1_n,
			dev_array_neurongroup_1_not_refractory
        );

    CUDA_CHECK_ERROR("_run_kernel_neurongroup_1_stateupdater_codeobject");


}


