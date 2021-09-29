#include "objects.h"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject_1.h"
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
        
                        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
                        __device__ double* _namespace_timedarray_3_values;
                        #else
                        double* _namespace_timedarray_3_values;
                        #endif
    __host__ __device__
    static inline double _timedarray_3(const double t, const int i)
    {
        const double epsilon = 10.000000000000000000 / 1048576;
        if (i < 0 || i >= 250000)
            return NAN;
        int timestep = (int)((t/epsilon + 0.5)/1048576);
        if(timestep < 0)
           timestep = 0;
        else if(timestep >= 1)
            timestep = 1-1;
        return _namespace_timedarray_3_values[timestep*250000 + i];
    }
                        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
                        __device__ double* _namespace_timedarray_4_values;
                        #else
                        double* _namespace_timedarray_4_values;
                        #endif
    __host__ __device__
    static inline double _timedarray_4(const double t, const int i)
    {
        const double epsilon = 10.000000000000000000 / 1048576;
        if (i < 0 || i >= 250000)
            return NAN;
        int timestep = (int)((t/epsilon + 0.5)/1048576);
        if(timestep < 0)
           timestep = 0;
        else if(timestep >= 1)
            timestep = 1-1;
        return _namespace_timedarray_4_values[timestep*250000 + i];
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
kernel_synapses_1_group_variable_set_conditional_codeobject_1(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    int32_t* _ptr_array_synapses_1_N,
	double* _ptr_array_synapses_1_g_raw,
	const int _numg_raw,
	int32_t* _ptr_array_synapses_1__synaptic_pre,
	const int _numi,
	int32_t* _ptr_array_synapses_1__synaptic_post,
	const int _numj
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numN = 1;

    ///// kernel_lines /////
        
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    _namespace_timedarray_3_values = d_timedarray_3_values;
    #else
    _namespace_timedarray_3_values = _timedarray_3_values;
    #endif
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    _namespace_timedarray_4_values = d_timedarray_4_values;
    #else
    _namespace_timedarray_4_values = _timedarray_4_values;
    #endif


    assert(THREADS_PER_BLOCK == blockDim.x);


    if(_idx >= _N)
    {
        return;
    }

    ///// block kernel_maincode /////

    ///// scalar_code['condition'] /////
        


    ///// scalar_code['statement'] /////
        
    const double _lio_statement_1 = 1.0f*1.0/1.0;
    const double _lio_statement_2 = 2.5 * 1e-09;
    const double _lio_statement_3 = 0.5 * 1e-09;


    ///// vector_code['condition'] /////
        
    const int32_t j = _ptr_array_synapses_1__synaptic_post[_idx];
    const int32_t i = _ptr_array_synapses_1__synaptic_pre[_idx];
    const char _cond = _timedarray_3(0.0, i + (j * 2500)) < 0.2;


    if (_cond)
    {
        ///// vector_code['statement'] /////
                
        const int32_t j = _ptr_array_synapses_1__synaptic_post[_idx];
        const int32_t i = _ptr_array_synapses_1__synaptic_pre[_idx];
        double g_raw;
        g_raw = _lio_statement_1 * (_lio_statement_2 + (_lio_statement_3 * _timedarray_4(0.0, i + (j * 2500))));
        _ptr_array_synapses_1_g_raw[_idx] = g_raw;

    }

    ///// endblock kernel_maincode /////
}

void _run_synapses_1_group_variable_set_conditional_codeobject_1()
{
    using namespace brian;


    const int _N = _array_synapses_1_N[0];

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
		double* const dev_array_synapses_1_g_raw = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_g_raw[0]);
		const int _numg_raw = dev_dynamic_array_synapses_1_g_raw.size();
		int32_t* const dev_array_synapses_1__synaptic_pre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]);
		const int _numi = dev_dynamic_array_synapses_1__synaptic_pre.size();
		int32_t* const dev_array_synapses_1__synaptic_post = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]);
		const int _numj = dev_dynamic_array_synapses_1__synaptic_post.size();


    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    kernel_synapses_1_group_variable_set_conditional_codeobject_1, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;



        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_synapses_1_group_variable_set_conditional_codeobject_1, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_synapses_1_group_variable_set_conditional_codeobject_1)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_synapses_1_group_variable_set_conditional_codeobject_1 "
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
                        kernel_synapses_1_group_variable_set_conditional_codeobject_1, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_synapses_1_group_variable_set_conditional_codeobject_1\n"
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


    kernel_synapses_1_group_variable_set_conditional_codeobject_1<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            dev_array_synapses_1_N,
			dev_array_synapses_1_g_raw,
			_numg_raw,
			dev_array_synapses_1__synaptic_pre,
			_numi,
			dev_array_synapses_1__synaptic_post,
			_numj
        );

    CUDA_CHECK_ERROR("kernel_synapses_1_group_variable_set_conditional_codeobject_1");


}


