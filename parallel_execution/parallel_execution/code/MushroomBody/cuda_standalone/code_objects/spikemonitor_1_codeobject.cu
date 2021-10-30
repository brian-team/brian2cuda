#include "code_objects/spikemonitor_1_codeobject.h"
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
    // declare monitor cudaVectors
    __device__ cudaVector<double>* monitor_t;
    // declare monitor cudaVectors
    __device__ cudaVector<int32_t>* monitor_i;

    ///// support_code_lines /////
        
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



__global__ void _init_kernel_spikemonitor_1_codeobject()
{
        monitor_t = new cudaVector<double>();
        monitor_i = new cudaVector<int32_t>();
}

__global__ void
_run_kernel_spikemonitor_1_codeobject(
    int neurongroup_N,
    int32_t* count,
    // KERNEL_PARAMETERS
    int32_t* _ptr_array_spikemonitor_1_N,
	int32_t* _ptr_array_neurongroup_i,
	int32_t* _ptr_array_spikemonitor_1__source_idx,
	const double _value_array_defaultclock_t,
	int32_t* _ptr_array_neurongroup__spikespace,
	int32_t* _ptr_array_spikemonitor_1_count,
	int32_t* _ptr_array_spikemonitor_1_i,
	const int _numi,
	double* _ptr_array_spikemonitor_1_t,
	const int _numt
    )
{
    using namespace brian;
    int tid = threadIdx.x;
    int bid = blockIdx.x;


    // KERNEL_CONSTANTS
    const int _numN = 1;
	const int _num_source_i = 2500;
	const int _num_source_idx = 2500;
	const int _num_spikespace = 2501;
	const int _numcount = 2500;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    // scalar_code
        


    // using not parallel spikespace: filled from left with all spiking neuron IDs, -1 ends the list
    for(int i = 0; i < neurongroup_N; i++)
    {
        int32_t spiking_neuron = _ptr_array_neurongroup__spikespace[i];
        if(spiking_neuron != -1)
        {
            if(0 <= spiking_neuron && spiking_neuron < 2500)
            {
                int _idx = spiking_neuron;
                int _vectorisation_idx = _idx;

                // vector_code
                                
                const int32_t _source_i = _ptr_array_neurongroup_i[_idx];
                const double _source_t = _ptr_array_defaultclock_t[0];
                const double _to_record_t = _source_t;
                const int32_t _to_record_i = _source_i;


                // push to monitors
                monitor_t->push(_to_record_t);
                monitor_i->push(_to_record_i);

                count[_idx -0]++;

            }
        }
        else
        {

            break;
        }
    }
}


void _run_spikemonitor_1_codeobject()
{
    using namespace brian;


    const int _N = _array_spikemonitor_1_N[0];

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
		const int _num_source_i = 2500;
		const int _num_source_idx = 2500;
		const int _num_spikespace = 2501;
		const int _numcount = 2500;
		int32_t* const dev_array_spikemonitor_1_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_1_i[0]);
		const int _numi = dev_dynamic_array_spikemonitor_1_i.size();
		double* const dev_array_spikemonitor_1_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_1_t[0]);
		const int _numt = dev_dynamic_array_spikemonitor_1_t.size();


    static int num_threads, num_blocks;
    static size_t needed_shared_memory = 0;
    static bool first_run = true;
    if (first_run)
    {
_init_kernel_spikemonitor_1_codeobject<<<1,1>>>();

CUDA_CHECK_ERROR("_init_kernel_spikemonitor_1_codeobject");
num_blocks = 1;
num_threads = 1;


        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    _run_kernel_spikemonitor_1_codeobject, num_threads, needed_shared_memory)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, _run_kernel_spikemonitor_1_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "_run_kernel_spikemonitor_1_codeobject "
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
                        _run_kernel_spikemonitor_1_codeobject, num_threads, needed_shared_memory)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }


        else
        {
            printf("INFO _run_kernel_spikemonitor_1_codeobject\n"
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


_run_kernel_spikemonitor_1_codeobject<<<num_blocks, num_threads>>>(
        _num_spikespace-1,
        dev_array_spikemonitor_1_count,
        // HOST_PARAMETERS
        dev_array_spikemonitor_1_N,
			dev_array_neurongroup_i,
			dev_array_spikemonitor_1__source_idx,
			_array_defaultclock_t[0],
			dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace],
			dev_array_spikemonitor_1_count,
			dev_array_spikemonitor_1_i,
			_numi,
			dev_array_spikemonitor_1_t,
			_numt);

CUDA_CHECK_ERROR("_run_kernel_spikemonitor_1_codeobject");


}

__global__ void _debugmsg_kernel_spikemonitor_1_codeobject(
    // KERNEL_PARAMETERS
    int32_t* _ptr_array_spikemonitor_1_N,
	int32_t* _ptr_array_neurongroup_i,
	int32_t* _ptr_array_spikemonitor_1__source_idx,
	const double _value_array_defaultclock_t,
	int32_t* _ptr_array_neurongroup__spikespace,
	int32_t* _ptr_array_spikemonitor_1_count,
	int32_t* _ptr_array_spikemonitor_1_i,
	const int _numi,
	double* _ptr_array_spikemonitor_1_t,
	const int _numt
)
{
    using namespace brian;

    // KERNEL_CONSTANTS
    const int _numN = 1;
	const int _num_source_i = 2500;
	const int _num_source_idx = 2500;
	const int _num_spikespace = 2501;
	const int _numcount = 2500;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    printf("Number of spikes: %d\n", _ptr_array_spikemonitor_1_N[0]);
}

__global__ void _count_kernel_spikemonitor_1_codeobject(
    int* dev_num_events,
    // KERNEL_PARAMETERS
    int32_t* _ptr_array_spikemonitor_1_N,
	int32_t* _ptr_array_neurongroup_i,
	int32_t* _ptr_array_spikemonitor_1__source_idx,
	const double _value_array_defaultclock_t,
	int32_t* _ptr_array_neurongroup__spikespace,
	int32_t* _ptr_array_spikemonitor_1_count,
	int32_t* _ptr_array_spikemonitor_1_i,
	const int _numi,
	double* _ptr_array_spikemonitor_1_t,
	const int _numt
)
{
    using namespace brian;
    // TODO: fix int types, num_events and  cudaVector::size() are int but _ptr_array_spikemonitor_1_N[0] is size32_t
    int num_events;

    // KERNEL_CONSTANTS
    const int _numN = 1;
	const int _num_source_i = 2500;
	const int _num_source_idx = 2500;
	const int _num_spikespace = 2501;
	const int _numcount = 2500;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    num_events = monitor_t->size();
    _ptr_array_spikemonitor_1_N[0] = num_events;

    *dev_num_events = num_events;
}

__global__ void _copy_kernel_spikemonitor_1_codeobject(
    double* dev_monitor_t,
    int32_t* dev_monitor_i,
    int dummy  )
{
    using namespace brian;
    int index = 0;

    // copy monitors
    index = 0;
    for(int j = 0; j < monitor_t->size(); j++)
    {
        dev_monitor_t[index] = monitor_t->at(j);
        index++;
    }
    index = 0;
    for(int j = 0; j < monitor_i->size(); j++)
    {
        dev_monitor_i[index] = monitor_i->at(j);
        index++;
    }
}

void _copyToHost_spikemonitor_1_codeobject()
{
    using namespace brian;

    const std::clock_t _start_time = std::clock();

    // TODO: Use the correct dev_eventmonitor_N instead of dev_num_events
    //   and the correct _array_eventmonitor_N instead of host_num_events.
    //       use: dev_array_spikemonitor_1_N and _array_spikemonitor_1_N
    //   dev_array_.. gets copied to _array_... in objects.cu::write_arrays()
    //   copying it here would result in copying it twice.
    //   monitor_... and dev_monitor... store the exact same values, but we
    //   need monitor_... as cudaVector for changing size from device funtions.
    //   Maybe use cudaVector as default for dynamic arrays, then we would not
    //   need monitor... at all. This would mean changing the copying in objects.cu
    //   for dynamic arrays (currently we just use thrust device to host vector).
    int host_num_events;
    int* dev_num_events;

    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_num_events, sizeof(int))
            );

    // HOST_CONSTANTS
    const int _numN = 1;
		const int _num_source_i = 2500;
		const int _num_source_idx = 2500;
		const int _num_spikespace = 2501;
		const int _numcount = 2500;
		int32_t* const dev_array_spikemonitor_1_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_1_i[0]);
		const int _numi = dev_dynamic_array_spikemonitor_1_i.size();
		double* const dev_array_spikemonitor_1_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_1_t[0]);
		const int _numt = dev_dynamic_array_spikemonitor_1_t.size();

    _count_kernel_spikemonitor_1_codeobject<<<1,1>>>(
        dev_num_events,
        // HOST_PARAMETERS
        dev_array_spikemonitor_1_N,
			dev_array_neurongroup_i,
			dev_array_spikemonitor_1__source_idx,
			_array_defaultclock_t[0],
			dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace],
			dev_array_spikemonitor_1_count,
			dev_array_spikemonitor_1_i,
			_numi,
			dev_array_spikemonitor_1_t,
			_numt
        );

    CUDA_CHECK_ERROR("_count_kernel_spikemonitor_1_codeobject");

    CUDA_SAFE_CALL(
            cudaMemcpy(&host_num_events, dev_num_events, sizeof(int), cudaMemcpyDeviceToHost)
            );

    // resize monitor device vectors
    THRUST_CHECK_ERROR(
            dev_dynamic_array_spikemonitor_1_t.resize(host_num_events)
            );
    THRUST_CHECK_ERROR(
            dev_dynamic_array_spikemonitor_1_i.resize(host_num_events)
            );

    _copy_kernel_spikemonitor_1_codeobject<<<1,1>>>(
        thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_1_t[0]),
        thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_1_i[0]),
        0          );

    CUDA_CHECK_ERROR("_copy_kernel_spikemonitor_1_codeobject");
}

void _debugmsg_spikemonitor_1_codeobject()
{
    using namespace brian;

    // HOST_CONSTANTS
    const int _numN = 1;
		const int _num_source_i = 2500;
		const int _num_source_idx = 2500;
		const int _num_spikespace = 2501;
		const int _numcount = 2500;
		int32_t* const dev_array_spikemonitor_1_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_1_i[0]);
		const int _numi = dev_dynamic_array_spikemonitor_1_i.size();
		double* const dev_array_spikemonitor_1_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_1_t[0]);
		const int _numt = dev_dynamic_array_spikemonitor_1_t.size();

    // TODO: can't we acces the correct _array_eventmonitor_N[0]
    //   value here without any kernel call?
    //   Yes: use _array_spikemonitor_1_N
    _debugmsg_kernel_spikemonitor_1_codeobject<<<1,1>>>(
            // HOST_PARAMETERS
            dev_array_spikemonitor_1_N,
			dev_array_neurongroup_i,
			dev_array_spikemonitor_1__source_idx,
			_array_defaultclock_t[0],
			dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace],
			dev_array_spikemonitor_1_count,
			dev_array_spikemonitor_1_i,
			_numi,
			dev_array_spikemonitor_1_t,
			_numt
            );

    CUDA_CHECK_ERROR("_debugmsg_kernel_spikemonitor_1_codeobject");
}

