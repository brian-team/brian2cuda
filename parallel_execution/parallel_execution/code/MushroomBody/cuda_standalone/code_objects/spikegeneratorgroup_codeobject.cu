#include "code_objects/spikegeneratorgroup_codeobject.h"
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
    // Function to reset spikespace and set lastindex
    __global__ void
    _reset_spikegeneratorgroup_codeobject(
        //int32_t* _spikespace,
        int32_t* _previous_spikespace,
        ///// KERNEL_PARAMETERS /////
        int32_t* _ptr_array_spikegeneratorgroup__lastindex,
	int32_t* _ptr_array_spikegeneratorgroup__period_bins,
	int32_t* _ptr_array_spikegeneratorgroup__spikespace,
	int32_t* _ptr_array_spikegeneratorgroup__timebins,
	const int _num_timebins,
	int32_t* _ptr_array_spikegeneratorgroup_neuron_index,
	const int _numneuron_index,
	int32_t* _ptr_array_spikegeneratorgroup_spike_number,
	const int _numspike_number,
	const int64_t _value_array_defaultclock_timestep
        )
    {
        using namespace brian;

        int _idx = blockIdx.x * blockDim.x + threadIdx.x;

        // We need kernel_lines for time variables
        ///// kernel_lines /////
                
        const int64_t* _ptr_array_defaultclock_timestep = &_value_array_defaultclock_timestep;


        if (_idx >= 100) {
            return;
        }

        if (_idx == 0)
        {
            // The period in multiples of dt
            const int32_t _the_period = _ptr_array_spikegeneratorgroup__period_bins[0];
            // The spike times in multiples of dt
            int32_t _timebin          = _ptr_array_defaultclock_timestep[0];
            // index of the last spiking neuron in this spikespace
            int32_t _lastindex = _ptr_array_spikegeneratorgroup__lastindex[0];

            // Update the lastindex variable with the number of spikes from the
            // spikespace from the previous time step
            _lastindex += _previous_spikespace[100];

            // Now reset the _lastindex if the priod has passed
            if (_the_period > 0) {
                _timebin %= _the_period;
                // If there is a periodicity in the SpikeGenerator, we need to reset the
                // lastindex when the period has passed
                if (_lastindex > 0 && _ptr_array_spikegeneratorgroup__timebins[_lastindex - 1] >= _timebin)
                    _lastindex = 0;
            }
            _ptr_array_spikegeneratorgroup__lastindex[0] = _lastindex;

            // Reset spikespace counter for this time step
            _ptr_array_spikegeneratorgroup__spikespace[100] = 0;
        }

        // Reset the entire spikespace
        _ptr_array_spikegeneratorgroup__spikespace[_idx] = -1;
    }

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



__global__ void
_run_kernel_spikegeneratorgroup_codeobject(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    int32_t* _ptr_array_spikegeneratorgroup__lastindex,
	int32_t* _ptr_array_spikegeneratorgroup__period_bins,
	int32_t* _ptr_array_spikegeneratorgroup__spikespace,
	int32_t* _ptr_array_spikegeneratorgroup__timebins,
	const int _num_timebins,
	int32_t* _ptr_array_spikegeneratorgroup_neuron_index,
	const int _numneuron_index,
	int32_t* _ptr_array_spikegeneratorgroup_spike_number,
	const int _numspike_number,
	const int64_t _value_array_defaultclock_timestep
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _num_lastindex = 1;
	const int _num_period_bins = 1;
	const int _num_spikespace = 101;

    ///// kernel_lines /////
        
    const int64_t* _ptr_array_defaultclock_timestep = &_value_array_defaultclock_timestep;


    assert(THREADS_PER_BLOCK == blockDim.x);


    if(_idx >= _N)
    {
        return;
    }

    // The period in multiples of dt
    const int32_t _the_period = _ptr_array_spikegeneratorgroup__period_bins[0];
    // The spike times in multiples of dt
    int32_t _timebin = _ptr_array_defaultclock_timestep[0];

    if (_the_period > 0)
        _timebin %= _the_period;

    // We can have at most one spike per neuron per time step, which is the number of
    // threads we call this kernel with. Hence, no need for any loops.

    // _spike_idx runs through the spikes in the spike generator
    int _spike_idx = _idx + _ptr_array_spikegeneratorgroup__lastindex[0];

    // TODO: Solve this smarter. Currently, we will call the reset kernel and this
    // kernel at each time step even if the spikegenerator has emitted all its spikes!
    // Instead, we should know on the host when this happened and not call any kernels.
    // See also #193
    if (_spike_idx >= _num_timebins)
        return;

    // If the spike time of this spike comes after the current time bin, do nothing
    if (_ptr_array_spikegeneratorgroup__timebins[_spike_idx] > _timebin)
    {
        return;
    }

    // Else add the spiking neuron to the spikespace
    int32_t _neuron_id = _ptr_array_spikegeneratorgroup_neuron_index[_spike_idx];
    int32_t _spikespace_index = atomicAdd(&_ptr_array_spikegeneratorgroup__spikespace[100], 1);
    _ptr_array_spikegeneratorgroup__spikespace[_spikespace_index] = _neuron_id;

}


void _run_spikegeneratorgroup_codeobject()
{
    using namespace brian;


    const int _N = 100;

    ///// HOST_CONSTANTS ///////////
    const int _num_lastindex = 1;
		const int _num_period_bins = 1;
		const int _num_spikespace = 101;
		int32_t* const dev_array_spikegeneratorgroup__timebins = thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup__timebins[0]);
		const int _num_timebins = dev_dynamic_array_spikegeneratorgroup__timebins.size();
		int32_t* const dev_array_spikegeneratorgroup_neuron_index = thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_neuron_index[0]);
		const int _numneuron_index = dev_dynamic_array_spikegeneratorgroup_neuron_index.size();
		int32_t* const dev_array_spikegeneratorgroup_spike_number = thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_spike_number[0]);
		const int _numspike_number = dev_dynamic_array_spikegeneratorgroup_spike_number.size();


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
                    _run_kernel_spikegeneratorgroup_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;





        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    _run_kernel_spikegeneratorgroup_codeobject, num_threads, needed_shared_memory)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, _run_kernel_spikegeneratorgroup_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "_run_kernel_spikegeneratorgroup_codeobject "
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
                        _run_kernel_spikegeneratorgroup_codeobject, num_threads, needed_shared_memory)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }


        else
        {
            printf("INFO _run_kernel_spikegeneratorgroup_codeobject\n"
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

    // Note: If we have no delays, there is only one spikespace and
    //       current_idx equals previous_idx.
    _reset_spikegeneratorgroup_codeobject<<<num_blocks, num_threads, 0, spikegenerator_stream>>>(
            dev_array_spikegeneratorgroup__spikespace[previous_idx_array_spikegeneratorgroup__spikespace],
            ///// HOST_PARAMETERS /////
            dev_array_spikegeneratorgroup__lastindex,
			dev_array_spikegeneratorgroup__period_bins,
			dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace],
			dev_array_spikegeneratorgroup__timebins,
			_num_timebins,
			dev_array_spikegeneratorgroup_neuron_index,
			_numneuron_index,
			dev_array_spikegeneratorgroup_spike_number,
			_numspike_number,
			_array_defaultclock_timestep[0]
        );

    CUDA_CHECK_ERROR("_reset_spikegeneratorgroup_codeobject");

    _run_kernel_spikegeneratorgroup_codeobject<<<num_blocks, num_threads,0, spikegenerator_stream>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            dev_array_spikegeneratorgroup__lastindex,
			dev_array_spikegeneratorgroup__period_bins,
			dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace],
			dev_array_spikegeneratorgroup__timebins,
			_num_timebins,
			dev_array_spikegeneratorgroup_neuron_index,
			_numneuron_index,
			dev_array_spikegeneratorgroup_spike_number,
			_numspike_number,
			_array_defaultclock_timestep[0]
        );

    CUDA_CHECK_ERROR("_run_kernel_spikegeneratorgroup_codeobject");
    CUDA_SAFE_CALL(cudaDeviceSynchronize());


}


