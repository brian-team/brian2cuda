#include <stdlib.h>
#include <ctime>
#include <time.h>
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include <curand.h>
#include <brianlib/curand_buffer.h>

#include "code_objects/synapses_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_1_synapses_create_generator_codeobject.h"


#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include "cuda_profiler_api.h"

// Makro for file and line information in _cudaSafeCall
#define COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(a, b, c, d) \
    _copyHostArrayToDeviceSymbol(a, b, c, d, __FILE__, __LINE__)

// support code starts here 

////// SUPPORT CODE ///////
namespace {
    double _host_rand(const int _vectorisation_idx);
    double _host_randn(const int _vectorisation_idx);
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx);

    ///// block extra_device_helper /////

        __global__ void
        _reset_neurongroup_1_thresholder_codeobject(
            int32_t* eventspace
            )
        {
            using namespace brian;

            int _idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (_idx >= 100) {
                return;
            }

            if (_idx == 0) {
                // reset eventspace counter
                eventspace[100] = 0;
            }

            // reset eventspace
            eventspace[_idx] = -1;
        }

            __global__ void
        _reset_neurongroup_thresholder_codeobject(
            int32_t* eventspace
            )
        {
            using namespace brian;

            int _idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (_idx >= 2500) {
                return;
            }

            if (_idx == 0) {
                // reset eventspace counter
                eventspace[2500] = 0;
            }

            // reset eventspace
            eventspace[_idx] = -1;
        }

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

                        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
                        __device__ double* _namespace_timedarray_2_values;
                        #else
                        double* _namespace_timedarray_2_values;
                        #endif
    __host__ __device__
    static inline double _timedarray_2(const double t, const int i)
    {
        const double epsilon = 10.000000000000000000 / 1048576;
        if (i < 0 || i >= 250000)
            return NAN;
        int timestep = (int)((t/epsilon + 0.5)/1048576);
        if(timestep < 0)
           timestep = 0;
        else if(timestep >= 1)
            timestep = 1-1;
        return _namespace_timedarray_2_values[timestep*250000 + i];
    }

        // declare monitor cudaVectors
    __device__ cudaVector<double>* monitor_t;
    // declare monitor cudaVectors
    __device__ cudaVector<int32_t>* monitor_i;

        
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
                        __device__ double* _namespace_timedarray_1_values;
                        #else
                        double* _namespace_timedarray_1_values;
                        #endif
    __host__ __device__
    static inline double _timedarray_1(const double t, const int i)
    {
        const double epsilon = 10.000000000000000000 / 1048576;
        if (i < 0 || i >= 250000)
            return NAN;
        int timestep = (int)((t/epsilon + 0.5)/1048576);
        if(timestep < 0)
           timestep = 0;
        else if(timestep >= 1)
            timestep = 1-1;
        return _namespace_timedarray_1_values[timestep*250000 + i];
    }

    inline __host__ __device__
    double _brian_clip(const double value,
                              const double a_min,
                              const double a_max)
    {
        if (value < a_min)
            return a_min;
        if (value > a_max)
            return a_max;
        return value;
    }

        // vector_t<T> is an alias for thrust:host_vector<T>
    template <typename T> using vector_t = thrust::host_vector<T>;
    // tuple type typedef
    typedef std::tuple<std::string, size_t, int> tuple_t;

    std::vector<tuple_t> memory_recorder;

    // Functions for online update of mean and std
    // for a new value newValue, compute the new count, new mean, the new M2.
    // mean accumulates the mean of the entire dataset
    // M2 aggregates the squared distance from the mean
    // count aggregates the number of samples seen so far
    inline void updateMeanStd(int &count, double &mean, double& M2, double newValue){
        count += 1;
        double delta = newValue - mean;
        mean += delta / count;
        double delta2 = newValue - mean;
        M2 += delta * delta2;
    }

    // get std from aggregated M2 value
    double getStd(int count, double M2){
        if (count < 2){
            return NAN;
        }
        double variance = M2 / (count - 1);
        double stdValue = sqrt(variance);
        return stdValue;
    }

    // Copy the data from a host array to global device memory and copy the
    // symbol to a global device variable.
    // host_array: host array with data to copy
    // device_symbol: global __device__ variable of same type as `host_array`
    // num_elements: number of elements in host_array to copy
    // NOTE: T can be a pointer variable itself (when copying 2D arrays)
    template <typename T>
    inline void _copyHostArrayToDeviceSymbol(const T *host_array, T *&device_symbol,
            int num_elements, const char* name, const char* file,
            const int line){
        T *d_ptr_tmp;
        size_t bytes = sizeof(T) * num_elements;
        // allocate device memory
        _cudaSafeCall(
                cudaMalloc((void**)&d_ptr_tmp, bytes),
                file, line, "cudaMalloc");
        // copy data from host array to device
        _cudaSafeCall(
                cudaMemcpy(d_ptr_tmp, host_array, bytes, cudaMemcpyHostToDevice),
                file, line, "cudaMemcpy");
        // copy the device data pointer to the global device symbol
        _cudaSafeCall(
                cudaMemcpyToSymbol(device_symbol, &d_ptr_tmp, sizeof(T*)),
                file, line, "cudaMemcpyToSymbol");
        memory_recorder.push_back(std::make_tuple(name, bytes, num_elements));
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


// support code ends here 

// synapses_pre_push_spikes start here

__global__ void _run_synapses_pre_push_spikes_advance_kernel()
{
    using namespace brian;
    int tid = threadIdx.x;
    synapses_pre.queue->advance(
        tid);
}

__global__ void
_run_synapses_pre_push_spikes_push_kernel(
    int num_parallel_blocks,
    int _num_blocks,
    int _num_threads,
    int32_t* _ptr_array_spikegeneratorgroup__spikespace)
{
    // apperently this is not always true and that is why _num_threads is passed as function argument
    // if this assert never fails, we could remove the _num_threads form the argument list
    assert(blockDim.x == _num_threads);

    using namespace brian;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int post_neuron_bid = bid % num_parallel_blocks;
    int pre_neuron_idx = bid / num_parallel_blocks;

    int32_t spiking_neuron = _ptr_array_spikegeneratorgroup__spikespace[pre_neuron_idx];
    assert(spiking_neuron != -1);

    // push to spikequeue if spiking_neuron is in sources of current SynapticPathway
    if(synapses_pre.spikes_start <= spiking_neuron && spiking_neuron < synapses_pre.spikes_stop)
    {
        synapses_pre.queue->push_bundles(
            post_neuron_bid,
            tid,
            _num_threads,
            spiking_neuron - synapses_pre.spikes_start);
    }
}

void _run_synapses_pre_push_spikes()
{
    using namespace brian;


    ///// HOST_CONSTANTS /////
    const int _num_spikespace = 101;

    if (synapses_pre_scalar_delay)
    {
        int num_eventspaces = dev_array_spikegeneratorgroup__spikespace.size();
        synapses_pre_eventspace_idx = (current_idx_array_spikegeneratorgroup__spikespace - synapses_pre_delay + num_eventspaces) % num_eventspaces;

        //////////////////////////////////////////////
        //// No pushing in no_or_const_delay_mode ////
        //////////////////////////////////////////////
    }
    else if (synapses_pre_max_size > 0)
    {

        // get the number of spiking neurons
        int32_t num_spiking_neurons;
        CUDA_SAFE_CALL(
                cudaMemcpy(&num_spiking_neurons,
                    dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace] + _num_spikespace - 1,
                    sizeof(int32_t), cudaMemcpyDeviceToHost)
                );

        // advance spike queues
        _run_synapses_pre_push_spikes_advance_kernel<<<1, num_parallel_blocks>>>();

        CUDA_CHECK_ERROR("_run_synapses_pre_push_spikes_advance_kernel");

        static int num_threads, num_blocks;
        static size_t needed_shared_memory;
        static bool first_run = true;
        if (first_run)
        {

            needed_shared_memory = 0;

            // We don't need more then max(num_synapses) threads per block.
            num_threads = synapses_pre_max_size;
            if (num_threads > max_threads_per_block)
            {
                num_threads = max_threads_per_block;
            }

            // calculate theoretical occupancy
            int max_active_blocks;
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                        _run_synapses_pre_push_spikes_push_kernel, num_threads,
                        needed_shared_memory)
                    );

            float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                              (float)(max_threads_per_sm / num_threads_per_warp);

            // check if we have enough ressources to call kernel with given
            // number of blocks and threads
            struct cudaFuncAttributes funcAttrib;
            CUDA_SAFE_CALL(
                    cudaFuncGetAttributes(&funcAttrib, _run_synapses_pre_push_spikes_push_kernel)
                    );
            if (num_threads > funcAttrib.maxThreadsPerBlock)
            {
                // use the max num_threads before launch failure
                num_threads = funcAttrib.maxThreadsPerBlock;
                printf("WARNING Not enough ressources available to call "
                       "_run_synapses_pre_push_spikes_push_kernel "
                       "with maximum possible threads per block (%u). "
                       "Reducing num_threads to %u. (Kernel needs %i "
                       "registers per block, %i bytes of "
                       "statically-allocated shared memory per block, %i "
                       "bytes of local memory per thread and a total of %i "
                       "bytes of user-allocated constant memory)\n",
                       max_threads_per_block, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes);
            }
            else
            {
                printf("INFO _run_synapses_pre_push_spikes_push_kernel\n"
                       "\t%u blocks per spiking neuron\n"
                       "\t%u threads\n"
                       "\t%i registers per block\n"
                       "\t%i bytes statically-allocated shared memory per block\n"
                       "\t%i bytes local memory per thread\n"
                       "\t%i bytes user-allocated constant memory\n"
                       "\t%.3f theoretical occupancy\n",
                       num_parallel_blocks, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes, occupancy);
            }
            first_run = false;
        }


        if (num_spiking_neurons > 0)
        {
            num_blocks = num_parallel_blocks * num_spiking_neurons;

            _run_synapses_pre_push_spikes_push_kernel<<<num_blocks, num_threads, needed_shared_memory>>>(
                    num_parallel_blocks,
                    num_blocks,
                    num_threads,
                    dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace]);

            CUDA_CHECK_ERROR("_run_synapses_pre_push_spikes_push_kernel");
        }
    }

}

// synapses_pre_push_spikes end here

// synapses_pre_initialise_queue starts here

__global__ void _run_synapses_pre_initialise_queue_kernel(
    int _source_N,
    int _num_blocks,
    int _num_threads,
    double _dt,
    int _syn_N,
    int num_queues,
    bool new_mode)
{
    using namespace brian;

    int tid = threadIdx.x;

    synapses_pre.queue->prepare(
        tid,
        _num_threads,
        _num_blocks,
        0,
        _source_N,
        _syn_N,
        num_queues,
        synapses_pre_num_synapses_by_pre,
        synapses_pre_num_synapses_by_bundle,
        synapses_pre_num_unique_delays_by_pre,
        synapses_pre_unique_delays,
        synapses_pre_global_bundle_id_start_by_pre,
        synapses_pre_synapses_offset_by_bundle,
        synapses_pre_synapse_ids,
        synapses_pre_synapse_ids_by_pre,
        synapses_pre_unique_delays_offset_by_pre,
        synapses_pre_unique_delay_start_idcs);
    synapses_pre.no_or_const_delay_mode = new_mode;
}


void _run_synapses_pre_initialise_queue()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();
    const double to_MB = 1.0 / (1024.0 * 1024.0);

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
        double* const _array_synapses_delay = thrust::raw_pointer_cast(&_dynamic_array_synapses_delay[0]);
        const int _numdelay = _dynamic_array_synapses_delay.size();

    ///// pointers_lines /////
        
    int32_t*   _ptr_array_synapses_N = _array_synapses_N;
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    double* __restrict  _ptr_array_synapses_delay = _array_synapses_delay;


    int64_t syn_N_check = _ptr_array_synapses_N[0];

    if (syn_N_check == 0){
        return;
    }
    else if (syn_N_check > INT_MAX){
        printf("ERROR: There are more Synapses (%lu) than an int can "
               "hold on this system (%u).\n", syn_N_check, INT_MAX);
    }
    // total number of synapses
    int syn_N = (int)syn_N_check;

    // simulation time step
    double dt = _ptr_array_defaultclock_dt[0];
    // number of neurons in source group
    int source_N = 100;
    // number of neurons in target group
    int target_N = 2500;

    // TODO: for multiple SynapticPathways for the same Synapses object (on_pre and on_post) the following copy is identical in both pathways initialise templates
    // delay (on device) was potentially set in group_variable_set_conditional and needs to be copied to host
    _dynamic_array_synapses_delay = dev_dynamic_array_synapses_delay;

    //////////////////////
    // Scalar variables //
    //////////////////////

    // total number of (preID, postBlock) pairs
    int num_pre_post_blocks = num_parallel_blocks * source_N;
    // size of the connectivity matrix (equal number of synapses)
    int size_connectivity_matrix = 0;

    // statistics of number of synapses per (preID, postBlock) pair
    int sum_num_elements = 0;
    int count_num_elements = 0;
    double mean_num_elements = 0;
    double M2_num_elements = 0;

    // statistics of number of unique delays per (preID, postBlock) pair
    int sum_num_unique_elements = 0;
    int count_num_unique_elements = 0;
    double mean_num_unique_elements = 0;
    double M2_num_unique_elements = 0;

    // total number of bundles in all (preID, postBlock) pairs (not known yet)
    int num_bundle_ids = 0;

    // statistics of number of synapses per bundle
    int sum_bundle_sizes = 0;
    int count_bundle_sizes = 0;
    double mean_bundle_sizes = 0;
    double M2_bundle_sizes = 0;


    ////////////////////////////////////////////////////////
    // Create array and vector variables (in host memory) //
    ////////////////////////////////////////////////////////

    /* VARIABLE NAMING:
     * Not scalar variables are named after TYPE_NAME_STRUCTURE, with:
     * STRUCTURE: the first array dimensions structure (`by_pre`, `by_bundle` or none)
     *   `by_pre`: Array (host pointer type) of size `num_pre_post_blocks`,
     *             which is the number of (preID, postBlock) pairs.
     *   `by_bundle`: thrust::host_vector, size of total number of bundles,
     *                which is one for each delay in each (preID, postBlock) pair.
     *                Different (preID, postBlock) pairs can have different sets
     *                of delay values -> each bundle gets a global bundleID
     *   none: If no STRUCTURE given, it's a one dim array storing everything
     * TYPE: data type in STRUCTURE (`h`, `h_vec`, `h_ptr`, `d_ptr`), with
     *       `h`: host value, `h_vec`: host vector, `h_ptr`: host pointer,
     *       `d_ptr`: device pointer (pointing to device, stored in host memory)
     * NAME: the variable name
     *
     * EXAMPLES:
     * `h_vec_delays_by_pre` - an array [size = num_pre_post_blocks] of host
     *                         vectors, each storing delay values of a
     *                         (preID, postBlock) pair
     * `h_num_synapses_by_bundle` - a host vector of integers specifying the
     *                              number of synapses in a bundle
     * `d_ptr_synapse_ids` - a device pointer to synapse IDs (all of them)
     */

    // synapse IDs for each (preID, postBlock) pair
    vector_t<int32_t>* h_vec_synapse_ids_by_pre = new vector_t<int32_t>[num_pre_post_blocks];
    // array of synapse IDs in device memory for each (preID, postBlock) pair
    int32_t** d_ptr_synapse_ids_by_pre;
    // number of synapses for each (preID, postBlock) pair
    int* h_num_synapses_by_pre;

    // delay for each synapse in `h_vec_synapse_ids_by_pre`,
    // only used to sort synapses by delay
    vector_t<int>* h_vec_delays_by_pre = new vector_t<int>[num_pre_post_blocks];
    // array of vectors with unique delays and start indices in synapses arrays
    vector_t<int>* h_vec_unique_delays_by_pre;
    vector_t<int>* h_vec_unique_delay_start_idcs_by_pre;
    // offset in array of all synapse IDs sorted by bundles (we are storing the
    // offset as 32bit int instead of a 64bit pointer to the bundle start)
    vector_t<int> h_synapses_offset_by_bundle;
    // number of synapses in each bundle
    vector_t<int> h_num_synapses_by_bundle;
    // start of global bundle ID per (preID, postBlock) pair (+ total num bundles)
    int* h_global_bundle_id_start_by_pre = new int[num_pre_post_blocks + 1];


    // we need to allocate device memory for synapse IDs independent of delay mode
    int32_t* d_ptr_synapse_ids;
    size_t memory_synapse_ids = sizeof(int32_t) * syn_N;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&d_ptr_synapse_ids, memory_synapse_ids)
            );
    memory_recorder.push_back(std::make_tuple("synapse IDs", memory_synapse_ids, syn_N));


    //fill vectors of connectivity matrix with synapse IDs and delays (in units of simulation time step)
    int max_delay = (int)(_dynamic_array_synapses_delay[0] / dt + 0.5);
    int min_delay = max_delay;
    for(int syn_id = 0; syn_id < syn_N; syn_id++)  // loop through all synapses
    {


        // Code generation checks
        assert(0 == 0);

        assert(0 == 0);

        // pre/post_neuron_id are integers from 0 to Nsource/Ntarget (from corresponding
        // SynapticPathway) this is relevant only when using Subgroups where they might
        // be NOT equal to the idx in their NeuronGroup
        int32_t pre_neuron_id = _dynamic_array_synapses__synaptic_pre[syn_id] - 0;
        int32_t post_neuron_id = _dynamic_array_synapses__synaptic_post[syn_id] - 0;

        int delay = (int)(_dynamic_array_synapses_delay[syn_id] / dt + 0.5);
        if (delay > max_delay)
            max_delay = delay;
        if (delay < min_delay)
            min_delay = delay;

        // each parallel executed cuda block gets an equal part of post neuron IDs
        int post_block_id = (post_neuron_id * num_parallel_blocks) / target_N;
        // we store synapses for each pre neuron and post block
        int pre_post_block_id = pre_neuron_id * num_parallel_blocks + post_block_id;

        h_vec_synapse_ids_by_pre[pre_post_block_id].push_back(syn_id);
        h_vec_delays_by_pre[pre_post_block_id].push_back(delay);
    }
    int num_queues = max_delay + 1;  // we also need a current step

    bool scalar_delay = (max_delay == min_delay);
    if (scalar_delay)
        synapses_pre_delay = max_delay;
    // Delete delay (in sec) on device, we don't need it
    // TODO: don't copy these delays to the device in first place, see #83
    dev_dynamic_array_synapses_delay.clear();
    dev_dynamic_array_synapses_delay.shrink_to_fit();
    CUDA_CHECK_MEMORY();
    size_t used_device_memory_after_dealloc = used_device_memory;

    ///////////////////////////////////////////////////////
    // Memory allocations which depend on the delay mode //
    ///////////////////////////////////////////////////////

    if (scalar_delay)
    {
        h_num_synapses_by_pre = new int[num_pre_post_blocks];
        d_ptr_synapse_ids_by_pre = new int32_t*[num_pre_post_blocks];
    }

    // allocate memory only if the delays are not all the same
    if (!scalar_delay)
    {

        h_vec_unique_delay_start_idcs_by_pre = new vector_t<int>[num_pre_post_blocks];
        h_vec_unique_delays_by_pre = new vector_t<int>[num_pre_post_blocks];

    }
    int global_bundle_id_start = 0;

    // loop through connectivity matrix [(preID, postBlock) pairs]
    for(int i = 0; i < num_pre_post_blocks; i++)
    {
        // i is pre_post_block_id

        int num_elements = h_vec_synapse_ids_by_pre[i].size();
        size_connectivity_matrix += num_elements;
        if (num_elements > synapses_pre_max_size)
            synapses_pre_max_size = num_elements;

        if (!scalar_delay)
        {
            // for this (preID, postBlock), sort all synapses by delay,
            // reduce the delay arrays to unique delays and store the
            // start indices in the synapses array for each unique delay

            typedef vector_t<int>::iterator itr;

            // sort synapses (values) and delays (keys) by delay
            thrust::sort_by_key(
                    h_vec_delays_by_pre[i].begin(),     // keys start
                    h_vec_delays_by_pre[i].end(),       // keys end
                    h_vec_synapse_ids_by_pre[i].begin() // values start
                    );

            // worst case: number of unique delays is num_elements
            h_vec_unique_delay_start_idcs_by_pre[i].resize(num_elements);

            // Initialise the unique delay start idcs array as a sequence
            thrust::sequence(h_vec_unique_delay_start_idcs_by_pre[i].begin(),
                    h_vec_unique_delay_start_idcs_by_pre[i].end());

            // get delays (keys) and values (indices) for first occurence of each delay value
            thrust::pair<itr, itr> end_pair = thrust::unique_by_key(
                    h_vec_delays_by_pre[i].begin(),                 // keys start
                    h_vec_delays_by_pre[i].end(),                   // keys end
                    h_vec_unique_delay_start_idcs_by_pre[i].begin() // values start (position in original delay array)
                    );

            itr unique_delay_end = end_pair.first;
            itr idx_end = end_pair.second;

            // erase unneded vector entries
            h_vec_unique_delay_start_idcs_by_pre[i].erase(
                    idx_end, h_vec_unique_delay_start_idcs_by_pre[i].end());
            // free not used but allocated host memory
            h_vec_unique_delay_start_idcs_by_pre[i].shrink_to_fit();
            h_vec_delays_by_pre[i].erase(unique_delay_end,
                    h_vec_delays_by_pre[i].end());
            // delay_by_pre holds the set of unique delays now
            // we don't need shrink_to_fit, swap takes care of that
            h_vec_unique_delays_by_pre[i].swap(h_vec_delays_by_pre[i]);

            int num_unique_elements = h_vec_unique_delays_by_pre[i].size();
            sum_num_unique_elements += num_unique_elements;

            if (num_unique_elements > synapses_pre_max_num_unique_delays)
                synapses_pre_max_num_unique_delays = num_unique_elements;

            // we need a start ID per i (pre_post_block_id) to calc the global
            // bundle ID from the local bundle_idx when pushing
            h_global_bundle_id_start_by_pre[i] = global_bundle_id_start;
            global_bundle_id_start += num_unique_elements;
            // the local bundle_idx goes from 0 to num_bundles for each i (pre_post_block_id)
            for (int bundle_idx = 0; bundle_idx < num_unique_elements; bundle_idx++)
            {
                // find the start idx in the synapses array for this delay (bundle)
                int synapses_start_idx = h_vec_unique_delay_start_idcs_by_pre[i][bundle_idx];
                // find the number of synapses for this delay (bundle)
                int num_synapses;
                if (bundle_idx == num_unique_elements - 1)
                    num_synapses = num_elements - synapses_start_idx;
                else
                    num_synapses = h_vec_unique_delay_start_idcs_by_pre[i][bundle_idx + 1] - synapses_start_idx;
                h_num_synapses_by_bundle.push_back(num_synapses);
                if (num_synapses > synapses_pre_max_bundle_size)
                    synapses_pre_max_bundle_size = num_synapses;

                // copy this bundle to device and store the device pointer
                int32_t* d_this_bundle = d_ptr_synapse_ids + sum_bundle_sizes;
                int32_t* h_this_bundle = thrust::raw_pointer_cast(&h_vec_synapse_ids_by_pre[i][synapses_start_idx]);
                size_t memory_size = sizeof(int32_t) * num_synapses;
                CUDA_SAFE_CALL(
                        cudaMemcpy(d_this_bundle, h_this_bundle, memory_size, cudaMemcpyHostToDevice)
                        );

                h_synapses_offset_by_bundle.push_back(sum_bundle_sizes);
                sum_bundle_sizes += num_synapses;
                updateMeanStd(count_bundle_sizes, mean_bundle_sizes, M2_bundle_sizes, num_synapses);
            }

            updateMeanStd(count_num_unique_elements, mean_num_unique_elements,
                    M2_num_unique_elements, num_unique_elements);

        }  // end if (!scalar_delay)
        else   // scalar_delay
        {
            // copy the synapse IDs and the number of synapses for this
            // (preID, postBlock) to device and store the device pointer

            h_num_synapses_by_pre[i] = num_elements;

            d_ptr_synapse_ids_by_pre[i] = d_ptr_synapse_ids + sum_num_elements;
            CUDA_SAFE_CALL(
                    cudaMemcpy(d_ptr_synapse_ids_by_pre[i],
                        thrust::raw_pointer_cast(&(h_vec_synapse_ids_by_pre[i][0])),
                        sizeof(int32_t) * num_elements,
                        cudaMemcpyHostToDevice)
                    );
        }

        sum_num_elements += num_elements;
        updateMeanStd(count_num_elements, mean_num_elements, M2_num_elements, num_elements);
    }  // end for loop through connectivity matrix
    printf("INFO connectivity matrix has size %i, number of (pre neuron ID, post neuron block) pairs is %u\n",
            size_connectivity_matrix, num_pre_post_blocks);

    if (scalar_delay)
    {
        // synapses size
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(h_num_synapses_by_pre,
                synapses_pre_num_synapses_by_pre, num_pre_post_blocks,
                "number of synapses per pre/post block");
        // synapses id
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(d_ptr_synapse_ids_by_pre,
                synapses_pre_synapse_ids_by_pre, num_pre_post_blocks,
                "pointers to synapse IDs");
    }

    else  // not scalar_delay
    {
        // Since we now know the total number of unique delays over all
        // (preID, postBlock) pairs, we can allocate the device memory
        size_t memory_unique_delays_by_pre = sizeof(int) * sum_num_unique_elements;
        assert(sum_bundle_sizes == syn_N);

        // array of all unique delas, sorted first by pre_post_block and per
        // pre_post_block by delay
        int *d_ptr_unique_delays;
        CUDA_SAFE_CALL(
                cudaMalloc((void**)&d_ptr_unique_delays, memory_unique_delays_by_pre)
                );
        memory_recorder.push_back(std::make_tuple(
                    "unique delays", memory_unique_delays_by_pre,
                    sum_num_unique_elements));

        int sum_num_unique_elements_bak = sum_num_unique_elements;

        // reset sum_num_unique_elements, we will use it to offset cudaMemcy correctly
        sum_num_unique_elements = 0;
        for(int i = 0; i < num_pre_post_blocks; i++)  // loop through connectivity matrix again
        {

            int num_elements = h_vec_synapse_ids_by_pre[i].size();
            int num_unique_elements = h_vec_unique_delays_by_pre[i].size();

            if(num_elements > 0)
            {
                // copy the unique delays to the device and store the device pointers
                CUDA_SAFE_CALL(
                        cudaMemcpy(d_ptr_unique_delays
                                       + sum_num_unique_elements,
                                   thrust::raw_pointer_cast(
                                       &(h_vec_unique_delays_by_pre[i][0])),
                                   sizeof(int)*num_unique_elements,
                                   cudaMemcpyHostToDevice)
                        );


                sum_num_unique_elements += num_unique_elements;
            }  // end if(num_elements < 0)
        }  // end second loop connectivity matrix
        assert(sum_num_unique_elements_bak == sum_num_unique_elements);

        // pointer to start of unique delays array
        CUDA_SAFE_CALL(
                cudaMemcpyToSymbol(synapses_pre_unique_delays,
                                   &d_ptr_unique_delays,
                                   sizeof(d_ptr_unique_delays))
                );

        num_bundle_ids = sum_num_unique_elements;

        // add num_bundle_ids as last entry
        h_global_bundle_id_start_by_pre[num_pre_post_blocks] = num_bundle_ids;

        // floor(mean(h_num_synapses_by_bundle))
        synapses_pre_mean_bundle_size = sum_bundle_sizes / num_bundle_ids;

        // pointer to start of synapse IDs array
        CUDA_SAFE_CALL(
                cudaMemcpyToSymbol(synapses_pre_synapse_ids, &d_ptr_synapse_ids,
                                   sizeof(d_ptr_synapse_ids))
                );

        // size by bundle
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(
                thrust::raw_pointer_cast(&h_num_synapses_by_bundle[0]),
                synapses_pre_num_synapses_by_bundle, num_bundle_ids,
                "number of synapses per bundle");

        // synapses offset by bundle
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(
                thrust::raw_pointer_cast(&h_synapses_offset_by_bundle[0]),
                synapses_pre_synapses_offset_by_bundle, num_bundle_ids,
                "synapses bundle offset");

        // global bundle id start idx by pre
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(
                h_global_bundle_id_start_by_pre,
                synapses_pre_global_bundle_id_start_by_pre,
                num_pre_post_blocks + 1, "global bundle ID start");


    }  // end if (!scalar_delay)

    ////////////////////////////////////////////////////
    //// PRINT INFORMATION ON MEMORY USAGE AND TIME ////
    ////////////////////////////////////////////////////

    // TODO print statistics!

    // sum all allocated memory
    size_t total_memory = 0;
    int max_string_length = 0;
    for(auto const& tuple: memory_recorder){
        total_memory += std::get<1>(tuple);
        int str_len = std::get<0>(tuple).length();
        if (str_len > max_string_length)
            max_string_length = str_len;
    }
    double total_memory_MB = total_memory * to_MB;
    max_string_length += 5;

    // sort tuples by used memory
    std::sort(begin(memory_recorder), end(memory_recorder),
            [](tuple_t const &t1, tuple_t const &t2) {
            return std::get<1>(t1) > std::get<1>(t2); // or use a custom compare function
            }
            );

    double std_num_elements = getStd(count_num_elements, M2_num_elements);
    double std_bundle_sizes = getStd(count_bundle_sizes, M2_bundle_sizes);
    double std_num_unique_elements = getStd(count_num_unique_elements, M2_num_unique_elements);

    // print memory information
    std::cout.precision(1);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << "INFO: synapse statistics and memory usage for synapses_pre:\n"
        << "\tnumber of synapses: " << syn_N << "\n"
        << "\tnumber of bundles: " << num_bundle_ids << "\n"
        << "\tnumber of pre/post blocks: " << num_pre_post_blocks << "\n"
        << "\tnumber of synapses over all pre/post blocks:\n"
        << "\t\tmean: " << mean_num_elements << "\tstd: "
            << std_num_elements << "\n"
        << "\tnumber of unique delays over all pre/post blocks:\n"
        << "\t\tmean: " << mean_num_unique_elements << "\tstd: "
            << std_num_unique_elements << "\n"
    << "\tbundle size over all bundles:\n"
        << "\t\tmean: " << mean_bundle_sizes << "\tstd: "
        << std_bundle_sizes << "\n"
    << "\n\tmemory usage: TOTAL: " << total_memory_MB << " MB (~"
        << total_memory_MB / syn_N * 1024.0 * 1024.0  << " byte per synapse)"
        << std::endl;

    for(auto const& tuple: memory_recorder){
        std::string name;
        size_t bytes;
        int num_elements;
        std::tie(name, bytes, num_elements) = tuple;
        double memory = bytes * to_MB;
        double fraction = memory / total_memory_MB * 100;
        std::cout << "\t\t" << std::setprecision(1) << std::fixed << fraction
            << "%\t" << std::setprecision(3) << std::fixed << memory << " MB\t"
            << name << " [" << num_elements << "]" << std::endl;
    }


    // Create circular eventspaces in no_or_const_delay_mode
    if (scalar_delay)
    {
        int num_spikespaces = dev_array_spikegeneratorgroup__spikespace.size();
        if (num_queues > num_spikespaces)
        {
            for (int i = num_spikespaces; i < num_queues; i++)
            {
                int32_t* new_eventspace;
                cudaError_t status = cudaMalloc((void**)&new_eventspace,
                        sizeof(int32_t)*_num__array_spikegeneratorgroup__spikespace);
                if (status != cudaSuccess)
                {
                    printf("ERROR while allocating momory for dev_array_spikegeneratorgroup__spikespace[%i] on device: %s %s %d\n",
                            i, cudaGetErrorString(status), __FILE__, __LINE__);
                    exit(status);
                }
                dev_array_spikegeneratorgroup__spikespace.push_back(new_eventspace);
            }
        }
    }

    int num_threads = num_queues;
    if(num_threads >= max_threads_per_block)
    {
        num_threads = max_threads_per_block;
    }
    int num_blocks = 1;

    // check if we have enough ressources to call kernel with given number
    // of blocks and threads
    struct cudaFuncAttributes funcAttrib;
    cudaFuncGetAttributes(&funcAttrib, _run_synapses_pre_initialise_queue_kernel);
    if (num_threads > funcAttrib.maxThreadsPerBlock)
    {
        // use the max num_threads before launch failure
        num_threads = funcAttrib.maxThreadsPerBlock;
        printf("WARNING Not enough ressources available to call "
               "_run_synapses_pre_initialise_queue_kernel "
               "with maximum possible threads per block (%u). "
               "Reducing num_threads to %u. (Kernel needs %i "
               "registers per block, %i bytes of "
               "statically-allocated shared memory per block, %i "
               "bytes of local memory per thread and a total of %i "
               "bytes of user-allocated constant memory)\n",
               max_threads_per_block, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }
    else
    {
        printf("INFO _run_synapses_pre_initialise_queue_kernel\n"
               "\t%u blocks\n"
               "\t%u threads\n"
               "\t%i registers per block\n"
               "\t%i bytes statically-allocated shared memory per block\n"
               "\t%i bytes local memory per thread\n"
               "\t%i bytes user-allocated constant memory\n"
               "",
               num_blocks, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }

    _run_synapses_pre_initialise_queue_kernel<<<num_blocks, num_threads>>>(
        source_N,
        num_parallel_blocks,
        num_threads,
        dt,
        syn_N,
        num_queues,
        scalar_delay
    );

    if (scalar_delay)
    {
        delete [] h_num_synapses_by_pre;
        delete [] d_ptr_synapse_ids_by_pre;
    }

    //delete temp arrays
    delete [] h_vec_synapse_ids_by_pre;
    delete [] h_vec_delays_by_pre;
    if (!scalar_delay)
    {
        delete [] h_vec_unique_delay_start_idcs_by_pre;
        delete [] h_vec_unique_delays_by_pre;
        delete [] h_global_bundle_id_start_by_pre;
    }

    synapses_pre_scalar_delay = scalar_delay;

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        printf("ERROR initialising synapses_pre in %s:%d %s\n",
                __FILE__, __LINE__, cudaGetErrorString(status));
        _dealloc_arrays();
        exit(status);
    }

    CUDA_CHECK_MEMORY();
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: synapses_pre initialisation took " <<  time_passed << "s";
    if (used_device_memory_after_dealloc < used_device_memory_start){
        size_t freed_bytes = used_device_memory_start - used_device_memory_after_dealloc;
        std::cout << ", freed " << freed_bytes * to_MB << "MB";
    }
    if (used_device_memory > used_device_memory_start){
        size_t used_bytes = used_device_memory - used_device_memory_start;
        std::cout << " and used " << used_bytes * to_MB << "MB of device memory.";
    }
    std::cout << std::endl;
}

// synapses_pre_initialise_queue ends here


// synapses_pre_codeobject starts here 

__global__ void
kernel_synapses_pre_codeobject(
    int _N,
    int bid_offset,
    int timestep,
    int THREADS_PER_BLOCK,
    int threads_per_bundle,
    int32_t* eventspace,
    int num_spiking_neurons,
    ///// KERNEL_PARAMETERS /////
    int32_t* _ptr_array_synapses_N,
    int32_t* _ptr_array_synapses__synaptic_post,
    const int _num_postsynaptic_idx,
    int32_t* _ptr_array_synapses__synaptic_pre,
    const int _num_synaptic_pre,
    double* _ptr_array_neurongroup_g_PN_iKC,
    double* _ptr_array_synapses_weight,
    const int _numweight
    )
{
    using namespace brian;

    assert(THREADS_PER_BLOCK == blockDim.x);

    int tid = threadIdx.x;
    int bid = blockIdx.x + bid_offset;
    //TODO: do we need _idx here? if no, get also rid of scoping after scalar code
    // scalar_code can depend on _idx (e.g. if the state update depends on a
    // subexpression that is the same for all synapses, ?)
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numN = 1;
    const int _numg_PN_iKC = 2500;

    ///// kernel_lines /////
        



    ///// scalar_code /////
        


    {  // _idx is defined in outer and inner scope (for `scalar_code`)
        if (synapses_pre.no_or_const_delay_mode)
        {
            // TODO: pass as kernel parameter instead?
            int num_parallel_blocks = synapses_pre.queue->num_blocks;
            int32_t spikes_start = synapses_pre.spikes_start;
            int32_t spikes_stop = synapses_pre.spikes_stop;

            // for the first delay timesteps the eventspace is not yet filled
            // note that num_queues is the number of eventspaces, num_queues-1 the delay in timesteps
            if (timestep >= synapses_pre.queue->num_queues - 1)
            {
                // `spiking_neuron_idx` runs through the eventspace
                // `post_block_idx` runs through the post neuron blocks of the connectivity matrix
                int spiking_neuron_idx = bid / num_parallel_blocks;
                int post_block_idx = bid % num_parallel_blocks;
                {

                    // spiking_neuron is index in NeuronGroup
                    int32_t spiking_neuron = eventspace[spiking_neuron_idx];

                    assert(spiking_neuron != -1);

                    // apply effects if event neuron is in sources of current SynapticPathway
                    if(spikes_start <= spiking_neuron && spiking_neuron < spikes_stop)
                    {
                        int pre_post_block_id = (spiking_neuron - spikes_start) * num_parallel_blocks + post_block_idx;
                        int num_synapses = synapses_pre_num_synapses_by_pre[pre_post_block_id];
                        int32_t* propagating_synapses = synapses_pre_synapse_ids_by_pre[pre_post_block_id];
                        for(int j = tid; j < num_synapses; j+=THREADS_PER_BLOCK)
                        {
                            // _idx is the synapse id
                            int32_t _idx = propagating_synapses[j];
                            _vectorisation_idx = j;

                            ///// vector_code /////
                                                        
                            //  Abstract code:  g_PN_iKC += 0.675 * weight
                            const int32_t _postsynaptic_idx = _ptr_array_synapses__synaptic_post[_idx];
                            const double weight = _ptr_array_synapses_weight[_idx];
                            _brian_atomicAdd(&_ptr_array_neurongroup_g_PN_iKC[_postsynaptic_idx], (double)(0.675 * weight));

                        }
                    }

                    __syncthreads();
                }
            }
        }
        else  // heterogeneous delay mode
        {
            cudaVector<int32_t>* synapses_queue;
            synapses_pre.queue->peek(&synapses_queue);

            int queue_size = synapses_queue[bid].size();

            // use a fixed number of threads per bundle, i runs through all those threads of all bundles
            // for threads_per_bundle == 1, we have one thread per bundle (parallel)
            for (int i = tid; i < queue_size*threads_per_bundle; i+=THREADS_PER_BLOCK)
            {
                // bundle_idx runs through all bundles
                int bundle_idx = i / threads_per_bundle;
                // syn_in_bundle_idx runs through all threads in a single bundle
                int syn_in_bundle_idx = i % threads_per_bundle;

                int bundle_id = synapses_queue[bid].at(bundle_idx);
                int bundle_size = synapses_pre_num_synapses_by_bundle[bundle_id];
                int synapses_offset = synapses_pre_synapses_offset_by_bundle[bundle_id];
                int32_t* synapse_ids = synapses_pre_synapse_ids;
                int32_t* synapse_bundle = synapse_ids + synapses_offset;

                // loop through synapses of this bundle with all available threads_per_bundle
                // if threads_per_bundle == 1, this is serial
                for (int j = syn_in_bundle_idx; j < bundle_size; j+=threads_per_bundle)
                {
                    int32_t _idx = synapse_bundle[j];


                            ///// vector_code /////
                                                        
                            //  Abstract code:  g_PN_iKC += 0.675 * weight
                            const int32_t _postsynaptic_idx = _ptr_array_synapses__synaptic_post[_idx];
                            const double weight = _ptr_array_synapses_weight[_idx];
                            _brian_atomicAdd(&_ptr_array_neurongroup_g_PN_iKC[_postsynaptic_idx], (double)(0.675 * weight));

                        }
                    }
                }
            }
        }


void _run_synapses_pre_codeobject()
{
    using namespace brian;


    const int _N = _array_synapses_N[0];

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
        int32_t* const dev_array_synapses__synaptic_post = thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]);
        const int _num_postsynaptic_idx = dev_dynamic_array_synapses__synaptic_post.size();
        int32_t* const dev_array_synapses__synaptic_pre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]);
        const int _num_synaptic_pre = dev_dynamic_array_synapses__synaptic_pre.size();
        const int _numg_PN_iKC = 2500;
        double* const dev_array_synapses_weight = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_weight[0]);
        const int _numweight = dev_dynamic_array_synapses_weight.size();

static int num_threads_per_bundle;
static int num_loops;

    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
// We are using atomics, we can fully parallelise.
num_blocks = num_parallel_blocks;
num_threads = max_threads_per_block;
// TODO: effect of mean instead of max?
num_threads_per_bundle = synapses_pre_max_bundle_size;
num_loops = 1;


        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_synapses_pre_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_synapses_pre_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_synapses_pre_codeobject "
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
                        kernel_synapses_pre_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
else if (synapses_pre_max_size <= 0)
{
    printf("INFO there are no synapses in the synapses_pre pathway. Skipping synapses_push and synapses kernels.\n");
}
        else
        {
            printf("INFO kernel_synapses_pre_codeobject\n"
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


// only call kernel if we have synapses (otherwise we skipped the push kernel)
if (synapses_pre_max_size > 0)
{
    int32_t num_spiking_neurons;
    // we only need the number of spiking neurons if we parallelise effect
    // application over spiking neurons in homogeneous delay mode
    if (synapses_pre_scalar_delay)
    {
        if (defaultclock.timestep[0] >= synapses_pre_delay)
        {
            cudaMemcpy(&num_spiking_neurons,
                    &dev_array_spikegeneratorgroup__spikespace[synapses_pre_eventspace_idx][_num__array_spikegeneratorgroup__spikespace - 1],
                    sizeof(int32_t), cudaMemcpyDeviceToHost);
            num_blocks = num_parallel_blocks * num_spiking_neurons;
            //TODO collect info abt mean, std of num spiking neurons per time
            //step and print INFO at end of simulation
        }
    }
    // only call kernel if neurons spiked (else num_blocks is zero)
    if (num_blocks != 0) {
        for(int bid_offset = 0; bid_offset < num_loops; bid_offset++)
        {
            kernel_synapses_pre_codeobject<<<num_blocks, num_threads>>>(
                _N,
                bid_offset,
                defaultclock.timestep[0],
                num_threads,
                num_threads_per_bundle,
                dev_array_spikegeneratorgroup__spikespace[synapses_pre_eventspace_idx],
                num_spiking_neurons,
                ///// HOST_PARAMETERS /////
                dev_array_synapses_N,
            dev_array_synapses__synaptic_post,
            _num_postsynaptic_idx,
            dev_array_synapses__synaptic_pre,
            _num_synaptic_pre,
            dev_array_neurongroup_g_PN_iKC,
            dev_array_synapses_weight,
            _numweight
            );
        }
    }

    CUDA_CHECK_ERROR("kernel_synapses_pre_codeobject");
}


}

void _debugmsg_synapses_pre_codeobject()
{
    using namespace brian;
    std::cout << "Number of synapses: " << _array_synapses_N[0] << endl;
}

// synapses_pre_codeobject ends here

// synapses_group_variable_set_conditional_codeobject starts here

__global__ void
kernel_synapses_group_variable_set_conditional_codeobject(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    int32_t* _ptr_array_synapses_N,
    int32_t* _ptr_array_synapses__synaptic_pre,
    const int _numi,
    int32_t* _ptr_array_synapses__synaptic_post,
    const int _numj,
    double* _ptr_array_synapses_weight,
    const int _numweight
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
    _namespace_timedarray_1_values = d_timedarray_1_values;
    #else
    _namespace_timedarray_1_values = _timedarray_1_values;
    #endif


    assert(THREADS_PER_BLOCK == blockDim.x);


    if(_idx >= _N)
    {
        return;
    }

    ///// block kernel_maincode /////

    ///// scalar_code['condition'] /////
        


    ///// scalar_code['statement'] /////
        
    const double _lio_statement_1 = 10.0 * 1e-09;
    const double _lio_statement_2 = 1.25 * 1e-09;


    ///// vector_code['condition'] /////
        
    const char _cond = true;


    if (_cond)
    {
        ///// vector_code['statement'] /////
                
        const int32_t j = _ptr_array_synapses__synaptic_post[_idx];
        const int32_t i = _ptr_array_synapses__synaptic_pre[_idx];
        double weight;
        weight = _lio_statement_1 + (_lio_statement_2 * _timedarray_1(0.0, i + (j * 100)));
        _ptr_array_synapses_weight[_idx] = weight;

    }

    ///// endblock kernel_maincode /////
}

void _run_synapses_group_variable_set_conditional_codeobject()
{
    using namespace brian;


    const int _N = _array_synapses_N[0];

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
        int32_t* const dev_array_synapses__synaptic_pre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]);
        const int _numi = dev_dynamic_array_synapses__synaptic_pre.size();
        int32_t* const dev_array_synapses__synaptic_post = thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]);
        const int _numj = dev_dynamic_array_synapses__synaptic_post.size();
        double* const dev_array_synapses_weight = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_weight[0]);
        const int _numweight = dev_dynamic_array_synapses_weight.size();


    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    kernel_synapses_group_variable_set_conditional_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;



        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_synapses_group_variable_set_conditional_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_synapses_group_variable_set_conditional_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_synapses_group_variable_set_conditional_codeobject "
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
                        kernel_synapses_group_variable_set_conditional_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_synapses_group_variable_set_conditional_codeobject\n"
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


    kernel_synapses_group_variable_set_conditional_codeobject<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            dev_array_synapses_N,
            dev_array_synapses__synaptic_pre,
            _numi,
            dev_array_synapses__synaptic_post,
            _numj,
            dev_array_synapses_weight,
            _numweight
        );

    CUDA_CHECK_ERROR("kernel_synapses_group_variable_set_conditional_codeobject");


}

// synapses_group_variable_set_conditional_codeobject ends here


// synapses_2_pre_push_spikes starts here 

__global__ void _run_synapses_2_pre_push_spikes_advance_kernel()
{
    using namespace brian;
    int tid = threadIdx.x;
    synapses_2_pre.queue->advance(
        tid);
}

__global__ void
_run_synapses_2_pre_push_spikes_push_kernel(
    int num_parallel_blocks,
    int _num_blocks,
    int _num_threads,
    int32_t* _ptr_array_neurongroup_1__spikespace)
{
    // apperently this is not always true and that is why _num_threads is passed as function argument
    // if this assert never fails, we could remove the _num_threads form the argument list
    assert(blockDim.x == _num_threads);

    using namespace brian;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int post_neuron_bid = bid % num_parallel_blocks;
    int pre_neuron_idx = bid / num_parallel_blocks;

    int32_t spiking_neuron = _ptr_array_neurongroup_1__spikespace[pre_neuron_idx];
    assert(spiking_neuron != -1);

    // push to spikequeue if spiking_neuron is in sources of current SynapticPathway
    if(synapses_2_pre.spikes_start <= spiking_neuron && spiking_neuron < synapses_2_pre.spikes_stop)
    {
        synapses_2_pre.queue->push_bundles(
            post_neuron_bid,
            tid,
            _num_threads,
            spiking_neuron - synapses_2_pre.spikes_start);
    }
}

void _run_synapses_2_pre_push_spikes()
{
    using namespace brian;


    ///// HOST_CONSTANTS /////
    const int _num_spikespace = 101;

    if (synapses_2_pre_scalar_delay)
    {
        int num_eventspaces = dev_array_neurongroup_1__spikespace.size();
        synapses_2_pre_eventspace_idx = (current_idx_array_neurongroup_1__spikespace - synapses_2_pre_delay + num_eventspaces) % num_eventspaces;

        //////////////////////////////////////////////
        //// No pushing in no_or_const_delay_mode ////
        //////////////////////////////////////////////
    }
    else if (synapses_2_pre_max_size > 0)
    {

        // get the number of spiking neurons
        int32_t num_spiking_neurons;
        CUDA_SAFE_CALL(
                cudaMemcpy(&num_spiking_neurons,
                    dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace] + _num_spikespace - 1,
                    sizeof(int32_t), cudaMemcpyDeviceToHost)
                );

        // advance spike queues
        _run_synapses_2_pre_push_spikes_advance_kernel<<<1, num_parallel_blocks>>>();

        CUDA_CHECK_ERROR("_run_synapses_2_pre_push_spikes_advance_kernel");

        static int num_threads, num_blocks;
        static size_t needed_shared_memory;
        static bool first_run = true;
        if (first_run)
        {

            needed_shared_memory = 0;

            // We don't need more then max(num_synapses) threads per block.
            num_threads = synapses_2_pre_max_size;
            if (num_threads > max_threads_per_block)
            {
                num_threads = max_threads_per_block;
            }

            // calculate theoretical occupancy
            int max_active_blocks;
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                        _run_synapses_2_pre_push_spikes_push_kernel, num_threads,
                        needed_shared_memory)
                    );

            float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                              (float)(max_threads_per_sm / num_threads_per_warp);

            // check if we have enough ressources to call kernel with given
            // number of blocks and threads
            struct cudaFuncAttributes funcAttrib;
            CUDA_SAFE_CALL(
                    cudaFuncGetAttributes(&funcAttrib, _run_synapses_2_pre_push_spikes_push_kernel)
                    );
            if (num_threads > funcAttrib.maxThreadsPerBlock)
            {
                // use the max num_threads before launch failure
                num_threads = funcAttrib.maxThreadsPerBlock;
                printf("WARNING Not enough ressources available to call "
                       "_run_synapses_2_pre_push_spikes_push_kernel "
                       "with maximum possible threads per block (%u). "
                       "Reducing num_threads to %u. (Kernel needs %i "
                       "registers per block, %i bytes of "
                       "statically-allocated shared memory per block, %i "
                       "bytes of local memory per thread and a total of %i "
                       "bytes of user-allocated constant memory)\n",
                       max_threads_per_block, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes);
            }
            else
            {
                printf("INFO _run_synapses_2_pre_push_spikes_push_kernel\n"
                       "\t%u blocks per spiking neuron\n"
                       "\t%u threads\n"
                       "\t%i registers per block\n"
                       "\t%i bytes statically-allocated shared memory per block\n"
                       "\t%i bytes local memory per thread\n"
                       "\t%i bytes user-allocated constant memory\n"
                       "\t%.3f theoretical occupancy\n",
                       num_parallel_blocks, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes, occupancy);
            }
            first_run = false;
        }


        if (num_spiking_neurons > 0)
        {
            num_blocks = num_parallel_blocks * num_spiking_neurons;

            _run_synapses_2_pre_push_spikes_push_kernel<<<num_blocks, num_threads, needed_shared_memory>>>(
                    num_parallel_blocks,
                    num_blocks,
                    num_threads,
                    dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace]);

            CUDA_CHECK_ERROR("_run_synapses_2_pre_push_spikes_push_kernel");
        }
    }

}

// synapses_2_pre_push_spikes ends here

// synapses_2_pre_initialise_queue starts here 

__global__ void _run_synapses_2_pre_initialise_queue_kernel(
    int _source_N,
    int _num_blocks,
    int _num_threads,
    double _dt,
    int _syn_N,
    int num_queues,
    bool new_mode)
{
    using namespace brian;

    int tid = threadIdx.x;

    synapses_2_pre.queue->prepare(
        tid,
        _num_threads,
        _num_blocks,
        0,
        _source_N,
        _syn_N,
        num_queues,
        synapses_2_pre_num_synapses_by_pre,
        synapses_2_pre_num_synapses_by_bundle,
        synapses_2_pre_num_unique_delays_by_pre,
        synapses_2_pre_unique_delays,
        synapses_2_pre_global_bundle_id_start_by_pre,
        synapses_2_pre_synapses_offset_by_bundle,
        synapses_2_pre_synapse_ids,
        synapses_2_pre_synapse_ids_by_pre,
        synapses_2_pre_unique_delays_offset_by_pre,
        synapses_2_pre_unique_delay_start_idcs);
    synapses_2_pre.no_or_const_delay_mode = new_mode;
}


void _run_synapses_2_pre_initialise_queue()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();
    const double to_MB = 1.0 / (1024.0 * 1024.0);

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
        double* const _array_synapses_2_delay = thrust::raw_pointer_cast(&_dynamic_array_synapses_2_delay[0]);
        const int _numdelay = _dynamic_array_synapses_2_delay.size();

    ///// pointers_lines /////
        
    int32_t*   _ptr_array_synapses_2_N = _array_synapses_2_N;
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    double*   _ptr_array_synapses_2_delay = _array_synapses_2_delay;


    int64_t syn_N_check = _ptr_array_synapses_2_N[0];

    if (syn_N_check == 0){
        return;
    }
    else if (syn_N_check > INT_MAX){
        printf("ERROR: There are more Synapses (%lu) than an int can "
               "hold on this system (%u).\n", syn_N_check, INT_MAX);
    }
    // total number of synapses
    int syn_N = (int)syn_N_check;

    // simulation time step
    double dt = _ptr_array_defaultclock_dt[0];
    // number of neurons in source group
    int source_N = 100;
    // number of neurons in target group
    int target_N = 100;

    // TODO: for multiple SynapticPathways for the same Synapses object (on_pre and on_post) the following copy is identical in both pathways initialise templates

    //////////////////////
    // Scalar variables //
    //////////////////////

    // total number of (preID, postBlock) pairs
    int num_pre_post_blocks = num_parallel_blocks * source_N;
    // size of the connectivity matrix (equal number of synapses)
    int size_connectivity_matrix = 0;

    // statistics of number of synapses per (preID, postBlock) pair
    int sum_num_elements = 0;
    int count_num_elements = 0;
    double mean_num_elements = 0;
    double M2_num_elements = 0;


    ////////////////////////////////////////////////////////
    // Create array and vector variables (in host memory) //
    ////////////////////////////////////////////////////////

    /* VARIABLE NAMING:
     * Not scalar variables are named after TYPE_NAME_STRUCTURE, with:
     * STRUCTURE: the first array dimensions structure (`by_pre`, `by_bundle` or none)
     *   `by_pre`: Array (host pointer type) of size `num_pre_post_blocks`,
     *             which is the number of (preID, postBlock) pairs.
     *   `by_bundle`: thrust::host_vector, size of total number of bundles,
     *                which is one for each delay in each (preID, postBlock) pair.
     *                Different (preID, postBlock) pairs can have different sets
     *                of delay values -> each bundle gets a global bundleID
     *   none: If no STRUCTURE given, it's a one dim array storing everything
     * TYPE: data type in STRUCTURE (`h`, `h_vec`, `h_ptr`, `d_ptr`), with
     *       `h`: host value, `h_vec`: host vector, `h_ptr`: host pointer,
     *       `d_ptr`: device pointer (pointing to device, stored in host memory)
     * NAME: the variable name
     *
     * EXAMPLES:
     * `h_vec_delays_by_pre` - an array [size = num_pre_post_blocks] of host
     *                         vectors, each storing delay values of a
     *                         (preID, postBlock) pair
     * `h_num_synapses_by_bundle` - a host vector of integers specifying the
     *                              number of synapses in a bundle
     * `d_ptr_synapse_ids` - a device pointer to synapse IDs (all of them)
     */

    // synapse IDs for each (preID, postBlock) pair
    vector_t<int32_t>* h_vec_synapse_ids_by_pre = new vector_t<int32_t>[num_pre_post_blocks];
    // array of synapse IDs in device memory for each (preID, postBlock) pair
    int32_t** d_ptr_synapse_ids_by_pre;
    // number of synapses for each (preID, postBlock) pair
    int* h_num_synapses_by_pre;



    // we need to allocate device memory for synapse IDs independent of delay mode
    int32_t* d_ptr_synapse_ids;
    size_t memory_synapse_ids = sizeof(int32_t) * syn_N;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&d_ptr_synapse_ids, memory_synapse_ids)
            );
    memory_recorder.push_back(std::make_tuple("synapse IDs", memory_synapse_ids, syn_N));


    //fill vectors of connectivity matrix with synapse IDs and delays (in units of simulation time step)
    int max_delay = (int)(_dynamic_array_synapses_2_delay[0] / dt + 0.5);
    for(int syn_id = 0; syn_id < syn_N; syn_id++)  // loop through all synapses
    {


        // Code generation checks
        assert(0 == 0);

        assert(0 == 0);

        // pre/post_neuron_id are integers from 0 to Nsource/Ntarget (from corresponding
        // SynapticPathway) this is relevant only when using Subgroups where they might
        // be NOT equal to the idx in their NeuronGroup
        int32_t pre_neuron_id = _dynamic_array_synapses_2__synaptic_pre[syn_id] - 0;
        int32_t post_neuron_id = _dynamic_array_synapses_2__synaptic_post[syn_id] - 0;


        // each parallel executed cuda block gets an equal part of post neuron IDs
        int post_block_id = (post_neuron_id * num_parallel_blocks) / target_N;
        // we store synapses for each pre neuron and post block
        int pre_post_block_id = pre_neuron_id * num_parallel_blocks + post_block_id;

        h_vec_synapse_ids_by_pre[pre_post_block_id].push_back(syn_id);
    }
    int num_queues = max_delay + 1;  // we also need a current step

    synapses_2_pre_delay = max_delay;
    // Delete delay (in sec) on device, we don't need it
    // TODO: don't copy these delays to the device in first place, see #83
    dev_dynamic_array_synapses_2_delay.clear();
    dev_dynamic_array_synapses_2_delay.shrink_to_fit();
    CUDA_CHECK_MEMORY();
    size_t used_device_memory_after_dealloc = used_device_memory;

    ///////////////////////////////////////////////////////
    // Memory allocations which depend on the delay mode //
    ///////////////////////////////////////////////////////

    {
        h_num_synapses_by_pre = new int[num_pre_post_blocks];
        d_ptr_synapse_ids_by_pre = new int32_t*[num_pre_post_blocks];
    }


    // loop through connectivity matrix [(preID, postBlock) pairs]
    for(int i = 0; i < num_pre_post_blocks; i++)
    {
        // i is pre_post_block_id

        int num_elements = h_vec_synapse_ids_by_pre[i].size();
        size_connectivity_matrix += num_elements;
        if (num_elements > synapses_2_pre_max_size)
            synapses_2_pre_max_size = num_elements;

        {
            // copy the synapse IDs and the number of synapses for this
            // (preID, postBlock) to device and store the device pointer

            h_num_synapses_by_pre[i] = num_elements;

            d_ptr_synapse_ids_by_pre[i] = d_ptr_synapse_ids + sum_num_elements;
            CUDA_SAFE_CALL(
                    cudaMemcpy(d_ptr_synapse_ids_by_pre[i],
                        thrust::raw_pointer_cast(&(h_vec_synapse_ids_by_pre[i][0])),
                        sizeof(int32_t) * num_elements,
                        cudaMemcpyHostToDevice)
                    );
        }

        sum_num_elements += num_elements;
        updateMeanStd(count_num_elements, mean_num_elements, M2_num_elements, num_elements);
    }  // end for loop through connectivity matrix
    printf("INFO connectivity matrix has size %i, number of (pre neuron ID, post neuron block) pairs is %u\n",
            size_connectivity_matrix, num_pre_post_blocks);

    {
        // synapses size
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(h_num_synapses_by_pre,
                synapses_2_pre_num_synapses_by_pre, num_pre_post_blocks,
                "number of synapses per pre/post block");
        // synapses id
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(d_ptr_synapse_ids_by_pre,
                synapses_2_pre_synapse_ids_by_pre, num_pre_post_blocks,
                "pointers to synapse IDs");
    }


    ////////////////////////////////////////////////////
    //// PRINT INFORMATION ON MEMORY USAGE AND TIME ////
    ////////////////////////////////////////////////////

    // TODO print statistics!

    // sum all allocated memory
    size_t total_memory = 0;
    int max_string_length = 0;
    for(auto const& tuple: memory_recorder){
        total_memory += std::get<1>(tuple);
        int str_len = std::get<0>(tuple).length();
        if (str_len > max_string_length)
            max_string_length = str_len;
    }
    double total_memory_MB = total_memory * to_MB;
    max_string_length += 5;

    // sort tuples by used memory
    std::sort(begin(memory_recorder), end(memory_recorder),
            [](tuple_t const &t1, tuple_t const &t2) {
            return std::get<1>(t1) > std::get<1>(t2); // or use a custom compare function
            }
            );

    double std_num_elements = getStd(count_num_elements, M2_num_elements);

    // print memory information
    std::cout.precision(1);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << "INFO: synapse statistics and memory usage for synapses_2_pre:\n"
        << "\tnumber of synapses: " << syn_N << "\n"
        << "\tnumber of pre/post blocks: " << num_pre_post_blocks << "\n"
        << "\tnumber of synapses over all pre/post blocks:\n"
        << "\t\tmean: " << mean_num_elements << "\tstd: "
            << std_num_elements << "\n"
    << "\n\tmemory usage: TOTAL: " << total_memory_MB << " MB (~"
        << total_memory_MB / syn_N * 1024.0 * 1024.0  << " byte per synapse)"
        << std::endl;

    for(auto const& tuple: memory_recorder){
        std::string name;
        size_t bytes;
        int num_elements;
        std::tie(name, bytes, num_elements) = tuple;
        double memory = bytes * to_MB;
        double fraction = memory / total_memory_MB * 100;
        std::cout << "\t\t" << std::setprecision(1) << std::fixed << fraction
            << "%\t" << std::setprecision(3) << std::fixed << memory << " MB\t"
            << name << " [" << num_elements << "]" << std::endl;
    }


    // Create circular eventspaces in no_or_const_delay_mode
    {
        int num_spikespaces = dev_array_neurongroup_1__spikespace.size();
        if (num_queues > num_spikespaces)
        {
            for (int i = num_spikespaces; i < num_queues; i++)
            {
                int32_t* new_eventspace;
                cudaError_t status = cudaMalloc((void**)&new_eventspace,
                        sizeof(int32_t)*_num__array_neurongroup_1__spikespace);
                if (status != cudaSuccess)
                {
                    printf("ERROR while allocating momory for dev_array_neurongroup_1__spikespace[%i] on device: %s %s %d\n",
                            i, cudaGetErrorString(status), __FILE__, __LINE__);
                    exit(status);
                }
                dev_array_neurongroup_1__spikespace.push_back(new_eventspace);
            }
        }
    }

    int num_threads = num_queues;
    if(num_threads >= max_threads_per_block)
    {
        num_threads = max_threads_per_block;
    }
    int num_blocks = 1;

    // check if we have enough ressources to call kernel with given number
    // of blocks and threads
    struct cudaFuncAttributes funcAttrib;
    cudaFuncGetAttributes(&funcAttrib, _run_synapses_2_pre_initialise_queue_kernel);
    if (num_threads > funcAttrib.maxThreadsPerBlock)
    {
        // use the max num_threads before launch failure
        num_threads = funcAttrib.maxThreadsPerBlock;
        printf("WARNING Not enough ressources available to call "
               "_run_synapses_2_pre_initialise_queue_kernel "
               "with maximum possible threads per block (%u). "
               "Reducing num_threads to %u. (Kernel needs %i "
               "registers per block, %i bytes of "
               "statically-allocated shared memory per block, %i "
               "bytes of local memory per thread and a total of %i "
               "bytes of user-allocated constant memory)\n",
               max_threads_per_block, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }
    else
    {
        printf("INFO _run_synapses_2_pre_initialise_queue_kernel\n"
               "\t%u blocks\n"
               "\t%u threads\n"
               "\t%i registers per block\n"
               "\t%i bytes statically-allocated shared memory per block\n"
               "\t%i bytes local memory per thread\n"
               "\t%i bytes user-allocated constant memory\n"
               "",
               num_blocks, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }

    _run_synapses_2_pre_initialise_queue_kernel<<<num_blocks, num_threads>>>(
        source_N,
        num_parallel_blocks,
        num_threads,
        dt,
        syn_N,
        num_queues,
        true
    );

    {
        delete [] h_num_synapses_by_pre;
        delete [] d_ptr_synapse_ids_by_pre;
    }

    //delete temp arrays
    delete [] h_vec_synapse_ids_by_pre;

    synapses_2_pre_scalar_delay = true;

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        printf("ERROR initialising synapses_2_pre in %s:%d %s\n",
                __FILE__, __LINE__, cudaGetErrorString(status));
        _dealloc_arrays();
        exit(status);
    }

    CUDA_CHECK_MEMORY();
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: synapses_2_pre initialisation took " <<  time_passed << "s";
    if (used_device_memory_after_dealloc < used_device_memory_start){
        size_t freed_bytes = used_device_memory_start - used_device_memory_after_dealloc;
        std::cout << ", freed " << freed_bytes * to_MB << "MB";
    }
    if (used_device_memory > used_device_memory_start){
        size_t used_bytes = used_device_memory - used_device_memory_start;
        std::cout << " and used " << used_bytes * to_MB << "MB of device memory.";
    }
    std::cout << std::endl;
}

// synapses_2_pre_initialise_queue ends here


// synapses_2_pre_codeobject.cu starts here 

__global__ void
kernel_synapses_2_pre_codeobject(
    int _N,
    int bid_offset,
    int timestep,
    int THREADS_PER_BLOCK,
    int threads_per_bundle,
    int32_t* eventspace,
    int num_spiking_neurons,
    ///// KERNEL_PARAMETERS /////
    int32_t* _ptr_array_synapses_2_N,
    int32_t* _ptr_array_synapses_2__synaptic_post,
    const int _num_postsynaptic_idx,
    int32_t* _ptr_array_synapses_2__synaptic_pre,
    const int _num_synaptic_pre,
    double* _ptr_array_neurongroup_1_g_eKC_eKC
    )
{
    using namespace brian;

    assert(THREADS_PER_BLOCK == blockDim.x);

    int tid = threadIdx.x;
    int bid = blockIdx.x + bid_offset;
    //TODO: do we need _idx here? if no, get also rid of scoping after scalar code
    // scalar_code can depend on _idx (e.g. if the state update depends on a
    // subexpression that is the same for all synapses, ?)
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numN = 1;
    const int _numg_eKC_eKC = 100;

    ///// kernel_lines /////
        



    ///// scalar_code /////
        
    const double _lio_1 = 0.675 * 7.500000000000001e-08;


    {  // _idx is defined in outer and inner scope (for `scalar_code`)
        if (synapses_2_pre.no_or_const_delay_mode)
        {
            // TODO: pass as kernel parameter instead?
            int num_parallel_blocks = synapses_2_pre.queue->num_blocks;
            int32_t spikes_start = synapses_2_pre.spikes_start;
            int32_t spikes_stop = synapses_2_pre.spikes_stop;

            // for the first delay timesteps the eventspace is not yet filled
            // note that num_queues is the number of eventspaces, num_queues-1 the delay in timesteps
            if (timestep >= synapses_2_pre.queue->num_queues - 1)
            {
                // `spiking_neuron_idx` runs through the eventspace
                // `post_block_idx` runs through the post neuron blocks of the connectivity matrix
                int spiking_neuron_idx = bid / num_parallel_blocks;
                int post_block_idx = bid % num_parallel_blocks;
                {

                    // spiking_neuron is index in NeuronGroup
                    int32_t spiking_neuron = eventspace[spiking_neuron_idx];

                    assert(spiking_neuron != -1);

                    // apply effects if event neuron is in sources of current SynapticPathway
                    if(spikes_start <= spiking_neuron && spiking_neuron < spikes_stop)
                    {
                        int pre_post_block_id = (spiking_neuron - spikes_start) * num_parallel_blocks + post_block_idx;
                        int num_synapses = synapses_2_pre_num_synapses_by_pre[pre_post_block_id];
                        int32_t* propagating_synapses = synapses_2_pre_synapse_ids_by_pre[pre_post_block_id];
                        for(int j = tid; j < num_synapses; j+=THREADS_PER_BLOCK)
                        {
                            // _idx is the synapse id
                            int32_t _idx = propagating_synapses[j];
                            _vectorisation_idx = j;

                            ///// vector_code /////
                                                        
                            //  Abstract code:  g_eKC_eKC += _lio_1
                            const int32_t _postsynaptic_idx = _ptr_array_synapses_2__synaptic_post[_idx];
                            _brian_atomicAdd(&_ptr_array_neurongroup_1_g_eKC_eKC[_postsynaptic_idx], (double)(_lio_1));

                        }
                    }

                    __syncthreads();
                }
            }
        }
        else  // heterogeneous delay mode
        {
            cudaVector<int32_t>* synapses_queue;
            synapses_2_pre.queue->peek(&synapses_queue);

            int queue_size = synapses_queue[bid].size();

            // use a fixed number of threads per bundle, i runs through all those threads of all bundles
            // for threads_per_bundle == 1, we have one thread per bundle (parallel)
            for (int i = tid; i < queue_size*threads_per_bundle; i+=THREADS_PER_BLOCK)
            {
                // bundle_idx runs through all bundles
                int bundle_idx = i / threads_per_bundle;
                // syn_in_bundle_idx runs through all threads in a single bundle
                int syn_in_bundle_idx = i % threads_per_bundle;

                int bundle_id = synapses_queue[bid].at(bundle_idx);
                int bundle_size = synapses_2_pre_num_synapses_by_bundle[bundle_id];
                int synapses_offset = synapses_2_pre_synapses_offset_by_bundle[bundle_id];
                int32_t* synapse_ids = synapses_2_pre_synapse_ids;
                int32_t* synapse_bundle = synapse_ids + synapses_offset;

                // loop through synapses of this bundle with all available threads_per_bundle
                // if threads_per_bundle == 1, this is serial
                for (int j = syn_in_bundle_idx; j < bundle_size; j+=threads_per_bundle)
                {
                    int32_t _idx = synapse_bundle[j];


                            ///// vector_code /////
                                                        
                            //  Abstract code:  g_eKC_eKC += _lio_1
                            const int32_t _postsynaptic_idx = _ptr_array_synapses_2__synaptic_post[_idx];
                            _brian_atomicAdd(&_ptr_array_neurongroup_1_g_eKC_eKC[_postsynaptic_idx], (double)(_lio_1));

                        }
                    }
                }
            }
        }


void _run_synapses_2_pre_codeobject()
{
    using namespace brian;


    const int _N = _array_synapses_2_N[0];

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
        int32_t* const dev_array_synapses_2__synaptic_post = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_post[0]);
        const int _num_postsynaptic_idx = dev_dynamic_array_synapses_2__synaptic_post.size();
        int32_t* const dev_array_synapses_2__synaptic_pre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_pre[0]);
        const int _num_synaptic_pre = dev_dynamic_array_synapses_2__synaptic_pre.size();
        const int _numg_eKC_eKC = 100;

static int num_threads_per_bundle;
static int num_loops;

    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
// We are using atomics, we can fully parallelise.
num_blocks = num_parallel_blocks;
num_threads = max_threads_per_block;
// TODO: effect of mean instead of max?
num_threads_per_bundle = synapses_2_pre_max_bundle_size;
num_loops = 1;


        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_synapses_2_pre_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_synapses_2_pre_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_synapses_2_pre_codeobject "
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
                        kernel_synapses_2_pre_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
else if (synapses_2_pre_max_size <= 0)
{
    printf("INFO there are no synapses in the synapses_2_pre pathway. Skipping synapses_push and synapses kernels.\n");
}
        else
        {
            printf("INFO kernel_synapses_2_pre_codeobject\n"
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


// only call kernel if we have synapses (otherwise we skipped the push kernel)
if (synapses_2_pre_max_size > 0)
{
    int32_t num_spiking_neurons;
    // we only need the number of spiking neurons if we parallelise effect
    // application over spiking neurons in homogeneous delay mode
    if (synapses_2_pre_scalar_delay)
    {
        if (defaultclock.timestep[0] >= synapses_2_pre_delay)
        {
            cudaMemcpy(&num_spiking_neurons,
                    &dev_array_neurongroup_1__spikespace[synapses_2_pre_eventspace_idx][_num__array_neurongroup_1__spikespace - 1],
                    sizeof(int32_t), cudaMemcpyDeviceToHost);
            num_blocks = num_parallel_blocks * num_spiking_neurons;
            //TODO collect info abt mean, std of num spiking neurons per time
            //step and print INFO at end of simulation
        }
    }
    // only call kernel if neurons spiked (else num_blocks is zero)
    if (num_blocks != 0) {
        for(int bid_offset = 0; bid_offset < num_loops; bid_offset++)
        {
            kernel_synapses_2_pre_codeobject<<<num_blocks, num_threads>>>(
                _N,
                bid_offset,
                defaultclock.timestep[0],
                num_threads,
                num_threads_per_bundle,
                dev_array_neurongroup_1__spikespace[synapses_2_pre_eventspace_idx],
                num_spiking_neurons,
                ///// HOST_PARAMETERS /////
                dev_array_synapses_2_N,
            dev_array_synapses_2__synaptic_post,
            _num_postsynaptic_idx,
            dev_array_synapses_2__synaptic_pre,
            _num_synaptic_pre,
            dev_array_neurongroup_1_g_eKC_eKC
            );
        }
    }

    CUDA_CHECK_ERROR("kernel_synapses_2_pre_codeobject");
}


}

void _debugmsg_synapses_2_pre_codeobject()
{
    using namespace brian;
    std::cout << "Number of synapses: " << _array_synapses_2_N[0] << endl;
}

// synapses_2_pre_codeobject.cu ends here



// synapses_1_pre_initialise_queue starts here 

__global__ void _run_synapses_1_pre_initialise_queue_kernel(
    int _source_N,
    int _num_blocks,
    int _num_threads,
    double _dt,
    int _syn_N,
    int num_queues,
    bool new_mode)
{
    using namespace brian;

    int tid = threadIdx.x;

    synapses_1_pre.queue->prepare(
        tid,
        _num_threads,
        _num_blocks,
        0,
        _source_N,
        _syn_N,
        num_queues,
        synapses_1_pre_num_synapses_by_pre,
        synapses_1_pre_num_synapses_by_bundle,
        synapses_1_pre_num_unique_delays_by_pre,
        synapses_1_pre_unique_delays,
        synapses_1_pre_global_bundle_id_start_by_pre,
        synapses_1_pre_synapses_offset_by_bundle,
        synapses_1_pre_synapse_ids,
        synapses_1_pre_synapse_ids_by_pre,
        synapses_1_pre_unique_delays_offset_by_pre,
        synapses_1_pre_unique_delay_start_idcs);
    synapses_1_pre.no_or_const_delay_mode = new_mode;
}


void _run_synapses_1_pre_initialise_queue()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();
    const double to_MB = 1.0 / (1024.0 * 1024.0);

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
        double* const _array_synapses_1_delay = thrust::raw_pointer_cast(&_dynamic_array_synapses_1_delay[0]);
        const int _numdelay = _dynamic_array_synapses_1_delay.size();

    ///// pointers_lines /////
        
    int32_t*   _ptr_array_synapses_1_N = _array_synapses_1_N;
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    double*   _ptr_array_synapses_1_delay = _array_synapses_1_delay;


    int64_t syn_N_check = _ptr_array_synapses_1_N[0];

    if (syn_N_check == 0){
        return;
    }
    else if (syn_N_check > INT_MAX){
        printf("ERROR: There are more Synapses (%lu) than an int can "
               "hold on this system (%u).\n", syn_N_check, INT_MAX);
    }
    // total number of synapses
    int syn_N = (int)syn_N_check;

    // simulation time step
    double dt = _ptr_array_defaultclock_dt[0];
    // number of neurons in source group
    int source_N = 2500;
    // number of neurons in target group
    int target_N = 100;

    // TODO: for multiple SynapticPathways for the same Synapses object (on_pre and on_post) the following copy is identical in both pathways initialise templates

    //////////////////////
    // Scalar variables //
    //////////////////////

    // total number of (preID, postBlock) pairs
    int num_pre_post_blocks = num_parallel_blocks * source_N;
    // size of the connectivity matrix (equal number of synapses)
    int size_connectivity_matrix = 0;

    // statistics of number of synapses per (preID, postBlock) pair
    int sum_num_elements = 0;
    int count_num_elements = 0;
    double mean_num_elements = 0;
    double M2_num_elements = 0;


    ////////////////////////////////////////////////////////
    // Create array and vector variables (in host memory) //
    ////////////////////////////////////////////////////////

    /* VARIABLE NAMING:
     * Not scalar variables are named after TYPE_NAME_STRUCTURE, with:
     * STRUCTURE: the first array dimensions structure (`by_pre`, `by_bundle` or none)
     *   `by_pre`: Array (host pointer type) of size `num_pre_post_blocks`,
     *             which is the number of (preID, postBlock) pairs.
     *   `by_bundle`: thrust::host_vector, size of total number of bundles,
     *                which is one for each delay in each (preID, postBlock) pair.
     *                Different (preID, postBlock) pairs can have different sets
     *                of delay values -> each bundle gets a global bundleID
     *   none: If no STRUCTURE given, it's a one dim array storing everything
     * TYPE: data type in STRUCTURE (`h`, `h_vec`, `h_ptr`, `d_ptr`), with
     *       `h`: host value, `h_vec`: host vector, `h_ptr`: host pointer,
     *       `d_ptr`: device pointer (pointing to device, stored in host memory)
     * NAME: the variable name
     *
     * EXAMPLES:
     * `h_vec_delays_by_pre` - an array [size = num_pre_post_blocks] of host
     *                         vectors, each storing delay values of a
     *                         (preID, postBlock) pair
     * `h_num_synapses_by_bundle` - a host vector of integers specifying the
     *                              number of synapses in a bundle
     * `d_ptr_synapse_ids` - a device pointer to synapse IDs (all of them)
     */

    // synapse IDs for each (preID, postBlock) pair
    vector_t<int32_t>* h_vec_synapse_ids_by_pre = new vector_t<int32_t>[num_pre_post_blocks];
    // array of synapse IDs in device memory for each (preID, postBlock) pair
    int32_t** d_ptr_synapse_ids_by_pre;
    // number of synapses for each (preID, postBlock) pair
    int* h_num_synapses_by_pre;



    // we need to allocate device memory for synapse IDs independent of delay mode
    int32_t* d_ptr_synapse_ids;
    size_t memory_synapse_ids = sizeof(int32_t) * syn_N;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&d_ptr_synapse_ids, memory_synapse_ids)
            );
    memory_recorder.push_back(std::make_tuple("synapse IDs", memory_synapse_ids, syn_N));


    //fill vectors of connectivity matrix with synapse IDs and delays (in units of simulation time step)
    int max_delay = (int)(_dynamic_array_synapses_1_delay[0] / dt + 0.5);
    for(int syn_id = 0; syn_id < syn_N; syn_id++)  // loop through all synapses
    {


        // Code generation checks
        assert(0 == 0);

        assert(0 == 0);

        // pre/post_neuron_id are integers from 0 to Nsource/Ntarget (from corresponding
        // SynapticPathway) this is relevant only when using Subgroups where they might
        // be NOT equal to the idx in their NeuronGroup
        int32_t pre_neuron_id = _dynamic_array_synapses_1__synaptic_pre[syn_id] - 0;
        int32_t post_neuron_id = _dynamic_array_synapses_1__synaptic_post[syn_id] - 0;


        // each parallel executed cuda block gets an equal part of post neuron IDs
        int post_block_id = (post_neuron_id * num_parallel_blocks) / target_N;
        // we store synapses for each pre neuron and post block
        int pre_post_block_id = pre_neuron_id * num_parallel_blocks + post_block_id;

        h_vec_synapse_ids_by_pre[pre_post_block_id].push_back(syn_id);
    }
    int num_queues = max_delay + 1;  // we also need a current step

    synapses_1_pre_delay = max_delay;
    // Delete delay (in sec) on device, we don't need it
    // TODO: don't copy these delays to the device in first place, see #83
    dev_dynamic_array_synapses_1_delay.clear();
    dev_dynamic_array_synapses_1_delay.shrink_to_fit();
    CUDA_CHECK_MEMORY();
    size_t used_device_memory_after_dealloc = used_device_memory;

    ///////////////////////////////////////////////////////
    // Memory allocations which depend on the delay mode //
    ///////////////////////////////////////////////////////

    {
        h_num_synapses_by_pre = new int[num_pre_post_blocks];
        d_ptr_synapse_ids_by_pre = new int32_t*[num_pre_post_blocks];
    }


    // loop through connectivity matrix [(preID, postBlock) pairs]
    for(int i = 0; i < num_pre_post_blocks; i++)
    {
        // i is pre_post_block_id

        int num_elements = h_vec_synapse_ids_by_pre[i].size();
        size_connectivity_matrix += num_elements;
        if (num_elements > synapses_1_pre_max_size)
            synapses_1_pre_max_size = num_elements;

        {
            // copy the synapse IDs and the number of synapses for this
            // (preID, postBlock) to device and store the device pointer

            h_num_synapses_by_pre[i] = num_elements;

            d_ptr_synapse_ids_by_pre[i] = d_ptr_synapse_ids + sum_num_elements;
            CUDA_SAFE_CALL(
                    cudaMemcpy(d_ptr_synapse_ids_by_pre[i],
                        thrust::raw_pointer_cast(&(h_vec_synapse_ids_by_pre[i][0])),
                        sizeof(int32_t) * num_elements,
                        cudaMemcpyHostToDevice)
                    );
        }

        sum_num_elements += num_elements;
        updateMeanStd(count_num_elements, mean_num_elements, M2_num_elements, num_elements);
    }  // end for loop through connectivity matrix
    printf("INFO connectivity matrix has size %i, number of (pre neuron ID, post neuron block) pairs is %u\n",
            size_connectivity_matrix, num_pre_post_blocks);

    {
        // synapses size
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(h_num_synapses_by_pre,
                synapses_1_pre_num_synapses_by_pre, num_pre_post_blocks,
                "number of synapses per pre/post block");
        // synapses id
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(d_ptr_synapse_ids_by_pre,
                synapses_1_pre_synapse_ids_by_pre, num_pre_post_blocks,
                "pointers to synapse IDs");
    }


    ////////////////////////////////////////////////////
    //// PRINT INFORMATION ON MEMORY USAGE AND TIME ////
    ////////////////////////////////////////////////////

    // TODO print statistics!

    // sum all allocated memory
    size_t total_memory = 0;
    int max_string_length = 0;
    for(auto const& tuple: memory_recorder){
        total_memory += std::get<1>(tuple);
        int str_len = std::get<0>(tuple).length();
        if (str_len > max_string_length)
            max_string_length = str_len;
    }
    double total_memory_MB = total_memory * to_MB;
    max_string_length += 5;

    // sort tuples by used memory
    std::sort(begin(memory_recorder), end(memory_recorder),
            [](tuple_t const &t1, tuple_t const &t2) {
            return std::get<1>(t1) > std::get<1>(t2); // or use a custom compare function
            }
            );

    double std_num_elements = getStd(count_num_elements, M2_num_elements);

    // print memory information
    std::cout.precision(1);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << "INFO: synapse statistics and memory usage for synapses_1_pre:\n"
        << "\tnumber of synapses: " << syn_N << "\n"
        << "\tnumber of pre/post blocks: " << num_pre_post_blocks << "\n"
        << "\tnumber of synapses over all pre/post blocks:\n"
        << "\t\tmean: " << mean_num_elements << "\tstd: "
            << std_num_elements << "\n"
    << "\n\tmemory usage: TOTAL: " << total_memory_MB << " MB (~"
        << total_memory_MB / syn_N * 1024.0 * 1024.0  << " byte per synapse)"
        << std::endl;

    for(auto const& tuple: memory_recorder){
        std::string name;
        size_t bytes;
        int num_elements;
        std::tie(name, bytes, num_elements) = tuple;
        double memory = bytes * to_MB;
        double fraction = memory / total_memory_MB * 100;
        std::cout << "\t\t" << std::setprecision(1) << std::fixed << fraction
            << "%\t" << std::setprecision(3) << std::fixed << memory << " MB\t"
            << name << " [" << num_elements << "]" << std::endl;
    }


    // Create circular eventspaces in no_or_const_delay_mode
    {
        int num_spikespaces = dev_array_neurongroup__spikespace.size();
        if (num_queues > num_spikespaces)
        {
            for (int i = num_spikespaces; i < num_queues; i++)
            {
                int32_t* new_eventspace;
                cudaError_t status = cudaMalloc((void**)&new_eventspace,
                        sizeof(int32_t)*_num__array_neurongroup__spikespace);
                if (status != cudaSuccess)
                {
                    printf("ERROR while allocating momory for dev_array_neurongroup__spikespace[%i] on device: %s %s %d\n",
                            i, cudaGetErrorString(status), __FILE__, __LINE__);
                    exit(status);
                }
                dev_array_neurongroup__spikespace.push_back(new_eventspace);
            }
        }
    }

    int num_threads = num_queues;
    if(num_threads >= max_threads_per_block)
    {
        num_threads = max_threads_per_block;
    }
    int num_blocks = 1;

    // check if we have enough ressources to call kernel with given number
    // of blocks and threads
    struct cudaFuncAttributes funcAttrib;
    cudaFuncGetAttributes(&funcAttrib, _run_synapses_1_pre_initialise_queue_kernel);
    if (num_threads > funcAttrib.maxThreadsPerBlock)
    {
        // use the max num_threads before launch failure
        num_threads = funcAttrib.maxThreadsPerBlock;
        printf("WARNING Not enough ressources available to call "
               "_run_synapses_1_pre_initialise_queue_kernel "
               "with maximum possible threads per block (%u). "
               "Reducing num_threads to %u. (Kernel needs %i "
               "registers per block, %i bytes of "
               "statically-allocated shared memory per block, %i "
               "bytes of local memory per thread and a total of %i "
               "bytes of user-allocated constant memory)\n",
               max_threads_per_block, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }
    else
    {
        printf("INFO _run_synapses_1_pre_initialise_queue_kernel\n"
               "\t%u blocks\n"
               "\t%u threads\n"
               "\t%i registers per block\n"
               "\t%i bytes statically-allocated shared memory per block\n"
               "\t%i bytes local memory per thread\n"
               "\t%i bytes user-allocated constant memory\n"
               "",
               num_blocks, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }

    _run_synapses_1_pre_initialise_queue_kernel<<<num_blocks, num_threads>>>(
        source_N,
        num_parallel_blocks,
        num_threads,
        dt,
        syn_N,
        num_queues,
        true
    );

    {
        delete [] h_num_synapses_by_pre;
        delete [] d_ptr_synapse_ids_by_pre;
    }

    //delete temp arrays
    delete [] h_vec_synapse_ids_by_pre;

    synapses_1_pre_scalar_delay = true;

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        printf("ERROR initialising synapses_1_pre in %s:%d %s\n",
                __FILE__, __LINE__, cudaGetErrorString(status));
        _dealloc_arrays();
        exit(status);
    }

    CUDA_CHECK_MEMORY();
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: synapses_1_pre initialisation took " <<  time_passed << "s";
    if (used_device_memory_after_dealloc < used_device_memory_start){
        size_t freed_bytes = used_device_memory_start - used_device_memory_after_dealloc;
        std::cout << ", freed " << freed_bytes * to_MB << "MB";
    }
    if (used_device_memory > used_device_memory_start){
        size_t used_bytes = used_device_memory - used_device_memory_start;
        std::cout << " and used " << used_bytes * to_MB << "MB of device memory.";
    }
    std::cout << std::endl;
}

// synapses_1_pre_initialise_queue ends here 

// synapses_1_pre_push_spikes starts here 

__global__ void _run_synapses_1_pre_push_spikes_advance_kernel()
{
    using namespace brian;
    int tid = threadIdx.x;
    synapses_1_pre.queue->advance(
        tid);
}

__global__ void
_run_synapses_1_pre_push_spikes_push_kernel(
    int num_parallel_blocks,
    int _num_blocks,
    int _num_threads,
    int32_t* _ptr_array_neurongroup__spikespace)
{
    // apperently this is not always true and that is why _num_threads is passed as function argument
    // if this assert never fails, we could remove the _num_threads form the argument list
    assert(blockDim.x == _num_threads);

    using namespace brian;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int post_neuron_bid = bid % num_parallel_blocks;
    int pre_neuron_idx = bid / num_parallel_blocks;

    int32_t spiking_neuron = _ptr_array_neurongroup__spikespace[pre_neuron_idx];
    assert(spiking_neuron != -1);

    // push to spikequeue if spiking_neuron is in sources of current SynapticPathway
    if(synapses_1_pre.spikes_start <= spiking_neuron && spiking_neuron < synapses_1_pre.spikes_stop)
    {
        synapses_1_pre.queue->push_bundles(
            post_neuron_bid,
            tid,
            _num_threads,
            spiking_neuron - synapses_1_pre.spikes_start);
    }
}

void _run_synapses_1_pre_push_spikes()
{
    using namespace brian;


    ///// HOST_CONSTANTS /////
    const int _num_spikespace = 2501;

    if (synapses_1_pre_scalar_delay)
    {
        int num_eventspaces = dev_array_neurongroup__spikespace.size();
        synapses_1_pre_eventspace_idx = (current_idx_array_neurongroup__spikespace - synapses_1_pre_delay + num_eventspaces) % num_eventspaces;

        //////////////////////////////////////////////
        //// No pushing in no_or_const_delay_mode ////
        //////////////////////////////////////////////
    }
    else if (synapses_1_pre_max_size > 0)
    {

        // get the number of spiking neurons
        int32_t num_spiking_neurons;
        CUDA_SAFE_CALL(
                cudaMemcpy(&num_spiking_neurons,
                    dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace] + _num_spikespace - 1,
                    sizeof(int32_t), cudaMemcpyDeviceToHost)
                );

        // advance spike queues
        _run_synapses_1_pre_push_spikes_advance_kernel<<<1, num_parallel_blocks>>>();

        CUDA_CHECK_ERROR("_run_synapses_1_pre_push_spikes_advance_kernel");

        static int num_threads, num_blocks;
        static size_t needed_shared_memory;
        static bool first_run = true;
        if (first_run)
        {

            needed_shared_memory = 0;

            // We don't need more then max(num_synapses) threads per block.
            num_threads = synapses_1_pre_max_size;
            if (num_threads > max_threads_per_block)
            {
                num_threads = max_threads_per_block;
            }

            // calculate theoretical occupancy
            int max_active_blocks;
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                        _run_synapses_1_pre_push_spikes_push_kernel, num_threads,
                        needed_shared_memory)
                    );

            float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                              (float)(max_threads_per_sm / num_threads_per_warp);

            // check if we have enough ressources to call kernel with given
            // number of blocks and threads
            struct cudaFuncAttributes funcAttrib;
            CUDA_SAFE_CALL(
                    cudaFuncGetAttributes(&funcAttrib, _run_synapses_1_pre_push_spikes_push_kernel)
                    );
            if (num_threads > funcAttrib.maxThreadsPerBlock)
            {
                // use the max num_threads before launch failure
                num_threads = funcAttrib.maxThreadsPerBlock;
                printf("WARNING Not enough ressources available to call "
                       "_run_synapses_1_pre_push_spikes_push_kernel "
                       "with maximum possible threads per block (%u). "
                       "Reducing num_threads to %u. (Kernel needs %i "
                       "registers per block, %i bytes of "
                       "statically-allocated shared memory per block, %i "
                       "bytes of local memory per thread and a total of %i "
                       "bytes of user-allocated constant memory)\n",
                       max_threads_per_block, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes);
            }
            else
            {
                printf("INFO _run_synapses_1_pre_push_spikes_push_kernel\n"
                       "\t%u blocks per spiking neuron\n"
                       "\t%u threads\n"
                       "\t%i registers per block\n"
                       "\t%i bytes statically-allocated shared memory per block\n"
                       "\t%i bytes local memory per thread\n"
                       "\t%i bytes user-allocated constant memory\n"
                       "\t%.3f theoretical occupancy\n",
                       num_parallel_blocks, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes, occupancy);
            }
            first_run = false;
        }


        if (num_spiking_neurons > 0)
        {
            num_blocks = num_parallel_blocks * num_spiking_neurons;

            _run_synapses_1_pre_push_spikes_push_kernel<<<num_blocks, num_threads, needed_shared_memory>>>(
                    num_parallel_blocks,
                    num_blocks,
                    num_threads,
                    dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace]);

            CUDA_CHECK_ERROR("_run_synapses_1_pre_push_spikes_push_kernel");
        }
    }

}

// synapses_1_pre_push_spikes ends here

// synapses_1_pre_codeobject starts here 

__global__ void
kernel_synapses_1_pre_codeobject(
    int _N,
    int bid_offset,
    int timestep,
    int THREADS_PER_BLOCK,
    int threads_per_bundle,
    int32_t* eventspace,
    int num_spiking_neurons,
    ///// KERNEL_PARAMETERS /////
    double* _ptr_array_synapses_1_Apost,
    const int _numApost,
    double* _ptr_array_synapses_1_Apre,
    const int _numApre,
    int32_t* _ptr_array_synapses_1_N,
    int32_t* _ptr_array_synapses_1__synaptic_post,
    const int _num_postsynaptic_idx,
    int32_t* _ptr_array_synapses_1__synaptic_pre,
    const int _num_synaptic_pre,
    double* _ptr_array_neurongroup_1_g_iKC_eKC,
    double* _ptr_array_synapses_1_g_raw,
    const int _numg_raw,
    double* _ptr_array_synapses_1_lastupdate,
    const int _numlastupdate,
    const double _value_array_defaultclock_t
    )
{
    using namespace brian;

    assert(THREADS_PER_BLOCK == blockDim.x);

    int tid = threadIdx.x;
    int bid = blockIdx.x + bid_offset;
    //TODO: do we need _idx here? if no, get also rid of scoping after scalar code
    // scalar_code can depend on _idx (e.g. if the state update depends on a
    // subexpression that is the same for all synapses, ?)
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numN = 1;
    const int _numg_iKC_eKC = 100;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;



    ///// scalar_code /////
        
    const double _lio_1 = 1.0f*1.0/0.01;
    const double _lio_2 = 1.0f*1.0/0.01;


    {  // _idx is defined in outer and inner scope (for `scalar_code`)
        if (synapses_1_pre.no_or_const_delay_mode)
        {
            // TODO: pass as kernel parameter instead?
            int num_parallel_blocks = synapses_1_pre.queue->num_blocks;
            int32_t spikes_start = synapses_1_pre.spikes_start;
            int32_t spikes_stop = synapses_1_pre.spikes_stop;

            // for the first delay timesteps the eventspace is not yet filled
            // note that num_queues is the number of eventspaces, num_queues-1 the delay in timesteps
            if (timestep >= synapses_1_pre.queue->num_queues - 1)
            {
                // `spiking_neuron_idx` runs through the eventspace
                // `post_block_idx` runs through the post neuron blocks of the connectivity matrix
                int spiking_neuron_idx = bid / num_parallel_blocks;
                int post_block_idx = bid % num_parallel_blocks;
                {

                    // spiking_neuron is index in NeuronGroup
                    int32_t spiking_neuron = eventspace[spiking_neuron_idx];

                    assert(spiking_neuron != -1);

                    // apply effects if event neuron is in sources of current SynapticPathway
                    if(spikes_start <= spiking_neuron && spiking_neuron < spikes_stop)
                    {
                        int pre_post_block_id = (spiking_neuron - spikes_start) * num_parallel_blocks + post_block_idx;
                        int num_synapses = synapses_1_pre_num_synapses_by_pre[pre_post_block_id];
                        int32_t* propagating_synapses = synapses_1_pre_synapse_ids_by_pre[pre_post_block_id];
                        for(int j = tid; j < num_synapses; j+=THREADS_PER_BLOCK)
                        {
                            // _idx is the synapse id
                            int32_t _idx = propagating_synapses[j];
                            _vectorisation_idx = j;

                            ///// vector_code /////
                                                        
                            //  Abstract code:  _Apost := Apost * exp(_lio_1 * (- (t - lastupdate)))
                            //  Abstract code:  _Apre := Apre * exp(_lio_2 * (- (t - lastupdate)))
                            //  Abstract code:  Apost = _Apost
                            //  Abstract code:  Apre = _Apre
                            //  Abstract code:  g_iKC_eKC += g_raw
                            //  Abstract code:  Apre += 1.0000000000000002e-10
                            //  Abstract code:  g_raw = clip(g_raw + Apost, 0, 3.7500000000000005e-09)
                            //  Abstract code:  lastupdate = t
                            const int32_t _postsynaptic_idx = _ptr_array_synapses_1__synaptic_post[_idx];
                            double Apre = _ptr_array_synapses_1_Apre[_idx];
                            double Apost = _ptr_array_synapses_1_Apost[_idx];
                            double g_raw = _ptr_array_synapses_1_g_raw[_idx];
                            double lastupdate = _ptr_array_synapses_1_lastupdate[_idx];
                            const double t = _ptr_array_defaultclock_t[0];
                            const double _Apost = Apost * _brian_exp(_lio_1 * (- (t - lastupdate)));
                            const double _Apre = Apre * _brian_exp(_lio_2 * (- (t - lastupdate)));
                            Apost = _Apost;
                            Apre = _Apre;
                            _brian_atomicAdd(&_ptr_array_neurongroup_1_g_iKC_eKC[_postsynaptic_idx], (double)(g_raw));
                            Apre += 1.0000000000000002e-10;
                            g_raw = _brian_clip(g_raw + Apost, 0, 3.7500000000000005e-09);
                            lastupdate = t;
                            _ptr_array_synapses_1_Apre[_idx] = Apre;
                            _ptr_array_synapses_1_Apost[_idx] = Apost;
                            _ptr_array_synapses_1_g_raw[_idx] = g_raw;
                            _ptr_array_synapses_1_lastupdate[_idx] = lastupdate;

                        }
                    }

                    __syncthreads();
                }
            }
        }
        else  // heterogeneous delay mode
        {
            cudaVector<int32_t>* synapses_queue;
            synapses_1_pre.queue->peek(&synapses_queue);

            int queue_size = synapses_queue[bid].size();

            // use a fixed number of threads per bundle, i runs through all those threads of all bundles
            // for threads_per_bundle == 1, we have one thread per bundle (parallel)
            for (int i = tid; i < queue_size*threads_per_bundle; i+=THREADS_PER_BLOCK)
            {
                // bundle_idx runs through all bundles
                int bundle_idx = i / threads_per_bundle;
                // syn_in_bundle_idx runs through all threads in a single bundle
                int syn_in_bundle_idx = i % threads_per_bundle;

                int bundle_id = synapses_queue[bid].at(bundle_idx);
                int bundle_size = synapses_1_pre_num_synapses_by_bundle[bundle_id];
                int synapses_offset = synapses_1_pre_synapses_offset_by_bundle[bundle_id];
                int32_t* synapse_ids = synapses_1_pre_synapse_ids;
                int32_t* synapse_bundle = synapse_ids + synapses_offset;

                // loop through synapses of this bundle with all available threads_per_bundle
                // if threads_per_bundle == 1, this is serial
                for (int j = syn_in_bundle_idx; j < bundle_size; j+=threads_per_bundle)
                {
                    int32_t _idx = synapse_bundle[j];


                            ///// vector_code /////
                                                        
                            //  Abstract code:  _Apost := Apost * exp(_lio_1 * (- (t - lastupdate)))
                            //  Abstract code:  _Apre := Apre * exp(_lio_2 * (- (t - lastupdate)))
                            //  Abstract code:  Apost = _Apost
                            //  Abstract code:  Apre = _Apre
                            //  Abstract code:  g_iKC_eKC += g_raw
                            //  Abstract code:  Apre += 1.0000000000000002e-10
                            //  Abstract code:  g_raw = clip(g_raw + Apost, 0, 3.7500000000000005e-09)
                            //  Abstract code:  lastupdate = t
                            const int32_t _postsynaptic_idx = _ptr_array_synapses_1__synaptic_post[_idx];
                            double Apre = _ptr_array_synapses_1_Apre[_idx];
                            double Apost = _ptr_array_synapses_1_Apost[_idx];
                            double g_raw = _ptr_array_synapses_1_g_raw[_idx];
                            double lastupdate = _ptr_array_synapses_1_lastupdate[_idx];
                            const double t = _ptr_array_defaultclock_t[0];
                            const double _Apost = Apost * _brian_exp(_lio_1 * (- (t - lastupdate)));
                            const double _Apre = Apre * _brian_exp(_lio_2 * (- (t - lastupdate)));
                            Apost = _Apost;
                            Apre = _Apre;
                            _brian_atomicAdd(&_ptr_array_neurongroup_1_g_iKC_eKC[_postsynaptic_idx], (double)(g_raw));
                            Apre += 1.0000000000000002e-10;
                            g_raw = _brian_clip(g_raw + Apost, 0, 3.7500000000000005e-09);
                            lastupdate = t;
                            _ptr_array_synapses_1_Apre[_idx] = Apre;
                            _ptr_array_synapses_1_Apost[_idx] = Apost;
                            _ptr_array_synapses_1_g_raw[_idx] = g_raw;
                            _ptr_array_synapses_1_lastupdate[_idx] = lastupdate;

                        }
                    }
                }
            }
        }


void _run_synapses_1_pre_codeobject()
{
    using namespace brian;


    const int _N = _array_synapses_1_N[0];

    ///// HOST_CONSTANTS ///////////
    double* const dev_array_synapses_1_Apost = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_Apost[0]);
        const int _numApost = dev_dynamic_array_synapses_1_Apost.size();
        double* const dev_array_synapses_1_Apre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_Apre[0]);
        const int _numApre = dev_dynamic_array_synapses_1_Apre.size();
        const int _numN = 1;
        int32_t* const dev_array_synapses_1__synaptic_post = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]);
        const int _num_postsynaptic_idx = dev_dynamic_array_synapses_1__synaptic_post.size();
        int32_t* const dev_array_synapses_1__synaptic_pre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]);
        const int _num_synaptic_pre = dev_dynamic_array_synapses_1__synaptic_pre.size();
        const int _numg_iKC_eKC = 100;
        double* const dev_array_synapses_1_g_raw = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_g_raw[0]);
        const int _numg_raw = dev_dynamic_array_synapses_1_g_raw.size();
        double* const dev_array_synapses_1_lastupdate = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_lastupdate[0]);
        const int _numlastupdate = dev_dynamic_array_synapses_1_lastupdate.size();

static int num_threads_per_bundle;
static int num_loops;

    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
// We are using atomics, we can fully parallelise.
num_blocks = num_parallel_blocks;
num_threads = max_threads_per_block;
// TODO: effect of mean instead of max?
num_threads_per_bundle = synapses_1_pre_max_bundle_size;
num_loops = 1;


        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_synapses_1_pre_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_synapses_1_pre_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_synapses_1_pre_codeobject "
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
                        kernel_synapses_1_pre_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
else if (synapses_1_pre_max_size <= 0)
{
    printf("INFO there are no synapses in the synapses_1_pre pathway. Skipping synapses_push and synapses kernels.\n");
}
        else
        {
            printf("INFO kernel_synapses_1_pre_codeobject\n"
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


// only call kernel if we have synapses (otherwise we skipped the push kernel)
if (synapses_1_pre_max_size > 0)
{
    int32_t num_spiking_neurons;
    // we only need the number of spiking neurons if we parallelise effect
    // application over spiking neurons in homogeneous delay mode
    if (synapses_1_pre_scalar_delay)
    {
        if (defaultclock.timestep[0] >= synapses_1_pre_delay)
        {
            cudaMemcpy(&num_spiking_neurons,
                    &dev_array_neurongroup__spikespace[synapses_1_pre_eventspace_idx][_num__array_neurongroup__spikespace - 1],
                    sizeof(int32_t), cudaMemcpyDeviceToHost);
            num_blocks = num_parallel_blocks * num_spiking_neurons;
            //TODO collect info abt mean, std of num spiking neurons per time
            //step and print INFO at end of simulation
        }
    }
    // only call kernel if neurons spiked (else num_blocks is zero)
    if (num_blocks != 0) {
        for(int bid_offset = 0; bid_offset < num_loops; bid_offset++)
        {
            kernel_synapses_1_pre_codeobject<<<num_blocks, num_threads>>>(
                _N,
                bid_offset,
                defaultclock.timestep[0],
                num_threads,
                num_threads_per_bundle,
                dev_array_neurongroup__spikespace[synapses_1_pre_eventspace_idx],
                num_spiking_neurons,
                ///// HOST_PARAMETERS /////
                dev_array_synapses_1_Apost,
            _numApost,
            dev_array_synapses_1_Apre,
            _numApre,
            dev_array_synapses_1_N,
            dev_array_synapses_1__synaptic_post,
            _num_postsynaptic_idx,
            dev_array_synapses_1__synaptic_pre,
            _num_synaptic_pre,
            dev_array_neurongroup_1_g_iKC_eKC,
            dev_array_synapses_1_g_raw,
            _numg_raw,
            dev_array_synapses_1_lastupdate,
            _numlastupdate,
            _array_defaultclock_t[0]
            );
        }
    }

    CUDA_CHECK_ERROR("kernel_synapses_1_pre_codeobject");
}


}

void _debugmsg_synapses_1_pre_codeobject()
{
    using namespace brian;
    std::cout << "Number of synapses: " << _array_synapses_1_N[0] << endl;
}

// synapses_1_pre_codeobject ends here

// synapses_1_post_push_spikes starts here 

__global__ void _run_synapses_1_post_push_spikes_advance_kernel()
{
    using namespace brian;
    int tid = threadIdx.x;
    synapses_1_post.queue->advance(
        tid);
}

__global__ void
_run_synapses_1_post_push_spikes_push_kernel(
    int num_parallel_blocks,
    int _num_blocks,
    int _num_threads,
    int32_t* _ptr_array_neurongroup_1__spikespace)
{
    // apperently this is not always true and that is why _num_threads is passed as function argument
    // if this assert never fails, we could remove the _num_threads form the argument list
    assert(blockDim.x == _num_threads);

    using namespace brian;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int post_neuron_bid = bid % num_parallel_blocks;
    int pre_neuron_idx = bid / num_parallel_blocks;

    int32_t spiking_neuron = _ptr_array_neurongroup_1__spikespace[pre_neuron_idx];
    assert(spiking_neuron != -1);

    // push to spikequeue if spiking_neuron is in sources of current SynapticPathway
    if(synapses_1_post.spikes_start <= spiking_neuron && spiking_neuron < synapses_1_post.spikes_stop)
    {
        synapses_1_post.queue->push_bundles(
            post_neuron_bid,
            tid,
            _num_threads,
            spiking_neuron - synapses_1_post.spikes_start);
    }
}

void _run_synapses_1_post_push_spikes()
{
    using namespace brian;


    ///// HOST_CONSTANTS /////
    const int _num_spikespace = 101;

    if (synapses_1_post_scalar_delay)
    {
        int num_eventspaces = dev_array_neurongroup_1__spikespace.size();
        synapses_1_post_eventspace_idx = (current_idx_array_neurongroup_1__spikespace - synapses_1_post_delay + num_eventspaces) % num_eventspaces;

        //////////////////////////////////////////////
        //// No pushing in no_or_const_delay_mode ////
        //////////////////////////////////////////////
    }
    else if (synapses_1_post_max_size > 0)
    {

        // get the number of spiking neurons
        int32_t num_spiking_neurons;
        CUDA_SAFE_CALL(
                cudaMemcpy(&num_spiking_neurons,
                    dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace] + _num_spikespace - 1,
                    sizeof(int32_t), cudaMemcpyDeviceToHost)
                );

        // advance spike queues
        _run_synapses_1_post_push_spikes_advance_kernel<<<1, num_parallel_blocks>>>();

        CUDA_CHECK_ERROR("_run_synapses_1_post_push_spikes_advance_kernel");

        static int num_threads, num_blocks;
        static size_t needed_shared_memory;
        static bool first_run = true;
        if (first_run)
        {

            needed_shared_memory = 0;

            // We don't need more then max(num_synapses) threads per block.
            num_threads = synapses_1_post_max_size;
            if (num_threads > max_threads_per_block)
            {
                num_threads = max_threads_per_block;
            }

            // calculate theoretical occupancy
            int max_active_blocks;
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                        _run_synapses_1_post_push_spikes_push_kernel, num_threads,
                        needed_shared_memory)
                    );

            float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                              (float)(max_threads_per_sm / num_threads_per_warp);

            // check if we have enough ressources to call kernel with given
            // number of blocks and threads
            struct cudaFuncAttributes funcAttrib;
            CUDA_SAFE_CALL(
                    cudaFuncGetAttributes(&funcAttrib, _run_synapses_1_post_push_spikes_push_kernel)
                    );
            if (num_threads > funcAttrib.maxThreadsPerBlock)
            {
                // use the max num_threads before launch failure
                num_threads = funcAttrib.maxThreadsPerBlock;
                printf("WARNING Not enough ressources available to call "
                       "_run_synapses_1_post_push_spikes_push_kernel "
                       "with maximum possible threads per block (%u). "
                       "Reducing num_threads to %u. (Kernel needs %i "
                       "registers per block, %i bytes of "
                       "statically-allocated shared memory per block, %i "
                       "bytes of local memory per thread and a total of %i "
                       "bytes of user-allocated constant memory)\n",
                       max_threads_per_block, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes);
            }
            else
            {
                printf("INFO _run_synapses_1_post_push_spikes_push_kernel\n"
                       "\t%u blocks per spiking neuron\n"
                       "\t%u threads\n"
                       "\t%i registers per block\n"
                       "\t%i bytes statically-allocated shared memory per block\n"
                       "\t%i bytes local memory per thread\n"
                       "\t%i bytes user-allocated constant memory\n"
                       "\t%.3f theoretical occupancy\n",
                       num_parallel_blocks, num_threads, funcAttrib.numRegs,
                       funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                       funcAttrib.constSizeBytes, occupancy);
            }
            first_run = false;
        }


        if (num_spiking_neurons > 0)
        {
            num_blocks = num_parallel_blocks * num_spiking_neurons;

            _run_synapses_1_post_push_spikes_push_kernel<<<num_blocks, num_threads, needed_shared_memory>>>(
                    num_parallel_blocks,
                    num_blocks,
                    num_threads,
                    dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace]);

            CUDA_CHECK_ERROR("_run_synapses_1_post_push_spikes_push_kernel");
        }
    }

}

// synapses_1_post_push_spikes end here

// synapases_1_post_initialise_queue starts here 

__global__ void _run_synapses_1_post_initialise_queue_kernel(
    int _source_N,
    int _num_blocks,
    int _num_threads,
    double _dt,
    int _syn_N,
    int num_queues,
    bool new_mode)
{
    using namespace brian;

    int tid = threadIdx.x;

    synapses_1_post.queue->prepare(
        tid,
        _num_threads,
        _num_blocks,
        0,
        _source_N,
        _syn_N,
        num_queues,
        synapses_1_post_num_synapses_by_pre,
        synapses_1_post_num_synapses_by_bundle,
        synapses_1_post_num_unique_delays_by_pre,
        synapses_1_post_unique_delays,
        synapses_1_post_global_bundle_id_start_by_pre,
        synapses_1_post_synapses_offset_by_bundle,
        synapses_1_post_synapse_ids,
        synapses_1_post_synapse_ids_by_pre,
        synapses_1_post_unique_delays_offset_by_pre,
        synapses_1_post_unique_delay_start_idcs);
    synapses_1_post.no_or_const_delay_mode = new_mode;
}


void _run_synapses_1_post_initialise_queue()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();
    const double to_MB = 1.0 / (1024.0 * 1024.0);

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
        double* const _array_synapses_1_delay_1 = thrust::raw_pointer_cast(&_dynamic_array_synapses_1_delay_1[0]);
        const int _numdelay = _dynamic_array_synapses_1_delay_1.size();

    ///// pointers_lines /////
        
    int32_t*   _ptr_array_synapses_1_N = _array_synapses_1_N;
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    double* __restrict  _ptr_array_synapses_1_delay_1 = _array_synapses_1_delay_1;


    int64_t syn_N_check = _ptr_array_synapses_1_N[0];

    if (syn_N_check == 0){
        return;
    }
    else if (syn_N_check > INT_MAX){
        printf("ERROR: There are more Synapses (%lu) than an int can "
               "hold on this system (%u).\n", syn_N_check, INT_MAX);
    }
    // total number of synapses
    int syn_N = (int)syn_N_check;

    // simulation time step
    double dt = _ptr_array_defaultclock_dt[0];
    // number of neurons in source group
    int source_N = 100;
    // number of neurons in target group
    int target_N = 2500;

    // TODO: for multiple SynapticPathways for the same Synapses object (on_pre and on_post) the following copy is identical in both pathways initialise templates
    // delay (on device) was potentially set in group_variable_set_conditional and needs to be copied to host
    _dynamic_array_synapses_1_delay_1 = dev_dynamic_array_synapses_1_delay_1;

    //////////////////////
    // Scalar variables //
    //////////////////////

    // total number of (preID, postBlock) pairs
    int num_pre_post_blocks = num_parallel_blocks * source_N;
    // size of the connectivity matrix (equal number of synapses)
    int size_connectivity_matrix = 0;

    // statistics of number of synapses per (preID, postBlock) pair
    int sum_num_elements = 0;
    int count_num_elements = 0;
    double mean_num_elements = 0;
    double M2_num_elements = 0;

    // statistics of number of unique delays per (preID, postBlock) pair
    int sum_num_unique_elements = 0;
    int count_num_unique_elements = 0;
    double mean_num_unique_elements = 0;
    double M2_num_unique_elements = 0;

    // total number of bundles in all (preID, postBlock) pairs (not known yet)
    int num_bundle_ids = 0;

    // statistics of number of synapses per bundle
    int sum_bundle_sizes = 0;
    int count_bundle_sizes = 0;
    double mean_bundle_sizes = 0;
    double M2_bundle_sizes = 0;


    ////////////////////////////////////////////////////////
    // Create array and vector variables (in host memory) //
    ////////////////////////////////////////////////////////

    /* VARIABLE NAMING:
     * Not scalar variables are named after TYPE_NAME_STRUCTURE, with:
     * STRUCTURE: the first array dimensions structure (`by_pre`, `by_bundle` or none)
     *   `by_pre`: Array (host pointer type) of size `num_pre_post_blocks`,
     *             which is the number of (preID, postBlock) pairs.
     *   `by_bundle`: thrust::host_vector, size of total number of bundles,
     *                which is one for each delay in each (preID, postBlock) pair.
     *                Different (preID, postBlock) pairs can have different sets
     *                of delay values -> each bundle gets a global bundleID
     *   none: If no STRUCTURE given, it's a one dim array storing everything
     * TYPE: data type in STRUCTURE (`h`, `h_vec`, `h_ptr`, `d_ptr`), with
     *       `h`: host value, `h_vec`: host vector, `h_ptr`: host pointer,
     *       `d_ptr`: device pointer (pointing to device, stored in host memory)
     * NAME: the variable name
     *
     * EXAMPLES:
     * `h_vec_delays_by_pre` - an array [size = num_pre_post_blocks] of host
     *                         vectors, each storing delay values of a
     *                         (preID, postBlock) pair
     * `h_num_synapses_by_bundle` - a host vector of integers specifying the
     *                              number of synapses in a bundle
     * `d_ptr_synapse_ids` - a device pointer to synapse IDs (all of them)
     */

    // synapse IDs for each (preID, postBlock) pair
    vector_t<int32_t>* h_vec_synapse_ids_by_pre = new vector_t<int32_t>[num_pre_post_blocks];
    // array of synapse IDs in device memory for each (preID, postBlock) pair
    int32_t** d_ptr_synapse_ids_by_pre;
    // number of synapses for each (preID, postBlock) pair
    int* h_num_synapses_by_pre;

    // delay for each synapse in `h_vec_synapse_ids_by_pre`,
    // only used to sort synapses by delay
    vector_t<int>* h_vec_delays_by_pre = new vector_t<int>[num_pre_post_blocks];
    // array of vectors with unique delays and start indices in synapses arrays
    vector_t<int>* h_vec_unique_delays_by_pre;
    vector_t<int>* h_vec_unique_delay_start_idcs_by_pre;
    // offset in array of all synapse IDs sorted by bundles (we are storing the
    // offset as 32bit int instead of a 64bit pointer to the bundle start)
    vector_t<int> h_synapses_offset_by_bundle;
    // number of synapses in each bundle
    vector_t<int> h_num_synapses_by_bundle;
    // start of global bundle ID per (preID, postBlock) pair (+ total num bundles)
    int* h_global_bundle_id_start_by_pre = new int[num_pre_post_blocks + 1];


    // we need to allocate device memory for synapse IDs independent of delay mode
    int32_t* d_ptr_synapse_ids;
    size_t memory_synapse_ids = sizeof(int32_t) * syn_N;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&d_ptr_synapse_ids, memory_synapse_ids)
            );
    memory_recorder.push_back(std::make_tuple("synapse IDs", memory_synapse_ids, syn_N));


    //fill vectors of connectivity matrix with synapse IDs and delays (in units of simulation time step)
    int max_delay = (int)(_dynamic_array_synapses_1_delay_1[0] / dt + 0.5);
    int min_delay = max_delay;
    for(int syn_id = 0; syn_id < syn_N; syn_id++)  // loop through all synapses
    {


        // Code generation checks
        assert(0 == 0);

        assert(0 == 0);

        // pre/post_neuron_id are integers from 0 to Nsource/Ntarget (from corresponding
        // SynapticPathway) this is relevant only when using Subgroups where they might
        // be NOT equal to the idx in their NeuronGroup
        int32_t pre_neuron_id = _dynamic_array_synapses_1__synaptic_post[syn_id] - 0;
        int32_t post_neuron_id = _dynamic_array_synapses_1__synaptic_pre[syn_id] - 0;

        int delay = (int)(_dynamic_array_synapses_1_delay_1[syn_id] / dt + 0.5);
        if (delay > max_delay)
            max_delay = delay;
        if (delay < min_delay)
            min_delay = delay;

        // each parallel executed cuda block gets an equal part of post neuron IDs
        int post_block_id = (post_neuron_id * num_parallel_blocks) / target_N;
        // we store synapses for each pre neuron and post block
        int pre_post_block_id = pre_neuron_id * num_parallel_blocks + post_block_id;

        h_vec_synapse_ids_by_pre[pre_post_block_id].push_back(syn_id);
        h_vec_delays_by_pre[pre_post_block_id].push_back(delay);
    }
    int num_queues = max_delay + 1;  // we also need a current step

    bool scalar_delay = (max_delay == min_delay);
    if (scalar_delay)
        synapses_1_post_delay = max_delay;
    // Delete delay (in sec) on device, we don't need it
    // TODO: don't copy these delays to the device in first place, see #83
    dev_dynamic_array_synapses_1_delay_1.clear();
    dev_dynamic_array_synapses_1_delay_1.shrink_to_fit();
    CUDA_CHECK_MEMORY();
    size_t used_device_memory_after_dealloc = used_device_memory;

    ///////////////////////////////////////////////////////
    // Memory allocations which depend on the delay mode //
    ///////////////////////////////////////////////////////

    if (scalar_delay)
    {
        h_num_synapses_by_pre = new int[num_pre_post_blocks];
        d_ptr_synapse_ids_by_pre = new int32_t*[num_pre_post_blocks];
    }

    // allocate memory only if the delays are not all the same
    if (!scalar_delay)
    {

        h_vec_unique_delay_start_idcs_by_pre = new vector_t<int>[num_pre_post_blocks];
        h_vec_unique_delays_by_pre = new vector_t<int>[num_pre_post_blocks];

    }
    int global_bundle_id_start = 0;

    // loop through connectivity matrix [(preID, postBlock) pairs]
    for(int i = 0; i < num_pre_post_blocks; i++)
    {
        // i is pre_post_block_id

        int num_elements = h_vec_synapse_ids_by_pre[i].size();
        size_connectivity_matrix += num_elements;
        if (num_elements > synapses_1_post_max_size)
            synapses_1_post_max_size = num_elements;

        if (!scalar_delay)
        {
            // for this (preID, postBlock), sort all synapses by delay,
            // reduce the delay arrays to unique delays and store the
            // start indices in the synapses array for each unique delay

            typedef vector_t<int>::iterator itr;

            // sort synapses (values) and delays (keys) by delay
            thrust::sort_by_key(
                    h_vec_delays_by_pre[i].begin(),     // keys start
                    h_vec_delays_by_pre[i].end(),       // keys end
                    h_vec_synapse_ids_by_pre[i].begin() // values start
                    );

            // worst case: number of unique delays is num_elements
            h_vec_unique_delay_start_idcs_by_pre[i].resize(num_elements);

            // Initialise the unique delay start idcs array as a sequence
            thrust::sequence(h_vec_unique_delay_start_idcs_by_pre[i].begin(),
                    h_vec_unique_delay_start_idcs_by_pre[i].end());

            // get delays (keys) and values (indices) for first occurence of each delay value
            thrust::pair<itr, itr> end_pair = thrust::unique_by_key(
                    h_vec_delays_by_pre[i].begin(),                 // keys start
                    h_vec_delays_by_pre[i].end(),                   // keys end
                    h_vec_unique_delay_start_idcs_by_pre[i].begin() // values start (position in original delay array)
                    );

            itr unique_delay_end = end_pair.first;
            itr idx_end = end_pair.second;

            // erase unneded vector entries
            h_vec_unique_delay_start_idcs_by_pre[i].erase(
                    idx_end, h_vec_unique_delay_start_idcs_by_pre[i].end());
            // free not used but allocated host memory
            h_vec_unique_delay_start_idcs_by_pre[i].shrink_to_fit();
            h_vec_delays_by_pre[i].erase(unique_delay_end,
                    h_vec_delays_by_pre[i].end());
            // delay_by_pre holds the set of unique delays now
            // we don't need shrink_to_fit, swap takes care of that
            h_vec_unique_delays_by_pre[i].swap(h_vec_delays_by_pre[i]);

            int num_unique_elements = h_vec_unique_delays_by_pre[i].size();
            sum_num_unique_elements += num_unique_elements;

            if (num_unique_elements > synapses_1_post_max_num_unique_delays)
                synapses_1_post_max_num_unique_delays = num_unique_elements;

            // we need a start ID per i (pre_post_block_id) to calc the global
            // bundle ID from the local bundle_idx when pushing
            h_global_bundle_id_start_by_pre[i] = global_bundle_id_start;
            global_bundle_id_start += num_unique_elements;
            // the local bundle_idx goes from 0 to num_bundles for each i (pre_post_block_id)
            for (int bundle_idx = 0; bundle_idx < num_unique_elements; bundle_idx++)
            {
                // find the start idx in the synapses array for this delay (bundle)
                int synapses_start_idx = h_vec_unique_delay_start_idcs_by_pre[i][bundle_idx];
                // find the number of synapses for this delay (bundle)
                int num_synapses;
                if (bundle_idx == num_unique_elements - 1)
                    num_synapses = num_elements - synapses_start_idx;
                else
                    num_synapses = h_vec_unique_delay_start_idcs_by_pre[i][bundle_idx + 1] - synapses_start_idx;
                h_num_synapses_by_bundle.push_back(num_synapses);
                if (num_synapses > synapses_1_post_max_bundle_size)
                    synapses_1_post_max_bundle_size = num_synapses;

                // copy this bundle to device and store the device pointer
                int32_t* d_this_bundle = d_ptr_synapse_ids + sum_bundle_sizes;
                int32_t* h_this_bundle = thrust::raw_pointer_cast(&h_vec_synapse_ids_by_pre[i][synapses_start_idx]);
                size_t memory_size = sizeof(int32_t) * num_synapses;
                CUDA_SAFE_CALL(
                        cudaMemcpy(d_this_bundle, h_this_bundle, memory_size, cudaMemcpyHostToDevice)
                        );

                h_synapses_offset_by_bundle.push_back(sum_bundle_sizes);
                sum_bundle_sizes += num_synapses;
                updateMeanStd(count_bundle_sizes, mean_bundle_sizes, M2_bundle_sizes, num_synapses);
            }

            updateMeanStd(count_num_unique_elements, mean_num_unique_elements,
                    M2_num_unique_elements, num_unique_elements);

        }  // end if (!scalar_delay)
        else   // scalar_delay
        {
            // copy the synapse IDs and the number of synapses for this
            // (preID, postBlock) to device and store the device pointer

            h_num_synapses_by_pre[i] = num_elements;

            d_ptr_synapse_ids_by_pre[i] = d_ptr_synapse_ids + sum_num_elements;
            CUDA_SAFE_CALL(
                    cudaMemcpy(d_ptr_synapse_ids_by_pre[i],
                        thrust::raw_pointer_cast(&(h_vec_synapse_ids_by_pre[i][0])),
                        sizeof(int32_t) * num_elements,
                        cudaMemcpyHostToDevice)
                    );
        }

        sum_num_elements += num_elements;
        updateMeanStd(count_num_elements, mean_num_elements, M2_num_elements, num_elements);
    }  // end for loop through connectivity matrix
    printf("INFO connectivity matrix has size %i, number of (pre neuron ID, post neuron block) pairs is %u\n",
            size_connectivity_matrix, num_pre_post_blocks);

    if (scalar_delay)
    {
        // synapses size
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(h_num_synapses_by_pre,
                synapses_1_post_num_synapses_by_pre, num_pre_post_blocks,
                "number of synapses per pre/post block");
        // synapses id
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(d_ptr_synapse_ids_by_pre,
                synapses_1_post_synapse_ids_by_pre, num_pre_post_blocks,
                "pointers to synapse IDs");
    }

    else  // not scalar_delay
    {
        // Since we now know the total number of unique delays over all
        // (preID, postBlock) pairs, we can allocate the device memory
        size_t memory_unique_delays_by_pre = sizeof(int) * sum_num_unique_elements;
        assert(sum_bundle_sizes == syn_N);

        // array of all unique delas, sorted first by pre_post_block and per
        // pre_post_block by delay
        int *d_ptr_unique_delays;
        CUDA_SAFE_CALL(
                cudaMalloc((void**)&d_ptr_unique_delays, memory_unique_delays_by_pre)
                );
        memory_recorder.push_back(std::make_tuple(
                    "unique delays", memory_unique_delays_by_pre,
                    sum_num_unique_elements));

        int sum_num_unique_elements_bak = sum_num_unique_elements;

        // reset sum_num_unique_elements, we will use it to offset cudaMemcy correctly
        sum_num_unique_elements = 0;
        for(int i = 0; i < num_pre_post_blocks; i++)  // loop through connectivity matrix again
        {

            int num_elements = h_vec_synapse_ids_by_pre[i].size();
            int num_unique_elements = h_vec_unique_delays_by_pre[i].size();

            if(num_elements > 0)
            {
                // copy the unique delays to the device and store the device pointers
                CUDA_SAFE_CALL(
                        cudaMemcpy(d_ptr_unique_delays
                                       + sum_num_unique_elements,
                                   thrust::raw_pointer_cast(
                                       &(h_vec_unique_delays_by_pre[i][0])),
                                   sizeof(int)*num_unique_elements,
                                   cudaMemcpyHostToDevice)
                        );


                sum_num_unique_elements += num_unique_elements;
            }  // end if(num_elements < 0)
        }  // end second loop connectivity matrix
        assert(sum_num_unique_elements_bak == sum_num_unique_elements);

        // pointer to start of unique delays array
        CUDA_SAFE_CALL(
                cudaMemcpyToSymbol(synapses_1_post_unique_delays,
                                   &d_ptr_unique_delays,
                                   sizeof(d_ptr_unique_delays))
                );

        num_bundle_ids = sum_num_unique_elements;

        // add num_bundle_ids as last entry
        h_global_bundle_id_start_by_pre[num_pre_post_blocks] = num_bundle_ids;

        // floor(mean(h_num_synapses_by_bundle))
        synapses_1_post_mean_bundle_size = sum_bundle_sizes / num_bundle_ids;

        // pointer to start of synapse IDs array
        CUDA_SAFE_CALL(
                cudaMemcpyToSymbol(synapses_1_post_synapse_ids, &d_ptr_synapse_ids,
                                   sizeof(d_ptr_synapse_ids))
                );

        // size by bundle
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(
                thrust::raw_pointer_cast(&h_num_synapses_by_bundle[0]),
                synapses_1_post_num_synapses_by_bundle, num_bundle_ids,
                "number of synapses per bundle");

        // synapses offset by bundle
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(
                thrust::raw_pointer_cast(&h_synapses_offset_by_bundle[0]),
                synapses_1_post_synapses_offset_by_bundle, num_bundle_ids,
                "synapses bundle offset");

        // global bundle id start idx by pre
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(
                h_global_bundle_id_start_by_pre,
                synapses_1_post_global_bundle_id_start_by_pre,
                num_pre_post_blocks + 1, "global bundle ID start");


    }  // end if (!scalar_delay)

    ////////////////////////////////////////////////////
    //// PRINT INFORMATION ON MEMORY USAGE AND TIME ////
    ////////////////////////////////////////////////////

    // TODO print statistics!

    // sum all allocated memory
    size_t total_memory = 0;
    int max_string_length = 0;
    for(auto const& tuple: memory_recorder){
        total_memory += std::get<1>(tuple);
        int str_len = std::get<0>(tuple).length();
        if (str_len > max_string_length)
            max_string_length = str_len;
    }
    double total_memory_MB = total_memory * to_MB;
    max_string_length += 5;

    // sort tuples by used memory
    std::sort(begin(memory_recorder), end(memory_recorder),
            [](tuple_t const &t1, tuple_t const &t2) {
            return std::get<1>(t1) > std::get<1>(t2); // or use a custom compare function
            }
            );

    double std_num_elements = getStd(count_num_elements, M2_num_elements);
    double std_bundle_sizes = getStd(count_bundle_sizes, M2_bundle_sizes);
    double std_num_unique_elements = getStd(count_num_unique_elements, M2_num_unique_elements);

    // print memory information
    std::cout.precision(1);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << "INFO: synapse statistics and memory usage for synapses_1_post:\n"
        << "\tnumber of synapses: " << syn_N << "\n"
        << "\tnumber of bundles: " << num_bundle_ids << "\n"
        << "\tnumber of pre/post blocks: " << num_pre_post_blocks << "\n"
        << "\tnumber of synapses over all pre/post blocks:\n"
        << "\t\tmean: " << mean_num_elements << "\tstd: "
            << std_num_elements << "\n"
        << "\tnumber of unique delays over all pre/post blocks:\n"
        << "\t\tmean: " << mean_num_unique_elements << "\tstd: "
            << std_num_unique_elements << "\n"
    << "\tbundle size over all bundles:\n"
        << "\t\tmean: " << mean_bundle_sizes << "\tstd: "
        << std_bundle_sizes << "\n"
    << "\n\tmemory usage: TOTAL: " << total_memory_MB << " MB (~"
        << total_memory_MB / syn_N * 1024.0 * 1024.0  << " byte per synapse)"
        << std::endl;

    for(auto const& tuple: memory_recorder){
        std::string name;
        size_t bytes;
        int num_elements;
        std::tie(name, bytes, num_elements) = tuple;
        double memory = bytes * to_MB;
        double fraction = memory / total_memory_MB * 100;
        std::cout << "\t\t" << std::setprecision(1) << std::fixed << fraction
            << "%\t" << std::setprecision(3) << std::fixed << memory << " MB\t"
            << name << " [" << num_elements << "]" << std::endl;
    }


    // Create circular eventspaces in no_or_const_delay_mode
    if (scalar_delay)
    {
        int num_spikespaces = dev_array_neurongroup_1__spikespace.size();
        if (num_queues > num_spikespaces)
        {
            for (int i = num_spikespaces; i < num_queues; i++)
            {
                int32_t* new_eventspace;
                cudaError_t status = cudaMalloc((void**)&new_eventspace,
                        sizeof(int32_t)*_num__array_neurongroup_1__spikespace);
                if (status != cudaSuccess)
                {
                    printf("ERROR while allocating momory for dev_array_neurongroup_1__spikespace[%i] on device: %s %s %d\n",
                            i, cudaGetErrorString(status), __FILE__, __LINE__);
                    exit(status);
                }
                dev_array_neurongroup_1__spikespace.push_back(new_eventspace);
            }
        }
    }

    int num_threads = num_queues;
    if(num_threads >= max_threads_per_block)
    {
        num_threads = max_threads_per_block;
    }
    int num_blocks = 1;

    // check if we have enough ressources to call kernel with given number
    // of blocks and threads
    struct cudaFuncAttributes funcAttrib;
    cudaFuncGetAttributes(&funcAttrib, _run_synapses_1_post_initialise_queue_kernel);
    if (num_threads > funcAttrib.maxThreadsPerBlock)
    {
        // use the max num_threads before launch failure
        num_threads = funcAttrib.maxThreadsPerBlock;
        printf("WARNING Not enough ressources available to call "
               "_run_synapses_1_post_initialise_queue_kernel "
               "with maximum possible threads per block (%u). "
               "Reducing num_threads to %u. (Kernel needs %i "
               "registers per block, %i bytes of "
               "statically-allocated shared memory per block, %i "
               "bytes of local memory per thread and a total of %i "
               "bytes of user-allocated constant memory)\n",
               max_threads_per_block, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }
    else
    {
        printf("INFO _run_synapses_1_post_initialise_queue_kernel\n"
               "\t%u blocks\n"
               "\t%u threads\n"
               "\t%i registers per block\n"
               "\t%i bytes statically-allocated shared memory per block\n"
               "\t%i bytes local memory per thread\n"
               "\t%i bytes user-allocated constant memory\n"
               "",
               num_blocks, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }

    _run_synapses_1_post_initialise_queue_kernel<<<num_blocks, num_threads>>>(
        source_N,
        num_parallel_blocks,
        num_threads,
        dt,
        syn_N,
        num_queues,
        scalar_delay
    );

    if (scalar_delay)
    {
        delete [] h_num_synapses_by_pre;
        delete [] d_ptr_synapse_ids_by_pre;
    }

    //delete temp arrays
    delete [] h_vec_synapse_ids_by_pre;
    delete [] h_vec_delays_by_pre;
    if (!scalar_delay)
    {
        delete [] h_vec_unique_delay_start_idcs_by_pre;
        delete [] h_vec_unique_delays_by_pre;
        delete [] h_global_bundle_id_start_by_pre;
    }

    synapses_1_post_scalar_delay = scalar_delay;

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        printf("ERROR initialising synapses_1_post in %s:%d %s\n",
                __FILE__, __LINE__, cudaGetErrorString(status));
        _dealloc_arrays();
        exit(status);
    }

    CUDA_CHECK_MEMORY();
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: synapses_1_post initialisation took " <<  time_passed << "s";
    if (used_device_memory_after_dealloc < used_device_memory_start){
        size_t freed_bytes = used_device_memory_start - used_device_memory_after_dealloc;
        std::cout << ", freed " << freed_bytes * to_MB << "MB";
    }
    if (used_device_memory > used_device_memory_start){
        size_t used_bytes = used_device_memory - used_device_memory_start;
        std::cout << " and used " << used_bytes * to_MB << "MB of device memory.";
    }
    std::cout << std::endl;
}

// synapses_1_post_initialise_queue ends here 

// synapses_1_post_codeobject starts here 

__global__ void
kernel_synapses_1_post_codeobject(
    int _N,
    int bid_offset,
    int timestep,
    int THREADS_PER_BLOCK,
    int threads_per_bundle,
    int32_t* eventspace,
    int neurongroup_size,
    ///// KERNEL_PARAMETERS /////
    double* _ptr_array_synapses_1_Apost,
    const int _numApost,
    double* _ptr_array_synapses_1_Apre,
    const int _numApre,
    int32_t* _ptr_array_synapses_1_N,
    int32_t* _ptr_array_synapses_1__synaptic_pre,
    const int _num_synaptic_pre,
    double* _ptr_array_synapses_1_g_raw,
    const int _numg_raw,
    double* _ptr_array_synapses_1_lastupdate,
    const int _numlastupdate,
    const double _value_array_defaultclock_t
    )
{
    using namespace brian;

    assert(THREADS_PER_BLOCK == blockDim.x);

    int tid = threadIdx.x;
    int bid = blockIdx.x + bid_offset;
    //TODO: do we need _idx here? if no, get also rid of scoping after scalar code
    // scalar_code can depend on _idx (e.g. if the state update depends on a
    // subexpression that is the same for all synapses, ?)
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numN = 1;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;



    ///// scalar_code /////
        
    const double _lio_1 = 1.0f*1.0/0.01;
    const double _lio_2 = 1.0f*1.0/0.01;


    {  // _idx is defined in outer and inner scope (for `scalar_code`)
        if (synapses_1_post.no_or_const_delay_mode)
        {
            // TODO: pass as kernel parameter instead?
            int num_parallel_blocks = synapses_1_post.queue->num_blocks;
            int32_t spikes_start = synapses_1_post.spikes_start;
            int32_t spikes_stop = synapses_1_post.spikes_stop;

            // for the first delay timesteps the eventspace is not yet filled
            // note that num_queues is the number of eventspaces, num_queues-1 the delay in timesteps
            if (timestep >= synapses_1_post.queue->num_queues - 1)
            {
                // `spiking_neuron_idx` runs through the eventspace
                // `post_block_idx` runs through the post neuron blocks of the connectivity matrix
                int post_block_idx = bid;
                // loop through neurons in eventspace (indices of event neurons, rest -1)
                for(int spiking_neuron_idx = 0;
                        spiking_neuron_idx < neurongroup_size;
                        spiking_neuron_idx++)
                {

                    // spiking_neuron is index in NeuronGroup
                    int32_t spiking_neuron = eventspace[spiking_neuron_idx];

                    if(spiking_neuron == -1) // end of spiking neurons
                    {
                        assert(spiking_neuron_idx == eventspace[neurongroup_size]);
                        return;
                    }

                    // apply effects if event neuron is in sources of current SynapticPathway
                    if(spikes_start <= spiking_neuron && spiking_neuron < spikes_stop)
                    {
                        int pre_post_block_id = (spiking_neuron - spikes_start) * num_parallel_blocks + post_block_idx;
                        int num_synapses = synapses_1_post_num_synapses_by_pre[pre_post_block_id];
                        int32_t* propagating_synapses = synapses_1_post_synapse_ids_by_pre[pre_post_block_id];
                        for(int j = tid; j < num_synapses; j+=THREADS_PER_BLOCK)
                        {
                            // _idx is the synapse id
                            int32_t _idx = propagating_synapses[j];
                            _vectorisation_idx = j;

                            ///// vector_code /////
                                                        
                            double Apre = _ptr_array_synapses_1_Apre[_idx];
                            double Apost = _ptr_array_synapses_1_Apost[_idx];
                            double g_raw = _ptr_array_synapses_1_g_raw[_idx];
                            const double t = _ptr_array_defaultclock_t[0];
                            double lastupdate = _ptr_array_synapses_1_lastupdate[_idx];
                            const double _Apost = Apost * _brian_exp(_lio_1 * (- (t - lastupdate)));
                            const double _Apre = Apre * _brian_exp(_lio_2 * (- (t - lastupdate)));
                            Apost = _Apost;
                            Apre = _Apre;
                            Apost += (- 1.0000000000000002e-10);
                            g_raw = _brian_clip(g_raw + Apre, 0, 3.7500000000000005e-09);
                            lastupdate = t;
                            _ptr_array_synapses_1_Apre[_idx] = Apre;
                            _ptr_array_synapses_1_Apost[_idx] = Apost;
                            _ptr_array_synapses_1_g_raw[_idx] = g_raw;
                            _ptr_array_synapses_1_lastupdate[_idx] = lastupdate;

                        }
                    }

                    __syncthreads();
                }
            }
        }
        else  // heterogeneous delay mode
        {
            cudaVector<int32_t>* synapses_queue;
            synapses_1_post.queue->peek(&synapses_queue);

            int queue_size = synapses_queue[bid].size();

            // use a fixed number of threads per bundle, i runs through all those threads of all bundles
            // for threads_per_bundle == 1, we have one thread per bundle (parallel)
            for (int i = tid; i < queue_size*threads_per_bundle; i+=THREADS_PER_BLOCK)
            {
                // bundle_idx runs through all bundles
                int bundle_idx = i / threads_per_bundle;
                // syn_in_bundle_idx runs through all threads in a single bundle
                int syn_in_bundle_idx = i % threads_per_bundle;

                int bundle_id = synapses_queue[bid].at(bundle_idx);
                int bundle_size = synapses_1_post_num_synapses_by_bundle[bundle_id];
                int synapses_offset = synapses_1_post_synapses_offset_by_bundle[bundle_id];
                int32_t* synapse_ids = synapses_1_post_synapse_ids;
                int32_t* synapse_bundle = synapse_ids + synapses_offset;

                // loop through synapses of this bundle with all available threads_per_bundle
                // if threads_per_bundle == 1, this is serial
                for (int j = syn_in_bundle_idx; j < bundle_size; j+=threads_per_bundle)
                {
                    int32_t _idx = synapse_bundle[j];


                            ///// vector_code /////
                                                        
                            double Apre = _ptr_array_synapses_1_Apre[_idx];
                            double Apost = _ptr_array_synapses_1_Apost[_idx];
                            double g_raw = _ptr_array_synapses_1_g_raw[_idx];
                            const double t = _ptr_array_defaultclock_t[0];
                            double lastupdate = _ptr_array_synapses_1_lastupdate[_idx];
                            const double _Apost = Apost * _brian_exp(_lio_1 * (- (t - lastupdate)));
                            const double _Apre = Apre * _brian_exp(_lio_2 * (- (t - lastupdate)));
                            Apost = _Apost;
                            Apre = _Apre;
                            Apost += (- 1.0000000000000002e-10);
                            g_raw = _brian_clip(g_raw + Apre, 0, 3.7500000000000005e-09);
                            lastupdate = t;
                            _ptr_array_synapses_1_Apre[_idx] = Apre;
                            _ptr_array_synapses_1_Apost[_idx] = Apost;
                            _ptr_array_synapses_1_g_raw[_idx] = g_raw;
                            _ptr_array_synapses_1_lastupdate[_idx] = lastupdate;

                        }
                    }
                }
            }
        }


void _run_synapses_1_post_codeobject()
{
    using namespace brian;


    const int _N = _array_synapses_1_N[0];

    ///// HOST_CONSTANTS ///////////
    double* const dev_array_synapses_1_Apost = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_Apost[0]);
        const int _numApost = dev_dynamic_array_synapses_1_Apost.size();
        double* const dev_array_synapses_1_Apre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_Apre[0]);
        const int _numApre = dev_dynamic_array_synapses_1_Apre.size();
        const int _numN = 1;
        int32_t* const dev_array_synapses_1__synaptic_pre = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]);
        const int _num_synaptic_pre = dev_dynamic_array_synapses_1__synaptic_pre.size();
        double* const dev_array_synapses_1_g_raw = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_g_raw[0]);
        const int _numg_raw = dev_dynamic_array_synapses_1_g_raw.size();
        double* const dev_array_synapses_1_lastupdate = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_lastupdate[0]);
        const int _numlastupdate = dev_dynamic_array_synapses_1_lastupdate.size();

static int num_threads_per_bundle;
static int num_loops;

    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
// Synaptic effects modify only synapse variables.
num_blocks = num_parallel_blocks;
num_threads = max_threads_per_block;
// TODO: effect of mean instead of max?
num_threads_per_bundle = synapses_1_post_max_bundle_size;
num_loops = 1;


        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_synapses_1_post_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_synapses_1_post_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_synapses_1_post_codeobject "
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
                        kernel_synapses_1_post_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
else if (synapses_1_post_max_size <= 0)
{
    printf("INFO there are no synapses in the synapses_1_post pathway. Skipping synapses_push and synapses kernels.\n");
}
        else
        {
            printf("INFO kernel_synapses_1_post_codeobject\n"
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


// only call kernel if we have synapses (otherwise we skipped the push kernel)
if (synapses_1_post_max_size > 0)
{
        for(int bid_offset = 0; bid_offset < num_loops; bid_offset++)
        {
            kernel_synapses_1_post_codeobject<<<num_blocks, num_threads>>>(
                _N,
                bid_offset,
                defaultclock.timestep[0],
                num_threads,
                num_threads_per_bundle,
                dev_array_neurongroup_1__spikespace[synapses_1_post_eventspace_idx],
                _num__array_neurongroup_1__spikespace-1,
                ///// HOST_PARAMETERS /////
                dev_array_synapses_1_Apost,
            _numApost,
            dev_array_synapses_1_Apre,
            _numApre,
            dev_array_synapses_1_N,
            dev_array_synapses_1__synaptic_pre,
            _num_synaptic_pre,
            dev_array_synapses_1_g_raw,
            _numg_raw,
            dev_array_synapses_1_lastupdate,
            _numlastupdate,
            _array_defaultclock_t[0]
            );
        }

    CUDA_CHECK_ERROR("kernel_synapses_1_post_codeobject");
}


}

void _debugmsg_synapses_1_post_codeobject()
{
    using namespace brian;
    std::cout << "Number of synapses: " << _array_synapses_1_N[0] << endl;
}

// synapses_1_push_codeobject ends here 





// synapses_1_group_variable_set_conditional_codeobject_1 starts here 

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
    _namespace_timedarray_4_values = d_timedarray_4_values;
    #else
    _namespace_timedarray_4_values = _timedarray_4_values;
    #endif
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    _namespace_timedarray_3_values = d_timedarray_3_values;
    #else
    _namespace_timedarray_3_values = _timedarray_3_values;
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

// synapses_11_group_variable_set_conditional_codeobject ends here

// synapses_1_group_cariable_set_conditional_codeobject starts here 

__global__ void
kernel_synapses_1_group_variable_set_conditional_codeobject(
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
    _namespace_timedarray_2_values = d_timedarray_2_values;
    #else
    _namespace_timedarray_2_values = _timedarray_2_values;
    #endif


    assert(THREADS_PER_BLOCK == blockDim.x);


    if(_idx >= _N)
    {
        return;
    }

    ///// block kernel_maincode /////

    ///// scalar_code['condition'] /////
        


    ///// scalar_code['statement'] /////
        
    const double _lio_statement_1 = 1.0f*(0.1 * 3.7500000000000005e-09)/1.0;


    ///// vector_code['condition'] /////
        
    const char _cond = true;


    if (_cond)
    {
        ///// vector_code['statement'] /////
                
        const int32_t j = _ptr_array_synapses_1__synaptic_post[_idx];
        const int32_t i = _ptr_array_synapses_1__synaptic_pre[_idx];
        double g_raw;
        g_raw = _lio_statement_1 * _timedarray_2(0.0, i + (j * 2500));
        _ptr_array_synapses_1_g_raw[_idx] = g_raw;

    }

    ///// endblock kernel_maincode /////
}

void _run_synapses_1_group_variable_set_conditional_codeobject()
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
                    kernel_synapses_1_group_variable_set_conditional_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;



        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_synapses_1_group_variable_set_conditional_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_synapses_1_group_variable_set_conditional_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_synapses_1_group_variable_set_conditional_codeobject "
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
                        kernel_synapses_1_group_variable_set_conditional_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_synapses_1_group_variable_set_conditional_codeobject\n"
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


    kernel_synapses_1_group_variable_set_conditional_codeobject<<<num_blocks, num_threads>>>(
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

    CUDA_CHECK_ERROR("kernel_synapses_1_group_variable_set_conditional_codeobject");


}

// synapses_1_group_variable_set_conditional_codeobject ends here




// spikemonitor_codeobject starts here 

__global__ void _run_spikemonitor_codeobject_init()
{
        monitor_t = new cudaVector<double>();
        monitor_i = new cudaVector<int32_t>();
}

__global__ void
kernel_spikemonitor_codeobject(
    int neurongroup_N,
    int32_t* count,
    // KERNEL_PARAMETERS
    int32_t* _ptr_array_spikemonitor_N,
    int32_t* _ptr_array_spikegeneratorgroup_i,
    int32_t* _ptr_array_spikemonitor__source_idx,
    const double _value_array_defaultclock_t,
    int32_t* _ptr_array_spikegeneratorgroup__spikespace,
    int32_t* _ptr_array_spikemonitor_count,
    int32_t* _ptr_array_spikemonitor_i,
    const int _numi,
    double* _ptr_array_spikemonitor_t,
    const int _numt
    )
{
    using namespace brian;
    int tid = threadIdx.x;
    int bid = blockIdx.x;


    // KERNEL_CONSTANTS
    const int _numN = 1;
    const int _num_source_i = 100;
    const int _num_source_idx = 100;
    const int _num_spikespace = 101;
    const int _numcount = 100;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    // scalar_code
        


    // using not parallel spikespace: filled from left with all spiking neuron IDs, -1 ends the list
    for(int i = 0; i < neurongroup_N; i++)
    {
        int32_t spiking_neuron = _ptr_array_spikegeneratorgroup__spikespace[i];
        if(spiking_neuron != -1)
        {
            if(0 <= spiking_neuron && spiking_neuron < 100)
            {
                int _idx = spiking_neuron;
                int _vectorisation_idx = _idx;

                // vector_code
                                
                const double _source_t = _ptr_array_defaultclock_t[0];
                const int32_t _source_i = _ptr_array_spikegeneratorgroup_i[_idx];
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

void _run_spikemonitor_codeobject()
{
    using namespace brian;


    const int _N = _array_spikemonitor_N[0];

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
        const int _num_source_i = 100;
        const int _num_source_idx = 100;
        const int _num_spikespace = 101;
        const int _numcount = 100;
        int32_t* const dev_array_spikemonitor_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]);
        const int _numi = dev_dynamic_array_spikemonitor_i.size();
        double* const dev_array_spikemonitor_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]);
        const int _numt = dev_dynamic_array_spikemonitor_t.size();


    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
_run_spikemonitor_codeobject_init<<<1,1>>>();

CUDA_CHECK_ERROR("_run_spikemonitor_codeobject_init");
num_blocks = 1;
num_threads = 1;

        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_spikemonitor_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_spikemonitor_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_spikemonitor_codeobject "
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
                        kernel_spikemonitor_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_spikemonitor_codeobject\n"
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


kernel_spikemonitor_codeobject<<<num_blocks, num_threads>>>(
        _num_spikespace-1,
        dev_array_spikemonitor_count,
        // HOST_PARAMETERS
        dev_array_spikemonitor_N,
            dev_array_spikegeneratorgroup_i,
            dev_array_spikemonitor__source_idx,
            _array_defaultclock_t[0],
            dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace],
            dev_array_spikemonitor_count,
            dev_array_spikemonitor_i,
            _numi,
            dev_array_spikemonitor_t,
            _numt);

CUDA_CHECK_ERROR("kernel_spikemonitor_codeobject");


}

__global__ void _run_debugmsg_spikemonitor_codeobject_kernel(
    // KERNEL_PARAMETERS
    int32_t* _ptr_array_spikemonitor_N,
    int32_t* _ptr_array_spikegeneratorgroup_i,
    int32_t* _ptr_array_spikemonitor__source_idx,
    const double _value_array_defaultclock_t,
    int32_t* _ptr_array_spikegeneratorgroup__spikespace,
    int32_t* _ptr_array_spikemonitor_count,
    int32_t* _ptr_array_spikemonitor_i,
    const int _numi,
    double* _ptr_array_spikemonitor_t,
    const int _numt
)
{
    using namespace brian;

    // KERNEL_CONSTANTS
    const int _numN = 1;
    const int _num_source_i = 100;
    const int _num_source_idx = 100;
    const int _num_spikespace = 101;
    const int _numcount = 100;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    printf("Number of spikes: %d\n", _ptr_array_spikemonitor_N[0]);
}

__global__ void _count_spikemonitor_codeobject_kernel(
    int* dev_num_events,
    // KERNEL_PARAMETERS
    int32_t* _ptr_array_spikemonitor_N,
    int32_t* _ptr_array_spikegeneratorgroup_i,
    int32_t* _ptr_array_spikemonitor__source_idx,
    const double _value_array_defaultclock_t,
    int32_t* _ptr_array_spikegeneratorgroup__spikespace,
    int32_t* _ptr_array_spikemonitor_count,
    int32_t* _ptr_array_spikemonitor_i,
    const int _numi,
    double* _ptr_array_spikemonitor_t,
    const int _numt
)
{
    using namespace brian;
    // TODO: fix int types, num_events and  cudaVector::size() are int but _ptr_array_spikemonitor_N[0] is size32_t
    int num_events;

    // KERNEL_CONSTANTS
    const int _numN = 1;
    const int _num_source_i = 100;
    const int _num_source_idx = 100;
    const int _num_spikespace = 101;
    const int _numcount = 100;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    num_events = monitor_t->size();
    _ptr_array_spikemonitor_N[0] = num_events;

    *dev_num_events = num_events;
}

__global__ void _copy_spikemonitor_codeobject_kernel(
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

void _copyToHost_spikemonitor_codeobject()
{
    using namespace brian;

    const std::clock_t _start_time = std::clock();

    // TODO: Use the correct dev_eventmonitor_N instead of dev_num_events
    //   and the correct _array_eventmonitor_N instead of host_num_events.
    //       use: dev_array_spikemonitor_N and _array_spikemonitor_N
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
        const int _num_source_i = 100;
        const int _num_source_idx = 100;
        const int _num_spikespace = 101;
        const int _numcount = 100;
        int32_t* const dev_array_spikemonitor_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]);
        const int _numi = dev_dynamic_array_spikemonitor_i.size();
        double* const dev_array_spikemonitor_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]);
        const int _numt = dev_dynamic_array_spikemonitor_t.size();

    _count_spikemonitor_codeobject_kernel<<<1,1>>>(
        dev_num_events,
        // HOST_PARAMETERS
        dev_array_spikemonitor_N,
            dev_array_spikegeneratorgroup_i,
            dev_array_spikemonitor__source_idx,
            _array_defaultclock_t[0],
            dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace],
            dev_array_spikemonitor_count,
            dev_array_spikemonitor_i,
            _numi,
            dev_array_spikemonitor_t,
            _numt
        );

    CUDA_CHECK_ERROR("_count_spikemonitor_codeobject_kernel");

    CUDA_SAFE_CALL(
            cudaMemcpy(&host_num_events, dev_num_events, sizeof(int), cudaMemcpyDeviceToHost)
            );

    // resize monitor device vectors
    THRUST_CHECK_ERROR(
            dev_dynamic_array_spikemonitor_t.resize(host_num_events)
            );
    THRUST_CHECK_ERROR(
            dev_dynamic_array_spikemonitor_i.resize(host_num_events)
            );

    _copy_spikemonitor_codeobject_kernel<<<1,1>>>(
        thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]),
        thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]),
        0          );

    CUDA_CHECK_ERROR("_copy_spikemonitor_codeobject_kernel");
}

void _debugmsg_spikemonitor_codeobject()
{
    using namespace brian;

    // HOST_CONSTANTS
    const int _numN = 1;
        const int _num_source_i = 100;
        const int _num_source_idx = 100;
        const int _num_spikespace = 101;
        const int _numcount = 100;
        int32_t* const dev_array_spikemonitor_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]);
        const int _numi = dev_dynamic_array_spikemonitor_i.size();
        double* const dev_array_spikemonitor_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]);
        const int _numt = dev_dynamic_array_spikemonitor_t.size();

    // TODO: can't we acces the correct _array_eventmonitor_N[0]
    //   value here without any kernel call?
    //   Yes: use _array_spikemonitor_N
    _run_debugmsg_spikemonitor_codeobject_kernel<<<1,1>>>(
            // HOST_PARAMETERS
            dev_array_spikemonitor_N,
            dev_array_spikegeneratorgroup_i,
            dev_array_spikemonitor__source_idx,
            _array_defaultclock_t[0],
            dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace],
            dev_array_spikemonitor_count,
            dev_array_spikemonitor_i,
            _numi,
            dev_array_spikemonitor_t,
            _numt
            );

    CUDA_CHECK_ERROR("_run_debugmsg_spikemonitor_codeobject_kernel");
}

// spikemonitor_codeobject ends here

// spikemonitor_2_codeobject starts here

__global__ void _run_spikemonitor_2_codeobject_init()
{
        monitor_t = new cudaVector<double>();
        monitor_i = new cudaVector<int32_t>();
}

__global__ void
kernel_spikemonitor_2_codeobject(
    int neurongroup_N,
    int32_t* count,
    // KERNEL_PARAMETERS
    int32_t* _ptr_array_spikemonitor_2_N,
    int32_t* _ptr_array_neurongroup_1_i,
    int32_t* _ptr_array_spikemonitor_2__source_idx,
    const double _value_array_defaultclock_t,
    int32_t* _ptr_array_neurongroup_1__spikespace,
    int32_t* _ptr_array_spikemonitor_2_count,
    int32_t* _ptr_array_spikemonitor_2_i,
    const int _numi,
    double* _ptr_array_spikemonitor_2_t,
    const int _numt
    )
{
    using namespace brian;
    int tid = threadIdx.x;
    int bid = blockIdx.x;


    // KERNEL_CONSTANTS
    const int _numN = 1;
    const int _num_source_i = 100;
    const int _num_source_idx = 100;
    const int _num_spikespace = 101;
    const int _numcount = 100;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    // scalar_code
        


    // using not parallel spikespace: filled from left with all spiking neuron IDs, -1 ends the list
    for(int i = 0; i < neurongroup_N; i++)
    {
        int32_t spiking_neuron = _ptr_array_neurongroup_1__spikespace[i];
        if(spiking_neuron != -1)
        {
            if(0 <= spiking_neuron && spiking_neuron < 100)
            {
                int _idx = spiking_neuron;
                int _vectorisation_idx = _idx;

                // vector_code
                                
                const double _source_t = _ptr_array_defaultclock_t[0];
                const int32_t _source_i = _ptr_array_neurongroup_1_i[_idx];
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

void _run_spikemonitor_2_codeobject()
{
    using namespace brian;


    const int _N = _array_spikemonitor_2_N[0];

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
        const int _num_source_i = 100;
        const int _num_source_idx = 100;
        const int _num_spikespace = 101;
        const int _numcount = 100;
        int32_t* const dev_array_spikemonitor_2_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_2_i[0]);
        const int _numi = dev_dynamic_array_spikemonitor_2_i.size();
        double* const dev_array_spikemonitor_2_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_2_t[0]);
        const int _numt = dev_dynamic_array_spikemonitor_2_t.size();


    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
_run_spikemonitor_2_codeobject_init<<<1,1>>>();

CUDA_CHECK_ERROR("_run_spikemonitor_2_codeobject_init");
num_blocks = 1;
num_threads = 1;

        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_spikemonitor_2_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_spikemonitor_2_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_spikemonitor_2_codeobject "
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
                        kernel_spikemonitor_2_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_spikemonitor_2_codeobject\n"
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


kernel_spikemonitor_2_codeobject<<<num_blocks, num_threads>>>(
        _num_spikespace-1,
        dev_array_spikemonitor_2_count,
        // HOST_PARAMETERS
        dev_array_spikemonitor_2_N,
            dev_array_neurongroup_1_i,
            dev_array_spikemonitor_2__source_idx,
            _array_defaultclock_t[0],
            dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace],
            dev_array_spikemonitor_2_count,
            dev_array_spikemonitor_2_i,
            _numi,
            dev_array_spikemonitor_2_t,
            _numt);

CUDA_CHECK_ERROR("kernel_spikemonitor_2_codeobject");


}

__global__ void _run_debugmsg_spikemonitor_2_codeobject_kernel(
    // KERNEL_PARAMETERS
    int32_t* _ptr_array_spikemonitor_2_N,
    int32_t* _ptr_array_neurongroup_1_i,
    int32_t* _ptr_array_spikemonitor_2__source_idx,
    const double _value_array_defaultclock_t,
    int32_t* _ptr_array_neurongroup_1__spikespace,
    int32_t* _ptr_array_spikemonitor_2_count,
    int32_t* _ptr_array_spikemonitor_2_i,
    const int _numi,
    double* _ptr_array_spikemonitor_2_t,
    const int _numt
)
{
    using namespace brian;

    // KERNEL_CONSTANTS
    const int _numN = 1;
    const int _num_source_i = 100;
    const int _num_source_idx = 100;
    const int _num_spikespace = 101;
    const int _numcount = 100;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    printf("Number of spikes: %d\n", _ptr_array_spikemonitor_2_N[0]);
}

__global__ void _count_spikemonitor_2_codeobject_kernel(
    int* dev_num_events,
    // KERNEL_PARAMETERS
    int32_t* _ptr_array_spikemonitor_2_N,
    int32_t* _ptr_array_neurongroup_1_i,
    int32_t* _ptr_array_spikemonitor_2__source_idx,
    const double _value_array_defaultclock_t,
    int32_t* _ptr_array_neurongroup_1__spikespace,
    int32_t* _ptr_array_spikemonitor_2_count,
    int32_t* _ptr_array_spikemonitor_2_i,
    const int _numi,
    double* _ptr_array_spikemonitor_2_t,
    const int _numt
)
{
    using namespace brian;
    // TODO: fix int types, num_events and  cudaVector::size() are int but _ptr_array_spikemonitor_2_N[0] is size32_t
    int num_events;

    // KERNEL_CONSTANTS
    const int _numN = 1;
    const int _num_source_i = 100;
    const int _num_source_idx = 100;
    const int _num_spikespace = 101;
    const int _numcount = 100;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    num_events = monitor_t->size();
    _ptr_array_spikemonitor_2_N[0] = num_events;

    *dev_num_events = num_events;
}

__global__ void _copy_spikemonitor_2_codeobject_kernel(
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

void _copyToHost_spikemonitor_2_codeobject()
{
    using namespace brian;

    const std::clock_t _start_time = std::clock();

    // TODO: Use the correct dev_eventmonitor_N instead of dev_num_events
    //   and the correct _array_eventmonitor_N instead of host_num_events.
    //       use: dev_array_spikemonitor_2_N and _array_spikemonitor_2_N
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
        const int _num_source_i = 100;
        const int _num_source_idx = 100;
        const int _num_spikespace = 101;
        const int _numcount = 100;
        int32_t* const dev_array_spikemonitor_2_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_2_i[0]);
        const int _numi = dev_dynamic_array_spikemonitor_2_i.size();
        double* const dev_array_spikemonitor_2_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_2_t[0]);
        const int _numt = dev_dynamic_array_spikemonitor_2_t.size();

    _count_spikemonitor_2_codeobject_kernel<<<1,1>>>(
        dev_num_events,
        // HOST_PARAMETERS
        dev_array_spikemonitor_2_N,
            dev_array_neurongroup_1_i,
            dev_array_spikemonitor_2__source_idx,
            _array_defaultclock_t[0],
            dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace],
            dev_array_spikemonitor_2_count,
            dev_array_spikemonitor_2_i,
            _numi,
            dev_array_spikemonitor_2_t,
            _numt
        );

    CUDA_CHECK_ERROR("_count_spikemonitor_2_codeobject_kernel");

    CUDA_SAFE_CALL(
            cudaMemcpy(&host_num_events, dev_num_events, sizeof(int), cudaMemcpyDeviceToHost)
            );

    // resize monitor device vectors
    THRUST_CHECK_ERROR(
            dev_dynamic_array_spikemonitor_2_t.resize(host_num_events)
            );
    THRUST_CHECK_ERROR(
            dev_dynamic_array_spikemonitor_2_i.resize(host_num_events)
            );

    _copy_spikemonitor_2_codeobject_kernel<<<1,1>>>(
        thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_2_t[0]),
        thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_2_i[0]),
        0          );

    CUDA_CHECK_ERROR("_copy_spikemonitor_2_codeobject_kernel");
}

void _debugmsg_spikemonitor_2_codeobject()
{
    using namespace brian;

    // HOST_CONSTANTS
    const int _numN = 1;
        const int _num_source_i = 100;
        const int _num_source_idx = 100;
        const int _num_spikespace = 101;
        const int _numcount = 100;
        int32_t* const dev_array_spikemonitor_2_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_2_i[0]);
        const int _numi = dev_dynamic_array_spikemonitor_2_i.size();
        double* const dev_array_spikemonitor_2_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_2_t[0]);
        const int _numt = dev_dynamic_array_spikemonitor_2_t.size();

    // TODO: can't we acces the correct _array_eventmonitor_N[0]
    //   value here without any kernel call?
    //   Yes: use _array_spikemonitor_2_N
    _run_debugmsg_spikemonitor_2_codeobject_kernel<<<1,1>>>(
            // HOST_PARAMETERS
            dev_array_spikemonitor_2_N,
            dev_array_neurongroup_1_i,
            dev_array_spikemonitor_2__source_idx,
            _array_defaultclock_t[0],
            dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace],
            dev_array_spikemonitor_2_count,
            dev_array_spikemonitor_2_i,
            _numi,
            dev_array_spikemonitor_2_t,
            _numt
            );

    CUDA_CHECK_ERROR("_run_debugmsg_spikemonitor_2_codeobject_kernel");
}

// spikemonitor_codeobject ends here


// spikemonitor_1_codeobject starts here

__global__ void _run_spikemonitor_1_codeobject_init()
{
        monitor_t = new cudaVector<double>();
        monitor_i = new cudaVector<int32_t>();
}

__global__ void
kernel_spikemonitor_1_codeobject(
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
                                
                const double _source_t = _ptr_array_defaultclock_t[0];
                const int32_t _source_i = _ptr_array_neurongroup_i[_idx];
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
    static bool first_run = true;
    if (first_run)
    {
_run_spikemonitor_1_codeobject_init<<<1,1>>>();

CUDA_CHECK_ERROR("_run_spikemonitor_1_codeobject_init");
num_blocks = 1;
num_threads = 1;

        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_spikemonitor_1_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_spikemonitor_1_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_spikemonitor_1_codeobject "
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
                        kernel_spikemonitor_1_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_spikemonitor_1_codeobject\n"
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


kernel_spikemonitor_1_codeobject<<<num_blocks, num_threads>>>(
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

CUDA_CHECK_ERROR("kernel_spikemonitor_1_codeobject");


}

__global__ void _run_debugmsg_spikemonitor_1_codeobject_kernel(
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

__global__ void _count_spikemonitor_1_codeobject_kernel(
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

__global__ void _copy_spikemonitor_1_codeobject_kernel(
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

    _count_spikemonitor_1_codeobject_kernel<<<1,1>>>(
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

    CUDA_CHECK_ERROR("_count_spikemonitor_1_codeobject_kernel");

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

    _copy_spikemonitor_1_codeobject_kernel<<<1,1>>>(
        thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_1_t[0]),
        thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_1_i[0]),
        0          );

    CUDA_CHECK_ERROR("_copy_spikemonitor_1_codeobject_kernel");
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
    _run_debugmsg_spikemonitor_1_codeobject_kernel<<<1,1>>>(
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

    CUDA_CHECK_ERROR("_run_debugmsg_spikemonitor_1_codeobject_kernel");
}

// spikemonitor_1_codeobject ends here 


// neurongroup_1_stateupdater.cu starts here 
__global__ void
kernel_neurongroup_1_stateupdater_codeobject(
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
    const double _lio_26 = 20.0855369231877 * (((_brian_pow(0.001, 1.0)) * 0.001) * _brian_exp(1.0f*(0.2 * (- 0.063))/0.001));
    const double _lio_27 = 0.032 * (- 0.063);
    const double _lio_28 = 0.48 * 0.001;
    const double _lio_29 = 1.0f*(0.642012708343871 * _brian_exp(1.0f*(0.025 * (- 0.063))/0.001))/0.001;
    const double _lio_30 = 1.0f*(- 0.025)/0.001;


    {
        ///// vector_code /////
                
        const double dt = _ptr_array_defaultclock_dt[0];
        double g_iKC_eKC = _ptr_array_neurongroup_1_g_iKC_eKC[_idx];
        double V = _ptr_array_neurongroup_1_V[_idx];
        double m = _ptr_array_neurongroup_1_m[_idx];
        double h = _ptr_array_neurongroup_1_h[_idx];
        double n = _ptr_array_neurongroup_1_n[_idx];
        char not_refractory = _ptr_array_neurongroup_1_not_refractory[_idx];
        double g_eKC_eKC = _ptr_array_neurongroup_1_g_eKC_eKC[_idx];
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
        const double _BA_n = 1.0f*(((1.0f*((- 0.032) * V)/(_lio_16 + (_lio_26 * _brian_exp(_lio_15 * V)))) + (1.0f*_lio_27/(_lio_16 + (_lio_26 * _brian_exp(_lio_15 * V))))) + (1.0f*_lio_28/(_lio_16 + (_lio_26 * _brian_exp(_lio_15 * V)))))/((1.0f*(0.032 * V)/(_lio_16 + (_lio_26 * _brian_exp(_lio_15 * V)))) - (((1.0f*_lio_27/(_lio_16 + (_lio_26 * _brian_exp(_lio_15 * V)))) + (1.0f*_lio_28/(_lio_16 + (_lio_26 * _brian_exp(_lio_15 * V))))) + (_lio_29 * _brian_exp(_lio_30 * V))));
        const double _n = (- _BA_n) + ((_BA_n + n) * _brian_exp(dt * ((1.0f*(0.032 * V)/(_lio_16 + (_lio_26 * _brian_exp(_lio_15 * V)))) - (((1.0f*_lio_27/(_lio_16 + (_lio_26 * _brian_exp(_lio_15 * V)))) + (1.0f*_lio_28/(_lio_16 + (_lio_26 * _brian_exp(_lio_15 * V))))) + (_lio_29 * _brian_exp(_lio_30 * V))))));
        V = _V;
        g_eKC_eKC = _g_eKC_eKC;
        g_iKC_eKC = _g_iKC_eKC;
        h = _h;
        m = _m;
        n = _n;
        _ptr_array_neurongroup_1_g_iKC_eKC[_idx] = g_iKC_eKC;
        _ptr_array_neurongroup_1_V[_idx] = V;
        _ptr_array_neurongroup_1_m[_idx] = m;
        _ptr_array_neurongroup_1_h[_idx] = h;
        _ptr_array_neurongroup_1_n[_idx] = n;
        _ptr_array_neurongroup_1_not_refractory[_idx] = not_refractory;
        _ptr_array_neurongroup_1_g_eKC_eKC[_idx] = g_eKC_eKC;


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
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    kernel_neurongroup_1_stateupdater_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;



        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_neurongroup_1_stateupdater_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_neurongroup_1_stateupdater_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_neurongroup_1_stateupdater_codeobject "
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
                        kernel_neurongroup_1_stateupdater_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_neurongroup_1_stateupdater_codeobject\n"
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


    kernel_neurongroup_1_stateupdater_codeobject<<<num_blocks, num_threads>>>(
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

    CUDA_CHECK_ERROR("kernel_neurongroup_1_stateupdater_codeobject");


}

// neurongroup_1_stateupdater_codeobject.cu ends here

// neurongroup_1_thresholder_codeobject.cu starts here 

__global__ void
kernel_neurongroup_1_thresholder_codeobject(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    double* _ptr_array_neurongroup_1_V,
    int32_t* _ptr_array_neurongroup_1__spikespace,
    double* _ptr_array_neurongroup_1_lastspike,
    char* _ptr_array_neurongroup_1_not_refractory,
    const double _value_array_defaultclock_t
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numV = 100;
    const int _num_spikespace = 101;
    const int _numlastspike = 100;
    const int _numnot_refractory = 100;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    assert(THREADS_PER_BLOCK == blockDim.x);


    if(_idx >= _N)
    {
        return;
    }


    ///// scalar_code /////
        



    {// there might be the same variable defined in scalar and vector code
    ///// vector_code /////
        
    const double V = _ptr_array_neurongroup_1_V[_idx];
    const char not_refractory = _ptr_array_neurongroup_1_not_refractory[_idx];
    char _cond;
    if(!not_refractory)
        _cond = (V > 0.0) && false;
    else 
        _cond = (V > 0.0) && true;


    if (_cond)
    {
        int32_t spike_index = atomicAdd(&_ptr_array_neurongroup_1__spikespace[_N], 1);
        _ptr_array_neurongroup_1__spikespace[spike_index] = _idx;
        // We have to use the pointer names directly here: The condition
        // might contain references to not_refractory or lastspike and in
        // that case the names will refer to a single entry.
        _ptr_array_neurongroup_1_not_refractory[_idx] = false;
        _ptr_array_neurongroup_1_lastspike[_idx] = _ptr_array_defaultclock_t[0];
    }
    }
}

void _run_neurongroup_1_thresholder_codeobject()
{
    using namespace brian;


    const int _N = 100;

    ///// HOST_CONSTANTS ///////////
    const int _numV = 100;
        const int _num_spikespace = 101;
        const int _numlastspike = 100;
        const int _numnot_refractory = 100;


    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    kernel_neurongroup_1_thresholder_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;



        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_neurongroup_1_thresholder_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_neurongroup_1_thresholder_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_neurongroup_1_thresholder_codeobject "
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
                        kernel_neurongroup_1_thresholder_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_neurongroup_1_thresholder_codeobject\n"
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

        _reset_neurongroup_1_thresholder_codeobject<<<num_blocks, num_threads>>>(
                dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace]
            );

        CUDA_CHECK_ERROR("_reset_neurongroup_1_thresholder_codeobject");

    kernel_neurongroup_1_thresholder_codeobject<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            dev_array_neurongroup_1_V,
            dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace],
            dev_array_neurongroup_1_lastspike,
            dev_array_neurongroup_1_not_refractory,
            _array_defaultclock_t[0]
        );

    CUDA_CHECK_ERROR("kernel_neurongroup_1_thresholder_codeobject");


}

// neurongroup_1_thresholder_codeobject.cu ends here

// neurongroup_stateupdater_codeobject.cu starts here

__global__ void
kernel_neurongroup_stateupdater_codeobject(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    double* _ptr_array_neurongroup_V,
    const double _value_array_defaultclock_dt,
    double* _ptr_array_neurongroup_g_PN_iKC,
    double* _ptr_array_neurongroup_h,
    double* _ptr_array_neurongroup_m,
    double* _ptr_array_neurongroup_n,
    char* _ptr_array_neurongroup_not_refractory
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numV = 2500;
    const int _numg_PN_iKC = 2500;
    const int _numh = 2500;
    const int _numm = 2500;
    const int _numn = 2500;
    const int _numnot_refractory = 2500;

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
    const double _lio_5 = 0.0 - (1.0f*2.67e-08/3e-10);
    const double _lio_6 = 1.0f*((- 1.0) * 1.4299999999999999e-06)/3e-10;
    const double _lio_7 = 1.0f*7.15e-06/3e-10;
    const double _lio_8 = 1.0f*1.0/3e-10;
    const double _lio_9 = _brian_exp(1.0f*(- dt)/0.002);
    const double _lio_10 = 1.0f*(0.329137207652868 * _brian_exp(1.0f*(0.0555555555555556 * (- 0.063))/0.001))/0.001;
    const double _lio_11 = 1.0f*(- 0.0555555555555556)/0.001;
    const double _lio_12 = 2980.95798704173 * (0.001 * _brian_exp(1.0f*(0.2 * (- 0.063))/0.001));
    const double _lio_13 = 1.0f*(- 0.2)/0.001;
    const double _lio_14 = ((- 1.0) * (_brian_pow(0.001, 1.0))) * 0.001;
    const double _lio_15 = 25.7903399171931 * (((_brian_pow(0.001, 1.0)) * 0.001) * _brian_exp(1.0f*(0.25 * (- 0.063))/0.001));
    const double _lio_16 = 1.0f*(- 0.25)/0.001;
    const double _lio_17 = 0.32 * (- 0.063);
    const double _lio_18 = 4.16 * 0.001;
    const double _lio_19 = 0.0 - ((_brian_pow(0.001, 1.0)) * 0.001);
    const double _lio_20 = 0.000335462627902512 * (((_brian_pow(0.001, 1.0)) * 0.001) * _brian_exp(1.0f*((- 0.2) * (- 0.063))/0.001));
    const double _lio_21 = 1.0f*0.2/0.001;
    const double _lio_22 = 0.28 * (- 0.063);
    const double _lio_23 = 11.2 * 0.001;
    const double _lio_24 = 20.0855369231877 * (((_brian_pow(0.001, 1.0)) * 0.001) * _brian_exp(1.0f*(0.2 * (- 0.063))/0.001));
    const double _lio_25 = 0.032 * (- 0.063);
    const double _lio_26 = 0.48 * 0.001;
    const double _lio_27 = 1.0f*(0.642012708343871 * _brian_exp(1.0f*(0.025 * (- 0.063))/0.001))/0.001;
    const double _lio_28 = 1.0f*(- 0.025)/0.001;


    {
        ///// vector_code /////
                
        const double dt = _ptr_array_defaultclock_dt[0];
        double g_PN_iKC = _ptr_array_neurongroup_g_PN_iKC[_idx];
        double V = _ptr_array_neurongroup_V[_idx];
        double m = _ptr_array_neurongroup_m[_idx];
        double h = _ptr_array_neurongroup_h[_idx];
        double n = _ptr_array_neurongroup_n[_idx];
        char not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
        if(!not_refractory)
            not_refractory = false || (! (V > 0.0));
        else 
            not_refractory = true || (! (V > 0.0));
        const double _BA_V = 1.0f*(_lio_1 + (((_lio_2 * (_brian_pow(n, 4.0))) + (_lio_3 * (h * (_brian_pow(m, 3.0))))) + (_lio_4 * g_PN_iKC)))/((_lio_5 + (_lio_6 * (_brian_pow(n, 4.0)))) - ((_lio_7 * (h * (_brian_pow(m, 3.0)))) + (_lio_8 * g_PN_iKC)));
        const double _V = (- _BA_V) + ((V + _BA_V) * _brian_exp(dt * ((_lio_5 + (_lio_6 * (_brian_pow(n, 4.0)))) - ((_lio_7 * (h * (_brian_pow(m, 3.0)))) + (_lio_8 * g_PN_iKC)))));
        const double _g_PN_iKC = _lio_9 * g_PN_iKC;
        const double _BA_h = 1.0f*(_lio_10 * _brian_exp(_lio_11 * V))/((1.0f*(- 4.0)/(0.001 + (_lio_12 * _brian_exp(_lio_13 * V)))) - (_lio_10 * _brian_exp(_lio_11 * V)));
        const double _h = (- _BA_h) + ((_BA_h + h) * _brian_exp(dt * ((1.0f*(- 4.0)/(0.001 + (_lio_12 * _brian_exp(_lio_13 * V)))) - (_lio_10 * _brian_exp(_lio_11 * V)))));
        const double _BA_m = 1.0f*(((1.0f*((- 0.32) * V)/(_lio_14 + (_lio_15 * _brian_exp(_lio_16 * V)))) + (1.0f*_lio_17/(_lio_14 + (_lio_15 * _brian_exp(_lio_16 * V))))) + (1.0f*_lio_18/(_lio_14 + (_lio_15 * _brian_exp(_lio_16 * V)))))/(((((1.0f*((- 0.28) * V)/(_lio_19 + (_lio_20 * _brian_exp(_lio_21 * V)))) + (1.0f*(0.32 * V)/(_lio_14 + (_lio_15 * _brian_exp(_lio_16 * V))))) + (1.0f*_lio_22/(_lio_19 + (_lio_20 * _brian_exp(_lio_21 * V))))) + (1.0f*_lio_23/(_lio_19 + (_lio_20 * _brian_exp(_lio_21 * V))))) - ((1.0f*_lio_17/(_lio_14 + (_lio_15 * _brian_exp(_lio_16 * V)))) + (1.0f*_lio_18/(_lio_14 + (_lio_15 * _brian_exp(_lio_16 * V))))));
        const double _m = (- _BA_m) + ((_BA_m + m) * _brian_exp(dt * (((((1.0f*((- 0.28) * V)/(_lio_19 + (_lio_20 * _brian_exp(_lio_21 * V)))) + (1.0f*(0.32 * V)/(_lio_14 + (_lio_15 * _brian_exp(_lio_16 * V))))) + (1.0f*_lio_22/(_lio_19 + (_lio_20 * _brian_exp(_lio_21 * V))))) + (1.0f*_lio_23/(_lio_19 + (_lio_20 * _brian_exp(_lio_21 * V))))) - ((1.0f*_lio_17/(_lio_14 + (_lio_15 * _brian_exp(_lio_16 * V)))) + (1.0f*_lio_18/(_lio_14 + (_lio_15 * _brian_exp(_lio_16 * V))))))));
        const double _BA_n = 1.0f*(((1.0f*((- 0.032) * V)/(_lio_14 + (_lio_24 * _brian_exp(_lio_13 * V)))) + (1.0f*_lio_25/(_lio_14 + (_lio_24 * _brian_exp(_lio_13 * V))))) + (1.0f*_lio_26/(_lio_14 + (_lio_24 * _brian_exp(_lio_13 * V)))))/((1.0f*(0.032 * V)/(_lio_14 + (_lio_24 * _brian_exp(_lio_13 * V)))) - (((1.0f*_lio_25/(_lio_14 + (_lio_24 * _brian_exp(_lio_13 * V)))) + (1.0f*_lio_26/(_lio_14 + (_lio_24 * _brian_exp(_lio_13 * V))))) + (_lio_27 * _brian_exp(_lio_28 * V))));
        const double _n = (- _BA_n) + ((_BA_n + n) * _brian_exp(dt * ((1.0f*(0.032 * V)/(_lio_14 + (_lio_24 * _brian_exp(_lio_13 * V)))) - (((1.0f*_lio_25/(_lio_14 + (_lio_24 * _brian_exp(_lio_13 * V)))) + (1.0f*_lio_26/(_lio_14 + (_lio_24 * _brian_exp(_lio_13 * V))))) + (_lio_27 * _brian_exp(_lio_28 * V))))));
        V = _V;
        g_PN_iKC = _g_PN_iKC;
        h = _h;
        m = _m;
        n = _n;
        _ptr_array_neurongroup_g_PN_iKC[_idx] = g_PN_iKC;
        _ptr_array_neurongroup_V[_idx] = V;
        _ptr_array_neurongroup_m[_idx] = m;
        _ptr_array_neurongroup_h[_idx] = h;
        _ptr_array_neurongroup_n[_idx] = n;
        _ptr_array_neurongroup_not_refractory[_idx] = not_refractory;


    }
}

void _run_neurongroup_stateupdater_codeobject()
{
    using namespace brian;


    const int _N = 2500;

    ///// HOST_CONSTANTS ///////////
    const int _numV = 2500;
        const int _numg_PN_iKC = 2500;
        const int _numh = 2500;
        const int _numm = 2500;
        const int _numn = 2500;
        const int _numnot_refractory = 2500;


    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    kernel_neurongroup_stateupdater_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;



        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_neurongroup_stateupdater_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_neurongroup_stateupdater_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_neurongroup_stateupdater_codeobject "
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
                        kernel_neurongroup_stateupdater_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_neurongroup_stateupdater_codeobject\n"
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


    kernel_neurongroup_stateupdater_codeobject<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            dev_array_neurongroup_V,
            _array_defaultclock_dt[0],
            dev_array_neurongroup_g_PN_iKC,
            dev_array_neurongroup_h,
            dev_array_neurongroup_m,
            dev_array_neurongroup_n,
            dev_array_neurongroup_not_refractory
        );

    CUDA_CHECK_ERROR("kernel_neurongroup_stateupdater_codeobject");


}

// neurongroup_stateupdater_codeobject ends here

// spikegeneratorgroup starts here

__global__ void
kernel_spikegeneratorgroup_codeobject(
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
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    kernel_spikegeneratorgroup_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;



        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_spikegeneratorgroup_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_spikegeneratorgroup_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_spikegeneratorgroup_codeobject "
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
                        kernel_spikegeneratorgroup_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_spikegeneratorgroup_codeobject\n"
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
    _reset_spikegeneratorgroup_codeobject<<<num_blocks, num_threads>>>(
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

    kernel_spikegeneratorgroup_codeobject<<<num_blocks, num_threads>>>(
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

    CUDA_CHECK_ERROR("kernel_spikegeneratorgroup_codeobject");


}

// spikegeneratorgroup ends here

// neurongroup_thresholder_codeobject starts here 

__global__ void
kernel_neurongroup_thresholder_codeobject(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    double* _ptr_array_neurongroup_V,
    int32_t* _ptr_array_neurongroup__spikespace,
    double* _ptr_array_neurongroup_lastspike,
    char* _ptr_array_neurongroup_not_refractory,
    const double _value_array_defaultclock_t
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numV = 2500;
    const int _num_spikespace = 2501;
    const int _numlastspike = 2500;
    const int _numnot_refractory = 2500;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;


    assert(THREADS_PER_BLOCK == blockDim.x);


    if(_idx >= _N)
    {
        return;
    }


    ///// scalar_code /////
        



    {// there might be the same variable defined in scalar and vector code
    ///// vector_code /////
        
    const double V = _ptr_array_neurongroup_V[_idx];
    const char not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
    char _cond;
    if(!not_refractory)
        _cond = (V > 0.0) && false;
    else 
        _cond = (V > 0.0) && true;


    if (_cond)
    {
        int32_t spike_index = atomicAdd(&_ptr_array_neurongroup__spikespace[_N], 1);
        _ptr_array_neurongroup__spikespace[spike_index] = _idx;
        // We have to use the pointer names directly here: The condition
        // might contain references to not_refractory or lastspike and in
        // that case the names will refer to a single entry.
        _ptr_array_neurongroup_not_refractory[_idx] = false;
        _ptr_array_neurongroup_lastspike[_idx] = _ptr_array_defaultclock_t[0];
    }
    }
}

void _run_neurongroup_thresholder_codeobject()
{
    using namespace brian;


    const int _N = 2500;

    ///// HOST_CONSTANTS ///////////
    const int _numV = 2500;
        const int _num_spikespace = 2501;
        const int _numlastspike = 2500;
        const int _numnot_refractory = 2500;


    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    kernel_neurongroup_thresholder_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;



        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_neurongroup_thresholder_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_neurongroup_thresholder_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_neurongroup_thresholder_codeobject "
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
                        kernel_neurongroup_thresholder_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_neurongroup_thresholder_codeobject\n"
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

        _reset_neurongroup_thresholder_codeobject<<<num_blocks, num_threads>>>(
                dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace]
            );

        CUDA_CHECK_ERROR("_reset_neurongroup_thresholder_codeobject");

    kernel_neurongroup_thresholder_codeobject<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            dev_array_neurongroup_V,
            dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace],
            dev_array_neurongroup_lastspike,
            dev_array_neurongroup_not_refractory,
            _array_defaultclock_t[0]
        );

    CUDA_CHECK_ERROR("kernel_neurongroup_thresholder_codeobject");


}

// neurongroup_thresholder_codeobject ends here

// network.cu starts here 

#define Clock_epsilon 1e-14

double Network::_last_run_time = 0.0;
double Network::_last_run_completed_fraction = 0.0;

Network::Network()
{
    t = 0.0;
}

void Network::clear()
{
    objects.clear();
}

void Network::add(Clock *clock, codeobj_func func)
{
#if defined(_MSC_VER) && (_MSC_VER>=1700)
    objects.push_back(std::make_pair(std::move(clock), std::move(func)));
#else
    objects.push_back(std::make_pair(clock, func));
#endif
}

void Network::run(const double duration, void (*report_func)(const double, const double, const double, const double), const double report_period)
{
    std::clock_t start, current;
    const double t_start = t;
    const double t_end = t + duration;
    double next_report_time = report_period;
    // compute the set of clocks
    compute_clocks();
    // set interval for all clocks

    for(std::set<Clock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
        (*i)->set_interval(t, t_end);

    start = std::clock();
    if (report_func)
    {
        report_func(0.0, 0.0, t_start, duration);
    }

    Clock* clock = next_clocks();
    double elapsed_realtime;
    bool did_break_early = false;

    while(clock && clock->running())
    {
        t = clock->t[0];

        for(int i=0; i<objects.size(); i++)
        {
            if (report_func)
            {
                current = std::clock();
                const double elapsed = (double)(current - start) / CLOCKS_PER_SEC;
                if (elapsed > next_report_time)
                {
                    report_func(elapsed, (clock->t[0]-t_start)/duration, t_start, duration);
                    next_report_time += report_period;
                }
            }
            Clock *obj_clock = objects[i].first;
            // Only execute the object if it uses the right clock for this step
            if (curclocks.find(obj_clock) != curclocks.end())
            {
                codeobj_func func = objects[i].second;
                if (func)  // code objects can be NULL in cases where we store just the clock
                {
                    func();
                }
            }
        }
        for(std::set<Clock*>::iterator i=curclocks.begin(); i!=curclocks.end(); i++)
            (*i)->tick();
        clock = next_clocks();

        // Advance index for circular eventspace vector (for no_or_const_delay_mode)
        brian::current_idx_array_neurongroup_1__spikespace = (brian::current_idx_array_neurongroup_1__spikespace + 1) % brian::dev_array_neurongroup_1__spikespace.size();
        brian::current_idx_array_neurongroup__spikespace = (brian::current_idx_array_neurongroup__spikespace + 1) % brian::dev_array_neurongroup__spikespace.size();
        brian::previous_idx_array_spikegeneratorgroup__spikespace = brian::current_idx_array_spikegeneratorgroup__spikespace;
        brian::current_idx_array_spikegeneratorgroup__spikespace = (brian::current_idx_array_spikegeneratorgroup__spikespace + 1) % brian::dev_array_spikegeneratorgroup__spikespace.size();

        current = std::clock();
        elapsed_realtime = (double)(current - start)/CLOCKS_PER_SEC;


    }

    if(!did_break_early) t = t_end;

    _last_run_time = elapsed_realtime;
    if(duration>0)
    {
        _last_run_completed_fraction = (t-t_start)/duration;
    } else {
        _last_run_completed_fraction = 1.0;
    }
    if (report_func)
    {
        report_func(elapsed_realtime, 1.0, t_start, duration);
    }
}

void Network::compute_clocks()
{
    clocks.clear();
    for(int i=0; i<objects.size(); i++)
    {
        Clock *clock = objects[i].first;
        clocks.insert(clock);
    }
}

Clock* Network::next_clocks()
{
    // find minclock, clock with smallest t value
    Clock *minclock = *clocks.begin();
    if (!minclock) // empty list of clocks
        return NULL;

    for(std::set<Clock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
    {
        Clock *clock = *i;
        if(clock->t[0]<minclock->t[0])
            minclock = clock;
    }
    // find set of equal clocks
    curclocks.clear();

    double t = minclock->t[0];
    for(std::set<Clock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
    {
        Clock *clock = *i;
        double s = clock->t[0];
        if(s==t || fabs(s-t)<=Clock_epsilon)
            curclocks.insert(clock);
    }
    return minclock;
}

// network.cu ends here


// objects.cu starts here 

size_t brian::used_device_memory = 0;

//////////////// clocks ///////////////////
Clock brian::defaultclock;

//////////////// networks /////////////////
Network brian::magicnetwork;

//////////////// arrays ///////////////////
double * brian::_array_defaultclock_dt;
double * brian::dev_array_defaultclock_dt;
__device__ double * brian::d_array_defaultclock_dt;
const int brian::_num__array_defaultclock_dt = 1;

double * brian::_array_defaultclock_t;
double * brian::dev_array_defaultclock_t;
__device__ double * brian::d_array_defaultclock_t;
const int brian::_num__array_defaultclock_t = 1;

int64_t * brian::_array_defaultclock_timestep;
int64_t * brian::dev_array_defaultclock_timestep;
__device__ int64_t * brian::d_array_defaultclock_timestep;
const int brian::_num__array_defaultclock_timestep = 1;

double * brian::_array_neurongroup_1_g_eKC_eKC;
double * brian::dev_array_neurongroup_1_g_eKC_eKC;
__device__ double * brian::d_array_neurongroup_1_g_eKC_eKC;
const int brian::_num__array_neurongroup_1_g_eKC_eKC = 100;

double * brian::_array_neurongroup_1_g_iKC_eKC;
double * brian::dev_array_neurongroup_1_g_iKC_eKC;
__device__ double * brian::d_array_neurongroup_1_g_iKC_eKC;
const int brian::_num__array_neurongroup_1_g_iKC_eKC = 100;

double * brian::_array_neurongroup_1_h;
double * brian::dev_array_neurongroup_1_h;
__device__ double * brian::d_array_neurongroup_1_h;
const int brian::_num__array_neurongroup_1_h = 100;

int32_t * brian::_array_neurongroup_1_i;
int32_t * brian::dev_array_neurongroup_1_i;
__device__ int32_t * brian::d_array_neurongroup_1_i;
const int brian::_num__array_neurongroup_1_i = 100;

double * brian::_array_neurongroup_1_lastspike;
double * brian::dev_array_neurongroup_1_lastspike;
__device__ double * brian::d_array_neurongroup_1_lastspike;
const int brian::_num__array_neurongroup_1_lastspike = 100;

double * brian::_array_neurongroup_1_m;
double * brian::dev_array_neurongroup_1_m;
__device__ double * brian::d_array_neurongroup_1_m;
const int brian::_num__array_neurongroup_1_m = 100;

double * brian::_array_neurongroup_1_n;
double * brian::dev_array_neurongroup_1_n;
__device__ double * brian::d_array_neurongroup_1_n;
const int brian::_num__array_neurongroup_1_n = 100;

char * brian::_array_neurongroup_1_not_refractory;
char * brian::dev_array_neurongroup_1_not_refractory;
__device__ char * brian::d_array_neurongroup_1_not_refractory;
const int brian::_num__array_neurongroup_1_not_refractory = 100;

double * brian::_array_neurongroup_1_V;
double * brian::dev_array_neurongroup_1_V;
__device__ double * brian::d_array_neurongroup_1_V;
const int brian::_num__array_neurongroup_1_V = 100;

double * brian::_array_neurongroup_g_PN_iKC;
double * brian::dev_array_neurongroup_g_PN_iKC;
__device__ double * brian::d_array_neurongroup_g_PN_iKC;
const int brian::_num__array_neurongroup_g_PN_iKC = 2500;

double * brian::_array_neurongroup_h;
double * brian::dev_array_neurongroup_h;
__device__ double * brian::d_array_neurongroup_h;
const int brian::_num__array_neurongroup_h = 2500;

int32_t * brian::_array_neurongroup_i;
int32_t * brian::dev_array_neurongroup_i;
__device__ int32_t * brian::d_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 2500;

double * brian::_array_neurongroup_lastspike;
double * brian::dev_array_neurongroup_lastspike;
__device__ double * brian::d_array_neurongroup_lastspike;
const int brian::_num__array_neurongroup_lastspike = 2500;

double * brian::_array_neurongroup_m;
double * brian::dev_array_neurongroup_m;
__device__ double * brian::d_array_neurongroup_m;
const int brian::_num__array_neurongroup_m = 2500;

double * brian::_array_neurongroup_n;
double * brian::dev_array_neurongroup_n;
__device__ double * brian::d_array_neurongroup_n;
const int brian::_num__array_neurongroup_n = 2500;

char * brian::_array_neurongroup_not_refractory;
char * brian::dev_array_neurongroup_not_refractory;
__device__ char * brian::d_array_neurongroup_not_refractory;
const int brian::_num__array_neurongroup_not_refractory = 2500;

double * brian::_array_neurongroup_V;
double * brian::dev_array_neurongroup_V;
__device__ double * brian::d_array_neurongroup_V;
const int brian::_num__array_neurongroup_V = 2500;

int32_t * brian::_array_spikegeneratorgroup__lastindex;
int32_t * brian::dev_array_spikegeneratorgroup__lastindex;
__device__ int32_t * brian::d_array_spikegeneratorgroup__lastindex;
const int brian::_num__array_spikegeneratorgroup__lastindex = 1;

int32_t * brian::_array_spikegeneratorgroup__period_bins;
int32_t * brian::dev_array_spikegeneratorgroup__period_bins;
__device__ int32_t * brian::d_array_spikegeneratorgroup__period_bins;
const int brian::_num__array_spikegeneratorgroup__period_bins = 1;

int32_t * brian::_array_spikegeneratorgroup_i;
int32_t * brian::dev_array_spikegeneratorgroup_i;
__device__ int32_t * brian::d_array_spikegeneratorgroup_i;
const int brian::_num__array_spikegeneratorgroup_i = 100;

double * brian::_array_spikegeneratorgroup_period;
double * brian::dev_array_spikegeneratorgroup_period;
__device__ double * brian::d_array_spikegeneratorgroup_period;
const int brian::_num__array_spikegeneratorgroup_period = 1;

int32_t * brian::_array_spikemonitor_1__source_idx;
int32_t * brian::dev_array_spikemonitor_1__source_idx;
__device__ int32_t * brian::d_array_spikemonitor_1__source_idx;
const int brian::_num__array_spikemonitor_1__source_idx = 2500;

int32_t * brian::_array_spikemonitor_1_count;
int32_t * brian::dev_array_spikemonitor_1_count;
__device__ int32_t * brian::d_array_spikemonitor_1_count;
const int brian::_num__array_spikemonitor_1_count = 2500;

int32_t * brian::_array_spikemonitor_1_N;
int32_t * brian::dev_array_spikemonitor_1_N;
__device__ int32_t * brian::d_array_spikemonitor_1_N;
const int brian::_num__array_spikemonitor_1_N = 1;

int32_t * brian::_array_spikemonitor_2__source_idx;
int32_t * brian::dev_array_spikemonitor_2__source_idx;
__device__ int32_t * brian::d_array_spikemonitor_2__source_idx;
const int brian::_num__array_spikemonitor_2__source_idx = 100;

int32_t * brian::_array_spikemonitor_2_count;
int32_t * brian::dev_array_spikemonitor_2_count;
__device__ int32_t * brian::d_array_spikemonitor_2_count;
const int brian::_num__array_spikemonitor_2_count = 100;

int32_t * brian::_array_spikemonitor_2_N;
int32_t * brian::dev_array_spikemonitor_2_N;
__device__ int32_t * brian::d_array_spikemonitor_2_N;
const int brian::_num__array_spikemonitor_2_N = 1;

int32_t * brian::_array_spikemonitor__source_idx;
int32_t * brian::dev_array_spikemonitor__source_idx;
__device__ int32_t * brian::d_array_spikemonitor__source_idx;
const int brian::_num__array_spikemonitor__source_idx = 100;

int32_t * brian::_array_spikemonitor_count;
int32_t * brian::dev_array_spikemonitor_count;
__device__ int32_t * brian::d_array_spikemonitor_count;
const int brian::_num__array_spikemonitor_count = 100;

int32_t * brian::_array_spikemonitor_N;
int32_t * brian::dev_array_spikemonitor_N;
__device__ int32_t * brian::d_array_spikemonitor_N;
const int brian::_num__array_spikemonitor_N = 1;

int32_t * brian::_array_synapses_1_N;
int32_t * brian::dev_array_synapses_1_N;
__device__ int32_t * brian::d_array_synapses_1_N;
const int brian::_num__array_synapses_1_N = 1;

int32_t * brian::_array_synapses_2_N;
int32_t * brian::dev_array_synapses_2_N;
__device__ int32_t * brian::d_array_synapses_2_N;
const int brian::_num__array_synapses_2_N = 1;

int32_t * brian::_array_synapses_N;
int32_t * brian::dev_array_synapses_N;
__device__ int32_t * brian::d_array_synapses_N;
const int brian::_num__array_synapses_N = 1;


//////////////// eventspaces ///////////////
// we dynamically create multiple eventspaces in no_or_const_delay_mode
// for initiating the first spikespace, we need a host pointer
// for choosing the right spikespace, we need a global index variable
int32_t * brian::_array_neurongroup_1__spikespace;
const int brian::_num__array_neurongroup_1__spikespace = 101;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_1__spikespace(1);
int brian::current_idx_array_neurongroup_1__spikespace = 0;
int32_t * brian::_array_neurongroup__spikespace;
const int brian::_num__array_neurongroup__spikespace = 2501;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup__spikespace(1);
int brian::current_idx_array_neurongroup__spikespace = 0;
int32_t * brian::_array_spikegeneratorgroup__spikespace;
const int brian::_num__array_spikegeneratorgroup__spikespace = 101;
thrust::host_vector<int32_t*> brian::dev_array_spikegeneratorgroup__spikespace(1);
int brian::current_idx_array_spikegeneratorgroup__spikespace = 0;
int brian::previous_idx_array_spikegeneratorgroup__spikespace;

//////////////// dynamic arrays 1d /////////
thrust::host_vector<int32_t> brian::_dynamic_array_spikegeneratorgroup__timebins;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup__timebins;
thrust::host_vector<int32_t> brian::_dynamic_array_spikegeneratorgroup_neuron_index;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup_neuron_index;
thrust::host_vector<int32_t> brian::_dynamic_array_spikegeneratorgroup_spike_number;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup_spike_number;
thrust::host_vector<double> brian::_dynamic_array_spikegeneratorgroup_spike_time;
thrust::device_vector<double> brian::dev_dynamic_array_spikegeneratorgroup_spike_time;
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_1_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_1_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_1_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_1_t;
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_2_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_2_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_2_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_2_t;
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_t;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_Apost;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_Apost;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_Apre;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_Apre;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_delay;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_delay_1;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_delay_1;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_g_raw;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_g_raw;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_lastupdate;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_lastupdate;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_outgoing;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_2_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_2_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2_N_outgoing;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_weight;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_weight;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
int32_t * brian::_static_array__dynamic_array_spikegeneratorgroup__timebins;
int32_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup__timebins;
__device__ int32_t * brian::d_static_array__dynamic_array_spikegeneratorgroup__timebins;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup__timebins = 19676;
int64_t * brian::_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
int64_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
__device__ int64_t * brian::d_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_neuron_index = 19676;
int64_t * brian::_static_array__dynamic_array_spikegeneratorgroup_spike_number;
int64_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup_spike_number;
__device__ int64_t * brian::d_static_array__dynamic_array_spikegeneratorgroup_spike_number;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_spike_number = 19676;
double * brian::_static_array__dynamic_array_spikegeneratorgroup_spike_time;
double * brian::dev_static_array__dynamic_array_spikegeneratorgroup_spike_time;
__device__ double * brian::d_static_array__dynamic_array_spikegeneratorgroup_spike_time;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_spike_time = 19676;
double * brian::_timedarray_1_values;
double * brian::dev_timedarray_1_values;
__device__ double * brian::d_timedarray_1_values;
const int brian::_num__timedarray_1_values = 250000;
double * brian::_timedarray_2_values;
double * brian::dev_timedarray_2_values;
__device__ double * brian::d_timedarray_2_values;
const int brian::_num__timedarray_2_values = 250000;
double * brian::_timedarray_3_values;
double * brian::dev_timedarray_3_values;
__device__ double * brian::d_timedarray_3_values;
const int brian::_num__timedarray_3_values = 250000;
double * brian::_timedarray_4_values;
double * brian::dev_timedarray_4_values;
__device__ double * brian::d_timedarray_4_values;
const int brian::_num__timedarray_4_values = 250000;
double * brian::_timedarray_values;
double * brian::dev_timedarray_values;
__device__ double * brian::d_timedarray_values;
const int brian::_num__timedarray_values = 250000;

//////////////// synapses /////////////////
// synapses
int32_t synapses_source_start_index;
int32_t synapses_source_stop_index;
bool brian::synapses_multiple_pre_post = false;
// synapses_pre
__device__ int* brian::synapses_pre_num_synapses_by_pre;
__device__ int* brian::synapses_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_pre_unique_delays;
__device__ int* brian::synapses_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_pre_global_bundle_id_start_by_pre;
int brian::synapses_pre_max_bundle_size = 0;
int brian::synapses_pre_mean_bundle_size = 0;
int brian::synapses_pre_max_size = 0;
__device__ int* brian::synapses_pre_num_unique_delays_by_pre;
int brian::synapses_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_pre_synapse_ids;
__device__ int* brian::synapses_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_pre;
int brian::synapses_pre_eventspace_idx = 0;
int brian::synapses_pre_delay;
bool brian::synapses_pre_scalar_delay;
// synapses_1
int32_t synapses_1_source_start_index;
int32_t synapses_1_source_stop_index;
bool brian::synapses_1_multiple_pre_post = false;
// synapses_1_post
__device__ int* brian::synapses_1_post_num_synapses_by_pre;
__device__ int* brian::synapses_1_post_num_synapses_by_bundle;
__device__ int* brian::synapses_1_post_unique_delays;
__device__ int* brian::synapses_1_post_synapses_offset_by_bundle;
__device__ int* brian::synapses_1_post_global_bundle_id_start_by_pre;
int brian::synapses_1_post_max_bundle_size = 0;
int brian::synapses_1_post_mean_bundle_size = 0;
int brian::synapses_1_post_max_size = 0;
__device__ int* brian::synapses_1_post_num_unique_delays_by_pre;
int brian::synapses_1_post_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_1_post_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_1_post_synapse_ids;
__device__ int* brian::synapses_1_post_unique_delay_start_idcs;
__device__ int* brian::synapses_1_post_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_1_post;
int brian::synapses_1_post_eventspace_idx = 0;
int brian::synapses_1_post_delay;
bool brian::synapses_1_post_scalar_delay;
// synapses_1_pre
__device__ int* brian::synapses_1_pre_num_synapses_by_pre;
__device__ int* brian::synapses_1_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_1_pre_unique_delays;
__device__ int* brian::synapses_1_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_1_pre_global_bundle_id_start_by_pre;
int brian::synapses_1_pre_max_bundle_size = 0;
int brian::synapses_1_pre_mean_bundle_size = 0;
int brian::synapses_1_pre_max_size = 0;
__device__ int* brian::synapses_1_pre_num_unique_delays_by_pre;
int brian::synapses_1_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_1_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_1_pre_synapse_ids;
__device__ int* brian::synapses_1_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_1_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_1_pre;
int brian::synapses_1_pre_eventspace_idx = 0;
int brian::synapses_1_pre_delay;
bool brian::synapses_1_pre_scalar_delay;
// synapses_2
int32_t synapses_2_source_start_index;
int32_t synapses_2_source_stop_index;
bool brian::synapses_2_multiple_pre_post = false;
// synapses_2_pre
__device__ int* brian::synapses_2_pre_num_synapses_by_pre;
__device__ int* brian::synapses_2_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_2_pre_unique_delays;
__device__ int* brian::synapses_2_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_2_pre_global_bundle_id_start_by_pre;
int brian::synapses_2_pre_max_bundle_size = 0;
int brian::synapses_2_pre_mean_bundle_size = 0;
int brian::synapses_2_pre_max_size = 0;
__device__ int* brian::synapses_2_pre_num_unique_delays_by_pre;
int brian::synapses_2_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_2_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_2_pre_synapse_ids;
__device__ int* brian::synapses_2_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_2_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_2_pre;
int brian::synapses_2_pre_eventspace_idx = 0;
int brian::synapses_2_pre_delay;
bool brian::synapses_2_pre_scalar_delay;

int brian::num_parallel_blocks;
int brian::max_threads_per_block;
int brian::max_threads_per_sm;
int brian::max_shared_mem_size;
int brian::num_threads_per_warp;

__global__ void synapses_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_1_post_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_1_post.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_1_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_1_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_2_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_2_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}

// Profiling information for each code object

//////////////random numbers//////////////////
curandGenerator_t brian::curand_generator;
__device__ unsigned long long* brian::d_curand_seed;
unsigned long long* brian::dev_curand_seed;
// dev_{co.name}_{rng_type}_allocator
//      pointer to start of generated random numbers array
//      at each generation cycle this array is refilled
// dev_{co.name}_{rng_type}
//      pointer moving through generated random number array
//      until it is regenerated at the next generation cycle
curandState* brian::dev_curand_states;
__device__ curandState* brian::d_curand_states;
RandomNumberBuffer brian::random_number_buffer;

void _init_arrays()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );

    num_parallel_blocks = props.multiProcessorCount * 1;
    printf("objects cu num par blocks %d\n", num_parallel_blocks);
    max_threads_per_block = props.maxThreadsPerBlock;
    max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    max_shared_mem_size = props.sharedMemPerBlock;
    num_threads_per_warp = props.warpSize;

    // Random seeds might be overwritten in main.cu
    unsigned long long seed = time(0);

    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_curand_seed,
                sizeof(unsigned long long))
            );

    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_curand_seed, &dev_curand_seed,
                sizeof(unsigned long long*))
            );

    CUDA_SAFE_CALL(
            curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT)
            );


    // this sets seed for host and device api RNG
    random_number_buffer.set_seed(seed);

    synapses_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_pre_init");
    synapses_1_post_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_1_post_init");
    synapses_1_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2500
            );
    CUDA_CHECK_ERROR("synapses_1_pre_init");
    synapses_2_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_2_pre_init");

    // Arrays initialized to 0
            _array_defaultclock_dt = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_dt, _array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_t = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_t, _array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_timestep = new int64_t[1];
            for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_timestep, _array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_g_eKC_eKC = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_g_eKC_eKC[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_g_eKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_eKC_eKC)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_g_eKC_eKC, _array_neurongroup_1_g_eKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_eKC_eKC, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_g_iKC_eKC = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_g_iKC_eKC[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_g_iKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_iKC_eKC)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_g_iKC_eKC, _array_neurongroup_1_g_iKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_iKC_eKC, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_h = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_h[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_h, sizeof(double)*_num__array_neurongroup_1_h)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_h, _array_neurongroup_1_h, sizeof(double)*_num__array_neurongroup_1_h, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_i = new int32_t[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_i, _array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_lastspike = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_lastspike[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_lastspike, sizeof(double)*_num__array_neurongroup_1_lastspike)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_lastspike, _array_neurongroup_1_lastspike, sizeof(double)*_num__array_neurongroup_1_lastspike, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_m = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_m[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_m, sizeof(double)*_num__array_neurongroup_1_m)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_m, _array_neurongroup_1_m, sizeof(double)*_num__array_neurongroup_1_m, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_n = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_n[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_n, sizeof(double)*_num__array_neurongroup_1_n)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_n, _array_neurongroup_1_n, sizeof(double)*_num__array_neurongroup_1_n, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_not_refractory = new char[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_not_refractory[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_not_refractory, sizeof(char)*_num__array_neurongroup_1_not_refractory)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_not_refractory, _array_neurongroup_1_not_refractory, sizeof(char)*_num__array_neurongroup_1_not_refractory, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_V = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_1_V[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_V, sizeof(double)*_num__array_neurongroup_1_V)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_V, _array_neurongroup_1_V, sizeof(double)*_num__array_neurongroup_1_V, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_g_PN_iKC = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_g_PN_iKC[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_g_PN_iKC, sizeof(double)*_num__array_neurongroup_g_PN_iKC)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_g_PN_iKC, _array_neurongroup_g_PN_iKC, sizeof(double)*_num__array_neurongroup_g_PN_iKC, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_h = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_h[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_h, sizeof(double)*_num__array_neurongroup_h)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_h, _array_neurongroup_h, sizeof(double)*_num__array_neurongroup_h, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_i = new int32_t[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_lastspike = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_lastspike[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_lastspike, _array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_m = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_m[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_m, sizeof(double)*_num__array_neurongroup_m)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_m, _array_neurongroup_m, sizeof(double)*_num__array_neurongroup_m, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_n = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_n[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_n, sizeof(double)*_num__array_neurongroup_n)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_n, _array_neurongroup_n, sizeof(double)*_num__array_neurongroup_n, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_not_refractory = new char[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_not_refractory[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_not_refractory, sizeof(char)*_num__array_neurongroup_not_refractory)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_not_refractory, _array_neurongroup_not_refractory, sizeof(char)*_num__array_neurongroup_not_refractory, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_V = new double[2500];
            for(int i=0; i<2500; i++) _array_neurongroup_V[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_V, sizeof(double)*_num__array_neurongroup_V)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_V, _array_neurongroup_V, sizeof(double)*_num__array_neurongroup_V, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup__lastindex = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikegeneratorgroup__lastindex[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup__lastindex, _array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup__period_bins = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikegeneratorgroup__period_bins[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup__period_bins, sizeof(int32_t)*_num__array_spikegeneratorgroup__period_bins)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup__period_bins, _array_spikegeneratorgroup__period_bins, sizeof(int32_t)*_num__array_spikegeneratorgroup__period_bins, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup_i = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikegeneratorgroup_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup_i, _array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup_period = new double[1];
            for(int i=0; i<1; i++) _array_spikegeneratorgroup_period[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup_period, _array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_1__source_idx = new int32_t[2500];
            for(int i=0; i<2500; i++) _array_spikemonitor_1__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_1__source_idx, _array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_1_count = new int32_t[2500];
            for(int i=0; i<2500; i++) _array_spikemonitor_1_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_1_count, sizeof(int32_t)*_num__array_spikemonitor_1_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_1_count, _array_spikemonitor_1_count, sizeof(int32_t)*_num__array_spikemonitor_1_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_1_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_1_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_1_N, sizeof(int32_t)*_num__array_spikemonitor_1_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_1_N, _array_spikemonitor_1_N, sizeof(int32_t)*_num__array_spikemonitor_1_N, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_2__source_idx = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikemonitor_2__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_2__source_idx, _array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_2_count = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikemonitor_2_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_2_count, sizeof(int32_t)*_num__array_spikemonitor_2_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_2_count, _array_spikemonitor_2_count, sizeof(int32_t)*_num__array_spikemonitor_2_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_2_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_2_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_2_N, sizeof(int32_t)*_num__array_spikemonitor_2_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_2_N, _array_spikemonitor_2_N, sizeof(int32_t)*_num__array_spikemonitor_2_N, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor__source_idx = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_count = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikemonitor_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_count, _array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_N, _array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_1_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_1_N, _array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_2_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_2_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_2_N, _array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_N, _array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyHostToDevice)
                    );
            _dynamic_array_spikegeneratorgroup__timebins.resize(19676);
            THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup__timebins.resize(19676));
            for(int i=0; i<19676; i++)
            {
                _dynamic_array_spikegeneratorgroup__timebins[i] = 0;
                dev_dynamic_array_spikegeneratorgroup__timebins[i] = 0;
            }
            _dynamic_array_synapses_1_delay.resize(1);
            THRUST_CHECK_ERROR(dev_dynamic_array_synapses_1_delay.resize(1));
            for(int i=0; i<1; i++)
            {
                _dynamic_array_synapses_1_delay[i] = 0;
                dev_dynamic_array_synapses_1_delay[i] = 0;
            }
            _dynamic_array_synapses_2_delay.resize(1);
            THRUST_CHECK_ERROR(dev_dynamic_array_synapses_2_delay.resize(1));
            for(int i=0; i<1; i++)
            {
                _dynamic_array_synapses_2_delay[i] = 0;
                dev_dynamic_array_synapses_2_delay[i] = 0;
            }

    // Arrays initialized to an "arange"
    _array_neurongroup_1_i = new int32_t[100];
    for(int i=0; i<100; i++) _array_neurongroup_1_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_1_i, _array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_i = new int32_t[2500];
    for(int i=0; i<2500; i++) _array_neurongroup_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
            );
    _array_spikegeneratorgroup_i = new int32_t[100];
    for(int i=0; i<100; i++) _array_spikegeneratorgroup_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikegeneratorgroup_i, _array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor_1__source_idx = new int32_t[2500];
    for(int i=0; i<2500; i++) _array_spikemonitor_1__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor_1__source_idx, _array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor_2__source_idx = new int32_t[100];
    for(int i=0; i<100; i++) _array_spikemonitor_2__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor_2__source_idx, _array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor__source_idx = new int32_t[100];
    for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice)
            );

    // static arrays
    _static_array__dynamic_array_spikegeneratorgroup__timebins = new int32_t[19676];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup__timebins, sizeof(int32_t)*19676)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup__timebins, &dev_static_array__dynamic_array_spikegeneratorgroup__timebins, sizeof(int32_t*))
            );
    _static_array__dynamic_array_spikegeneratorgroup_neuron_index = new int64_t[19676];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t)*19676)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_neuron_index, &dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t*))
            );
    _static_array__dynamic_array_spikegeneratorgroup_spike_number = new int64_t[19676];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int64_t)*19676)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_spike_number, &dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int64_t*))
            );
    _static_array__dynamic_array_spikegeneratorgroup_spike_time = new double[19676];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double)*19676)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_spike_time, &dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double*))
            );
    _timedarray_1_values = new double[250000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_1_values, sizeof(double)*250000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_1_values, &dev_timedarray_1_values, sizeof(double*))
            );
    _timedarray_2_values = new double[250000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_2_values, sizeof(double)*250000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_2_values, &dev_timedarray_2_values, sizeof(double*))
            );
    _timedarray_3_values = new double[250000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_3_values, sizeof(double)*250000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_3_values, &dev_timedarray_3_values, sizeof(double*))
            );
    _timedarray_4_values = new double[250000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_4_values, sizeof(double)*250000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_4_values, &dev_timedarray_4_values, sizeof(double*))
            );
    _timedarray_values = new double[250000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_values, sizeof(double)*250000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_values, &dev_timedarray_values, sizeof(double*))
            );


    // eventspace_arrays
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_1__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_1__spikespace)
            );
    _array_neurongroup_1__spikespace = new int32_t[101];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup__spikespace[0], sizeof(int32_t)*_num__array_neurongroup__spikespace)
            );
    _array_neurongroup__spikespace = new int32_t[2501];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikegeneratorgroup__spikespace[0], sizeof(int32_t)*_num__array_spikegeneratorgroup__spikespace)
            );
    _array_spikegeneratorgroup__spikespace = new int32_t[101];

    CUDA_CHECK_MEMORY();
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    double tot_memory_MB = (used_device_memory - used_device_memory_start) * to_MB;
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: _init_arrays() took " <<  time_passed << "s";
    if (tot_memory_MB > 0)
        std::cout << " and used " << tot_memory_MB << "MB of device memory.";
    std::cout << std::endl;
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__dynamic_array_spikegeneratorgroup__timebins;
    f_static_array__dynamic_array_spikegeneratorgroup__timebins.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup__timebins", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup__timebins.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup__timebins.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup__timebins), 19676*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup__timebins." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup__timebins, _static_array__dynamic_array_spikegeneratorgroup__timebins, sizeof(int32_t)*19676, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
    f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_neuron_index", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_neuron_index), 19676*sizeof(int64_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_neuron_index." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, _static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t)*19676, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_number;
    f_static_array__dynamic_array_spikegeneratorgroup_spike_number.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_number", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_spike_number.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_spike_number.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_number), 19676*sizeof(int64_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_number." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, _static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int64_t)*19676, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_time;
    f_static_array__dynamic_array_spikegeneratorgroup_spike_time.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_time", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_spike_time.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_spike_time.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_time), 19676*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_time." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, _static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double)*19676, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_1_values;
    f_timedarray_1_values.open("static_arrays/_timedarray_1_values", ios::in | ios::binary);
    if(f_timedarray_1_values.is_open())
    {
        f_timedarray_1_values.read(reinterpret_cast<char*>(_timedarray_1_values), 250000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_1_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_1_values, _timedarray_1_values, sizeof(double)*250000, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_2_values;
    f_timedarray_2_values.open("static_arrays/_timedarray_2_values", ios::in | ios::binary);
    if(f_timedarray_2_values.is_open())
    {
        f_timedarray_2_values.read(reinterpret_cast<char*>(_timedarray_2_values), 250000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_2_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_2_values, _timedarray_2_values, sizeof(double)*250000, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_3_values;
    f_timedarray_3_values.open("static_arrays/_timedarray_3_values", ios::in | ios::binary);
    if(f_timedarray_3_values.is_open())
    {
        f_timedarray_3_values.read(reinterpret_cast<char*>(_timedarray_3_values), 250000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_3_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_3_values, _timedarray_3_values, sizeof(double)*250000, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_4_values;
    f_timedarray_4_values.open("static_arrays/_timedarray_4_values", ios::in | ios::binary);
    if(f_timedarray_4_values.is_open())
    {
        f_timedarray_4_values.read(reinterpret_cast<char*>(_timedarray_4_values), 250000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_4_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_4_values, _timedarray_4_values, sizeof(double)*250000, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_values;
    f_timedarray_values.open("static_arrays/_timedarray_values", ios::in | ios::binary);
    if(f_timedarray_values.is_open())
    {
        f_timedarray_values.read(reinterpret_cast<char*>(_timedarray_values), 250000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_values, _timedarray_values, sizeof(double)*250000, cudaMemcpyHostToDevice)
            );
}

void _write_arrays()
{
    using namespace brian;

    CUDA_SAFE_CALL(
            cudaMemcpy(_array_defaultclock_dt, dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_-2498286751126143934", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(double));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_defaultclock_t, dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open("results/_array_defaultclock_t_-2737290110509227905", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(double));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_defaultclock_timestep, dev_array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_-8079704882989719448", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(int64_t));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_g_eKC_eKC, dev_array_neurongroup_1_g_eKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_eKC_eKC, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_g_eKC_eKC;
    outfile__array_neurongroup_1_g_eKC_eKC.open("results/_array_neurongroup_1_g_eKC_eKC_-5583994226418931441", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_g_eKC_eKC.is_open())
    {
        outfile__array_neurongroup_1_g_eKC_eKC.write(reinterpret_cast<char*>(_array_neurongroup_1_g_eKC_eKC), 100*sizeof(double));
        outfile__array_neurongroup_1_g_eKC_eKC.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_g_eKC_eKC." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_g_iKC_eKC, dev_array_neurongroup_1_g_iKC_eKC, sizeof(double)*_num__array_neurongroup_1_g_iKC_eKC, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_g_iKC_eKC;
    outfile__array_neurongroup_1_g_iKC_eKC.open("results/_array_neurongroup_1_g_iKC_eKC_698381179621967883", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_g_iKC_eKC.is_open())
    {
        outfile__array_neurongroup_1_g_iKC_eKC.write(reinterpret_cast<char*>(_array_neurongroup_1_g_iKC_eKC), 100*sizeof(double));
        outfile__array_neurongroup_1_g_iKC_eKC.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_g_iKC_eKC." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_h, dev_array_neurongroup_1_h, sizeof(double)*_num__array_neurongroup_1_h, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_h;
    outfile__array_neurongroup_1_h.open("results/_array_neurongroup_1_h_5344468206502301276", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_h.is_open())
    {
        outfile__array_neurongroup_1_h.write(reinterpret_cast<char*>(_array_neurongroup_1_h), 100*sizeof(double));
        outfile__array_neurongroup_1_h.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_h." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_i, dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_i;
    outfile__array_neurongroup_1_i.open("results/_array_neurongroup_1_i_-3789611295489125583", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_i.is_open())
    {
        outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 100*sizeof(int32_t));
        outfile__array_neurongroup_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_lastspike, dev_array_neurongroup_1_lastspike, sizeof(double)*_num__array_neurongroup_1_lastspike, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_lastspike;
    outfile__array_neurongroup_1_lastspike.open("results/_array_neurongroup_1_lastspike_7243448513092147373", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_lastspike.is_open())
    {
        outfile__array_neurongroup_1_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_1_lastspike), 100*sizeof(double));
        outfile__array_neurongroup_1_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_lastspike." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_m, dev_array_neurongroup_1_m, sizeof(double)*_num__array_neurongroup_1_m, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_m;
    outfile__array_neurongroup_1_m.open("results/_array_neurongroup_1_m_-2120397286057579911", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_m.is_open())
    {
        outfile__array_neurongroup_1_m.write(reinterpret_cast<char*>(_array_neurongroup_1_m), 100*sizeof(double));
        outfile__array_neurongroup_1_m.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_m." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_n, dev_array_neurongroup_1_n, sizeof(double)*_num__array_neurongroup_1_n, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_n;
    outfile__array_neurongroup_1_n.open("results/_array_neurongroup_1_n_-8428575863822087678", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_n.is_open())
    {
        outfile__array_neurongroup_1_n.write(reinterpret_cast<char*>(_array_neurongroup_1_n), 100*sizeof(double));
        outfile__array_neurongroup_1_n.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_n." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_not_refractory, dev_array_neurongroup_1_not_refractory, sizeof(char)*_num__array_neurongroup_1_not_refractory, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_not_refractory;
    outfile__array_neurongroup_1_not_refractory.open("results/_array_neurongroup_1_not_refractory_7368107638204237228", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_not_refractory.is_open())
    {
        outfile__array_neurongroup_1_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_1_not_refractory), 100*sizeof(char));
        outfile__array_neurongroup_1_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_not_refractory." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_V, dev_array_neurongroup_1_V, sizeof(double)*_num__array_neurongroup_1_V, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_V;
    outfile__array_neurongroup_1_V.open("results/_array_neurongroup_1_V_-6137459464929975667", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_V.is_open())
    {
        outfile__array_neurongroup_1_V.write(reinterpret_cast<char*>(_array_neurongroup_1_V), 100*sizeof(double));
        outfile__array_neurongroup_1_V.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_V." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_g_PN_iKC, dev_array_neurongroup_g_PN_iKC, sizeof(double)*_num__array_neurongroup_g_PN_iKC, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_g_PN_iKC;
    outfile__array_neurongroup_g_PN_iKC.open("results/_array_neurongroup_g_PN_iKC_1280755048330296697", ios::binary | ios::out);
    if(outfile__array_neurongroup_g_PN_iKC.is_open())
    {
        outfile__array_neurongroup_g_PN_iKC.write(reinterpret_cast<char*>(_array_neurongroup_g_PN_iKC), 2500*sizeof(double));
        outfile__array_neurongroup_g_PN_iKC.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_g_PN_iKC." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_h, dev_array_neurongroup_h, sizeof(double)*_num__array_neurongroup_h, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_h;
    outfile__array_neurongroup_h.open("results/_array_neurongroup_h_-3092669250522089359", ios::binary | ios::out);
    if(outfile__array_neurongroup_h.is_open())
    {
        outfile__array_neurongroup_h.write(reinterpret_cast<char*>(_array_neurongroup_h), 2500*sizeof(double));
        outfile__array_neurongroup_h.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_h." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_i, dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_i;
    outfile__array_neurongroup_i.open("results/_array_neurongroup_i_-5243747817659055085", ios::binary | ios::out);
    if(outfile__array_neurongroup_i.is_open())
    {
        outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 2500*sizeof(int32_t));
        outfile__array_neurongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_lastspike, dev_array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_lastspike;
    outfile__array_neurongroup_lastspike.open("results/_array_neurongroup_lastspike_5887856098539198058", ios::binary | ios::out);
    if(outfile__array_neurongroup_lastspike.is_open())
    {
        outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), 2500*sizeof(double));
        outfile__array_neurongroup_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_m, dev_array_neurongroup_m, sizeof(double)*_num__array_neurongroup_m, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_m;
    outfile__array_neurongroup_m.open("results/_array_neurongroup_m_-1318907606515658718", ios::binary | ios::out);
    if(outfile__array_neurongroup_m.is_open())
    {
        outfile__array_neurongroup_m.write(reinterpret_cast<char*>(_array_neurongroup_m), 2500*sizeof(double));
        outfile__array_neurongroup_m.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_m." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_n, dev_array_neurongroup_n, sizeof(double)*_num__array_neurongroup_n, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_n;
    outfile__array_neurongroup_n.open("results/_array_neurongroup_n_-8712203438137663296", ios::binary | ios::out);
    if(outfile__array_neurongroup_n.is_open())
    {
        outfile__array_neurongroup_n.write(reinterpret_cast<char*>(_array_neurongroup_n), 2500*sizeof(double));
        outfile__array_neurongroup_n.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_n." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_not_refractory, dev_array_neurongroup_not_refractory, sizeof(char)*_num__array_neurongroup_not_refractory, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_not_refractory;
    outfile__array_neurongroup_not_refractory.open("results/_array_neurongroup_not_refractory_2056821189576763631", ios::binary | ios::out);
    if(outfile__array_neurongroup_not_refractory.is_open())
    {
        outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), 2500*sizeof(char));
        outfile__array_neurongroup_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_V, dev_array_neurongroup_V, sizeof(double)*_num__array_neurongroup_V, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_V;
    outfile__array_neurongroup_V.open("results/_array_neurongroup_V_1012544495117118507", ios::binary | ios::out);
    if(outfile__array_neurongroup_V.is_open())
    {
        outfile__array_neurongroup_V.write(reinterpret_cast<char*>(_array_neurongroup_V), 2500*sizeof(double));
        outfile__array_neurongroup_V.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_V." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup__lastindex, dev_array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup__lastindex;
    outfile__array_spikegeneratorgroup__lastindex.open("results/_array_spikegeneratorgroup__lastindex_-4562001451387750606", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup__lastindex.is_open())
    {
        outfile__array_spikegeneratorgroup__lastindex.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__lastindex), 1*sizeof(int32_t));
        outfile__array_spikegeneratorgroup__lastindex.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup__lastindex." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup__period_bins, dev_array_spikegeneratorgroup__period_bins, sizeof(int32_t)*_num__array_spikegeneratorgroup__period_bins, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup__period_bins;
    outfile__array_spikegeneratorgroup__period_bins.open("results/_array_spikegeneratorgroup__period_bins_3814106930303245699", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup__period_bins.is_open())
    {
        outfile__array_spikegeneratorgroup__period_bins.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__period_bins), 1*sizeof(int32_t));
        outfile__array_spikegeneratorgroup__period_bins.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup__period_bins." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup_i, dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup_i;
    outfile__array_spikegeneratorgroup_i.open("results/_array_spikegeneratorgroup_i_-397239631630174056", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup_i.is_open())
    {
        outfile__array_spikegeneratorgroup_i.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_i), 100*sizeof(int32_t));
        outfile__array_spikegeneratorgroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup_period, dev_array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup_period;
    outfile__array_spikegeneratorgroup_period.open("results/_array_spikegeneratorgroup_period_7725887948559186023", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup_period.is_open())
    {
        outfile__array_spikegeneratorgroup_period.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_period), 1*sizeof(double));
        outfile__array_spikegeneratorgroup_period.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup_period." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_1__source_idx, dev_array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_1__source_idx;
    outfile__array_spikemonitor_1__source_idx.open("results/_array_spikemonitor_1__source_idx_1057894290895839432", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1__source_idx.is_open())
    {
        outfile__array_spikemonitor_1__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_1__source_idx), 2500*sizeof(int32_t));
        outfile__array_spikemonitor_1__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_1_count, dev_array_spikemonitor_1_count, sizeof(int32_t)*_num__array_spikemonitor_1_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_1_count;
    outfile__array_spikemonitor_1_count.open("results/_array_spikemonitor_1_count_-9137222686626857787", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1_count.is_open())
    {
        outfile__array_spikemonitor_1_count.write(reinterpret_cast<char*>(_array_spikemonitor_1_count), 2500*sizeof(int32_t));
        outfile__array_spikemonitor_1_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1_count." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_1_N, dev_array_spikemonitor_1_N, sizeof(int32_t)*_num__array_spikemonitor_1_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_1_N;
    outfile__array_spikemonitor_1_N.open("results/_array_spikemonitor_1_N_7812431033937921391", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1_N.is_open())
    {
        outfile__array_spikemonitor_1_N.write(reinterpret_cast<char*>(_array_spikemonitor_1_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_2__source_idx, dev_array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_2__source_idx;
    outfile__array_spikemonitor_2__source_idx.open("results/_array_spikemonitor_2__source_idx_1362393325657685696", ios::binary | ios::out);
    if(outfile__array_spikemonitor_2__source_idx.is_open())
    {
        outfile__array_spikemonitor_2__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_2__source_idx), 100*sizeof(int32_t));
        outfile__array_spikemonitor_2__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_2__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_2_count, dev_array_spikemonitor_2_count, sizeof(int32_t)*_num__array_spikemonitor_2_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_2_count;
    outfile__array_spikemonitor_2_count.open("results/_array_spikemonitor_2_count_7626239611073350978", ios::binary | ios::out);
    if(outfile__array_spikemonitor_2_count.is_open())
    {
        outfile__array_spikemonitor_2_count.write(reinterpret_cast<char*>(_array_spikemonitor_2_count), 100*sizeof(int32_t));
        outfile__array_spikemonitor_2_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_2_count." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_2_N, dev_array_spikemonitor_2_N, sizeof(int32_t)*_num__array_spikemonitor_2_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_2_N;
    outfile__array_spikemonitor_2_N.open("results/_array_spikemonitor_2_N_4348563937531214317", ios::binary | ios::out);
    if(outfile__array_spikemonitor_2_N.is_open())
    {
        outfile__array_spikemonitor_2_N.write(reinterpret_cast<char*>(_array_spikemonitor_2_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_2_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor__source_idx, dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor__source_idx;
    outfile__array_spikemonitor__source_idx.open("results/_array_spikemonitor__source_idx_4492520236303501020", ios::binary | ios::out);
    if(outfile__array_spikemonitor__source_idx.is_open())
    {
        outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 100*sizeof(int32_t));
        outfile__array_spikemonitor__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_count, dev_array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_count;
    outfile__array_spikemonitor_count.open("results/_array_spikemonitor_count_-8939381533980173332", ios::binary | ios::out);
    if(outfile__array_spikemonitor_count.is_open())
    {
        outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 100*sizeof(int32_t));
        outfile__array_spikemonitor_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_N, dev_array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_N;
    outfile__array_spikemonitor_N.open("results/_array_spikemonitor_N_-4176126692401210049", ios::binary | ios::out);
    if(outfile__array_spikemonitor_N.is_open())
    {
        outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_1_N, dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_1_N;
    outfile__array_synapses_1_N.open("results/_array_synapses_1_N_293610487777644577", ios::binary | ios::out);
    if(outfile__array_synapses_1_N.is_open())
    {
        outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(int32_t));
        outfile__array_synapses_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_2_N, dev_array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_2_N;
    outfile__array_synapses_2_N.open("results/_array_synapses_2_N_6367731152988484261", ios::binary | ios::out);
    if(outfile__array_synapses_2_N.is_open())
    {
        outfile__array_synapses_2_N.write(reinterpret_cast<char*>(_array_synapses_2_N), 1*sizeof(int32_t));
        outfile__array_synapses_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_2_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_N, dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_N;
    outfile__array_synapses_N.open("results/_array_synapses_N_-4267478581729905340", ios::binary | ios::out);
    if(outfile__array_synapses_N.is_open())
    {
        outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(int32_t));
        outfile__array_synapses_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_N." << endl;
    }

    _dynamic_array_spikegeneratorgroup__timebins = dev_dynamic_array_spikegeneratorgroup__timebins;
    ofstream outfile__dynamic_array_spikegeneratorgroup__timebins;
    outfile__dynamic_array_spikegeneratorgroup__timebins.open("results/_dynamic_array_spikegeneratorgroup__timebins_5801608633011989326", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup__timebins.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup__timebins.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup__timebins[0])), _dynamic_array_spikegeneratorgroup__timebins.size()*sizeof(int32_t));
        outfile__dynamic_array_spikegeneratorgroup__timebins.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup__timebins." << endl;
    }
    _dynamic_array_spikegeneratorgroup_neuron_index = dev_dynamic_array_spikegeneratorgroup_neuron_index;
    ofstream outfile__dynamic_array_spikegeneratorgroup_neuron_index;
    outfile__dynamic_array_spikegeneratorgroup_neuron_index.open("results/_dynamic_array_spikegeneratorgroup_neuron_index_3327538713134410383", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup_neuron_index.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup_neuron_index[0])), _dynamic_array_spikegeneratorgroup_neuron_index.size()*sizeof(int32_t));
        outfile__dynamic_array_spikegeneratorgroup_neuron_index.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_neuron_index." << endl;
    }
    _dynamic_array_spikegeneratorgroup_spike_number = dev_dynamic_array_spikegeneratorgroup_spike_number;
    ofstream outfile__dynamic_array_spikegeneratorgroup_spike_number;
    outfile__dynamic_array_spikegeneratorgroup_spike_number.open("results/_dynamic_array_spikegeneratorgroup_spike_number_-4677374471276286859", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_spike_number.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup_spike_number.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup_spike_number[0])), _dynamic_array_spikegeneratorgroup_spike_number.size()*sizeof(int32_t));
        outfile__dynamic_array_spikegeneratorgroup_spike_number.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_number." << endl;
    }
    _dynamic_array_spikegeneratorgroup_spike_time = dev_dynamic_array_spikegeneratorgroup_spike_time;
    ofstream outfile__dynamic_array_spikegeneratorgroup_spike_time;
    outfile__dynamic_array_spikegeneratorgroup_spike_time.open("results/_dynamic_array_spikegeneratorgroup_spike_time_5281423122437343888", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_spike_time.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup_spike_time.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup_spike_time[0])), _dynamic_array_spikegeneratorgroup_spike_time.size()*sizeof(double));
        outfile__dynamic_array_spikegeneratorgroup_spike_time.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_time." << endl;
    }
    _dynamic_array_spikemonitor_1_i = dev_dynamic_array_spikemonitor_1_i;
    ofstream outfile__dynamic_array_spikemonitor_1_i;
    outfile__dynamic_array_spikemonitor_1_i.open("results/_dynamic_array_spikemonitor_1_i_7158638244621399231", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_1_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_1_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_1_i[0])), _dynamic_array_spikemonitor_1_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_i." << endl;
    }
    _dynamic_array_spikemonitor_1_t = dev_dynamic_array_spikemonitor_1_t;
    ofstream outfile__dynamic_array_spikemonitor_1_t;
    outfile__dynamic_array_spikemonitor_1_t.open("results/_dynamic_array_spikemonitor_1_t_6409207889459703585", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_1_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_1_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_1_t[0])), _dynamic_array_spikemonitor_1_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_1_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_t." << endl;
    }
    _dynamic_array_spikemonitor_2_i = dev_dynamic_array_spikemonitor_2_i;
    ofstream outfile__dynamic_array_spikemonitor_2_i;
    outfile__dynamic_array_spikemonitor_2_i.open("results/_dynamic_array_spikemonitor_2_i_-2319234593350803970", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_2_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_2_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_2_i[0])), _dynamic_array_spikemonitor_2_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_2_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_2_i." << endl;
    }
    _dynamic_array_spikemonitor_2_t = dev_dynamic_array_spikemonitor_2_t;
    ofstream outfile__dynamic_array_spikemonitor_2_t;
    outfile__dynamic_array_spikemonitor_2_t.open("results/_dynamic_array_spikemonitor_2_t_-7213119999462114063", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_2_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_2_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_2_t[0])), _dynamic_array_spikemonitor_2_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_2_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_2_t." << endl;
    }
    _dynamic_array_spikemonitor_i = dev_dynamic_array_spikemonitor_i;
    ofstream outfile__dynamic_array_spikemonitor_i;
    outfile__dynamic_array_spikemonitor_i.open("results/_dynamic_array_spikemonitor_i_7932276293891085154", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_i[0])), _dynamic_array_spikemonitor_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
    }
    _dynamic_array_spikemonitor_t = dev_dynamic_array_spikemonitor_t;
    ofstream outfile__dynamic_array_spikemonitor_t;
    outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t_-2690765867727077152", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_t[0])), _dynamic_array_spikemonitor_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_post;
    outfile__dynamic_array_synapses_1__synaptic_post.open("results/_dynamic_array_synapses_1__synaptic_post_3887552400782846830", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1__synaptic_post[0])), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
    outfile__dynamic_array_synapses_1__synaptic_pre.open("results/_dynamic_array_synapses_1__synaptic_pre_-3465909282819894897", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1__synaptic_pre[0])), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
    }
    _dynamic_array_synapses_1_Apost = dev_dynamic_array_synapses_1_Apost;
    ofstream outfile__dynamic_array_synapses_1_Apost;
    outfile__dynamic_array_synapses_1_Apost.open("results/_dynamic_array_synapses_1_Apost_-2802562643289696327", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_Apost.is_open())
    {
        outfile__dynamic_array_synapses_1_Apost.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_Apost[0])), _dynamic_array_synapses_1_Apost.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_Apost.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_Apost." << endl;
    }
    _dynamic_array_synapses_1_Apre = dev_dynamic_array_synapses_1_Apre;
    ofstream outfile__dynamic_array_synapses_1_Apre;
    outfile__dynamic_array_synapses_1_Apre.open("results/_dynamic_array_synapses_1_Apre_-4131744976984627475", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_Apre.is_open())
    {
        outfile__dynamic_array_synapses_1_Apre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_Apre[0])), _dynamic_array_synapses_1_Apre.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_Apre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_Apre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay;
    outfile__dynamic_array_synapses_1_delay.open("results/_dynamic_array_synapses_1_delay_-2231186514412628865", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay.is_open())
    {
        outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_delay[0])), _dynamic_array_synapses_1_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay_1;
    outfile__dynamic_array_synapses_1_delay_1.open("results/_dynamic_array_synapses_1_delay_1_-7861599511511956763", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay_1.is_open())
    {
        outfile__dynamic_array_synapses_1_delay_1.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_delay_1[0])), _dynamic_array_synapses_1_delay_1.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_delay_1.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay_1." << endl;
    }
    _dynamic_array_synapses_1_g_raw = dev_dynamic_array_synapses_1_g_raw;
    ofstream outfile__dynamic_array_synapses_1_g_raw;
    outfile__dynamic_array_synapses_1_g_raw.open("results/_dynamic_array_synapses_1_g_raw_-8690563007348565991", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_g_raw.is_open())
    {
        outfile__dynamic_array_synapses_1_g_raw.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_g_raw[0])), _dynamic_array_synapses_1_g_raw.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_g_raw.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_g_raw." << endl;
    }
    _dynamic_array_synapses_1_lastupdate = dev_dynamic_array_synapses_1_lastupdate;
    ofstream outfile__dynamic_array_synapses_1_lastupdate;
    outfile__dynamic_array_synapses_1_lastupdate.open("results/_dynamic_array_synapses_1_lastupdate_1417193071217998245", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_lastupdate.is_open())
    {
        outfile__dynamic_array_synapses_1_lastupdate.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_lastupdate[0])), _dynamic_array_synapses_1_lastupdate.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_lastupdate.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_lastupdate." << endl;
    }
    _dynamic_array_synapses_1_N_incoming = dev_dynamic_array_synapses_1_N_incoming;
    ofstream outfile__dynamic_array_synapses_1_N_incoming;
    outfile__dynamic_array_synapses_1_N_incoming.open("results/_dynamic_array_synapses_1_N_incoming_-4730020938361073333", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_N_incoming[0])), _dynamic_array_synapses_1_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
    }
    _dynamic_array_synapses_1_N_outgoing = dev_dynamic_array_synapses_1_N_outgoing;
    ofstream outfile__dynamic_array_synapses_1_N_outgoing;
    outfile__dynamic_array_synapses_1_N_outgoing.open("results/_dynamic_array_synapses_1_N_outgoing_5847266960107717791", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_N_outgoing[0])), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_post;
    outfile__dynamic_array_synapses_2__synaptic_post.open("results/_dynamic_array_synapses_2__synaptic_post_5774352615214944728", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_2__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_post[0])), _dynamic_array_synapses_2__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_pre;
    outfile__dynamic_array_synapses_2__synaptic_pre.open("results/_dynamic_array_synapses_2__synaptic_pre_-9012074162976892665", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_2__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_pre[0])), _dynamic_array_synapses_2__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_delay;
    outfile__dynamic_array_synapses_2_delay.open("results/_dynamic_array_synapses_2_delay_1607835271570489182", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_delay.is_open())
    {
        outfile__dynamic_array_synapses_2_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_delay[0])), _dynamic_array_synapses_2_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_2_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_delay." << endl;
    }
    _dynamic_array_synapses_2_N_incoming = dev_dynamic_array_synapses_2_N_incoming;
    ofstream outfile__dynamic_array_synapses_2_N_incoming;
    outfile__dynamic_array_synapses_2_N_incoming.open("results/_dynamic_array_synapses_2_N_incoming_8882135263890869691", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_2_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_incoming[0])), _dynamic_array_synapses_2_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_incoming." << endl;
    }
    _dynamic_array_synapses_2_N_outgoing = dev_dynamic_array_synapses_2_N_outgoing;
    ofstream outfile__dynamic_array_synapses_2_N_outgoing;
    outfile__dynamic_array_synapses_2_N_outgoing.open("results/_dynamic_array_synapses_2_N_outgoing_-7531657551927571654", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_2_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_outgoing[0])), _dynamic_array_synapses_2_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_post;
    outfile__dynamic_array_synapses__synaptic_post.open("results/_dynamic_array_synapses__synaptic_post_-5399118445813332823", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0])), _dynamic_array_synapses__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_pre;
    outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre_4910052875650969924", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0])), _dynamic_array_synapses__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay;
    outfile__dynamic_array_synapses_delay.open("results/_dynamic_array_synapses_delay_-7446265175536355663", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay.is_open())
    {
        outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_delay[0])), _dynamic_array_synapses_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
    }
    _dynamic_array_synapses_N_incoming = dev_dynamic_array_synapses_N_incoming;
    ofstream outfile__dynamic_array_synapses_N_incoming;
    outfile__dynamic_array_synapses_N_incoming.open("results/_dynamic_array_synapses_N_incoming_2716837380347339885", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_incoming[0])), _dynamic_array_synapses_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
    }
    _dynamic_array_synapses_N_outgoing = dev_dynamic_array_synapses_N_outgoing;
    ofstream outfile__dynamic_array_synapses_N_outgoing;
    outfile__dynamic_array_synapses_N_outgoing.open("results/_dynamic_array_synapses_N_outgoing_-946603928985537374", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_outgoing[0])), _dynamic_array_synapses_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
    }
    _dynamic_array_synapses_weight = dev_dynamic_array_synapses_weight;
    ofstream outfile__dynamic_array_synapses_weight;
    outfile__dynamic_array_synapses_weight.open("results/_dynamic_array_synapses_weight_5741023513072131750", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_weight.is_open())
    {
        outfile__dynamic_array_synapses_weight.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_weight[0])), _dynamic_array_synapses_weight.size()*sizeof(double));
        outfile__dynamic_array_synapses_weight.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_weight." << endl;
    }


    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open("results/last_run_info.txt", ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        std::cout << "Error writing last run info to file." << std::endl;
    }
}

__global__ void synapses_pre_destroy()
{
    using namespace brian;

    synapses_pre.destroy();
}
__global__ void synapses_1_post_destroy()
{
    using namespace brian;

    synapses_1_post.destroy();
}
__global__ void synapses_1_pre_destroy()
{
    using namespace brian;

    synapses_1_pre.destroy();
}
__global__ void synapses_2_pre_destroy()
{
    using namespace brian;

    synapses_2_pre.destroy();
}

void _dealloc_arrays()
{
    using namespace brian;


    CUDA_SAFE_CALL(
            curandDestroyGenerator(curand_generator)
            );

    synapses_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_pre_destroy");
    synapses_1_post_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_1_post_destroy");
    synapses_1_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_1_pre_destroy");
    synapses_2_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_2_pre_destroy");

    dev_dynamic_array_spikegeneratorgroup__timebins.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup__timebins);
    _dynamic_array_spikegeneratorgroup__timebins.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikegeneratorgroup__timebins);
    dev_dynamic_array_spikegeneratorgroup_neuron_index.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup_neuron_index);
    _dynamic_array_spikegeneratorgroup_neuron_index.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikegeneratorgroup_neuron_index);
    dev_dynamic_array_spikegeneratorgroup_spike_number.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup_spike_number);
    _dynamic_array_spikegeneratorgroup_spike_number.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikegeneratorgroup_spike_number);
    dev_dynamic_array_spikegeneratorgroup_spike_time.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikegeneratorgroup_spike_time);
    _dynamic_array_spikegeneratorgroup_spike_time.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikegeneratorgroup_spike_time);
    dev_dynamic_array_spikemonitor_1_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_1_i);
    _dynamic_array_spikemonitor_1_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_1_i);
    dev_dynamic_array_spikemonitor_1_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_1_t);
    _dynamic_array_spikemonitor_1_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_1_t);
    dev_dynamic_array_spikemonitor_2_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_2_i);
    _dynamic_array_spikemonitor_2_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_2_i);
    dev_dynamic_array_spikemonitor_2_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_2_t);
    _dynamic_array_spikemonitor_2_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_2_t);
    dev_dynamic_array_spikemonitor_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_i);
    _dynamic_array_spikemonitor_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_i);
    dev_dynamic_array_spikemonitor_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_t);
    _dynamic_array_spikemonitor_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_t);
    dev_dynamic_array_synapses_1__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_post);
    _dynamic_array_synapses_1__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1__synaptic_post);
    dev_dynamic_array_synapses_1__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_pre);
    _dynamic_array_synapses_1__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1__synaptic_pre);
    dev_dynamic_array_synapses_1_Apost.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_Apost);
    _dynamic_array_synapses_1_Apost.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_Apost);
    dev_dynamic_array_synapses_1_Apre.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_Apre);
    _dynamic_array_synapses_1_Apre.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_Apre);
    dev_dynamic_array_synapses_1_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_delay);
    _dynamic_array_synapses_1_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_delay);
    dev_dynamic_array_synapses_1_delay_1.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_delay_1);
    _dynamic_array_synapses_1_delay_1.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_delay_1);
    dev_dynamic_array_synapses_1_g_raw.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_g_raw);
    _dynamic_array_synapses_1_g_raw.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_g_raw);
    dev_dynamic_array_synapses_1_lastupdate.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_lastupdate);
    _dynamic_array_synapses_1_lastupdate.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_lastupdate);
    dev_dynamic_array_synapses_1_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_incoming);
    _dynamic_array_synapses_1_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1_N_incoming);
    dev_dynamic_array_synapses_1_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_outgoing);
    _dynamic_array_synapses_1_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1_N_outgoing);
    dev_dynamic_array_synapses_2__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2__synaptic_post);
    _dynamic_array_synapses_2__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2__synaptic_post);
    dev_dynamic_array_synapses_2__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2__synaptic_pre);
    _dynamic_array_synapses_2__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2__synaptic_pre);
    dev_dynamic_array_synapses_2_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_2_delay);
    _dynamic_array_synapses_2_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_2_delay);
    dev_dynamic_array_synapses_2_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2_N_incoming);
    _dynamic_array_synapses_2_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2_N_incoming);
    dev_dynamic_array_synapses_2_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2_N_outgoing);
    _dynamic_array_synapses_2_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2_N_outgoing);
    dev_dynamic_array_synapses__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_post);
    _dynamic_array_synapses__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_post);
    dev_dynamic_array_synapses__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_pre);
    _dynamic_array_synapses__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_pre);
    dev_dynamic_array_synapses_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_delay);
    _dynamic_array_synapses_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_delay);
    dev_dynamic_array_synapses_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_incoming);
    _dynamic_array_synapses_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_incoming);
    dev_dynamic_array_synapses_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_outgoing);
    _dynamic_array_synapses_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_outgoing);
    dev_dynamic_array_synapses_weight.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_weight);
    _dynamic_array_synapses_weight.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_weight);

    if(_array_defaultclock_dt!=0)
    {
        delete [] _array_defaultclock_dt;
        _array_defaultclock_dt = 0;
    }
    if(dev_array_defaultclock_dt!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_dt)
                );
        dev_array_defaultclock_dt = 0;
    }
    if(_array_defaultclock_t!=0)
    {
        delete [] _array_defaultclock_t;
        _array_defaultclock_t = 0;
    }
    if(dev_array_defaultclock_t!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_t)
                );
        dev_array_defaultclock_t = 0;
    }
    if(_array_defaultclock_timestep!=0)
    {
        delete [] _array_defaultclock_timestep;
        _array_defaultclock_timestep = 0;
    }
    if(dev_array_defaultclock_timestep!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_timestep)
                );
        dev_array_defaultclock_timestep = 0;
    }
    if(_array_neurongroup_1_g_eKC_eKC!=0)
    {
        delete [] _array_neurongroup_1_g_eKC_eKC;
        _array_neurongroup_1_g_eKC_eKC = 0;
    }
    if(dev_array_neurongroup_1_g_eKC_eKC!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_g_eKC_eKC)
                );
        dev_array_neurongroup_1_g_eKC_eKC = 0;
    }
    if(_array_neurongroup_1_g_iKC_eKC!=0)
    {
        delete [] _array_neurongroup_1_g_iKC_eKC;
        _array_neurongroup_1_g_iKC_eKC = 0;
    }
    if(dev_array_neurongroup_1_g_iKC_eKC!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_g_iKC_eKC)
                );
        dev_array_neurongroup_1_g_iKC_eKC = 0;
    }
    if(_array_neurongroup_1_h!=0)
    {
        delete [] _array_neurongroup_1_h;
        _array_neurongroup_1_h = 0;
    }
    if(dev_array_neurongroup_1_h!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_h)
                );
        dev_array_neurongroup_1_h = 0;
    }
    if(_array_neurongroup_1_i!=0)
    {
        delete [] _array_neurongroup_1_i;
        _array_neurongroup_1_i = 0;
    }
    if(dev_array_neurongroup_1_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_i)
                );
        dev_array_neurongroup_1_i = 0;
    }
    if(_array_neurongroup_1_lastspike!=0)
    {
        delete [] _array_neurongroup_1_lastspike;
        _array_neurongroup_1_lastspike = 0;
    }
    if(dev_array_neurongroup_1_lastspike!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_lastspike)
                );
        dev_array_neurongroup_1_lastspike = 0;
    }
    if(_array_neurongroup_1_m!=0)
    {
        delete [] _array_neurongroup_1_m;
        _array_neurongroup_1_m = 0;
    }
    if(dev_array_neurongroup_1_m!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_m)
                );
        dev_array_neurongroup_1_m = 0;
    }
    if(_array_neurongroup_1_n!=0)
    {
        delete [] _array_neurongroup_1_n;
        _array_neurongroup_1_n = 0;
    }
    if(dev_array_neurongroup_1_n!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_n)
                );
        dev_array_neurongroup_1_n = 0;
    }
    if(_array_neurongroup_1_not_refractory!=0)
    {
        delete [] _array_neurongroup_1_not_refractory;
        _array_neurongroup_1_not_refractory = 0;
    }
    if(dev_array_neurongroup_1_not_refractory!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_not_refractory)
                );
        dev_array_neurongroup_1_not_refractory = 0;
    }
    if(_array_neurongroup_1_V!=0)
    {
        delete [] _array_neurongroup_1_V;
        _array_neurongroup_1_V = 0;
    }
    if(dev_array_neurongroup_1_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_V)
                );
        dev_array_neurongroup_1_V = 0;
    }
    if(_array_neurongroup_g_PN_iKC!=0)
    {
        delete [] _array_neurongroup_g_PN_iKC;
        _array_neurongroup_g_PN_iKC = 0;
    }
    if(dev_array_neurongroup_g_PN_iKC!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_g_PN_iKC)
                );
        dev_array_neurongroup_g_PN_iKC = 0;
    }
    if(_array_neurongroup_h!=0)
    {
        delete [] _array_neurongroup_h;
        _array_neurongroup_h = 0;
    }
    if(dev_array_neurongroup_h!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_h)
                );
        dev_array_neurongroup_h = 0;
    }
    if(_array_neurongroup_i!=0)
    {
        delete [] _array_neurongroup_i;
        _array_neurongroup_i = 0;
    }
    if(dev_array_neurongroup_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_i)
                );
        dev_array_neurongroup_i = 0;
    }
    if(_array_neurongroup_lastspike!=0)
    {
        delete [] _array_neurongroup_lastspike;
        _array_neurongroup_lastspike = 0;
    }
    if(dev_array_neurongroup_lastspike!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_lastspike)
                );
        dev_array_neurongroup_lastspike = 0;
    }
    if(_array_neurongroup_m!=0)
    {
        delete [] _array_neurongroup_m;
        _array_neurongroup_m = 0;
    }
    if(dev_array_neurongroup_m!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_m)
                );
        dev_array_neurongroup_m = 0;
    }
    if(_array_neurongroup_n!=0)
    {
        delete [] _array_neurongroup_n;
        _array_neurongroup_n = 0;
    }
    if(dev_array_neurongroup_n!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_n)
                );
        dev_array_neurongroup_n = 0;
    }
    if(_array_neurongroup_not_refractory!=0)
    {
        delete [] _array_neurongroup_not_refractory;
        _array_neurongroup_not_refractory = 0;
    }
    if(dev_array_neurongroup_not_refractory!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_not_refractory)
                );
        dev_array_neurongroup_not_refractory = 0;
    }
    if(_array_neurongroup_V!=0)
    {
        delete [] _array_neurongroup_V;
        _array_neurongroup_V = 0;
    }
    if(dev_array_neurongroup_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_V)
                );
        dev_array_neurongroup_V = 0;
    }
    if(_array_spikegeneratorgroup__lastindex!=0)
    {
        delete [] _array_spikegeneratorgroup__lastindex;
        _array_spikegeneratorgroup__lastindex = 0;
    }
    if(dev_array_spikegeneratorgroup__lastindex!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup__lastindex)
                );
        dev_array_spikegeneratorgroup__lastindex = 0;
    }
    if(_array_spikegeneratorgroup__period_bins!=0)
    {
        delete [] _array_spikegeneratorgroup__period_bins;
        _array_spikegeneratorgroup__period_bins = 0;
    }
    if(dev_array_spikegeneratorgroup__period_bins!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup__period_bins)
                );
        dev_array_spikegeneratorgroup__period_bins = 0;
    }
    if(_array_spikegeneratorgroup_i!=0)
    {
        delete [] _array_spikegeneratorgroup_i;
        _array_spikegeneratorgroup_i = 0;
    }
    if(dev_array_spikegeneratorgroup_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup_i)
                );
        dev_array_spikegeneratorgroup_i = 0;
    }
    if(_array_spikegeneratorgroup_period!=0)
    {
        delete [] _array_spikegeneratorgroup_period;
        _array_spikegeneratorgroup_period = 0;
    }
    if(dev_array_spikegeneratorgroup_period!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup_period)
                );
        dev_array_spikegeneratorgroup_period = 0;
    }
    if(_array_spikemonitor_1__source_idx!=0)
    {
        delete [] _array_spikemonitor_1__source_idx;
        _array_spikemonitor_1__source_idx = 0;
    }
    if(dev_array_spikemonitor_1__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_1__source_idx)
                );
        dev_array_spikemonitor_1__source_idx = 0;
    }
    if(_array_spikemonitor_1_count!=0)
    {
        delete [] _array_spikemonitor_1_count;
        _array_spikemonitor_1_count = 0;
    }
    if(dev_array_spikemonitor_1_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_1_count)
                );
        dev_array_spikemonitor_1_count = 0;
    }
    if(_array_spikemonitor_1_N!=0)
    {
        delete [] _array_spikemonitor_1_N;
        _array_spikemonitor_1_N = 0;
    }
    if(dev_array_spikemonitor_1_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_1_N)
                );
        dev_array_spikemonitor_1_N = 0;
    }
    if(_array_spikemonitor_2__source_idx!=0)
    {
        delete [] _array_spikemonitor_2__source_idx;
        _array_spikemonitor_2__source_idx = 0;
    }
    if(dev_array_spikemonitor_2__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_2__source_idx)
                );
        dev_array_spikemonitor_2__source_idx = 0;
    }
    if(_array_spikemonitor_2_count!=0)
    {
        delete [] _array_spikemonitor_2_count;
        _array_spikemonitor_2_count = 0;
    }
    if(dev_array_spikemonitor_2_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_2_count)
                );
        dev_array_spikemonitor_2_count = 0;
    }
    if(_array_spikemonitor_2_N!=0)
    {
        delete [] _array_spikemonitor_2_N;
        _array_spikemonitor_2_N = 0;
    }
    if(dev_array_spikemonitor_2_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_2_N)
                );
        dev_array_spikemonitor_2_N = 0;
    }
    if(_array_spikemonitor__source_idx!=0)
    {
        delete [] _array_spikemonitor__source_idx;
        _array_spikemonitor__source_idx = 0;
    }
    if(dev_array_spikemonitor__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor__source_idx)
                );
        dev_array_spikemonitor__source_idx = 0;
    }
    if(_array_spikemonitor_count!=0)
    {
        delete [] _array_spikemonitor_count;
        _array_spikemonitor_count = 0;
    }
    if(dev_array_spikemonitor_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_count)
                );
        dev_array_spikemonitor_count = 0;
    }
    if(_array_spikemonitor_N!=0)
    {
        delete [] _array_spikemonitor_N;
        _array_spikemonitor_N = 0;
    }
    if(dev_array_spikemonitor_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_N)
                );
        dev_array_spikemonitor_N = 0;
    }
    if(_array_synapses_1_N!=0)
    {
        delete [] _array_synapses_1_N;
        _array_synapses_1_N = 0;
    }
    if(dev_array_synapses_1_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_1_N)
                );
        dev_array_synapses_1_N = 0;
    }
    if(_array_synapses_2_N!=0)
    {
        delete [] _array_synapses_2_N;
        _array_synapses_2_N = 0;
    }
    if(dev_array_synapses_2_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_2_N)
                );
        dev_array_synapses_2_N = 0;
    }
    if(_array_synapses_N!=0)
    {
        delete [] _array_synapses_N;
        _array_synapses_N = 0;
    }
    if(dev_array_synapses_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_N)
                );
        dev_array_synapses_N = 0;
    }


    // static arrays
    if(_static_array__dynamic_array_spikegeneratorgroup__timebins!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup__timebins;
        _static_array__dynamic_array_spikegeneratorgroup__timebins = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_neuron_index!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_neuron_index;
        _static_array__dynamic_array_spikegeneratorgroup_neuron_index = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_spike_number!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_spike_number;
        _static_array__dynamic_array_spikegeneratorgroup_spike_number = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_spike_time!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_spike_time;
        _static_array__dynamic_array_spikegeneratorgroup_spike_time = 0;
    }
    if(_timedarray_1_values!=0)
    {
        delete [] _timedarray_1_values;
        _timedarray_1_values = 0;
    }
    if(_timedarray_2_values!=0)
    {
        delete [] _timedarray_2_values;
        _timedarray_2_values = 0;
    }
    if(_timedarray_3_values!=0)
    {
        delete [] _timedarray_3_values;
        _timedarray_3_values = 0;
    }
    if(_timedarray_4_values!=0)
    {
        delete [] _timedarray_4_values;
        _timedarray_4_values = 0;
    }
    if(_timedarray_values!=0)
    {
        delete [] _timedarray_values;
        _timedarray_values = 0;
    }

}


// objects.cu ends here

// run.cu starts here

void brian_start()
{
    _init_arrays();
    _load_arrays();
    srand(time(NULL));

    // Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::defaultclock.t = brian::_array_defaultclock_t;
}

void brian_end()
{
    _write_arrays();
    _dealloc_arrays();
}


// run.cu ends here

// rand.cu starts here 

// XXX: for some documentation on random number generation, check out our wiki:
//      https://github.com/brian-team/brian2cuda/wiki/Random-number-generation

using namespace brian;

// TODO make this a class member function
// TODO don't call one kernel per codeobject but instead on kernel which takes
//      care of all codeobjects, preferably called with as many threads/blocks
//      as necessary for all states and initializing in parallel with warp
//      level divergence [needs changing set_curand_device_api_states()]
namespace {

    __global__ void init_curand_states(int N, int sequence_offset)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N)
        {
            // Each thread gets the same seed, a different sequence number and
            // no offset
            // TODO: different seed and 0 sequence number is much faster, with
            // less security for independent sequences, add option as
            // preference!
            //curand_init(curand_seed + idx, 0, 0,
            curand_init(
                    *d_curand_seed,          // seed
                    sequence_offset + idx,   // sequence number
                    0,                       // offset
                    &d_curand_states[idx]);
        }
    }
}


// need a function pointer for Network::add(), can't pass a pointer to a class
// method, which is of different type
void _run_random_number_buffer()
{
    // random_number_buffer is a RandomNumberBuffer instance, declared in objects.cu
    random_number_buffer.next_time_step();
}


void RandomNumberBuffer::init()
{
    // check that we have enough memory available
    size_t free_byte;
    size_t total_byte;
    CUDA_SAFE_CALL(
            cudaMemGetInfo(&free_byte, &total_byte)
            );
    // TODO: This assumes all random number have randomNumber_t type, but poisson
    //       objects have different type
    size_t num_free_floats = free_byte / sizeof(randomNumber_t);

    if (run_counter == 0)
    {
        // number of time steps each codeobject is executed during current Network::run() call
        // XXX: we are assuming here that this function is only run in the first time step of a Network::run()


        // now check if the total number of generated floats fit into available memory
        int total_num_generated_floats = 0;
        if (num_free_floats < total_num_generated_floats)
        {
            // TODO: find a way to deal with this? E.g. looping over buffers sorted
            // by buffer size and reducing them until it fits.
            printf("MEMORY ERROR: Trying to generate more random numbers than fit "
                   "into available memory. Please report this as an issue on "
                   "GitHub: https://github.com/brian-team/brian2cuda/issues/new");
            _dealloc_arrays();
            exit(1);
        }

    } // if (run_counter == 0)

    // init curand states only in first run
    if (run_counter == 0)
    {

        // Update curand device api states once before anything is run. At this
        // point all N's (also from probabilistically generated synapses) are
        // known. This might update the number of needed curand states.
        ensure_enough_curand_states();
    }

}


void RandomNumberBuffer::allocate_device_curand_states()
{
    // allocate globabl memory for curand device api states
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_curand_states,
                sizeof(curandState) * num_curand_states)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_curand_states,
                &dev_curand_states, sizeof(curandState*))
            );
}



void RandomNumberBuffer::update_needed_number_curand_states()
{
    // Find the maximum number of threads generating random numbers in parallel
    // using the cuRAND device API. For synapses objects, the number of
    // synapses might not be known yet. This is the case when the first random
    // seed is set and for any seed() call before the synapses creation.
    num_threads_curand_init = max_threads_per_block;
    num_blocks_curand_init = num_curand_states / max_threads_per_block + 1;
    if (num_curand_states < num_threads_curand_init)
        num_threads_curand_init = num_curand_states;
}


void RandomNumberBuffer::set_curand_device_api_states(bool reset_seed)
{
    int sequence_offset = 0;
    int num_curand_states_old = num_curand_states;
    // Whenever curand states are set, check if enough states where
    // initialized. This will generate states the first time the seed is set.
    // But it can be that the seed is set before all network objects' N are
    // available (e.g. synapses not created yet) and before the network is
    // run. In such a case, once the network is run, missing curand states are
    // generated here. If the seed was not reset inbetween, the pervious states
    // should not be reinitialized (achieved by the `sequence_offset`
    // parameter). If the seed was reset, then all states should be
    // reinitialized.
    update_needed_number_curand_states();

    // number of curand states that need to be initialized
    int num_curand_states_to_init;

    if (reset_seed)
    {
        // initialize all curand states
        num_curand_states_to_init = num_curand_states;
        sequence_offset = 0;
    }
    else
    {
        // don't initialize existing curand states, only the new ones
        num_curand_states_to_init = num_curand_states - num_curand_states_old;
        sequence_offset = num_curand_states_old;
    }

    if (num_curand_states_old < num_curand_states)
    {
        // copy curand states to new array of updated size
        curandState* dev_curand_states_old = dev_curand_states;
        // allocate memory for new number of curand states
        allocate_device_curand_states();

        if ((!reset_seed) && (num_curand_states_old > 0))
        {
            // copy old states to new memory address on device
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_curand_states, dev_curand_states_old,
                        sizeof(curandState) * num_curand_states_old,
                        cudaMemcpyDeviceToDevice)
                    );
        }
    }

    if (num_curand_states_to_init > 0)
    {
        init_curand_states<<<num_blocks_curand_init, num_threads_curand_init>>>(
                num_curand_states_to_init,
                sequence_offset);
    }
}


void RandomNumberBuffer::ensure_enough_curand_states()
{
    // Separate public function needed for synapses codeobjects that are run
    // only once before the network
    // The N of synapses will not be known when setting the seed and needs to
    // be updated before using random numbers per synapse. This occurs e.g.
    // when initializing synaptic variables (synapses_group_conditional_....)
    bool reset_seed = false;
    set_curand_device_api_states(reset_seed);
}


void RandomNumberBuffer::run_finished()
{
    needs_init = true;
    run_counter += 1;
}


void RandomNumberBuffer::set_seed(unsigned long long seed)
{
    CUDA_SAFE_CALL(
            curandSetPseudoRandomGeneratorSeed(curand_generator, seed)
            );

    // generator offset needs to be reset to its default (=0)
    CUDA_SAFE_CALL(
            curandSetGeneratorOffset(curand_generator, 0ULL)
            );

    // set seed for curand device api calls
    // don't set the same seed for host api and device api random states, just in case
    unsigned long long curand_seed = seed + 1;
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_curand_seed, &curand_seed,
                sizeof(unsigned long long), cudaMemcpyHostToDevice)
            );

    bool reset_seed = true;
    set_curand_device_api_states(reset_seed);
    // We set all device api states for codeobjects run outside the network
    // since we don't know when they will be used.
    //set_curand_device_api_states_for_separate_calls();
    // Curand device api states for binomials during network runs will be set
    // only for the current run in init(), once the network starts.
}


void RandomNumberBuffer::refill_uniform_numbers(
        randomNumber_t* dev_rand_allocator,
        randomNumber_t* &dev_rand,
        int num_per_gen_rand,
        int &idx_rand)
{
    // generate uniform distributed random numbers and reset buffer index

    curandGenerateUniformDouble(curand_generator, dev_rand_allocator, num_per_gen_rand);
    // before: XXX dev_rand = &dev_rand_allocator[0];
    dev_rand = dev_rand_allocator;
    idx_rand = 1;
}


void RandomNumberBuffer::refill_normal_numbers(
        randomNumber_t* dev_randn_allocator,
        randomNumber_t* &dev_randn,
        int num_per_gen_randn,
        int &idx_randn)
{
    // generate normal distributed random numbers and reset buffer index

    curandGenerateNormalDouble(curand_generator, dev_randn_allocator, num_per_gen_randn, 0, 1);
    // before: XXX dev_randn = &dev_randn_allocator[0];
    dev_randn = dev_randn_allocator;
    idx_randn = 1;
}


void RandomNumberBuffer::refill_poisson_numbers(
        double lambda,
        unsigned int* dev_poisson_allocator,
        unsigned int* &dev_poisson,
        int num_per_gen_poisson,
        int &idx_poisson)
{
    // generate poisson distributed random numbers and reset buffer index

    printf("num_per_gen_poisson %d, lambda %f\n", num_per_gen_poisson, lambda);
    CUDA_SAFE_CALL(
            curandGeneratePoisson(curand_generator, dev_poisson_allocator, num_per_gen_poisson, lambda)
            );
    dev_poisson = dev_poisson_allocator;
    idx_poisson = 1;
}

void RandomNumberBuffer::next_time_step()
{
    // init buffers at fist time step of each run call
    if (needs_init)
    {
        // free device memory for random numbers used during last run call
        if (run_counter > 0)
        {
        }

        // init random number buffers
        init();
        needs_init = false;
    }

    if (run_counter == 0)
    {
    }// run_counter == 0
}

// rand.cu ends here


// main starts here

int main(int argc, char **argv)
{
    // seed variable set in Python through brian2.seed() calls can use this
    // variable (see device.py CUDAStandaloneDevice.generate_main_source())
    unsigned long long seed;

    const std::clock_t _start_time = std::clock();

    const std::clock_t _start_time2 = std::clock();

    CUDA_SAFE_CALL(
            cudaSetDevice(0)
            );

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );
    size_t limit = 128 * 1024 * 1024;
    CUDA_SAFE_CALL(
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit)
            );
    CUDA_SAFE_CALL(
            cudaDeviceSynchronize()
            );

    const double _run_time2 = (double)(std::clock() -_start_time2)/CLOCKS_PER_SEC;
    printf("INFO: setting cudaDevice stuff took %f seconds\n", _run_time2);

    brian_start();

    const std::clock_t _start_time3 = std::clock();
    {
        using namespace brian;

                
                        for(int i=0; i<_num__array_spikegeneratorgroup__spikespace; i++)
                        {
                            _array_spikegeneratorgroup__spikespace[i] = - 1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace],
                                           &_array_spikegeneratorgroup__spikespace[0],
                                           sizeof(_array_spikegeneratorgroup__spikespace[0])*_num__array_spikegeneratorgroup__spikespace,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_spikegeneratorgroup__spikespace[_num__array_spikegeneratorgroup__spikespace - 1] = 0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace] + _num__array_spikegeneratorgroup__spikespace - 1,
                                           &_array_spikegeneratorgroup__spikespace[_num__array_spikegeneratorgroup__spikespace - 1],
                                           sizeof(_array_spikegeneratorgroup__spikespace[_num__array_spikegeneratorgroup__spikespace - 1]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup__spikespace; i++)
                        {
                            _array_neurongroup__spikespace[i] = - 1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace],
                                           &_array_neurongroup__spikespace[0],
                                           sizeof(_array_neurongroup__spikespace[0])*_num__array_neurongroup__spikespace,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_neurongroup__spikespace[_num__array_neurongroup__spikespace - 1] = 0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace] + _num__array_neurongroup__spikespace - 1,
                                           &_array_neurongroup__spikespace[_num__array_neurongroup__spikespace - 1],
                                           sizeof(_array_neurongroup__spikespace[_num__array_neurongroup__spikespace - 1]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_1__spikespace; i++)
                        {
                            _array_neurongroup_1__spikespace[i] = - 1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace],
                                           &_array_neurongroup_1__spikespace[0],
                                           sizeof(_array_neurongroup_1__spikespace[0])*_num__array_neurongroup_1__spikespace,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_neurongroup_1__spikespace[_num__array_neurongroup_1__spikespace - 1] = 0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace] + _num__array_neurongroup_1__spikespace - 1,
                                           &_array_neurongroup_1__spikespace[_num__array_neurongroup_1__spikespace - 1],
                                           sizeof(_array_neurongroup_1__spikespace[_num__array_neurongroup_1__spikespace - 1]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_defaultclock_dt[0] = 0.0001;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_defaultclock_dt + 0,
                                           &_array_defaultclock_dt[0],
                                           sizeof(_array_defaultclock_dt[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_defaultclock_dt[0] = 0.0001;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_defaultclock_dt + 0,
                                           &_array_defaultclock_dt[0],
                                           sizeof(_array_defaultclock_dt[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_defaultclock_dt[0] = 0.0001;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_defaultclock_dt + 0,
                                           &_array_defaultclock_dt[0],
                                           sizeof(_array_defaultclock_dt[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                            _dynamic_array_spikegeneratorgroup_spike_number.resize(19676);
                            THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup_spike_number.resize(19676));
                        
        
                        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup_spike_number; i++)
                        {
                            _dynamic_array_spikegeneratorgroup_spike_number[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_number[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_spike_number[0]), &_dynamic_array_spikegeneratorgroup_spike_number[0],
                                        sizeof(_dynamic_array_spikegeneratorgroup_spike_number[0])*_dynamic_array_spikegeneratorgroup_spike_number.size(), cudaMemcpyHostToDevice)
                                );
                        
        
                            _dynamic_array_spikegeneratorgroup_neuron_index.resize(19676);
                            THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup_neuron_index.resize(19676));
                        
        
                        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup_neuron_index; i++)
                        {
                            _dynamic_array_spikegeneratorgroup_neuron_index[i] = _static_array__dynamic_array_spikegeneratorgroup_neuron_index[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_neuron_index[0]), &_dynamic_array_spikegeneratorgroup_neuron_index[0],
                                        sizeof(_dynamic_array_spikegeneratorgroup_neuron_index[0])*_dynamic_array_spikegeneratorgroup_neuron_index.size(), cudaMemcpyHostToDevice)
                                );
                        
        
                            _dynamic_array_spikegeneratorgroup_spike_time.resize(19676);
                            THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup_spike_time.resize(19676));
                        
        
                        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup_spike_time; i++)
                        {
                            _dynamic_array_spikegeneratorgroup_spike_time[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_time[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_spike_time[0]), &_dynamic_array_spikegeneratorgroup_spike_time[0],
                                        sizeof(_dynamic_array_spikegeneratorgroup_spike_time[0])*_dynamic_array_spikegeneratorgroup_spike_time.size(), cudaMemcpyHostToDevice)
                                );
                        
        
                            _dynamic_array_spikegeneratorgroup__timebins.resize(19676);
                            THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup__timebins.resize(19676));
                        
        
                        _array_spikegeneratorgroup__lastindex[0] = 0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_spikegeneratorgroup__lastindex + 0,
                                           &_array_spikegeneratorgroup__lastindex[0],
                                           sizeof(_array_spikegeneratorgroup__lastindex[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_spikegeneratorgroup_period[0] = 0.0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_spikegeneratorgroup_period + 0,
                                           &_array_spikegeneratorgroup_period[0],
                                           sizeof(_array_spikegeneratorgroup_period[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_lastspike; i++)
                        {
                            _array_neurongroup_lastspike[i] = - 10000.0;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_lastspike,
                                           &_array_neurongroup_lastspike[0],
                                           sizeof(_array_neurongroup_lastspike[0])*_num__array_neurongroup_lastspike,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_not_refractory; i++)
                        {
                            _array_neurongroup_not_refractory[i] = true;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_not_refractory,
                                           &_array_neurongroup_not_refractory[0],
                                           sizeof(_array_neurongroup_not_refractory[0])*_num__array_neurongroup_not_refractory,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_1_lastspike; i++)
                        {
                            _array_neurongroup_1_lastspike[i] = - 10000.0;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_1_lastspike,
                                           &_array_neurongroup_1_lastspike[0],
                                           sizeof(_array_neurongroup_1_lastspike[0])*_num__array_neurongroup_1_lastspike,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_1_not_refractory; i++)
                        {
                            _array_neurongroup_1_not_refractory[i] = true;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_1_not_refractory,
                                           &_array_neurongroup_1_not_refractory[0],
                                           sizeof(_array_neurongroup_1_not_refractory[0])*_num__array_neurongroup_1_not_refractory,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                            _dynamic_array_synapses_1_delay.resize(1);
                            THRUST_CHECK_ERROR(dev_dynamic_array_synapses_1_delay.resize(1));
                        
        
                            _dynamic_array_synapses_1_delay.resize(1);
                            THRUST_CHECK_ERROR(dev_dynamic_array_synapses_1_delay.resize(1));
                        
        
                        _dynamic_array_synapses_1_delay[0] = 0.0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_delay[0]) + 0,
                                           &_dynamic_array_synapses_1_delay[0],
                                           sizeof(_dynamic_array_synapses_1_delay[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                            _dynamic_array_synapses_2_delay.resize(1);
                            THRUST_CHECK_ERROR(dev_dynamic_array_synapses_2_delay.resize(1));
                        
        
                            _dynamic_array_synapses_2_delay.resize(1);
                            THRUST_CHECK_ERROR(dev_dynamic_array_synapses_2_delay.resize(1));
                        
        
                        _dynamic_array_synapses_2_delay[0] = 0.0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2_delay[0]) + 0,
                                           &_dynamic_array_synapses_2_delay[0],
                                           sizeof(_dynamic_array_synapses_2_delay[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_synapses_create_generator_codeobject();
        _run_synapses_1_synapses_create_generator_codeobject();
        _run_synapses_2_synapses_create_generator_codeobject();
        _run_synapses_group_variable_set_conditional_codeobject();
        _run_synapses_1_group_variable_set_conditional_codeobject();
        _run_synapses_1_group_variable_set_conditional_codeobject_1();
        
                        for(int i=0; i<_num__array_neurongroup_V; i++)
                        {
                            _array_neurongroup_V[i] = - 0.06356;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_V,
                                           &_array_neurongroup_V[0],
                                           sizeof(_array_neurongroup_V[0])*_num__array_neurongroup_V,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_h; i++)
                        {
                            _array_neurongroup_h[i] = 1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_h,
                                           &_array_neurongroup_h[0],
                                           sizeof(_array_neurongroup_h[0])*_num__array_neurongroup_h,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_m; i++)
                        {
                            _array_neurongroup_m[i] = 0;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_m,
                                           &_array_neurongroup_m[0],
                                           sizeof(_array_neurongroup_m[0])*_num__array_neurongroup_m,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_n; i++)
                        {
                            _array_neurongroup_n[i] = 0.5;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_n,
                                           &_array_neurongroup_n[0],
                                           sizeof(_array_neurongroup_n[0])*_num__array_neurongroup_n,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_1_V; i++)
                        {
                            _array_neurongroup_1_V[i] = - 0.06356;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_1_V,
                                           &_array_neurongroup_1_V[0],
                                           sizeof(_array_neurongroup_1_V[0])*_num__array_neurongroup_1_V,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_1_h; i++)
                        {
                            _array_neurongroup_1_h[i] = 1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_1_h,
                                           &_array_neurongroup_1_h[0],
                                           sizeof(_array_neurongroup_1_h[0])*_num__array_neurongroup_1_h,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_1_m; i++)
                        {
                            _array_neurongroup_1_m[i] = 0;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_1_m,
                                           &_array_neurongroup_1_m[0],
                                           sizeof(_array_neurongroup_1_m[0])*_num__array_neurongroup_1_m,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_1_n; i++)
                        {
                            _array_neurongroup_1_n[i] = 0.5;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_1_n,
                                           &_array_neurongroup_1_n[0],
                                           sizeof(_array_neurongroup_1_n[0])*_num__array_neurongroup_1_n,
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_defaultclock_timestep[0] = 0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_defaultclock_timestep + 0,
                                           &_array_defaultclock_timestep[0],
                                           sizeof(_array_defaultclock_timestep[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_defaultclock_t[0] = 0.0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_defaultclock_t + 0,
                                           &_array_defaultclock_t[0],
                                           sizeof(_array_defaultclock_t[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_spikegeneratorgroup__lastindex[0] = 0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_spikegeneratorgroup__lastindex + 0,
                                           &_array_spikegeneratorgroup__lastindex[0],
                                           sizeof(_array_spikegeneratorgroup__lastindex[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup__timebins; i++)
                        {
                            _dynamic_array_spikegeneratorgroup__timebins[i] = _static_array__dynamic_array_spikegeneratorgroup__timebins[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup__timebins[0]), &_dynamic_array_spikegeneratorgroup__timebins[0],
                                        sizeof(_dynamic_array_spikegeneratorgroup__timebins[0])*_dynamic_array_spikegeneratorgroup__timebins.size(), cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_spikegeneratorgroup__period_bins[0] = 0.0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_spikegeneratorgroup__period_bins + 0,
                                           &_array_spikegeneratorgroup__period_bins[0],
                                           sizeof(_array_spikegeneratorgroup__period_bins[0]),
                                           cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_1_pre_initialise_queue();
        _run_synapses_2_pre_initialise_queue();
        _run_synapses_pre_initialise_queue();
        _run_synapses_1_post_initialise_queue();
        
                                    dev_dynamic_array_synapses_1__synaptic_pre.clear();
                                    dev_dynamic_array_synapses_1__synaptic_pre.shrink_to_fit();
                                    
        
                                    dev_dynamic_array_synapses_2__synaptic_pre.clear();
                                    dev_dynamic_array_synapses_2__synaptic_pre.shrink_to_fit();
                                    
        
                                    dev_dynamic_array_synapses__synaptic_pre.clear();
                                    dev_dynamic_array_synapses__synaptic_pre.shrink_to_fit();
                                    
        magicnetwork.clear();
        magicnetwork.add(&defaultclock, _run_random_number_buffer);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_spikegeneratorgroup_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_1_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_2_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_1_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_1_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_2_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_2_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_1_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_1_post_codeobject);
        CUDA_SAFE_CALL(cudaProfilerStart());
        magicnetwork.run(10.0, NULL, 10.0);
        random_number_buffer.run_finished();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaProfilerStop());
        _copyToHost_spikemonitor_codeobject();
        _debugmsg_spikemonitor_codeobject();
        
        _copyToHost_spikemonitor_1_codeobject();
        _debugmsg_spikemonitor_1_codeobject();
        
        _copyToHost_spikemonitor_2_codeobject();
        _debugmsg_spikemonitor_2_codeobject();
        
        _debugmsg_synapses_1_pre_codeobject();
        
        _debugmsg_synapses_2_pre_codeobject();
        
        _debugmsg_synapses_pre_codeobject();
        
        _debugmsg_synapses_1_post_codeobject();

    }

    const double _run_time3 = (double)(std::clock() -_start_time3)/CLOCKS_PER_SEC;
    printf("INFO: main_lines took %f seconds\n", _run_time3);

    brian_end();

    // Profiling
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    printf("INFO: main function took %f seconds\n", _run_time);

    return 0;
}