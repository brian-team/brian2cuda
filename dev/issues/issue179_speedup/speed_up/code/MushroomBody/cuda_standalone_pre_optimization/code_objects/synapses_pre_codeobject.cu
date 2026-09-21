#include "objects.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>

#include <stdint.h>
#include "synapses_classes.h"


////// SUPPORT CODE ///////
namespace {
    double _host_rand(const int _vectorisation_idx);
    double _host_randn(const int _vectorisation_idx);
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx);

    ///// block extra_device_helper /////

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

