#include "objects.h"
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <iostream>
#include <ctime>
#include <limits.h>
#include <tuple>
#include <string>
#include <iomanip>
#include <vector>
#include "code_objects/synapses_pre_initialise_queue.h"
#include "brianlib/cuda_utils.h"

// Makro for file and line information in _cudaSafeCall
#define COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(a, b, c, d) \
    _copyHostArrayToDeviceSymbol(a, b, c, d, __FILE__, __LINE__)

namespace {
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
}


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

