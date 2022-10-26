{# USES_VARIABLES { N, delay, _n_sources, _n_targets, _source_dt } #}
{% extends 'common_group.cu' %}

{# Get the name of the array that stores these events (e.g. the spikespace array) #}
{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}

{### BEFORE RUN ###}
{# TEMPLATE INFO
 # This template creates the connectivity matrix for this SynapticPathway
 # ({{owner.name}}). It's form depends on the delay and for heterogeneous delays
 # on the propagation mode.
 #
 # DELAY MODE
 # When the delay is set using the `Synapses` constructor's `delay` keyword
 # (e.g. `syn = Synapses(..., delay=2*ms)`) or no delay is set at all, we
 # are in "no_or_const_delay_mode" (template parameter). If the delay is
 # set in the objects `delay` attribute (e.g.  `syn.delay = ...`), we are
 # not. If we are not in "no_or_const_delay_mode", we can still have the
 # same delay for all synapses, e.g. when `syn.delay = 2*ms` was given. But
 # we can't differentiate it on Python side from e.g. `syn.delay =
 # "rand()*ms", where each synapse gets a different delay. Therefore we
 # check in this template if all delays have the same value after
 # transforming them into integer multiples of the simulation time step. If
 # so, we set `scalar_delay = true` (cpp variable).
 #
 # PROPAGATION MODE
 # If we have `scalar_delay = false` (which implies we don't have
 # "no_or_const_delay_mode"), the connectivity information stored on the
 # device depends on the "bundle_mode" template variable. If True, we are
 # pushing synapse bundle IDs when a neuron spikes. All synapses of a
 # bundle habe the same (preID, postBlock, delay). All synapse IDs are
 # stored in these bundles, where each bundle has a unique global bundle ID.
 # If we have no "bundle_mode" or we have homogeneous delays, all synapses
 # IDs are sorted by (preID, postBlock) pairs and per (preID, postBlock)
 # sorted by their delay (if they have any).
 #}
{% block before_run_headers %}
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
#include "objects.h"
#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/cuda_utils.h"
{% endblock before_run_headers %}


{% block before_run_defines %}
// Makro for file and line information in _cudaSafeCall
#define COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(a, b, c, d, e) \
    _copyHostArrayToDeviceSymbol(a, b, c, d, e, __FILE__, __LINE__)

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
    // device_array: device pointer to allocate data for and which to copy to device symbol
    // host_array: host array with data to copy
    // device_symbol: global __device__ variable of same type as `host_array`
    // num_elements: number of elements in host_array to copy
    // NOTE: T can be a pointer variable itself (when copying 2D arrays)
    template <typename T>
    inline void _copyHostArrayToDeviceSymbol(T *device_array, const T *host_array,
            T *&device_symbol, int num_elements, const char* name, const char* file,
            const int line){
        size_t bytes = sizeof(T) * num_elements;
        // allocate device memory
        _cudaSafeCall(
                cudaMalloc((void**)&device_array, bytes),
                file, line, "cudaMalloc");
        // copy data from host array to device
        _cudaSafeCall(
                cudaMemcpy(device_array, host_array, bytes, cudaMemcpyHostToDevice),
                file, line, "cudaMemcpy");
        // copy the device data pointer to the global device symbol
        _cudaSafeCall(
                cudaMemcpyToSymbol(device_symbol, &device_array, sizeof(T*)),
                file, line, "cudaMemcpyToSymbol");
        memory_recorder.push_back(std::make_tuple(name, bytes, num_elements));
    }
}


__global__ void _before_run_kernel_{{codeobj_name}}(
    int _source_N,
    int _num_blocks,
    int _num_threads,
    double _dt,
    int _syn_N,
    int num_delays,
    bool scalar_delay)
{
    using namespace brian;

    int tid = threadIdx.x;

    if (scalar_delay)
    {
        if (tid == 0)
        {
            {{owner.name}}.queue->num_blocks = _num_blocks;
            {{owner.name}}.queue->num_delays = num_delays;
        }
    }
    else
    {
        {{owner.name}}.queue->prepare(
            tid,
            _num_threads,
            _num_blocks,
            0,
            _source_N,
            _syn_N,
            num_delays,
            {{owner.name}}_num_synapses_by_pre,
            {{owner.name}}_num_synapses_by_bundle,
            {{owner.name}}_num_unique_delays_by_pre,
            {{owner.name}}_unique_delays,
            {{owner.name}}_global_bundle_id_start_by_pre,
            {{owner.name}}_synapses_offset_by_bundle,
            {{owner.name}}_synapse_ids,
            {{owner.name}}_synapse_ids_by_pre,
            {{owner.name}}_unique_delays_offset_by_pre,
            {{owner.name}}_unique_delay_start_idcs
        );
    }
    {{owner.name}}.no_or_const_delay_mode = scalar_delay;
}
{% endblock before_run_defines %}


{% block define_N %}
{% endblock %}


{% block before_run_host_maincode %}
    std::clock_t start_timer = std::clock();
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    static bool first_run = true;

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    ///// HOST_CONSTANTS ///////////
    %HOST_CONSTANTS%

    ///// pointers_lines /////
    {{pointers_lines|autoindent}}

    int64_t syn_N_check = {{N}};

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
    double dt = {{_source_dt}};
    // number of neurons in source group
    int source_N = {{constant_or_scalar('_n_sources', variables['_n_sources'])}};
    // number of neurons in target group
    int target_N = {{constant_or_scalar('_n_targets', variables['_n_targets'])}};

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

    {% if not no_or_const_delay_mode %}
    // statistics of number of unique delays per (preID, postBlock) pair
    int sum_num_unique_elements = 0;
    int count_num_unique_elements = 0;
    double mean_num_unique_elements = 0;
    double M2_num_unique_elements = 0;

    {% if bundle_mode %}
    // total number of bundles in all (preID, postBlock) pairs (not known yet)
    int num_bundle_ids = 0;

    // statistics of number of synapses per bundle
    int sum_bundle_sizes = 0;
    int count_bundle_sizes = 0;
    double mean_bundle_sizes = 0;
    double M2_bundle_sizes = 0;
    {% endif %}{# bundle_mode #}

    {% endif %}{# no_or_const_delay_mode #}

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
    int32_t** h_ptr_d_ptr_synapse_ids_by_pre;
    static int32_t **d_ptr_d_ptr_synapse_ids_by_pre;
    // number of synapses for each (preID, postBlock) pair
    int* h_num_synapses_by_pre;
    static int* d_ptr_num_synapses_by_pre;

    {% if not no_or_const_delay_mode %}
    // delay for each synapse in `h_vec_synapse_ids_by_pre`,
    // only used to sort synapses by delay
    vector_t<int>* h_vec_delays_by_pre = new vector_t<int>[num_pre_post_blocks];
    // array of vectors with unique delays and start indices in synapses arrays
    vector_t<int>* h_vec_unique_delays_by_pre;
    vector_t<int>* h_vec_unique_delay_start_idcs_by_pre;
    {% if bundle_mode %}
    // offset in array of all synapse IDs sorted by bundles (we are storing the
    // offset as 32bit int instead of a 64bit pointer to the bundle start)
    vector_t<int> h_synapses_offset_by_bundle;
    static int* d_ptr_synapses_offset_by_bundle;
    // number of synapses in each bundle
    vector_t<int> h_num_synapses_by_bundle;
    static int* d_ptr_num_synapses_by_bundle;
    // start of global bundle ID per (preID, postBlock) pair (+ total num bundles)
    int* h_global_bundle_id_start_by_pre;
    static int* d_ptr_global_bundle_id_start_by_pre;
    {% else %}{# not bundle_mode #}
    // array of unique delays [in integer multiples of dt] in device memory
    int* h_unique_delays_offset_by_pre;
    static int* d_ptr_unique_delays_offset_by_pre;
    // number of unique delays for each (preID, postBlock) pair
    int* h_num_unique_delays_by_pre;
    static int* d_ptr_num_unique_delays_by_pre;
    {% endif %}{# bundle_mode #}
    {% endif %}{# not no_or_const_delay_mode #}


    // we need to allocate device memory for synapse IDs independent of delay mode
    static int32_t* d_ptr_synapse_ids;
    size_t memory_synapse_ids = sizeof(int32_t) * syn_N;
    if (!first_run) {
        CUDA_SAFE_CALL(cudaFree(d_ptr_synapse_ids));
    }
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&d_ptr_synapse_ids, memory_synapse_ids)
            );
    memory_recorder.push_back(std::make_tuple("synapse IDs", memory_synapse_ids, syn_N));


    //fill vectors of connectivity matrix with synapse IDs and delays (in units of simulation time step)
    int max_delay = (int)({{_dynamic_delay}}[0] / dt + 0.5);
    {% if not no_or_const_delay_mode %}
    int min_delay = max_delay;
    {% endif %}
    for(int syn_id = 0; syn_id < syn_N; syn_id++)  // loop through all synapses
    {
        {# We need to handle subgroups here. When source/target of this SynapticPathway
           is a Subgroup, it will have an `_offset` variable. By checking for its
           existence, we also makes sure that we don't try to access the
           source/target.start variable when it doesn't exist (e.g. if source/target
           is a Synapses object). What we are doing here is the same as is done to set
           the `_source_offset` and `_target_offset` variables in
           `Synapses._create_variables()` (but we can't use those variables directly
           since this codeobjects owner is the a `SynapticPathway`, not a `Synapses`
           object. And using owner.synapses.variables['_source_offset'] gives only the
           Synapses source, which doesn't have to the SynapticPathway source (e.g. for
           `on_post` SynapticPatwhays. Therefore, we define our own source_offset and
           target_offset for this SynapticPathway #}
        {% if '_offset' in owner.source.variables %}
        {% set source_offset = owner.source.variables['_offset'].get_value() %}
        {% else %}
        {% set source_offset = 0 %}
        {% endif %}

        {% if '_offset' in owner.target.variables %}
        {% set target_offset = owner.target.variables['_offset'].get_value() %}
        {% else %}
        {% set target_offset = 0 %}
        {% endif %}

        {# Sanity check that what I'm doing here is correct:
           - If the source/target is a Synapse, we can't use subgroups since we
             can't index the Synapse object before the synapses are created
           - SynapticPathway.source.start should be the same as
             SynapticPathway.source.variables['_offset'] #}
        // Code generation checks
        {% if owner.source.__class__.__name__ == 'Synapses' %}
        assert({{source_offset}} == 0);
        {% else %}
        assert({{source_offset}} == {{owner.source.start}});
        {% endif %}

        {% if owner.target.__class__.__name__ == 'Synapses' %}
        assert({{target_offset}} == 0);
        {% else %}
        assert({{target_offset}} == {{owner.target.start}});
        {% endif %}

        // pre/post_neuron_id are integers from 0 to Nsource/Ntarget (from corresponding
        // SynapticPathway) this is relevant only when using Subgroups where they might
        // be NOT equal to the idx in their NeuronGroup
        {% set source_ids = get_array_name(owner.synapse_sources, access_data=False) %}
        {% set target_ids = get_array_name(owner.synapse_targets, access_data=False) %}
        int32_t pre_neuron_id = {{source_ids}}[syn_id] - {{source_offset}};
        int32_t post_neuron_id = {{target_ids}}[syn_id] - {{target_offset}};

        {% if not no_or_const_delay_mode %}
        int delay = (int)({{_dynamic_delay}}[syn_id] / dt + 0.5);
        if (delay > max_delay)
            max_delay = delay;
        if (delay < min_delay)
            min_delay = delay;
        {% endif %}

        // each parallel executed cuda block gets an equal part of post neuron IDs
        int post_block_id = (post_neuron_id * num_parallel_blocks) / target_N;
        // we store synapses for each pre neuron and post block
        int pre_post_block_id = pre_neuron_id * num_parallel_blocks + post_block_id;

        h_vec_synapse_ids_by_pre[pre_post_block_id].push_back(syn_id);
        {% if not no_or_const_delay_mode %}
        h_vec_delays_by_pre[pre_post_block_id].push_back(delay);
        {% endif %}
    }
    int num_delays = max_delay;
    int num_queues = max_delay + 1;  // we also need a current step

    {% if no_or_const_delay_mode %}
    {{owner.name}}_delay = max_delay;
    {% else %}
    bool scalar_delay = (max_delay == min_delay);
    if (scalar_delay)
        {{owner.name}}_delay = max_delay;
    {% endif %}
    CUDA_CHECK_MEMORY();
    size_t used_device_memory_after_dealloc = used_device_memory;

    ///////////////////////////////////////////////////////
    // Memory allocations which depend on the delay mode //
    ///////////////////////////////////////////////////////

    {# We only NOT need size/synapses_by_pre if we are in bundle mode with
        heterogeneous delays.
     #}
    {% if bundle_mode and not no_or_const_delay_mode %}
    if (scalar_delay)
    {% endif %}
    {
        h_num_synapses_by_pre = new int[num_pre_post_blocks];
        h_ptr_d_ptr_synapse_ids_by_pre = new int32_t*[num_pre_post_blocks];
    }

    {% if not no_or_const_delay_mode %}
    // allocate memory only if the delays are not all the same
    if (!scalar_delay)
    {
        {% if bundle_mode %}
        h_global_bundle_id_start_by_pre = new int[num_pre_post_blocks + 1];
        {% else %}
        h_unique_delays_offset_by_pre =  new int[num_pre_post_blocks];
        h_num_unique_delays_by_pre = new int[num_pre_post_blocks];
        {% endif %}

        h_vec_unique_delay_start_idcs_by_pre = new vector_t<int>[num_pre_post_blocks];
        h_vec_unique_delays_by_pre = new vector_t<int>[num_pre_post_blocks];

    }
    {% if bundle_mode %}
    int global_bundle_id_start = 0;
    {% endif %}{# bundle_mode #}
    {% endif %}{# not no_or_const_delay_mode #}

    // loop through connectivity matrix [(preID, postBlock) pairs]
    for(int i = 0; i < num_pre_post_blocks; i++)
    {
        // i is pre_post_block_id

        int num_elements = h_vec_synapse_ids_by_pre[i].size();
        size_connectivity_matrix += num_elements;
        if (num_elements > {{owner.name}}_max_size)
            {{owner.name}}_max_size = num_elements;

        {% if not no_or_const_delay_mode %}
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

            if (num_unique_elements > {{owner.name}}_max_num_unique_delays)
                {{owner.name}}_max_num_unique_delays = num_unique_elements;

            {% if bundle_mode %}
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
                if (bundle_idx == num_unique_elements - 1){
                    num_synapses = num_elements - synapses_start_idx;
                }
                else {
                    num_synapses = h_vec_unique_delay_start_idcs_by_pre[i][bundle_idx + 1] - synapses_start_idx;
                }
                h_num_synapses_by_bundle.push_back(num_synapses);

                if (bundle_idx == 0){
                    {{owner.name}}_bundle_size_min = num_synapses;
                    {{owner.name}}_bundle_size_max = num_synapses;
                }
                else {
                    if (num_synapses > {{owner.name}}_bundle_size_max){
                        {{owner.name}}_bundle_size_max = num_synapses;
                    }
                    if (num_synapses < {{owner.name}}_bundle_size_min){
                        {{owner.name}}_bundle_size_min = num_synapses;
                    }
                }

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
            {% else %}{# not bundle_mode #}
            h_num_unique_delays_by_pre[i] = num_unique_elements;
            {% endif %}{# bundle_mode #}

            updateMeanStd(count_num_unique_elements, mean_num_unique_elements,
                    M2_num_unique_elements, num_unique_elements);

        }  // end if (!scalar_delay)
        {% if bundle_mode %}
        {# only in bundle_mode with not scalar_delay we dont need size/synapses_by_pre #}
        else   // scalar_delay
        {% endif %}{# bundle_mode #}
        {% endif %}{# not no_or_const_delay_mode #}
        {
            // copy the synapse IDs and the number of synapses for this
            // (preID, postBlock) to device and store the device pointer

            h_num_synapses_by_pre[i] = num_elements;

            h_ptr_d_ptr_synapse_ids_by_pre[i] = d_ptr_synapse_ids + sum_num_elements;
            CUDA_SAFE_CALL(
                    cudaMemcpy(h_ptr_d_ptr_synapse_ids_by_pre[i],
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

    {# If we have don't have heterogeneous delays, we just need to copy the
       synapse IDs and number of synapses per (preID, postBlock) to the device #}
    {% if bundle_mode and not no_or_const_delay_mode %}
    if (scalar_delay)
    {% endif %}
    {
        if (!first_run) {
            CUDA_SAFE_CALL(cudaFree(d_ptr_num_synapses_by_pre));
            CUDA_SAFE_CALL(cudaFree(d_ptr_d_ptr_synapse_ids_by_pre));
        }
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(d_ptr_num_synapses_by_pre,
                h_num_synapses_by_pre, {{owner.name}}_num_synapses_by_pre,
                num_pre_post_blocks, "number of synapses per pre/post block");
        // synapses id
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(d_ptr_d_ptr_synapse_ids_by_pre,
                h_ptr_d_ptr_synapse_ids_by_pre, {{owner.name}}_synapse_ids_by_pre,
                num_pre_post_blocks,
                "pointers to synapse IDs");
    }

    {# If we have heterogeneous delays, we need to loop once more through the
       connectivity matrix since we didn't know the total number of unique
       delays in all (preID, postBlock) pairs beforehand. Now we can allocate
       the correct amount of device memory and copy the bundles or unique delay
       start indices in the second loop.
       NOTE: we could have also allocated for each delay on the fly, but this
       would mean to call a lot of cudaMallocs, which takes long and seems
       to result in excessive memory usage. (cudaMalloc always allocates in
       fixed chunk sizes and when a new allocation needs more memory than
       available from the lust chunk, I assume that it allocates a new memory
       and the last bit of the previous chunk is lost. This is just an untested
       assumption since the memory usage decreased significantly when allocating
       all bundle memory in one cudaMalloc.)
    #}
    {% if not no_or_const_delay_mode %}
    {% if bundle_mode %}
    else  // not scalar_delay
    {% else %}{# not bundle_mode #}
    if (!scalar_delay)
    {% endif %}{# bundle_mode #}
    {
        // Since we now know the total number of unique delays over all
        // (preID, postBlock) pairs, we can allocate the device memory
        size_t memory_unique_delays_by_pre = sizeof(int) * sum_num_unique_elements;
        {% if bundle_mode %}
        assert(sum_bundle_sizes == syn_N);
        {% else %}{# not bundle_mode #}
        static int *d_ptr_unique_delay_start_idcs;
        if (!first_run) {
            CUDA_SAFE_CALL(cudaFree(d_ptr_unique_delay_start_idcs));
        }
        CUDA_SAFE_CALL(
                cudaMalloc((void**)&d_ptr_unique_delay_start_idcs,
                    memory_unique_delays_by_pre)
                );
        memory_recorder.push_back(std::make_tuple(
                    "unique delay start indices", memory_unique_delays_by_pre,
                    sum_num_unique_elements));
        {% endif %}{# bundle_mode #}

        // array of all unique delas, sorted first by pre_post_block and per
        // pre_post_block by delay
        static int *d_ptr_unique_delays;
        if (!first_run) {
            CUDA_SAFE_CALL(cudaFree(d_ptr_unique_delays));
        }
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

                {% if not bundle_mode %}
                h_unique_delays_offset_by_pre[i] = sum_num_unique_elements;

                // copy the unique delays start indices to the device
                CUDA_SAFE_CALL(
                        cudaMemcpy(d_ptr_unique_delay_start_idcs + sum_num_unique_elements,
                                   thrust::raw_pointer_cast(&(h_vec_unique_delay_start_idcs_by_pre[i][0])),
                                   sizeof(int)*num_unique_elements,
                                   cudaMemcpyHostToDevice)
                        );
                {% endif %}{# not bundle_mode #}

                sum_num_unique_elements += num_unique_elements;
            }  // end if(num_elements < 0)
        }  // end second loop connectivity matrix
        assert(sum_num_unique_elements_bak == sum_num_unique_elements);

        // pointer to start of unique delays array
        CUDA_SAFE_CALL(
                cudaMemcpyToSymbol({{owner.name}}_unique_delays,
                                   &d_ptr_unique_delays,
                                   sizeof(d_ptr_unique_delays))
                );

        {% if bundle_mode %}
        num_bundle_ids = sum_num_unique_elements;

        // add num_bundle_ids as last entry
        h_global_bundle_id_start_by_pre[num_pre_post_blocks] = num_bundle_ids;

        // floor(mean(h_num_synapses_by_bundle)) = sum_bundle_sizes / num_bundle_ids;
        assert(std::floor(mean_bundle_sizes) == sum_bundle_sizes / num_bundle_ids);
        {{owner.name}}_bundle_size_mean = mean_bundle_sizes;
        {{owner.name}}_bundle_size_std = getStd(count_bundle_sizes, M2_bundle_sizes);

        // pointer to start of synapse IDs array
        CUDA_SAFE_CALL(
                cudaMemcpyToSymbol({{owner.name}}_synapse_ids, &d_ptr_synapse_ids,
                                   sizeof(d_ptr_synapse_ids))
                );

        if (!first_run) {
            CUDA_SAFE_CALL(cudaFree(d_ptr_num_synapses_by_bundle));
            CUDA_SAFE_CALL(cudaFree(d_ptr_synapses_offset_by_bundle));
            CUDA_SAFE_CALL(cudaFree(d_ptr_global_bundle_id_start_by_pre));
        }
        // size by bundle
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(d_ptr_num_synapses_by_bundle,
                thrust::raw_pointer_cast(&h_num_synapses_by_bundle[0]),
                {{owner.name}}_num_synapses_by_bundle, num_bundle_ids,
                "number of synapses per bundle");

        // synapses offset by bundle
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(d_ptr_synapses_offset_by_bundle,
                thrust::raw_pointer_cast(&h_synapses_offset_by_bundle[0]),
                {{owner.name}}_synapses_offset_by_bundle, num_bundle_ids,
                "synapses bundle offset");

        // global bundle id start idx by pre
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(
                d_ptr_global_bundle_id_start_by_pre,
                h_global_bundle_id_start_by_pre,
                {{owner.name}}_global_bundle_id_start_by_pre,
                num_pre_post_blocks + 1, "global bundle ID start");

        {% else %}{# not bundle_mode #}
        // pointer to start of unique delay start indices array
        CUDA_SAFE_CALL(
                cudaMemcpyToSymbol({{owner.name}}_unique_delay_start_idcs,
                                   &d_ptr_unique_delay_start_idcs,
                                   sizeof(d_ptr_unique_delay_start_idcs))
                );

        if (!first_run) {
            CUDA_SAFE_CALL(cudaFree(d_ptr_unique_delays_offset_by_pre));
            CUDA_SAFE_CALL(cudaFree(d_ptr_num_unique_delays_by_pre));
        }
        // unique delay offset
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(
                d_ptr_unique_delays_offset_by_pre,
                h_unique_delays_offset_by_pre,
                {{owner.name}}_unique_delays_offset_by_pre,
                num_pre_post_blocks, "unique delays offset by pre");

        // unique delay size
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(
                d_ptr_num_unique_delays_by_pre,
                h_num_unique_delays_by_pre,
                {{owner.name}}_num_unique_delays_by_pre, num_pre_post_blocks,
                "number of unique delays");
        {% endif %}{# bundle_mode #}

    }  // end if (!scalar_delay)
    {% endif %}{# not no_or_const_delay_mode #}

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
    {% if not no_or_const_delay_mode %}
    {% if bundle_mode %}
    double std_bundle_sizes = getStd(count_bundle_sizes, M2_bundle_sizes);
    {% endif %}{# bundle_mode #}
    double std_num_unique_elements = getStd(count_num_unique_elements, M2_num_unique_elements);
    {% endif %}{# not no_or_const_delay_mode #}

    // print memory information
    std::cout.precision(1);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << "INFO: synapse statistics and memory usage for {{owner.name}}:\n"
        << "\tnumber of synapses: " << syn_N << "\n"
    {% if not no_or_const_delay_mode and bundle_mode %}
        << "\tnumber of bundles: " << num_bundle_ids << "\n"
    {% endif %}
        << "\tnumber of pre/post blocks: " << num_pre_post_blocks << "\n"
        << "\tnumber of synapses over all pre/post blocks:\n"
        << "\t\tmean: " << mean_num_elements << "\tstd: "
            << std_num_elements << "\n"
    {% if not no_or_const_delay_mode %}
        << "\tnumber of unique delays over all pre/post blocks:\n"
        << "\t\tmean: " << mean_num_unique_elements << "\tstd: "
            << std_num_unique_elements << "\n"
    {% if bundle_mode %}
    << "\tbundle size over all bundles:\n"
        << "\t\tmean: " << mean_bundle_sizes << "\tstd: "
        << std_bundle_sizes << "\n"
    {% endif %}{# bundle_mode #}
    {% endif %}{# not no_or_const_delay_mode #}
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
    {% if not no_or_const_delay_mode %}
    if (scalar_delay)
    {% endif %}
    {
        int num_eventspaces = dev{{_eventspace}}.size();
        bool require_new_eventspaces = (num_queues > num_eventspaces);

        if (require_new_eventspaces)
        {
            // rotate circular eventspace such that the current idx is at the start
            // (logic copied from CSpikeQueue.expand() in Brian's cspikequeue.cpp)
            std::rotate(
                dev{{_eventspace}}.begin(),
                dev{{_eventspace}}.begin() + current_idx{{_eventspace}},
                dev{{_eventspace}}.end()
            );
            current_idx{{_eventspace}} = 0;
            // add new eventspaces
            for (int i = num_eventspaces; i < num_queues; i++)
            {
                {{c_data_type(eventspace_variable.dtype)}}* new_eventspace;
                CUDA_SAFE_CALL(
                    cudaMalloc(
                        (void**)&new_eventspace,
                        sizeof({{c_data_type(eventspace_variable.dtype)}}) * _num_{{_eventspace}}
                    )
                );
                // initialize device eventspace with -1 and counter with 0
                CUDA_SAFE_CALL(
                    cudaMemcpy(
                        new_eventspace,
                        {{_eventspace}},  // defined in objects.cu
                        sizeof({{c_data_type(eventspace_variable.dtype)}}) * _num_{{_eventspace}},
                        cudaMemcpyHostToDevice
                    )
                );
                dev{{_eventspace}}.push_back(new_eventspace);
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
    cudaFuncGetAttributes(&funcAttrib, _before_run_kernel_{{codeobj_name}});
    if (num_threads > funcAttrib.maxThreadsPerBlock)
    {
        // use the max num_threads before launch failure
        num_threads = funcAttrib.maxThreadsPerBlock;
        printf("WARNING Not enough ressources available to call "
               "_before_run_kernel_{{codeobj_name}}"
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
        printf("INFO _before_run_kernel_{{codeobj_name}}\n"
               "\t%u blocks\n"
               "\t%u threads\n"
               "\t%i registers per thread\n"
               "\t%i bytes statically-allocated shared memory per block\n"
               "\t%i bytes local memory per thread\n"
               "\t%i bytes user-allocated constant memory\n"
               "",
               num_blocks, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }

    _before_run_kernel_{{codeobj_name}}<<<num_blocks, num_threads>>>(
        source_N,
        num_parallel_blocks,
        num_threads,
        dt,
        syn_N,
        num_delays,
    {% if no_or_const_delay_mode %}
        true
    {% else %}
        scalar_delay
    {% endif %}
    );

    {% if bundle_mode and not no_or_const_delay_mode %}
    if (scalar_delay)
    {% endif %}
    {
        delete [] h_num_synapses_by_pre;
        delete [] h_ptr_d_ptr_synapse_ids_by_pre;
    }

    //delete temp arrays
    delete [] h_vec_synapse_ids_by_pre;
    {% if not no_or_const_delay_mode %}
    delete [] h_vec_delays_by_pre;
    if (!scalar_delay)
    {
        delete [] h_vec_unique_delay_start_idcs_by_pre;
        delete [] h_vec_unique_delays_by_pre;
        {% if bundle_mode %}
        delete [] h_global_bundle_id_start_by_pre;
        {% else %}
        delete [] h_unique_delays_offset_by_pre;
        delete [] h_num_unique_delays_by_pre;
        {% endif %}
    }
    {% endif %}

    {% if no_or_const_delay_mode %}
    {{owner.name}}_scalar_delay = true;
    {% else %}
    {{owner.name}}_scalar_delay = scalar_delay;
    {% endif %}

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        printf("ERROR initialising {{owner.name}} in %s:%d %s\n",
                __FILE__, __LINE__, cudaGetErrorString(status));
        _dealloc_arrays();
        exit(status);
    }

    CUDA_CHECK_MEMORY();
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: {{owner.name}} initialisation took " <<  time_passed << "s";
    if (used_device_memory_after_dealloc < used_device_memory_start){
        size_t freed_bytes = used_device_memory_start - used_device_memory_after_dealloc;
        std::cout << ", freed " << freed_bytes * to_MB << "MB";
    }
    if (used_device_memory > used_device_memory_start){
        size_t used_bytes = used_device_memory - used_device_memory_start;
        std::cout << " and used " << used_bytes * to_MB << "MB of device memory.";
    }
    std::cout << std::endl;

    first_run = false;
{% endblock before_run_host_maincode %}


{### RUN ###}
{% block extra_device_helper %}
__global__ void _advance_kernel_{{codeobj_name}}()
{
    using namespace brian;
    int tid = threadIdx.x;
    {{owner.name}}.queue->advance(
        tid);
}
{% endblock extra_device_helper %}


{% block kernel %}
__global__ void
{% if launch_bounds or syn_launch_bounds %}
__launch_bounds__(1024, {{sm_multiplier}})
{% endif %}
_run_kernel_{{codeobj_name}}(
    int num_parallel_blocks,
    int _num_blocks,
    int _num_threads,
    int32_t* _eventspace)
{
    // apperently this is not always true and that is why _num_threads is passed as function argument
    // if this assert never fails, we could remove the _num_threads form the argument list
    assert(blockDim.x == _num_threads);

    using namespace brian;

    {% if not bundle_mode %}
    // TODO: check if static shared memory is faster / makes any difference
    extern __shared__ char shared_mem[];
    {% endif %}
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int post_neuron_bid = bid % num_parallel_blocks;
    int pre_neuron_idx = bid / num_parallel_blocks;

    int32_t spiking_neuron = _eventspace[pre_neuron_idx];
    assert(spiking_neuron != -1);

    // push to spikequeue if spiking_neuron is in sources of current SynapticPathway
    if({{owner.name}}.spikes_start <= spiking_neuron && spiking_neuron < {{owner.name}}.spikes_stop)
    {
        {% if bundle_mode %}
        {{owner.name}}.queue->push_bundles(
        {% else %}
        {{owner.name}}.queue->push_synapses(
            shared_mem,
        {% endif %}
            post_neuron_bid,
            tid,
            _num_threads,
            spiking_neuron - {{owner.name}}.spikes_start);
    }
}
{% endblock kernel %}


{% block host_maincode %}
    if ({{owner.name}}_scalar_delay)
    {
        int num_eventspaces = dev{{_eventspace}}.size();
        {{owner.name}}_eventspace_idx = (current_idx{{_eventspace}} - {{owner.name}}_delay + num_eventspaces) % num_eventspaces;

        //////////////////////////////////////////////
        //// No pushing in no_or_const_delay_mode ////
        //////////////////////////////////////////////
    }
    else if ({{owner.name}}_max_size > 0)
    {

        // get the number of spiking neurons
        int32_t num_spiking_neurons;
        CUDA_SAFE_CALL(
                cudaMemcpy(&num_spiking_neurons,
                    dev{{_eventspace}}[current_idx{{_eventspace}}] + _num_{{owner.event}}space - 1,
                    sizeof(int32_t), cudaMemcpyDeviceToHost)
                );

        // advance spike queues
        _advance_kernel_{{codeobj_name}}<<<1, num_parallel_blocks>>>();

        CUDA_CHECK_ERROR("_advance_kernel_{{codeobj_name}}");

    {# Don't close else bracket here, close it at end of block kernel_call, such that
       block prepare_kernel and block kernel_call are executed in this else clause #}
{% endblock host_maincode %}

{% block kernel_info_num_blocks_str %}
"\tvariable number of blocks (depends on number of spiking neurons)\n"
{% endblock %}
{% block kernel_info_num_blocks_var %}
{% endblock %}

{% block prepare_kernel_inner %}
    {% if not bundle_mode %}
    /* We are copying next_delay_start_idx and the atomic offset (both
     * size = num_unique_delays) into shared memory. Since
     * num_unique_delays varies for different combinations of pre
     * neuron and bid, we allocate for max(num_unique_delays). And +1
     * per block for copying size_before_resize into shared memory when
     * we need to use the outer loop.
     */
    needed_shared_memory = (2 * {{owner.name}}_max_num_unique_delays + 1) * sizeof(int);
    assert (needed_shared_memory <= max_shared_mem_size);
    {% else %}{# bundle_mode #}
    needed_shared_memory = 0;
    {% endif %}{# not bundle_mode #}

    // We don't need more then max(num_synapses) threads per block.
    num_threads = {{owner.name}}_max_size;
    if (num_threads > max_threads_per_block)
    {
        num_threads = max_threads_per_block;
    }
    // num_blocks depends on num_spiking_neurons, which changes each time step
{% endblock prepare_kernel_inner %}


{% block kernel_call %}
        if (num_spiking_neurons > 0)
        {
            num_blocks = num_parallel_blocks * num_spiking_neurons;

            _run_kernel_{{codeobj_name}}<<<num_blocks, num_threads, needed_shared_memory>>>(
                    num_parallel_blocks,
                    num_blocks,
                    num_threads,
                    dev{{_eventspace}}[current_idx{{_eventspace}}]);

            CUDA_CHECK_ERROR("_run_kernel_{{codeobj_name}}");
        }
    }  // end else if ({{owner.name}}_max_size > 0) {# from block host_maincode #}
{% endblock kernel_call %}
