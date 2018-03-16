{% macro cu_file() %}
{# USES_VARIABLES { delay } #}
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <set>
#include <iostream>
#include <ctime>
#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/cuda_utils.h"
{% set pathobj = owner.name %}


namespace {
	// Functions for online update of mean and std
	// for a new value newValue, compute the new count, new mean, the new M2.
	// mean accumulates the mean of the entire dataset
	// M2 aggregates the squared distance from the mean
	// count aggregates the number of samples seen so far
	inline void updateMeanStd(unsigned int &count, double &mean, double& M2, double newValue){
		count += 1;
		double delta = newValue - mean;
		mean += delta / count;
		double delta2 = newValue - mean;
		M2 += delta * delta2;
	}

	// get std from aggregated M2 value
	double getStd(unsigned int count, double M2){
		if (count < 2){
			printf("ERROR: getStd: count < 2\n");
			return NAN;
		}
		double variance = M2 / (count - 1);
		double stdValue = sqrt(variance);
		return stdValue;
	}
}


__global__ void _run_{{codeobj_name}}_kernel(
	unsigned int _source_N,
	unsigned int _num_blocks,
	unsigned int _num_threads,
	double _dt,
	unsigned int _syn_N,
	unsigned int num_queues,
	bool new_mode)
{
	using namespace brian;

	int tid = threadIdx.x;

	{{pathobj}}.queue->prepare(
		tid,
		_num_threads,
		_num_blocks,
		0,
		_source_N,
		_syn_N,
		num_queues,
		{{pathobj}}_size_by_pre,
		{{pathobj}}_size_by_bundle_id,
		{{pathobj}}_unique_delay_size_by_pre,
		{{pathobj}}_global_bundle_id_start_idx_by_pre,
		{{pathobj}}_synapses_id_by_pre,
		{{pathobj}}_synapses_id_by_bundle_id,
		{{pathobj}}_unique_delay_by_pre);
	{{pathobj}}.no_or_const_delay_mode = new_mode;
}

//POS(queue_id, neuron_id, neurons_N)
#define OFFSET(a, b, c)	(a*c + b)

void _run_{{pathobj}}_initialise_queue()
{
	using namespace brian;

	std::clock_t start_timer = std::clock();

	const double to_MB = 1.0 / (1024.0 * 1024.0);

	CUDA_CHECK_MEMORY();
	size_t used_device_memory_start = used_device_memory;

	{# we don't use {{N}} to avoid using {{pointer_lines}} which only work inside kernels with %DEVICE_PARAMETERS% #}
	unsigned int syn_N = {{get_array_name(owner.variables['N'], access_data=False)}}[0];
	if (syn_N == 0)
		return;

	double dt = {{owner.clock.name}}.dt[0];
	unsigned int source_N = {{owner.source.N}};
	unsigned int target_N = {{owner.target.N}};

    unsigned int num_pre_post_blocks = num_parallel_blocks * source_N;

	// DENIS: TODO check speed difference when using thrust host vectors instead for easier readability and programming comfort, e.g.:
	// thrust::host_vector<int32_t> h_synapses_synaptic_sources = dev_dynamic_array_{{owner.synapses.name}}_{{owner.synapse_sources.name}}

	{# TODO: explain this somewhere #}
	{# delay was set using Synapses object's delay attribute: `conn = Synapses(...); conn.delay = ...` #}
	{# all delays have the same value, e.g. `conn.delay = 2*ms` or because of small jitter + rounding to dt #}

	///////////////////////////////////
	// Create temporary host vectors //
	///////////////////////////////////

	// pre neuron IDs, post neuron IDs and delays for all synapses (sorted by synapse IDs)
	//TODO: for multiple SynapticPathways for the same Synapses object (on_pre and on_post) the following copy is identical in both pathways initialise templates
	{% if not no_or_const_delay_mode %}
	// delay (on device) was set in group_variable_set_conditional and needs to be copied to host
	{{_dynamic_delay}} = dev{{_dynamic_delay}};
	{% endif %}

	// synapse IDs and delays in connectivity matrix, projected to 1D arrays of vectors
	// sorted first by pre neuron ID, then by cuda blocks (corresponding to groups of post neuron IDs)
	// the index for one pre neuron ID and block ID is: ( pre_neuron_ID * num_blocks + block_ID )

	// vectors store synapse IDs and delays for each synapse, will be sorted by delay
	thrust::host_vector<int32_t>* h_synapses_by_pre_id = new thrust::host_vector<int32_t>[num_pre_post_blocks];
	{% if not no_or_const_delay_mode %}
	thrust::host_vector<unsigned int>* h_delay_by_pre_id = new thrust::host_vector<unsigned int>[num_pre_post_blocks];
	// nemo bundle stuff
	//// this is one vector of host pointers per right_offset (since we don't know how many bundles we will have per right_offset)
	//thrust::host_vector<int32_t*>* h_synapses_by_bundle_id_by_pre = new thrust::host_vector<int32_t*>[num_pre_post_blocks];
	//thrust::host_vector<unsigned int>* h_size_by_bundle_id_by_pre = new thrust::host_vector<unsigned int>[num_pre_post_blocks];
	// this is a vector of host pointers (since we don't know how many bundles we will have in total)
	thrust::host_vector<int32_t*> h_synapses_by_bundle_id;
	thrust::host_vector<unsigned int> h_size_by_bundle_id;
	// start index for local bundle_idx per right_offset
	unsigned int* global_bundle_id_start_idx_by_pre_id = new unsigned int[num_pre_post_blocks];
	{% endif %}

	//fill vectors of connectivity matrix with synapse IDs and delay IDs (in units of simulation time step)
	unsigned int max_delay = (int)({{_dynamic_delay}}[0] / dt + 0.5);
	{% if not no_or_const_delay_mode %}
	unsigned int min_delay = max_delay;
	// TODO: remove delay_set, we are using {{pathobj}}_max_num_bundles now
	std::set<unsigned int> delay_set;
	{% endif %}
	for(int syn_id = 0; syn_id < syn_N; syn_id++)  // loop through all synapses
	{
		// pre/post_neuron_id are integers from 0 to Nsource/Ntarget (from corresponding SynapticPathway)
		// this is relevant only when using Subgroups where they might be NOT equal to the idx in their NeuronGroup
		int32_t pre_neuron_id = {{get_array_name(owner.synapse_sources, access_data=False)}}[syn_id] - {{owner.source.start}};
		int32_t post_neuron_id = {{get_array_name(owner.synapse_targets, access_data=False)}}[syn_id] - {{owner.target.start}};

		{% if not no_or_const_delay_mode %}
		unsigned int delay = (int)({{_dynamic_delay}}[syn_id] / dt + 0.5);
		delay_set.insert(delay);
		if (delay > max_delay)
			max_delay = delay;
		if (delay < min_delay)
			min_delay = delay;
		{% endif %}

		unsigned int right_queue = (post_neuron_id*num_parallel_blocks)/target_N;
		unsigned int right_offset = pre_neuron_id * num_parallel_blocks + right_queue;
		h_synapses_by_pre_id[right_offset].push_back(syn_id);

		{% if not no_or_const_delay_mode %}
		h_delay_by_pre_id[right_offset].push_back(delay);
		{% endif %}
	}
	unsigned int num_queues = max_delay + 1;  // we also need a current step

	{% if no_or_const_delay_mode %}
	{{owner.name}}_delay = max_delay;
	{% else %}
	bool scalar_delay = (max_delay == min_delay);
	if (scalar_delay)
		{{owner.name}}_delay = max_delay;
	{% endif %}
	// Delete delay (in sec) on device, we don't need it
	// TODO: don't copy these delays to the device in first place, see #83
	dev{{_dynamic_delay}}.clear();
	dev{{_dynamic_delay}}.shrink_to_fit();
	CUDA_CHECK_MEMORY();
	size_t used_device_memory_after_dealloc = used_device_memory;

	///////////////////////////////////////
	// Create arrays for device pointers //
	///////////////////////////////////////

	// TODO rename temp
	unsigned int* temp_size_by_pre_id;
	int32_t** temp_synapses_by_pre_id;
	{% if not no_or_const_delay_mode %}
	int num_unique_elements;
	unsigned int* temp_unique_delay_size_by_pre_id;
	unsigned int** temp_unique_delay_by_pre_id;
	// vectors store only unique set of delays and the corresponding start index in the h_delay_by_pre_id vectors
	thrust::host_vector<unsigned int>* h_delay_count_by_pre_id;
	thrust::host_vector<unsigned int>* h_unique_delay_start_idx_by_pre_id;
	thrust::host_vector<unsigned int>* h_unique_delay_by_pre_id;
	if (scalar_delay)
	{% endif %}
	{
		temp_size_by_pre_id = new unsigned int[num_pre_post_blocks];
		temp_synapses_by_pre_id = new int32_t*[num_pre_post_blocks];
	}
	{% if not no_or_const_delay_mode %}
	else  // not scalar_delay
	{
		// allocate memory only if the delays are not all the same
		temp_unique_delay_size_by_pre_id = new unsigned int[num_pre_post_blocks];
		temp_unique_delay_by_pre_id =  new unsigned int*[num_pre_post_blocks];

		h_delay_count_by_pre_id = new thrust::host_vector<unsigned int>[num_pre_post_blocks];
		h_unique_delay_start_idx_by_pre_id = new thrust::host_vector<unsigned int>[num_pre_post_blocks];
		h_unique_delay_by_pre_id = new thrust::host_vector<unsigned int>[num_pre_post_blocks];
	}
	{% endif %}

    long unsigned int sum_num_synapses = 0;
    unsigned int count_num_synapses = 0;
    double mean_num_synapses = 0;
    double M2_num_synapses = 0;
    size_t sum_memory_synapse_ids = 0;

    long unsigned int sum_num_elements = 0;
    long unsigned int sum_num_unique_elements = 0;
    unsigned int count_num_unique_elements = 0;
    double mean_num_unique_elements = 0;
    double M2_num_unique_elements = 0;

    size_t sum_memory_delay_by_pre_id = 0;
    size_t memory_size_by_bundle_id = 0;  // num_bundle_ids are copied
    size_t memory_synapses_bundle_ptrs = 0;  // num_bundle_ids are copied
    size_t memory_unique_delay_size_by_pre = 0;  // num_pre_post_blocks are copied
    size_t memory_unique_delay_by_pre_ptrs = 0;  // num_pre_post_blocks are copied

	// we need to allocate memory for synapse IDs independent of delay mode
    int32_t* d_synapse_ids;
    size_t memory_synapse_ids = sizeof(int32_t) * syn_N;
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_synapse_ids, memory_synapse_ids) );

	std::cout << "Allocated memory for synapseIDs: " << memory_synapse_ids * to_MB << " MB" << std::endl;

	int size_connectivity_matrix = 0;
	unsigned int num_used_bundle_ids = 0;
	unsigned int global_bundle_id_start_idx = 0;
	//fill temp arrays with device pointers
	for(int i = 0; i < num_pre_post_blocks; i++)  // loop through connectivity matrix
	{
		int num_elements = h_synapses_by_pre_id[i].size();
		size_connectivity_matrix += num_elements;
		if (num_elements > {{pathobj}}_max_size)
			{{pathobj}}_max_size = num_elements;

		{% if not no_or_const_delay_mode %}
		if (scalar_delay)
		{% endif %}
		{
			temp_size_by_pre_id[i] = num_elements;

			temp_synapses_by_pre_id[i] = d_synapse_ids + sum_num_elements;
			CUDA_SAFE_CALL( cudaMemcpy(temp_synapses_by_pre_id[i],
						thrust::raw_pointer_cast(&(h_synapses_by_pre_id[i][0])),
						sizeof(unsigned int)*num_elements,
						cudaMemcpyHostToDevice) );
			sum_num_elements += num_elements;
		}
		{% if not no_or_const_delay_mode %}
		else  // not scalar_delay
		{

			// sort synapses (values) and delays (keys) by delay
			thrust::sort_by_key(
					h_delay_by_pre_id[i].begin(), 		// keys start
					h_delay_by_pre_id[i].end(), 		// keys end
					h_synapses_by_pre_id[i].begin()		// values start
					);

			// worst case: number of unique delays is num_elements
			h_unique_delay_by_pre_id[i].resize(num_elements);
			h_delay_count_by_pre_id[i].resize(num_elements);
			// TODO resize h_unique_delay_start_idx_by_pre_id after reduce_by_key and erasing h_delay_count_by_pre_id to correct size
			h_unique_delay_start_idx_by_pre_id[i].resize(num_elements);

			// create arrays of unique delays (keys) and corresponding number of occurences (values)
			thrust::pair<thrust::host_vector<unsigned int>::iterator, thrust::host_vector<unsigned int>::iterator> end_pair;
			end_pair = thrust::reduce_by_key(
					h_delay_by_pre_id[i].begin(), 		// keys start
					h_delay_by_pre_id[i].end(), 		// keys end
					thrust::make_constant_iterator(1),	// values start (each delay has count 1 before reduction)
					h_unique_delay_by_pre_id[i].begin(),  	// unique values
					h_delay_count_by_pre_id[i].begin()  	// reduced keys
					);

			thrust::host_vector<unsigned int>::iterator unique_delay_end = end_pair.first;
			thrust::host_vector<unsigned int>::iterator count_end = end_pair.second;

			// reduce count array to get start indices of unique delays in h_delay_by_pre_id (one delay for each synapse)
			thrust::host_vector<unsigned int>::iterator idx_end;
			idx_end = thrust::exclusive_scan(
					h_delay_count_by_pre_id[i].begin(),
					h_delay_count_by_pre_id[i].end(),
					h_unique_delay_start_idx_by_pre_id[i].begin()
					);

			// erase unused vector entries
			h_delay_count_by_pre_id[i].erase(count_end, h_delay_count_by_pre_id[i].end());
			h_unique_delay_by_pre_id[i].erase(unique_delay_end, h_unique_delay_by_pre_id[i].end());
			h_unique_delay_start_idx_by_pre_id[i].erase(idx_end, h_unique_delay_start_idx_by_pre_id[i].end());
			///////////////////////////////////////////////////////////

			///////////////////////////////////////////////////////////
			//// VERION FOR HAVING ONLY synapses_id_by_pre, unique_delays and delay_start_idx
			//// TODO: delete everything with ..._delay_id_by_pre, ..._delay_count_by_pre
			//
			//// worst case: number of unique delays is num_elements
			//h_unique_delay_start_idx_by_pre_id[i].resize(num_elements);
			//
			//// set the vector of indices for the original delay vector (not unique)
			//thrust::sequence(h_unique_delay_start_idx_by_pre_id[i].begin(), h_unique_delay_start_idx_by_pre_id[i].end());
			//
			//// get delays (keys) and values (indices) for first occurence of each delay value
			//thrust::pair<thrust::host_vector<unsigned int>::iterator, thrust::host_vector<unsigned int>::iterator> end_pair;
			//end_pair = thrust::unique_by_key(
			//		h_unique_delay_by_pre_id[i].begin(),  		// keys start
			//		h_unique_delay_by_pre_id[i].end(),  		// keys end
			//		h_unique_delay_start_idx_by_pre_id[i].begin() 	// values start (position in original delay array)
			//		);
			//unique_delay_end = end_pair.first;
			//idx_end = end_pair.second;
			//
			//// erase unneded vector entries
			//h_unique_delay_by_pre_id[i].erase(unique_delay_end, h_unique_delay_by_pre_id[i].end());
			//h_unique_delay_start_idx_by_pre_id[i].erase(idx_end, h_unique_delay_start_idx_by_pre_id[i].end());
			//
			///////////////////////////////////////////////////////////

			num_unique_elements = h_unique_delay_by_pre_id[i].size();
			temp_unique_delay_size_by_pre_id[i] = num_unique_elements;

			// nemo bundle stuff
			// we need the maximum number of delays over all i (right_offset) to calculate the global bundle ID when pushing
			if (num_unique_elements > {{pathobj}}_max_num_bundles)
				{{pathobj}}_max_num_bundles = num_unique_elements;
			num_used_bundle_ids += num_unique_elements;
			assert(num_unique_elements <= delay_set.size());

			// we need a start idx per i (right_offset) to calc the global bundle ID from the local bundle_idx when pushing
			global_bundle_id_start_idx_by_pre_id[i] = global_bundle_id_start_idx;
			global_bundle_id_start_idx += num_unique_elements;
			// the local bundle_idx goes from 0 to num_bundles for each i (right_offset)
			for (int bundle_idx = 0; bundle_idx < num_unique_elements; bundle_idx++)
			{
				// find the start idx in the synapses array for this delay (bundle)
				unsigned int synapses_start_idx = h_unique_delay_start_idx_by_pre_id[i][bundle_idx];
				// find the number of synapses for this delay (bundle)
				unsigned int num_synapses;
				if (bundle_idx == num_unique_elements - 1)
					num_synapses = num_elements - synapses_start_idx;
				else
					num_synapses = h_unique_delay_start_idx_by_pre_id[i][bundle_idx + 1] - synapses_start_idx;
				//h_size_by_bundle_id_by_pre[i].push_back(num_synapses);
				h_size_by_bundle_id.push_back(num_synapses);
				if (num_synapses > {{pathobj}}_max_bundle_size)
					{{pathobj}}_max_bundle_size = num_synapses;
				int32_t* synapse_bundle = new int32_t[num_synapses];
				// TODO: don't copy synapses to synapse_bundle, just cudaMemcpy it directly to the device with
				// CUDA_SAFE_CALL( cudaMemcpy(d_..., h_synapses_by_pre_id[i] + synapses_start_idx, ...)
				for (int j = 0; j < num_synapses; j++)
				{
					synapse_bundle[j] = h_synapses_by_pre_id[i][synapses_start_idx + j];
				}
				// copy this bundle to device
                int32_t* d_this_bundle = d_synapse_ids + sum_num_synapses;
				size_t memory_size = sizeof(int32_t) * num_synapses;
				CUDA_SAFE_CALL( cudaMemcpy(d_this_bundle, synapse_bundle, memory_size, cudaMemcpyHostToDevice) );
				//h_synapses_by_bundle_id_by_pre[i].push_back(d_this_bundle);
				h_synapses_by_bundle_id.push_back(d_this_bundle);
				delete [] synapse_bundle;

                sum_num_synapses += num_synapses;
                sum_memory_synapse_ids += memory_size;
                updateMeanStd(count_num_synapses, mean_num_synapses, M2_num_synapses, num_synapses);
			}
            sum_num_unique_elements += num_unique_elements;
            updateMeanStd(count_num_unique_elements, mean_num_unique_elements, M2_num_unique_elements, num_unique_elements);

		}  // end if (!scalar_delay)
		{% endif %}{# not no_or_const_delay_mode #}
	}  // end for loop through connectivity matrix
	printf("INFO connectivity matrix has size %i, number of (pre neuron ID, post neuron block) pairs is %u\n",
			size_connectivity_matrix, num_pre_post_blocks);


    {% if not no_or_const_delay_mode %}
    long unsigned int num_bundle_ids = 0;
	if (scalar_delay)
	{% endif %}
	{
		//copy temp arrays to device
		// DENIS: TODO: rename those temp1... variables AND: why sizeof(int32_t*) and not sizeof(unsigned int*) for last 3 cpys? typo? --> CHANGED!
		unsigned int* temp;
		CUDA_SAFE_CALL( cudaMalloc((void**)&temp, sizeof(unsigned int)*num_pre_post_blocks) );
		CUDA_SAFE_CALL( cudaMemcpy(temp, temp_size_by_pre_id, sizeof(unsigned int)*num_pre_post_blocks, cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol({{pathobj}}_size_by_pre, &temp, sizeof(unsigned int*)) );
		int32_t* temp2;
		CUDA_SAFE_CALL( cudaMalloc((void**)&temp2, sizeof(int32_t*)*num_pre_post_blocks) );
		CUDA_SAFE_CALL( cudaMemcpy(temp2, temp_synapses_by_pre_id, sizeof(int32_t*)*num_pre_post_blocks, cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol({{pathobj}}_synapses_id_by_pre, &temp2, sizeof(int32_t**)) );
	}
	{% if not no_or_const_delay_mode %}
	else  // not scalar_delay
    {
		// nemo bundle stuff
        assert(sum_num_synapses == syn_N);
        assert(sum_memory_synapse_ids = memory_synapse_ids);

        printf("AFTER CONN MATRIX\n");

        unsigned int *d_unique_delays;
        {
        size_t memory_size = sizeof(unsigned int) * sum_num_unique_elements;
        CUDA_SAFE_CALL( cudaMalloc((void**)&d_unique_delays, memory_size) );
        sum_memory_delay_by_pre_id += memory_size;
        }
        long unsigned int sum_num_unique_elements_bak = sum_num_unique_elements;
        sum_num_unique_elements = 0;

        printf("AFTER ALLOCATING UNIQUE DELAY BY PRE\n");
        std::cout << "Allocated memory for unique_delay_values: " << sum_memory_delay_by_pre_id * to_MB << " MB" << std::endl;

		for(int i = 0; i < num_pre_post_blocks; i++)  // loop through connectivity matrix again
		{

		    int num_elements = h_synapses_by_pre_id[i].size();
			num_unique_elements = h_unique_delay_by_pre_id[i].size();
            // TODO: here the copying to device happens, get rid of what we don't need to solve memory issues
            if(num_elements > 0)
            {
				temp_unique_delay_by_pre_id[i] = d_unique_delays + sum_num_unique_elements;
				CUDA_SAFE_CALL( cudaMemcpy(temp_unique_delay_by_pre_id[i],
					thrust::raw_pointer_cast(&(h_unique_delay_by_pre_id[i][0])),
					sizeof(unsigned int)*num_unique_elements,
					cudaMemcpyHostToDevice) );
				sum_num_unique_elements += num_unique_elements;
            }  // end if(num_elements < 0)
		}
		num_bundle_ids = sum_num_unique_elements;
		// floor(mean(h_size_by_bundle_id))
		{{pathobj}}_mean_bundle_size = sum_num_synapses / num_bundle_ids;
		//delete [] h_size_by_bundle_id_by_pre;
		//delete [] h_synapses_by_bundle_id_by_pre;

        assert(sum_num_unique_elements_bak == sum_num_unique_elements);
        printf("AFTER 2nd CONN MATRIX LOOP\n");

		// nemo bundle stuff info prints
		printf("INFO used bundle IDS %u, unused bundle IDs %u, would have unused bundle IDs with set %u, max(num_bundles) %u, delay_set.size() %u, \n",
				num_used_bundle_ids, num_pre_post_blocks * {{pathobj}}_max_num_bundles - num_used_bundle_ids,
				num_pre_post_blocks * delay_set.size() - num_used_bundle_ids,
				{{pathobj}}_max_num_bundles, delay_set.size());

        unsigned int* d_size_by_bundle_id;

        {
        size_t memory_size = sizeof(unsigned int) * num_bundle_ids;
        CUDA_SAFE_CALL( cudaMalloc((void**)&d_size_by_bundle_id, memory_size) );
        memory_size_by_bundle_id += memory_size;
        printf("AFTER ALLOCATING SIZE_BY_BUNDLE_ID\n");
        std::cout << "Allocated memory for size_by_bundle_id: " << memory_size * to_MB << " MB" << std::endl;
        //CUDA_SAFE_CALL( cudaMemcpy(d_size_by_bundle_id, h_size_by_bundle_id,
        CUDA_SAFE_CALL( cudaMemcpy(d_size_by_bundle_id, thrust::raw_pointer_cast(&h_size_by_bundle_id[0]),
                memory_size, cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemcpyToSymbol({{pathobj}}_size_by_bundle_id, &d_size_by_bundle_id, sizeof(unsigned int*)) );
        }

        {
        int32_t* d_synapses_by_bundle_id;
        size_t memory_size = sizeof(int32_t*) * num_bundle_ids;
        CUDA_SAFE_CALL( cudaMalloc((void**)&d_synapses_by_bundle_id, memory_size) );
        memory_synapses_bundle_ptrs += memory_size;
        std::cout << "Allocated memory for bundle ptrs: " << memory_size * to_MB << " MB" << std::endl;
        //CUDA_SAFE_CALL( cudaMemcpy(d_synapses_by_bundle_id, h_synapses_by_bundle_id,
        CUDA_SAFE_CALL( cudaMemcpy(d_synapses_by_bundle_id, thrust::raw_pointer_cast(&(h_synapses_by_bundle_id[0])),
                sizeof(int32_t*) * num_bundle_ids, cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemcpyToSymbol({{pathobj}}_synapses_id_by_bundle_id, &d_synapses_by_bundle_id, sizeof(int32_t**)) );
        }

        {
	    unsigned int* d_global_bundle_id_start_idx_by_pre_id;
        size_t memory_size = sizeof(unsigned int) * num_pre_post_blocks;
	    cudaMalloc((void**)&d_global_bundle_id_start_idx_by_pre_id, memory_size);
	    cudaMemcpy(d_global_bundle_id_start_idx_by_pre_id, global_bundle_id_start_idx_by_pre_id,
	    		memory_size, cudaMemcpyHostToDevice);
	    cudaMemcpyToSymbol({{pathobj}}_global_bundle_id_start_idx_by_pre, &d_global_bundle_id_start_idx_by_pre_id,
	    		sizeof(unsigned int*));
        }

        //delete [] h_synapses_by_bundle_id;
        //delete [] h_size_by_bundle_id;
    }  // end if (!scalar_delay)
    {% endif %}{# not no_or_const_delay_mode #}

	{% if not no_or_const_delay_mode %}
	if (!scalar_delay)
	{
		unsigned int* temp7;
        {
        size_t memory_size = sizeof(unsigned int)*num_pre_post_blocks;
		CUDA_SAFE_CALL( cudaMalloc((void**)&temp7, memory_size) );
        memory_unique_delay_size_by_pre += memory_size;
        std::cout << "Allocated memory for unique_delay_size_by_pre: " << memory_size * to_MB << " MB" << std::endl;
        }

		CUDA_SAFE_CALL( cudaMemcpy(temp7, temp_unique_delay_size_by_pre_id, sizeof(unsigned int)*num_pre_post_blocks, cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol({{pathobj}}_unique_delay_size_by_pre, &temp7, sizeof(unsigned int*)) );

		unsigned int* temp6;
        {
        size_t memory_size = sizeof(unsigned int*)*num_pre_post_blocks;
		CUDA_SAFE_CALL( cudaMalloc((void**)&temp6, memory_size) );
        memory_unique_delay_by_pre_ptrs += memory_size;
        std::cout << "Allocated memory for unique_delay_by_pre ptrs: " << memory_size * to_MB << " MB" << std::endl;
        }

		CUDA_SAFE_CALL( cudaMemcpy(temp6, temp_unique_delay_by_pre_id, sizeof(unsigned int*)*num_pre_post_blocks, cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol({{pathobj}}_unique_delay_by_pre, &temp6, sizeof(unsigned int**)) );

	}
	{% endif %}

    {% if not no_or_const_delay_mode %}
    if (scalar_delay)
    {% endif %}
    {
        // TODO: print memory consumption for scalar delay case
    }
    {% if not no_or_const_delay_mode %}
    else  // not scalar_delay
    {
        double std_num_synapses = getStd(count_num_synapses, M2_num_synapses);
        double std_num_unique_elements = getStd(count_num_unique_elements, M2_num_unique_elements);
        double MB_sum_memory_synapse_ids = sum_memory_synapse_ids * to_MB;
        double MB_sum_memory_delay_by_pre_id = sum_memory_delay_by_pre_id * to_MB;
        double MB_memory_size_by_bundle_id = memory_size_by_bundle_id * to_MB;  // num_bundle_ids are copied
        double MB_memory_synapses_bundle_ptrs = memory_synapses_bundle_ptrs * to_MB;  // num_bundle_ids are copied
        double MB_memory_unique_delay_size_by_pre = memory_unique_delay_size_by_pre * to_MB;  // num_pre_post_blocks are copied
        double MB_memory_unique_delay_by_pre_ptrs = memory_unique_delay_by_pre_ptrs * to_MB;  // num_pre_post_blocks are copied
        double MB_total_synapses_init = MB_sum_memory_synapse_ids +
            MB_sum_memory_delay_by_pre_id + MB_memory_size_by_bundle_id +
            MB_memory_synapses_bundle_ptrs + MB_memory_unique_delay_size_by_pre +
            MB_memory_unique_delay_by_pre_ptrs;

        // heterogeneous delays --> bundles
        printf("INFO: memory usage {{pathobj}}:\n"
               "\t bundle size over all pre/post blocks:\n"
               "\t\t mean: %.1f \t std: %.1f \t total: %lu\n"
               "\t num bundles per pre/post block:\n"
               "\t\t mean: %.1f \t std: %.1f \t total: %lu\n"
               "\t memory usage: TOTAL: %f MB\n"
               "\t\t synapse IDs (inside bundles):\n"
               "\t\t\t %f MB \t (size: %lu [total_num_synapses])\n"
               "\t\t delay_by_pre (unique delay values for all pre/post blocks):\n"
               "\t\t\t %f MB \t (size: %lu)\n"
               "\t\t bundle sizes:\n"
               "\t\t\t %f MB \t (size: %lu [num_bundles])\n"
               "\t\t ptrs to bundles:\n"
               "\t\t\t %f MB \t (size: %lu [num_bundles])\n"
               "\t\t num bundles per pre/post block:\n"
               "\t\t\t %f MB \t (size: %lu [num pre/post blocks])\n"
               "\t\t ptrs to delays per pre/post block:\n"
               "\t\t\t %f MB \t (size: %lu [num pre/post blocks])\n",
               mean_num_synapses, std_num_synapses, sum_num_synapses,
               mean_num_unique_elements, std_num_unique_elements, sum_num_unique_elements,
               MB_total_synapses_init,
               MB_sum_memory_synapse_ids, sum_num_synapses,
               MB_sum_memory_delay_by_pre_id, sum_num_unique_elements,
               MB_memory_size_by_bundle_id, num_bundle_ids,
               MB_memory_synapses_bundle_ptrs, num_bundle_ids,
               MB_memory_unique_delay_size_by_pre, num_pre_post_blocks,
               MB_memory_unique_delay_by_pre_ptrs, num_pre_post_blocks);
    }
    {% endif %}

	// Create circular eventspaces in no_or_const_delay_mode
	{% if not no_or_const_delay_mode %}
	if (scalar_delay)
	{% endif %}
	{
		{% set eventspace_variable = owner.variables[owner.eventspace_name] %}
		{% set _eventspace = get_array_name(eventspace_variable, access_data=False) %}
		unsigned int num_spikespaces = dev{{_eventspace}}.size();
		if (num_queues > num_spikespaces)
		{
			for (int i = num_spikespaces; i < num_queues; i++)
			{
				{{c_data_type(eventspace_variable.dtype)}}* new_eventspace;
				cudaError_t status = cudaMalloc((void**)&new_eventspace,
						sizeof({{c_data_type(eventspace_variable.dtype)}})*_num_{{_eventspace}});
				if (status != cudaSuccess)
				{
					printf("ERROR while allocating momory for dev{{_eventspace}}[%i] on device: %s %s %d\n",
							i, cudaGetErrorString(status), __FILE__, __LINE__);
					exit(status);
				}
				dev{{_eventspace}}.push_back(new_eventspace);
			}
		}
		// Check if we have multiple synapses per source-target pair in no_or_const_delay_mode
		if ({{owner.synapses.name}}_multiple_pre_post)
		{
			printf("WARNING Multiple synapses per source-target pair and scalar delays detected in Synapses object "
					"with name ``{{owner.synapses.name}}``. Application of synaptic effects will be "
					"serialized to avoid race conditions. Consider reformulating your "
					"model to avoid multiple synapses per source-target pair in a single Synapses object by using multiple "
					"Synapses objects instead. For scalar delays this is very likely to increase simulation "
					"performance significantly due to parallelisation of synaptic effect applications.\n");
		}
	}
	
	unsigned int num_threads = num_queues;
	if(num_threads >= max_threads_per_block)
	{
		num_threads = max_threads_per_block;
	}
    unsigned int num_blocks = 1;

    // check if we have enough ressources to call kernel with given number
    // of blocks and threads
    struct cudaFuncAttributes funcAttrib;
    cudaFuncGetAttributes(&funcAttrib, _run_{{codeobj_name}}_kernel);
    if (num_threads > funcAttrib.maxThreadsPerBlock)
    {
        // use the max num_threads before launch failure
        num_threads = funcAttrib.maxThreadsPerBlock;
        printf("WARNING Not enough ressources available to call "
               "_run_{{codeobj_name}}_kernel "
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
        printf("INFO _run_{{codeobj_name}}_kernel\n"
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

	_run_{{codeobj_name}}_kernel<<<num_blocks, num_threads>>>(
		source_N,
		num_parallel_blocks,
		num_threads,
		dt,
		syn_N,
		num_queues,
	{% if no_or_const_delay_mode %}
		true
	{% else %}
		scalar_delay
	{% endif %}
	);

	//delete temp arrays
	delete [] h_synapses_by_pre_id;
	{% if not no_or_const_delay_mode %}
	delete [] h_delay_by_pre_id;
	if (scalar_delay)
	{% endif %}
	{
		delete [] temp_size_by_pre_id;
		delete [] temp_synapses_by_pre_id;
	}
	{% if not no_or_const_delay_mode %}
	else
	{
		delete [] h_delay_count_by_pre_id;
		delete [] h_unique_delay_start_idx_by_pre_id;
		delete [] h_unique_delay_by_pre_id;
		delete [] temp_unique_delay_size_by_pre_id;
		delete [] temp_unique_delay_by_pre_id;
		delete [] global_bundle_id_start_idx_by_pre_id;

	}
	{% endif %}

	{% if no_or_const_delay_mode %}
	{{pathobj}}_scalar_delay = true;
	{% else %}
	{{pathobj}}_scalar_delay = scalar_delay;
	{% endif %}

	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		printf("ERROR initialising {{pathobj}} in %s:%d %s\n",
				__FILE__, __LINE__, cudaGetErrorString(status));
		_dealloc_arrays();
		exit(status);
	}

	CUDA_CHECK_MEMORY();
	double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
	std::cout << "INFO: {{pathobj}} initialisation took " <<  time_passed << "s";
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

{% endmacro %}

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
