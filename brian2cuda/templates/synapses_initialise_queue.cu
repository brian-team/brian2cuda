{% macro cu_file() %}
{# USES_VARIABLES { delay } #}
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <set>
#include "code_objects/{{codeobj_name}}.h"
{% set pathobj = owner.name %}

namespace {
	int num_blocks(int objects){
		return ceil(objects / (double)brian::max_threads_per_block);
	}
	int num_threads(int objects){
		return brian::max_threads_per_block;
	}
}

__global__ void _run_{{codeobj_name}}_kernel(
	unsigned int _source_N,
	unsigned int _num_blocks,
	unsigned int _num_threads,
	double _dt,
	unsigned int _syn_N,
	unsigned int num_queues,
	unsigned int max_num_delays_per_block,
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
		max_num_delays_per_block,
		//{{pathobj}}_size_by_pre,
		{{pathobj}}_size_by_bundle_id,
		{{pathobj}}_unique_delay_size_by_pre,
		//{{pathobj}}_synapses_id_by_pre,
		{{pathobj}}_synapses_id_by_bundle_id,
		{{pathobj}}_unique_delay_by_pre);
		//{{pathobj}}_unique_delay_start_idx_by_pre);
	{{pathobj}}.no_or_const_delay_mode = new_mode;
}

//POS(queue_id, neuron_id, neurons_N)
#define OFFSET(a, b, c)	(a*c + b)

void _run_{{pathobj}}_initialise_queue()
{
	using namespace brian;

	{# we don't use {{N}} to avoid using {{pointer_lines}} which only work inside kernels with %DEVICE_PARAMETERS% #}
	unsigned int syn_N = {{get_array_name(owner.variables['N'], access_data=False)}}[0];
	if (syn_N == 0)
		return;

	double dt = {{owner.clock.name}}.dt[0];
	unsigned int source_N = {{owner.source.N}};
	unsigned int target_N = {{owner.target.N}};

	// DENIS: TODO check speed difference when using thrust host vectors instead for easier readability and programming comfort, e.g.:
	// thrust::host_vector<int32_t> h_synapses_synaptic_sources = dev_dynamic_array_{{owner.synapses.name}}_{{owner.synapse_sources.name}}

	///////////////////////////////////
	// Create temporary host vectors //
	///////////////////////////////////

	// pre neuron IDs, post neuron IDs and delays for all synapses (sorted by synapse IDs)
	//TODO: for multiple SynapticPathways for the same Synapses object (on_pre and on_post) the following copy is identical in both pathways initialise templates
	{% if no_or_const_delay_mode %}
	// delay (on host) was potentially set in main and needs to be copied to device for later use
	dev{{_dynamic_delay}} = {{_dynamic_delay}};
	{% else %}
	// delay (on device) was set in group_variable_set_conditional and needs to be copied to host
	{{_dynamic_delay}} = dev{{_dynamic_delay}};
	{% endif %}

	// synapse IDs and delays in connectivity matrix, projected to 1D arrays of vectors
	// sorted first by pre neuron ID, then by cuda blocks (corresponding to groups of post neuron IDs)
	// the index for one pre neuron ID and block ID is: ( pre_neuron_ID * num_blocks + block_ID )

	// vectors store synapse IDs and delays for each synapse, will be sorted by delay
	thrust::host_vector<int32_t>* h_synapses_by_pre_id = new thrust::host_vector<int32_t>[num_parallel_blocks*source_N];
	{% if not no_or_const_delay_mode %}
	thrust::host_vector<unsigned int>* h_delay_by_pre_id = new thrust::host_vector<unsigned int>[num_parallel_blocks*source_N];
	// nemo bundle stuff
	// this is one vector of host pointers per right_offset (since we don't know how many bundles we will have per right_offset)
	thrust::host_vector<int32_t*>* h_synapses_by_bundle_id_by_pre = new thrust::host_vector<int32_t*>[num_parallel_blocks * source_N];
	thrust::host_vector<unsigned int>* h_size_by_bundle_id_by_pre = new thrust::host_vector<unsigned int>[num_parallel_blocks * source_N];
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

	///////////////////////////////////////
	// Create arrays for device pointers //
	///////////////////////////////////////

	// TODO rename temp
	//unsigned int* temp_size_by_pre_id = new unsigned int[num_parallel_blocks*source_N];
	//int32_t** temp_synapses_by_pre_id = new int32_t*[num_parallel_blocks*source_N];

	{% if not no_or_const_delay_mode %}
	int num_unique_elements;
	unsigned int* temp_unique_delay_size_by_pre_id;
	//unsigned int** temp_unique_delay_start_idx_by_pre_id;
	unsigned int** temp_unique_delay_by_pre_id;
	// vectors store only unique set of delays and the corresponding start index in the h_delay_by_pre_id vectors
	thrust::host_vector<unsigned int>* h_delay_count_by_pre_id;
	thrust::host_vector<unsigned int>* h_unique_delay_start_idx_by_pre_id;
	thrust::host_vector<unsigned int>* h_unique_delay_by_pre_id;
	if (!scalar_delay)
	{
		// allocate memory only if the delays are not all the same
		temp_unique_delay_size_by_pre_id = new unsigned int[num_parallel_blocks*source_N];
		//temp_unique_delay_start_idx_by_pre_id =  new unsigned int*[num_parallel_blocks*source_N];
		temp_unique_delay_by_pre_id =  new unsigned int*[num_parallel_blocks*source_N];

		h_delay_count_by_pre_id = new thrust::host_vector<unsigned int>[num_parallel_blocks*source_N];
		h_unique_delay_start_idx_by_pre_id = new thrust::host_vector<unsigned int>[num_parallel_blocks*source_N];
		h_unique_delay_by_pre_id = new thrust::host_vector<unsigned int>[num_parallel_blocks*source_N];
	}
	{% endif %}


	int size_connectivity_matrix = 0;
	unsigned int num_used_bundle_ids = 0;
	//fill temp arrays with device pointers
	for(int i = 0; i < num_parallel_blocks*source_N; i++)  // loop through connectivity matrix
	{
		int num_elements = h_synapses_by_pre_id[i].size();
		size_connectivity_matrix += num_elements;
		//temp_size_by_pre_id[i] = num_elements;
		if (num_elements > {{pathobj}}_max_size)
			{{pathobj}}_max_size = num_elements;

		{% if not no_or_const_delay_mode %}
		{# delay was set using Synapses object's delay attribute: `conn = Synapses(...); conn.delay = ...` #}
		if (!scalar_delay)
		{# all delays have the same value, e.g. `conn.delay = 2*ms` or because of small jitter + rounding to dt #}
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
			// the local bundle_idx goes from 0 to num_bundles for each i (right_offset)
			assert(num_unique_elements <= delay_set.size());
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
				h_size_by_bundle_id_by_pre[i].push_back(num_synapses);
				int32_t* synapse_bundle = new int32_t[num_synapses];
				// TODO: don't copy synapses to synapse_bundle, just cudaMemcpy it directly to the device with
				// cudaMemcpy(d_..., h_synapses_by_pre_id[i] + synapses_start_idx, ...)
				for (int j = 0; j < num_synapses; j++)
				{
					synapse_bundle[j] = h_synapses_by_pre_id[i][synapses_start_idx + j];
				}
				// copy this bundle to device
				int32_t* d_synapse_bundle;
				unsigned int memory_size = sizeof(int32_t) * num_synapses;
				cudaMalloc((void**)&d_synapse_bundle, memory_size);
				cudaMemcpy(d_synapse_bundle, synapse_bundle, memory_size, cudaMemcpyHostToDevice);
				h_synapses_by_bundle_id_by_pre[i].push_back(d_synapse_bundle);
				delete [] synapse_bundle;
			}
		}  // end if (!scalar_delay)
		{% endif %}{# not no_or_const_delay_mode #}

		// TODO: here the copying to device happens, get rid of what we don't need to solve memory issues
		if(num_elements > 0)
		{
			//cudaMalloc((void**)&temp_synapses_by_pre_id[i], sizeof(int32_t)*num_elements);
			//cudaMemcpy(temp_synapses_by_pre_id[i],
			//	thrust::raw_pointer_cast(&(h_synapses_by_pre_id[i][0])),
			//	sizeof(int32_t)*num_elements,
			//	cudaMemcpyHostToDevice);

			{% if not no_or_const_delay_mode %}
			if (!scalar_delay)
			{
				//cudaMalloc((void**)&temp_unique_delay_start_idx_by_pre_id[i], sizeof(unsigned int)*num_unique_elements);
				cudaMalloc((void**)&temp_unique_delay_by_pre_id[i], sizeof(unsigned int)*num_unique_elements);
				//cudaMemcpy(temp_unique_delay_start_idx_by_pre_id[i],
				//	thrust::raw_pointer_cast(&(h_unique_delay_start_idx_by_pre_id[i][0])),
				//	sizeof(unsigned int)*num_unique_elements,
				//	cudaMemcpyHostToDevice);
				cudaMemcpy(temp_unique_delay_by_pre_id[i],
					thrust::raw_pointer_cast(&(h_unique_delay_by_pre_id[i][0])),
					sizeof(unsigned int)*num_unique_elements,
					cudaMemcpyHostToDevice);
			}
			{% endif %}
		}  // end if(num_elements < 0)
	}  // end for loop through connectivity matrix
	printf("INFO connectivity matrix has size %i, number of (pre neuron ID, post neuron block) pairs is %u\n",
			size_connectivity_matrix, num_parallel_blocks * source_N);

	// nemo bundle stuff
	{% if not no_or_const_delay_mode %}
	if (!scalar_delay)
	{
		unsigned int num_bundle_ids = num_parallel_blocks * source_N * {{pathobj}}_max_num_bundles;
		int32_t** h_synapses_by_bundle_id = new int32_t*[num_bundle_ids];
		unsigned int* h_size_by_bundle_id = new unsigned int[num_bundle_ids];
		unsigned int bundle_size_sum = 0;
		for(int i = 0; i < num_parallel_blocks*source_N; i++)  // loop through connectivity matrix again
		{
			unsigned int num_unique_elements = h_size_by_bundle_id_by_pre[i].size();
			for (int j = 0; j < {{pathobj}}_max_num_bundles; j++)
			{
				int idx = i * {{pathobj}}_max_num_bundles + j;
				if (j < num_unique_elements)
				{
					unsigned int size = h_size_by_bundle_id_by_pre[i][j];
					h_size_by_bundle_id[idx] = size;
					if (size > {{pathobj}}_max_bundle_size)
					{{pathobj}}_max_bundle_size = size;
					bundle_size_sum += size;
					h_synapses_by_bundle_id[idx] = h_synapses_by_bundle_id_by_pre[i][j];
				}
				else
				{
					h_size_by_bundle_id[idx] = 0;
					h_synapses_by_bundle_id[idx] = NULL;
				}
			}
		}
		// floor(mean(h_size_by_bundle_id))
		{{pathobj}}_mean_bundle_size = bundle_size_sum / num_bundle_ids;
		delete [] h_size_by_bundle_id_by_pre;
		delete [] h_synapses_by_bundle_id_by_pre;

		// nemo bundle stuff info prints
		printf("INFO used bundle IDS %u, unused bundle IDs %u, would have unused bundle IDs with set %u, max(num_bundles) %u, delay_set.size() %u, \n",
				num_used_bundle_ids, num_parallel_blocks * source_N * {{pathobj}}_max_num_bundles - num_used_bundle_ids,
				num_parallel_blocks * source_N * delay_set.size() - num_used_bundle_ids,
				{{pathobj}}_max_num_bundles, delay_set.size());

        unsigned int* d_size_by_bundle_id;
        cudaMalloc((void**)&d_size_by_bundle_id, sizeof(unsigned int) * num_bundle_ids);
        cudaMemcpy(d_size_by_bundle_id, h_size_by_bundle_id,
                sizeof(unsigned int) * num_bundle_ids, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol({{pathobj}}_size_by_bundle_id, &d_size_by_bundle_id, sizeof(unsigned int*));
        int32_t* d_synapses_by_bundle_id;
        cudaMalloc((void**)&d_synapses_by_bundle_id, sizeof(int32_t*) * num_bundle_ids);
        cudaMemcpy(d_synapses_by_bundle_id, h_synapses_by_bundle_id,
                sizeof(int32_t*) * num_bundle_ids, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol({{pathobj}}_synapses_id_by_bundle_id, &d_synapses_by_bundle_id, sizeof(int32_t**));

        delete [] h_synapses_by_bundle_id;
        delete [] h_size_by_bundle_id;
	}
	{% endif %}

	//copy temp arrays to device
	// DENIS: TODO: rename those temp1... variables AND: why sizeof(int32_t*) and not sizeof(unsigned int*) for last 3 cpys? typo? --> CHANGED!
	//unsigned int* temp;
	//cudaMalloc((void**)&temp, sizeof(unsigned int)*num_parallel_blocks*source_N);
	//cudaMemcpy(temp, temp_size_by_pre_id, sizeof(unsigned int)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol({{pathobj}}_size_by_pre, &temp, sizeof(unsigned int*));
	//int32_t* temp2;
	//cudaMalloc((void**)&temp2, sizeof(int32_t*)*num_parallel_blocks*source_N);
	//cudaMemcpy(temp2, temp_synapses_by_pre_id, sizeof(int32_t*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol({{pathobj}}_synapses_id_by_pre, &temp2, sizeof(int32_t**));

	{% if not no_or_const_delay_mode %}
	if (!scalar_delay)
	{
		unsigned int* temp7;
		cudaMalloc((void**)&temp7, sizeof(unsigned int)*num_parallel_blocks*source_N);
		cudaMemcpy(temp7, temp_unique_delay_size_by_pre_id, sizeof(unsigned int)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol({{pathobj}}_unique_delay_size_by_pre, &temp7, sizeof(unsigned int*));
		//unsigned int* temp5;
		//cudaMalloc((void**)&temp5, sizeof(unsigned int*)*num_parallel_blocks*source_N);
		//cudaMemcpy(temp5, temp_unique_delay_start_idx_by_pre_id, sizeof(unsigned int*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
		//cudaMemcpyToSymbol({{pathobj}}_unique_delay_start_idx_by_pre, &temp5, sizeof(unsigned int**));
		unsigned int* temp6;
		cudaMalloc((void**)&temp6, sizeof(unsigned int*)*num_parallel_blocks*source_N);
		cudaMemcpy(temp6, temp_unique_delay_by_pre_id, sizeof(unsigned int*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol({{pathobj}}_unique_delay_by_pre, &temp6, sizeof(unsigned int**));
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
		1,
		true
	{% else %}
		{{pathobj}}_max_num_bundles,
		scalar_delay
	{% endif %}
	);

	//delete temp arrays
	delete [] h_synapses_by_pre_id;
	//delete [] temp_size_by_pre_id;
	//delete [] temp_synapses_by_pre_id;
	{% if not no_or_const_delay_mode %}
	delete [] h_delay_by_pre_id;
	if (!scalar_delay)
	{
		delete [] h_delay_count_by_pre_id;
		delete [] h_unique_delay_start_idx_by_pre_id;
		delete [] h_unique_delay_by_pre_id;
		delete [] temp_unique_delay_size_by_pre_id;
		//delete [] temp_unique_delay_start_idx_by_pre_id;
		delete [] temp_unique_delay_by_pre_id;

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
}

{% endmacro %}

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
