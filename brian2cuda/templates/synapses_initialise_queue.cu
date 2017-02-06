{% macro cu_file() %}
{# USES_VARIABLES { N } #}
#include <thrust/sort.h>
#include <thrust/reduce.h>
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
	unsigned int _num_threads_per_block,
	double _dt,
	unsigned int _syn_N,
	unsigned int max_delay,
	bool new_mode)
{
	using namespace brian;

	int tid = threadIdx.x;

	{{pathobj}}.queue->prepare(
		tid,
		_num_threads_per_block,
		_num_blocks,
		0,
		_source_N,
		_syn_N,
		max_delay,
		{{pathobj}}_size_by_pre,
		{{pathobj}}_unique_delay_size_by_pre,
		{{pathobj}}_synapses_id_by_pre,
		// TODO: delete _delay_by_pre
		{{pathobj}}_delay_by_pre,
		{{pathobj}}_unique_delay_by_pre,
		{{pathobj}}_unique_delay_start_idx_by_pre);
	{{pathobj}}.no_or_const_delay_mode = new_mode;
}

//POS(queue_id, neuron_id, neurons_N)
#define OFFSET(a, b, c)	(a*c + b)

void _run_{{pathobj}}_initialise_queue()
{
	using namespace brian;

	{% if no_or_const_delay_mode %}
	unsigned int save_num_blocks = num_parallel_blocks;
	num_parallel_blocks = 1;
	{% endif %}

	{{pointers_lines|autoindent}}

	double dt = {{owner.clock.name}}.dt[0];
	unsigned int syn_N = {{N}};
	unsigned int source_N = {{owner.source.N}};
	unsigned int target_N = {{owner.target.N}};

	// DENIS: TODO check speed difference when using thrust host vectors instead for easier readability and programming comfort, e.g.:
	// thrust::host_vector<int32_t> h_synapses_synaptic_sources = dev_dynamic_array_{{owner.synapses.name}}_{{owner.synapse_sources.name}}

	///////////////////////////////////
	// Create temporary host vectors //
	///////////////////////////////////

	// pre neuron IDs, post neuron IDs and delays for all synapses (sorted by synapse IDs)
	int32_t* h_synapses_synaptic_sources = new int32_t[syn_N];
	int32_t* h_synapses_synaptic_targets = new int32_t[syn_N];
	double* h_synapses_delay = new double[syn_N];

	// synapse IDs and delays in connectivity matrix, projected to 1D arrays of vectors
	// sorted first by pre neuron ID, then by cuda blocks (corresponding to groups of post neuron IDs)
	// the index for one pre neuron ID and block ID is: ( pre_neuron_ID * num_blocks + block_ID )

	// vector store synapse IDs and delays for each synapse, will be sorted by delay
	thrust::host_vector<int32_t>* h_synapses_by_pre_id = new thrust::host_vector<int32_t>[num_parallel_blocks*source_N];
	thrust::host_vector<unsigned int>* h_delay_by_pre_id = new thrust::host_vector<unsigned int>[num_parallel_blocks*source_N];

	// vectors store only unique set of delays and the corresponding start index in the h_delay_by_pre_id vectors
	thrust::host_vector<unsigned int>* h_delay_count_by_pre_id = new thrust::host_vector<unsigned int>[num_parallel_blocks*source_N];
	thrust::host_vector<unsigned int>* h_unique_delay_start_idx_by_pre_id = new thrust::host_vector<unsigned int>[num_parallel_blocks*source_N];
	thrust::host_vector<unsigned int>* h_unique_delay_by_pre_id = new thrust::host_vector<unsigned int>[num_parallel_blocks*source_N];

	// get neuron IDs and delays from device
	cudaMemcpy(h_synapses_synaptic_sources, thrust::raw_pointer_cast(&dev_dynamic_array_{{owner.synapses.name}}_{{owner.synapse_sources.name}}[0]), sizeof(int32_t) * syn_N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_synapses_synaptic_targets, thrust::raw_pointer_cast(&dev_dynamic_array_{{owner.synapses.name}}_{{owner.synapse_targets.name}}[0]), sizeof(int32_t) * syn_N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_synapses_delay, thrust::raw_pointer_cast(&dev_dynamic_array_{{owner.synapses.name}}_delay[0]), sizeof(double) * syn_N, cudaMemcpyDeviceToHost);

	//fill vectors of connectivity matrix with synapse IDs and delay IDs (in units of simulation time step)
	unsigned int max_delay = 0;
	for(int syn_id = 0; syn_id < syn_N; syn_id++)  // loop through all synapses
	{
		// pre/post_neuron_id are integers from 0 to Nsource/Ntarget (from corresponding SynapticPathway)
		// this is relevant only when using Subgroups where they might be NOT equal to the idx in their NeuronGroup
		int32_t pre_neuron_id = h_synapses_synaptic_sources[syn_id] - {{owner.source.start}};
		int32_t post_neuron_id = h_synapses_synaptic_targets[syn_id]  - {{owner.target.start}};
		unsigned int delay = (int)(h_synapses_delay[syn_id] / dt + 0.5);
		if(delay > max_delay)
		{
			max_delay = delay;
		}
		unsigned int right_queue = (post_neuron_id*num_parallel_blocks)/target_N;
		unsigned int right_offset = pre_neuron_id * num_parallel_blocks + right_queue;
		h_synapses_by_pre_id[right_offset].push_back(syn_id);
		h_delay_by_pre_id[right_offset].push_back(delay);
	}
	max_delay++;	//we also need a current step

	///////////////////////////////////////
	// Create arrays for device pointers //
	///////////////////////////////////////

	// TODO rename temp
	unsigned int* temp_size_by_pre_id = new unsigned int[num_parallel_blocks*source_N];
	unsigned int* temp_unique_delay_size_by_pre_id = new unsigned int[num_parallel_blocks*source_N];
	int32_t** temp_synapses_by_pre_id = new int32_t*[num_parallel_blocks*source_N];
	unsigned int** temp_delay_by_pre_id = new unsigned int*[num_parallel_blocks*source_N];
	unsigned int** temp_delay_count_by_pre_id =  new unsigned int*[num_parallel_blocks*source_N];
	unsigned int** temp_unique_delay_start_idx_by_pre_id =  new unsigned int*[num_parallel_blocks*source_N];
	unsigned int** temp_unique_delay_by_pre_id =  new unsigned int*[num_parallel_blocks*source_N];


	//fill temp arrays with device pointers
	for(int i = 0; i < num_parallel_blocks*source_N; i++)  // loop through connectivity matrix
	{
		int num_elements = h_synapses_by_pre_id[i].size();
		temp_size_by_pre_id[i] = num_elements;
		if (num_elements > {{pathobj}}_max_size)
			{{pathobj}}_max_size = num_elements;

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
		

		int num_unique_elements = h_unique_delay_by_pre_id[i].size();
		temp_unique_delay_size_by_pre_id[i] = num_unique_elements;
		if (num_unique_elements > {{pathobj}}_max_unique_delay_size)
			{{pathobj}}_max_unique_delay_size = num_unique_elements;

		if(num_elements > 0)
		{
			cudaMalloc((void**)&temp_synapses_by_pre_id[i], sizeof(int32_t)*num_elements);
			cudaMalloc((void**)&temp_delay_by_pre_id[i], sizeof(unsigned int)*num_elements);
			cudaMalloc((void**)&temp_delay_count_by_pre_id[i], sizeof(unsigned int)*num_unique_elements);
			cudaMalloc((void**)&temp_unique_delay_start_idx_by_pre_id[i], sizeof(unsigned int)*num_unique_elements);
			cudaMalloc((void**)&temp_unique_delay_by_pre_id[i], sizeof(unsigned int)*num_unique_elements);
			cudaMemcpy(temp_synapses_by_pre_id[i],
				thrust::raw_pointer_cast(&(h_synapses_by_pre_id[i][0])),
				sizeof(int32_t)*num_elements,
				cudaMemcpyHostToDevice);
			cudaMemcpy(temp_delay_by_pre_id[i],
				thrust::raw_pointer_cast(&(h_delay_by_pre_id[i][0])),
				sizeof(unsigned int)*num_elements,
				cudaMemcpyHostToDevice);
			cudaMemcpy(temp_delay_count_by_pre_id[i],
				thrust::raw_pointer_cast(&(h_delay_count_by_pre_id[i][0])),
				sizeof(unsigned int)*num_unique_elements,
				cudaMemcpyHostToDevice);
			cudaMemcpy(temp_unique_delay_start_idx_by_pre_id[i],
				thrust::raw_pointer_cast(&(h_unique_delay_start_idx_by_pre_id[i][0])),
				sizeof(unsigned int)*num_unique_elements,
				cudaMemcpyHostToDevice);
			cudaMemcpy(temp_unique_delay_by_pre_id[i],
				thrust::raw_pointer_cast(&(h_unique_delay_by_pre_id[i][0])),
				sizeof(unsigned int)*num_unique_elements,
				cudaMemcpyHostToDevice);
		}
	}


	//copy temp arrays to device
	// DENIS: TODO: rename those temp1... variables AND: why sizeof(int32_t*) and not sizeof(unsigned int*) for last 3 cpys? typo? --> CHANGED!
	unsigned int* temp;
	cudaMalloc((void**)&temp, sizeof(unsigned int)*num_parallel_blocks*source_N);
	cudaMemcpy(temp, temp_size_by_pre_id, sizeof(unsigned int)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol({{pathobj}}_size_by_pre, &temp, sizeof(unsigned int*));
	unsigned int* temp7;
	cudaMalloc((void**)&temp7, sizeof(unsigned int)*num_parallel_blocks*source_N);
	cudaMemcpy(temp7, temp_unique_delay_size_by_pre_id, sizeof(unsigned int)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol({{pathobj}}_unique_delay_size_by_pre, &temp7, sizeof(unsigned int*));
	int32_t* temp2;
	cudaMalloc((void**)&temp2, sizeof(int32_t*)*num_parallel_blocks*source_N);
	cudaMemcpy(temp2, temp_synapses_by_pre_id, sizeof(int32_t*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol({{pathobj}}_synapses_id_by_pre, &temp2, sizeof(int32_t**));
	unsigned int* temp3;
	cudaMalloc((void**)&temp3, sizeof(unsigned int*)*num_parallel_blocks*source_N);
	cudaMemcpy(temp3, temp_delay_by_pre_id, sizeof(unsigned int*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol({{pathobj}}_delay_by_pre, &temp3, sizeof(unsigned int**));
	unsigned int* temp4;
	cudaMalloc((void**)&temp4, sizeof(unsigned int*)*num_parallel_blocks*source_N);
	cudaMemcpy(temp4, temp_delay_count_by_pre_id, sizeof(unsigned int*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol({{pathobj}}_delay_count_by_pre, &temp4, sizeof(unsigned int**));
	unsigned int* temp5;
	cudaMalloc((void**)&temp5, sizeof(unsigned int*)*num_parallel_blocks*source_N);
	cudaMemcpy(temp5, temp_unique_delay_start_idx_by_pre_id, sizeof(unsigned int*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol({{pathobj}}_unique_delay_start_idx_by_pre, &temp5, sizeof(unsigned int**));
	unsigned int* temp6;
	cudaMalloc((void**)&temp6, sizeof(unsigned int*)*num_parallel_blocks*source_N);
	cudaMemcpy(temp6, temp_unique_delay_by_pre_id, sizeof(unsigned int*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol({{pathobj}}_unique_delay_by_pre, &temp6, sizeof(unsigned int**));

	
	unsigned int num_threads = max_delay;
	if(num_threads >= max_threads_per_block)
	{
		num_threads = max_threads_per_block;
	}
	_run_{{codeobj_name}}_kernel<<<1, num_threads>>>(
		source_N,
		num_parallel_blocks,
		max_threads_per_block,
		dt,
		syn_N,
		max_delay,
	{% if no_or_const_delay_mode %}
		true
	{% else %}
		false
	{% endif %}
	);

	//delete temp arrays
	delete [] h_synapses_synaptic_sources;
	delete [] h_synapses_synaptic_targets;
	delete [] h_synapses_delay;
	delete [] h_synapses_by_pre_id;
	delete [] h_delay_by_pre_id;
	delete [] h_delay_count_by_pre_id;
	delete [] h_unique_delay_start_idx_by_pre_id;
	delete [] h_unique_delay_by_pre_id;
	delete [] temp_size_by_pre_id;
	delete [] temp_unique_delay_size_by_pre_id;
	delete [] temp_synapses_by_pre_id;
	delete [] temp_delay_by_pre_id;
	delete [] temp_delay_count_by_pre_id;
	delete [] temp_unique_delay_start_idx_by_pre_id;
	delete [] temp_unique_delay_by_pre_id;

	{% if no_or_const_delay_mode %}
	num_parallel_blocks = save_num_blocks;
	{% endif %}
}

{% endmacro %}

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
