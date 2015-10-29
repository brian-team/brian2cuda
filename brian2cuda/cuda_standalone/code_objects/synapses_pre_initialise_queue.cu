#include "objects.h"
#include "code_objects/synapses_pre_initialise_queue.h"

namespace {
	int num_blocks(int objects){
		return ceil(objects / (double)brian::max_threads_per_block);
	}
	int num_threads(int objects){
		return brian::max_threads_per_block;
	}
}

__global__ void _run_synapses_pre_initialise_queue_kernel(
	unsigned int _target_N,
	unsigned int _num_blocks,
	unsigned int _num_threads_per_block,
	double _dt,
	unsigned int _syn_N,
	unsigned int max_delay,
	bool new_mode)
{
	using namespace brian;

	int tid = threadIdx.x;

	synapses_pre.queue->prepare(
		tid,
		_num_threads_per_block,
		_num_blocks,
		0,
		_target_N,
		_syn_N,
		max_delay,
		synapses_pre_size_by_pre,
		synapses_pre_synapses_id_by_pre,
		synapses_pre_delay_by_pre);
	synapses_pre.no_or_const_delay_mode = new_mode;
}

//POS(queue_id, neuron_id, neurons_N)
#define OFFSET(a, b, c)	(a*c + b)

void _run_synapses_pre_initialise_queue()
{
	using namespace brian;

	
	//NeuronGroup(clock=Clock(dt=100. * usecond, name='defaultclock'), when=start, order=0, name='neurongroup')
	//neurongroup

	double dt = defaultclock.dt[0];
	unsigned int syn_N = dev_dynamic_array_synapses_pre_delay.size();
	unsigned int source_N = 100;
	unsigned int target_N = 100;

	//Create temporary host vectors
	int32_t* h_synapses_synaptic_sources = new int32_t[syn_N];
	int32_t* h_synapses_synaptic_targets = new int32_t[syn_N];
	double* h_synapses_delay = new double[syn_N];

	cudaMemcpy(h_synapses_synaptic_sources, thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]), sizeof(int32_t) * syn_N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_synapses_synaptic_targets, thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]), sizeof(int32_t) * syn_N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_synapses_delay, thrust::raw_pointer_cast(&dev_dynamic_array_synapses_pre_delay[0]), sizeof(double) * syn_N, cudaMemcpyDeviceToHost);
	thrust::host_vector<int32_t>* h_synapses_by_pre_id = new thrust::host_vector<int32_t>[num_parallel_blocks*source_N];
	thrust::host_vector<unsigned int>* h_delay_by_pre_id = new thrust::host_vector<unsigned int>[num_parallel_blocks*source_N];

	//fill vectors with pre_neuron, post_neuron, delay data
	unsigned int max_delay = 0;
	for(int syn_id = 0; syn_id < syn_N; syn_id++)
	{
		int32_t pre_neuron_id = h_synapses_synaptic_sources[syn_id] - 0;
		int32_t post_neuron_id = h_synapses_synaptic_targets[syn_id]  - 0;
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


	//create array for device pointers
	unsigned int* temp_size_by_pre_id = new unsigned int[num_parallel_blocks*source_N];
	int32_t** temp_synapses_by_pre_id = new int32_t*[num_parallel_blocks*source_N];
	unsigned int** temp_delay_by_pre_id = new unsigned int*[num_parallel_blocks*source_N];
	//fill temp arrays with device pointers
	for(int i = 0; i < num_parallel_blocks*source_N; i++)
	{
		int num_elements = h_synapses_by_pre_id[i].size();
		temp_size_by_pre_id[i] = num_elements;
		if(num_elements > 0)
		{
			cudaMalloc((void**)&temp_synapses_by_pre_id[i], sizeof(int32_t)*num_elements);
			cudaMalloc((void**)&temp_delay_by_pre_id[i], sizeof(unsigned int)*num_elements);
			cudaMemcpy(temp_synapses_by_pre_id[i],
				thrust::raw_pointer_cast(&(h_synapses_by_pre_id[i][0])),
				sizeof(int32_t)*num_elements,
				cudaMemcpyHostToDevice);
			cudaMemcpy(temp_delay_by_pre_id[i],
				thrust::raw_pointer_cast(&(h_delay_by_pre_id[i][0])),
				sizeof(unsigned int)*num_elements,
				cudaMemcpyHostToDevice);
		}
	}

	//copy temp arrays to device
	unsigned int* temp;
	cudaMalloc((void**)&temp, sizeof(unsigned int)*num_parallel_blocks*source_N);
	cudaMemcpy(temp, temp_size_by_pre_id, sizeof(unsigned int)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(synapses_pre_size_by_pre, &temp, sizeof(unsigned int*));
	int32_t* temp2;
	cudaMalloc((void**)&temp2, sizeof(int32_t*)*num_parallel_blocks*source_N);
	cudaMemcpy(temp2, temp_synapses_by_pre_id, sizeof(int32_t*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(synapses_pre_synapses_id_by_pre, &temp2, sizeof(int32_t**));
	unsigned int* temp3;
	cudaMalloc((void**)&temp3, sizeof(unsigned int*)*num_parallel_blocks*source_N);
	cudaMemcpy(temp3, temp_delay_by_pre_id, sizeof(int32_t*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(synapses_pre_delay_by_pre, &temp3, sizeof(unsigned int**));
	
	unsigned int num_threads = max_delay;
	if(num_threads >= max_threads_per_block)
	{
		num_threads = max_threads_per_block;
	}
	_run_synapses_pre_initialise_queue_kernel<<<1, num_threads>>>(
		source_N,
		num_parallel_blocks,
		max_threads_per_block,
		dt,
		syn_N,
		max_delay,
		false
	);

	//delete temp arrays
	delete [] h_synapses_synaptic_sources;
	delete [] h_synapses_synaptic_targets;
	delete [] h_synapses_delay;
	delete [] h_synapses_by_pre_id;
	delete [] h_delay_by_pre_id;
	delete [] temp_size_by_pre_id;
	delete [] temp_synapses_by_pre_id;
	delete [] temp_delay_by_pre_id;

}

