
#include<stdint.h>
#include<vector>
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "network.h"
#include<iostream>
#include<fstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>

//////////////// clocks ///////////////////
Clock brian::defaultclock(0.0001);

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

uint64_t * brian::_array_defaultclock_timestep;
uint64_t * brian::dev_array_defaultclock_timestep;
__device__ uint64_t * brian::d_array_defaultclock_timestep;
const int brian::_num__array_defaultclock_timestep = 1;

int32_t * brian::_array_neurongroup__spikespace;
int32_t * brian::dev_array_neurongroup__spikespace;
__device__ int32_t * brian::d_array_neurongroup__spikespace;
const int brian::_num__array_neurongroup__spikespace = 101;

double * brian::_array_neurongroup_g;
double * brian::dev_array_neurongroup_g;
__device__ double * brian::d_array_neurongroup_g;
const int brian::_num__array_neurongroup_g = 100;

int32_t * brian::_array_neurongroup_i;
int32_t * brian::dev_array_neurongroup_i;
__device__ int32_t * brian::d_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 100;

double * brian::_array_neurongroup_lastspike;
double * brian::dev_array_neurongroup_lastspike;
__device__ double * brian::d_array_neurongroup_lastspike;
const int brian::_num__array_neurongroup_lastspike = 100;

char * brian::_array_neurongroup_not_refractory;
char * brian::dev_array_neurongroup_not_refractory;
__device__ char * brian::d_array_neurongroup_not_refractory;
const int brian::_num__array_neurongroup_not_refractory = 100;

double * brian::_array_neurongroup_v;
double * brian::dev_array_neurongroup_v;
__device__ double * brian::d_array_neurongroup_v;
const int brian::_num__array_neurongroup_v = 100;

int32_t * brian::_array_spikegeneratorgroup__lastindex;
int32_t * brian::dev_array_spikegeneratorgroup__lastindex;
__device__ int32_t * brian::d_array_spikegeneratorgroup__lastindex;
const int brian::_num__array_spikegeneratorgroup__lastindex = 1;

int32_t * brian::_array_spikegeneratorgroup__spikespace;
int32_t * brian::dev_array_spikegeneratorgroup__spikespace;
__device__ int32_t * brian::d_array_spikegeneratorgroup__spikespace;
const int brian::_num__array_spikegeneratorgroup__spikespace = 101;

int32_t * brian::_array_spikegeneratorgroup_i;
int32_t * brian::dev_array_spikegeneratorgroup_i;
__device__ int32_t * brian::d_array_spikegeneratorgroup_i;
const int brian::_num__array_spikegeneratorgroup_i = 100;

double * brian::_array_spikegeneratorgroup_period;
double * brian::dev_array_spikegeneratorgroup_period;
__device__ double * brian::d_array_spikegeneratorgroup_period;
const int brian::_num__array_spikegeneratorgroup_period = 1;

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

int32_t * brian::_array_statemonitor_1__indices;
int32_t * brian::dev_array_statemonitor_1__indices;
__device__ int32_t * brian::d_array_statemonitor_1__indices;
const int brian::_num__array_statemonitor_1__indices = 10;

int32_t * brian::_array_statemonitor_1_N;
int32_t * brian::dev_array_statemonitor_1_N;
__device__ int32_t * brian::d_array_statemonitor_1_N;
const int brian::_num__array_statemonitor_1_N = 1;

double * brian::_array_statemonitor_1_v;
double * brian::dev_array_statemonitor_1_v;
__device__ double * brian::d_array_statemonitor_1_v;
const int brian::_num__array_statemonitor_1_v = (0, 10);

int32_t * brian::_array_statemonitor__indices;
int32_t * brian::dev_array_statemonitor__indices;
__device__ int32_t * brian::d_array_statemonitor__indices;
const int brian::_num__array_statemonitor__indices = 10;

int32_t * brian::_array_statemonitor_N;
int32_t * brian::dev_array_statemonitor_N;
__device__ int32_t * brian::d_array_statemonitor_N;
const int brian::_num__array_statemonitor_N = 1;

double * brian::_array_statemonitor_w;
double * brian::dev_array_statemonitor_w;
__device__ double * brian::d_array_statemonitor_w;
const int brian::_num__array_statemonitor_w = (0, 10);

int32_t * brian::_array_synapses_1_N;
int32_t * brian::dev_array_synapses_1_N;
__device__ int32_t * brian::d_array_synapses_1_N;
const int brian::_num__array_synapses_1_N = 1;

int32_t * brian::_array_synapses_N;
int32_t * brian::dev_array_synapses_N;
__device__ int32_t * brian::d_array_synapses_N;
const int brian::_num__array_synapses_N = 1;


//////////////// dynamic arrays 1d /////////
std::vector<int32_t> brian::_dynamic_array_spikegeneratorgroup_neuron_index;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup_neuron_index;
std::vector<int32_t> brian::_dynamic_array_spikegeneratorgroup_spike_number;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup_spike_number;
std::vector<double> brian::_dynamic_array_spikegeneratorgroup_spike_time;
thrust::device_vector<double> brian::dev_dynamic_array_spikegeneratorgroup_spike_time;
std::vector<int32_t> brian::_dynamic_array_spikemonitor_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_i;
std::vector<double> brian::_dynamic_array_spikemonitor_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_t;
std::vector<double> brian::_dynamic_array_statemonitor_1_t;
thrust::device_vector<double> brian::dev_dynamic_array_statemonitor_1_t;
std::vector<double> brian::_dynamic_array_statemonitor_t;
thrust::device_vector<double> brian::dev_dynamic_array_statemonitor_t;
std::vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_post;
std::vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_pre;
std::vector<double> brian::_dynamic_array_synapses_1_lastupdate;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_lastupdate;
std::vector<int32_t> brian::_dynamic_array_synapses_1_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_incoming;
std::vector<int32_t> brian::_dynamic_array_synapses_1_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_outgoing;
std::vector<double> brian::_dynamic_array_synapses_1_pre_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_pre_delay;
std::vector<int32_t> brian::_dynamic_array_synapses_1_pre_spiking_synapses;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_pre_spiking_synapses;
std::vector<double> brian::_dynamic_array_synapses_1_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_w;
std::vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_post;
std::vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_pre;
std::vector<double> brian::_dynamic_array_synapses_Apost;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_Apost;
std::vector<double> brian::_dynamic_array_synapses_Apre;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_Apre;
std::vector<double> brian::_dynamic_array_synapses_lastupdate;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_lastupdate;
std::vector<int32_t> brian::_dynamic_array_synapses_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_incoming;
std::vector<int32_t> brian::_dynamic_array_synapses_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_outgoing;
std::vector<double> brian::_dynamic_array_synapses_post_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_post_delay;
std::vector<int32_t> brian::_dynamic_array_synapses_post_spiking_synapses;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_post_spiking_synapses;
std::vector<double> brian::_dynamic_array_synapses_pre_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_pre_delay;
std::vector<int32_t> brian::_dynamic_array_synapses_pre_spiking_synapses;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_pre_spiking_synapses;
std::vector<double> brian::_dynamic_array_synapses_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_w;

//////////////// dynamic arrays 2d /////////
thrust::device_vector<double*> brian::addresses_monitor__dynamic_array_statemonitor_1_v;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor_1_v;
thrust::device_vector<double*> brian::addresses_monitor__dynamic_array_statemonitor_w;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor_w;

/////////////// static arrays /////////////
double * brian::_static_array__array_neurongroup_v;
double * brian::dev_static_array__array_neurongroup_v;
__device__ double * brian::d_static_array__array_neurongroup_v;
const int brian::_num__static_array__array_neurongroup_v = 100;
int32_t * brian::_static_array__array_statemonitor_1__indices;
int32_t * brian::dev_static_array__array_statemonitor_1__indices;
__device__ int32_t * brian::d_static_array__array_statemonitor_1__indices;
const int brian::_num__static_array__array_statemonitor_1__indices = 10;
int32_t * brian::_static_array__array_statemonitor__indices;
int32_t * brian::dev_static_array__array_statemonitor__indices;
__device__ int32_t * brian::d_static_array__array_statemonitor__indices;
const int brian::_num__static_array__array_statemonitor__indices = 10;
int64_t * brian::_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
int64_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
__device__ int64_t * brian::d_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_neuron_index = 1000;
int64_t * brian::_static_array__dynamic_array_spikegeneratorgroup_spike_number;
int64_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup_spike_number;
__device__ int64_t * brian::d_static_array__dynamic_array_spikegeneratorgroup_spike_number;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_spike_number = 1000;
double * brian::_static_array__dynamic_array_spikegeneratorgroup_spike_time;
double * brian::dev_static_array__dynamic_array_spikegeneratorgroup_spike_time;
__device__ double * brian::d_static_array__dynamic_array_spikegeneratorgroup_spike_time;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_spike_time = 1000;
double * brian::_static_array__dynamic_array_synapses_w;
double * brian::dev_static_array__dynamic_array_synapses_w;
__device__ double * brian::d_static_array__dynamic_array_synapses_w;
const int brian::_num__static_array__dynamic_array_synapses_w = 10000;

//////////////// synapses /////////////////
// synapses
Synapses<double> brian::synapses(100, 100);
int32_t synapses_source_start_index;
int32_t synapses_source_stop_index;
// synapses_post
__device__ unsigned int* brian::synapses_post_size_by_pre;
__device__ int32_t** brian::synapses_post_synapses_id_by_pre;
__device__ unsigned int** brian::synapses_post_delay_by_pre;
__device__ SynapticPathway<double> brian::synapses_post;
// synapses_pre
__device__ unsigned int* brian::synapses_pre_size_by_pre;
__device__ int32_t** brian::synapses_pre_synapses_id_by_pre;
__device__ unsigned int** brian::synapses_pre_delay_by_pre;
__device__ SynapticPathway<double> brian::synapses_pre;
// synapses_1
Synapses<double> brian::synapses_1(100, 100);
int32_t synapses_1_source_start_index;
int32_t synapses_1_source_stop_index;
// synapses_1_pre
__device__ unsigned int* brian::synapses_1_pre_size_by_pre;
__device__ int32_t** brian::synapses_1_pre_synapses_id_by_pre;
__device__ unsigned int** brian::synapses_1_pre_delay_by_pre;
__device__ SynapticPathway<double> brian::synapses_1_pre;

unsigned int brian::num_parallel_blocks;
unsigned int brian::max_threads_per_block;
unsigned int brian::max_shared_mem_size;

__global__ void synapses_post_init(
				unsigned int Nsource,
				unsigned int Ntarget,
				double* delays,
				int32_t* sources,
				int32_t* targets,
				double dt,
				int32_t start,
				int32_t stop
		)
{
	using namespace brian;

	synapses_post.init(Nsource, Ntarget, delays, sources, targets, dt, start, stop);
}
__global__ void synapses_pre_init(
				unsigned int Nsource,
				unsigned int Ntarget,
				double* delays,
				int32_t* sources,
				int32_t* targets,
				double dt,
				int32_t start,
				int32_t stop
		)
{
	using namespace brian;

	synapses_pre.init(Nsource, Ntarget, delays, sources, targets, dt, start, stop);
}
__global__ void synapses_1_pre_init(
				unsigned int Nsource,
				unsigned int Ntarget,
				double* delays,
				int32_t* sources,
				int32_t* targets,
				double dt,
				int32_t start,
				int32_t stop
		)
{
	using namespace brian;

	synapses_1_pre.init(Nsource, Ntarget, delays, sources, targets, dt, start, stop);
}

//////////////random numbers//////////////////
curandGenerator_t brian::random_float_generator;

void _init_arrays()
{
	using namespace brian;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	

	num_parallel_blocks = props.multiProcessorCount * 1;
	max_threads_per_block = props.maxThreadsPerBlock;
	max_shared_mem_size = props.sharedMemPerBlock;
	
	curandCreateGenerator(&random_float_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(random_float_generator, time(0));

	//since random number generation is at the end of each clock_tick, also generate numbers for t = 0
	unsigned int needed_random_numbers;

	synapses_post_init<<<1,1>>>(
			100,
			100,
			thrust::raw_pointer_cast(&dev_dynamic_array_synapses_post_delay[0]),
			thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]),
			thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]),
			0,	//was dt, maybe irrelevant?
			0,
			100
			);
	synapses_pre_init<<<1,1>>>(
			100,
			100,
			thrust::raw_pointer_cast(&dev_dynamic_array_synapses_pre_delay[0]),
			thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]),
			thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]),
			0,	//was dt, maybe irrelevant?
			0,
			100
			);
	synapses_1_pre_init<<<1,1>>>(
			100,
			100,
			thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_pre_delay[0]),
			thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]),
			thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]),
			0,	//was dt, maybe irrelevant?
			0,
			100
			);

    // Arrays initialized to 0
	_array_statemonitor__indices = new int32_t[10];
	for(int i=0; i<10; i++) _array_statemonitor__indices[i] = 0;
	cudaMalloc((void**)&dev_array_statemonitor__indices, sizeof(int32_t)*_num__array_statemonitor__indices);
	if(!dev_array_statemonitor__indices)
	{
		printf("ERROR while allocating _array_statemonitor__indices on device with size %ld\n", sizeof(int32_t)*_num__array_statemonitor__indices);
	}
	cudaMemcpy(dev_array_statemonitor__indices, _array_statemonitor__indices, sizeof(int32_t)*_num__array_statemonitor__indices, cudaMemcpyHostToDevice);

	_array_statemonitor_1__indices = new int32_t[10];
	for(int i=0; i<10; i++) _array_statemonitor_1__indices[i] = 0;
	cudaMalloc((void**)&dev_array_statemonitor_1__indices, sizeof(int32_t)*_num__array_statemonitor_1__indices);
	if(!dev_array_statemonitor_1__indices)
	{
		printf("ERROR while allocating _array_statemonitor_1__indices on device with size %ld\n", sizeof(int32_t)*_num__array_statemonitor_1__indices);
	}
	cudaMemcpy(dev_array_statemonitor_1__indices, _array_statemonitor_1__indices, sizeof(int32_t)*_num__array_statemonitor_1__indices, cudaMemcpyHostToDevice);

	_array_spikegeneratorgroup__lastindex = new int32_t[1];
	for(int i=0; i<1; i++) _array_spikegeneratorgroup__lastindex[i] = 0;
	cudaMalloc((void**)&dev_array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex);
	if(!dev_array_spikegeneratorgroup__lastindex)
	{
		printf("ERROR while allocating _array_spikegeneratorgroup__lastindex on device with size %ld\n", sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex);
	}
	cudaMemcpy(dev_array_spikegeneratorgroup__lastindex, _array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex, cudaMemcpyHostToDevice);

	_array_spikemonitor__source_idx = new int32_t[100];
	for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0;
	cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx);
	if(!dev_array_spikemonitor__source_idx)
	{
		printf("ERROR while allocating _array_spikemonitor__source_idx on device with size %ld\n", sizeof(int32_t)*_num__array_spikemonitor__source_idx);
	}
	cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice);

	_array_neurongroup__spikespace = new int32_t[101];
	for(int i=0; i<101; i++) _array_neurongroup__spikespace[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace);
	if(!dev_array_neurongroup__spikespace)
	{
		printf("ERROR while allocating _array_neurongroup__spikespace on device with size %ld\n", sizeof(int32_t)*_num__array_neurongroup__spikespace);
	}
	cudaMemcpy(dev_array_neurongroup__spikespace, _array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyHostToDevice);

	_array_spikegeneratorgroup__spikespace = new int32_t[101];
	for(int i=0; i<101; i++) _array_spikegeneratorgroup__spikespace[i] = 0;
	cudaMalloc((void**)&dev_array_spikegeneratorgroup__spikespace, sizeof(int32_t)*_num__array_spikegeneratorgroup__spikespace);
	if(!dev_array_spikegeneratorgroup__spikespace)
	{
		printf("ERROR while allocating _array_spikegeneratorgroup__spikespace on device with size %ld\n", sizeof(int32_t)*_num__array_spikegeneratorgroup__spikespace);
	}
	cudaMemcpy(dev_array_spikegeneratorgroup__spikespace, _array_spikegeneratorgroup__spikespace, sizeof(int32_t)*_num__array_spikegeneratorgroup__spikespace, cudaMemcpyHostToDevice);

	_array_spikemonitor_count = new int32_t[100];
	for(int i=0; i<100; i++) _array_spikemonitor_count[i] = 0;
	cudaMalloc((void**)&dev_array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count);
	if(!dev_array_spikemonitor_count)
	{
		printf("ERROR while allocating _array_spikemonitor_count on device with size %ld\n", sizeof(int32_t)*_num__array_spikemonitor_count);
	}
	cudaMemcpy(dev_array_spikemonitor_count, _array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count, cudaMemcpyHostToDevice);

	_array_defaultclock_dt = new double[1];
	for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;
	cudaMalloc((void**)&dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt);
	if(!dev_array_defaultclock_dt)
	{
		printf("ERROR while allocating _array_defaultclock_dt on device with size %ld\n", sizeof(double)*_num__array_defaultclock_dt);
	}
	cudaMemcpy(dev_array_defaultclock_dt, _array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyHostToDevice);

	_array_neurongroup_g = new double[100];
	for(int i=0; i<100; i++) _array_neurongroup_g[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_g, sizeof(double)*_num__array_neurongroup_g);
	if(!dev_array_neurongroup_g)
	{
		printf("ERROR while allocating _array_neurongroup_g on device with size %ld\n", sizeof(double)*_num__array_neurongroup_g);
	}
	cudaMemcpy(dev_array_neurongroup_g, _array_neurongroup_g, sizeof(double)*_num__array_neurongroup_g, cudaMemcpyHostToDevice);

	_array_neurongroup_i = new int32_t[100];
	for(int i=0; i<100; i++) _array_neurongroup_i[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i);
	if(!dev_array_neurongroup_i)
	{
		printf("ERROR while allocating _array_neurongroup_i on device with size %ld\n", sizeof(int32_t)*_num__array_neurongroup_i);
	}
	cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice);

	_array_spikegeneratorgroup_i = new int32_t[100];
	for(int i=0; i<100; i++) _array_spikegeneratorgroup_i[i] = 0;
	cudaMalloc((void**)&dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i);
	if(!dev_array_spikegeneratorgroup_i)
	{
		printf("ERROR while allocating _array_spikegeneratorgroup_i on device with size %ld\n", sizeof(int32_t)*_num__array_spikegeneratorgroup_i);
	}
	cudaMemcpy(dev_array_spikegeneratorgroup_i, _array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyHostToDevice);

	_array_neurongroup_lastspike = new double[100];
	for(int i=0; i<100; i++) _array_neurongroup_lastspike[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike);
	if(!dev_array_neurongroup_lastspike)
	{
		printf("ERROR while allocating _array_neurongroup_lastspike on device with size %ld\n", sizeof(double)*_num__array_neurongroup_lastspike);
	}
	cudaMemcpy(dev_array_neurongroup_lastspike, _array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike, cudaMemcpyHostToDevice);

	_array_synapses_N = new int32_t[1];
	for(int i=0; i<1; i++) _array_synapses_N[i] = 0;
	cudaMalloc((void**)&dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N);
	if(!dev_array_synapses_N)
	{
		printf("ERROR while allocating _array_synapses_N on device with size %ld\n", sizeof(int32_t)*_num__array_synapses_N);
	}
	cudaMemcpy(dev_array_synapses_N, _array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyHostToDevice);

	_array_synapses_1_N = new int32_t[1];
	for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;
	cudaMalloc((void**)&dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N);
	if(!dev_array_synapses_1_N)
	{
		printf("ERROR while allocating _array_synapses_1_N on device with size %ld\n", sizeof(int32_t)*_num__array_synapses_1_N);
	}
	cudaMemcpy(dev_array_synapses_1_N, _array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyHostToDevice);

	_array_spikemonitor_N = new int32_t[1];
	for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;
	cudaMalloc((void**)&dev_array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N);
	if(!dev_array_spikemonitor_N)
	{
		printf("ERROR while allocating _array_spikemonitor_N on device with size %ld\n", sizeof(int32_t)*_num__array_spikemonitor_N);
	}
	cudaMemcpy(dev_array_spikemonitor_N, _array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N, cudaMemcpyHostToDevice);

	_array_statemonitor_N = new int32_t[1];
	for(int i=0; i<1; i++) _array_statemonitor_N[i] = 0;
	cudaMalloc((void**)&dev_array_statemonitor_N, sizeof(int32_t)*_num__array_statemonitor_N);
	if(!dev_array_statemonitor_N)
	{
		printf("ERROR while allocating _array_statemonitor_N on device with size %ld\n", sizeof(int32_t)*_num__array_statemonitor_N);
	}
	cudaMemcpy(dev_array_statemonitor_N, _array_statemonitor_N, sizeof(int32_t)*_num__array_statemonitor_N, cudaMemcpyHostToDevice);

	_array_statemonitor_1_N = new int32_t[1];
	for(int i=0; i<1; i++) _array_statemonitor_1_N[i] = 0;
	cudaMalloc((void**)&dev_array_statemonitor_1_N, sizeof(int32_t)*_num__array_statemonitor_1_N);
	if(!dev_array_statemonitor_1_N)
	{
		printf("ERROR while allocating _array_statemonitor_1_N on device with size %ld\n", sizeof(int32_t)*_num__array_statemonitor_1_N);
	}
	cudaMemcpy(dev_array_statemonitor_1_N, _array_statemonitor_1_N, sizeof(int32_t)*_num__array_statemonitor_1_N, cudaMemcpyHostToDevice);

	_array_neurongroup_not_refractory = new char[100];
	for(int i=0; i<100; i++) _array_neurongroup_not_refractory[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_not_refractory, sizeof(char)*_num__array_neurongroup_not_refractory);
	if(!dev_array_neurongroup_not_refractory)
	{
		printf("ERROR while allocating _array_neurongroup_not_refractory on device with size %ld\n", sizeof(char)*_num__array_neurongroup_not_refractory);
	}
	cudaMemcpy(dev_array_neurongroup_not_refractory, _array_neurongroup_not_refractory, sizeof(char)*_num__array_neurongroup_not_refractory, cudaMemcpyHostToDevice);

	_array_spikegeneratorgroup_period = new double[1];
	for(int i=0; i<1; i++) _array_spikegeneratorgroup_period[i] = 0;
	cudaMalloc((void**)&dev_array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period);
	if(!dev_array_spikegeneratorgroup_period)
	{
		printf("ERROR while allocating _array_spikegeneratorgroup_period on device with size %ld\n", sizeof(double)*_num__array_spikegeneratorgroup_period);
	}
	cudaMemcpy(dev_array_spikegeneratorgroup_period, _array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period, cudaMemcpyHostToDevice);

	_array_defaultclock_t = new double[1];
	for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;
	cudaMalloc((void**)&dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t);
	if(!dev_array_defaultclock_t)
	{
		printf("ERROR while allocating _array_defaultclock_t on device with size %ld\n", sizeof(double)*_num__array_defaultclock_t);
	}
	cudaMemcpy(dev_array_defaultclock_t, _array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyHostToDevice);

	_array_defaultclock_timestep = new uint64_t[1];
	for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;
	cudaMalloc((void**)&dev_array_defaultclock_timestep, sizeof(uint64_t)*_num__array_defaultclock_timestep);
	if(!dev_array_defaultclock_timestep)
	{
		printf("ERROR while allocating _array_defaultclock_timestep on device with size %ld\n", sizeof(uint64_t)*_num__array_defaultclock_timestep);
	}
	cudaMemcpy(dev_array_defaultclock_timestep, _array_defaultclock_timestep, sizeof(uint64_t)*_num__array_defaultclock_timestep, cudaMemcpyHostToDevice);

	_array_neurongroup_v = new double[100];
	for(int i=0; i<100; i++) _array_neurongroup_v[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v);
	if(!dev_array_neurongroup_v)
	{
		printf("ERROR while allocating _array_neurongroup_v on device with size %ld\n", sizeof(double)*_num__array_neurongroup_v);
	}
	cudaMemcpy(dev_array_neurongroup_v, _array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyHostToDevice);


	// Arrays initialized to an "arange"
	_array_spikemonitor__source_idx = new int32_t[100];
	for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0 + i;
	cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx);
	if(!dev_array_spikemonitor__source_idx)
	{
		printf("ERROR while allocating _array_spikemonitor__source_idx on device with size %ld\n", sizeof(int32_t)*_num__array_spikemonitor__source_idx);
	}
	cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice);

	_array_neurongroup_i = new int32_t[100];
	for(int i=0; i<100; i++) _array_neurongroup_i[i] = 0 + i;
	cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i);
	if(!dev_array_neurongroup_i)
	{
		printf("ERROR while allocating _array_neurongroup_i on device with size %ld\n", sizeof(int32_t)*_num__array_neurongroup_i);
	}
	cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice);

	_array_spikegeneratorgroup_i = new int32_t[100];
	for(int i=0; i<100; i++) _array_spikegeneratorgroup_i[i] = 0 + i;
	cudaMalloc((void**)&dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i);
	if(!dev_array_spikegeneratorgroup_i)
	{
		printf("ERROR while allocating _array_spikegeneratorgroup_i on device with size %ld\n", sizeof(int32_t)*_num__array_spikegeneratorgroup_i);
	}
	cudaMemcpy(dev_array_spikegeneratorgroup_i, _array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyHostToDevice);


	// static arrays
	_static_array__array_neurongroup_v = new double[100];
	cudaMalloc((void**)&dev_static_array__array_neurongroup_v, sizeof(double)*100);
	if(!dev_static_array__array_neurongroup_v)
	{
		printf("ERROR while allocating _static_array__array_neurongroup_v on device with size %ld\n", sizeof(double)*100);
	}
	cudaMemcpyToSymbol(d_static_array__array_neurongroup_v, &dev_static_array__array_neurongroup_v, sizeof(double*));
	_static_array__array_statemonitor_1__indices = new int32_t[10];
	cudaMalloc((void**)&dev_static_array__array_statemonitor_1__indices, sizeof(int32_t)*10);
	if(!dev_static_array__array_statemonitor_1__indices)
	{
		printf("ERROR while allocating _static_array__array_statemonitor_1__indices on device with size %ld\n", sizeof(int32_t)*10);
	}
	cudaMemcpyToSymbol(d_static_array__array_statemonitor_1__indices, &dev_static_array__array_statemonitor_1__indices, sizeof(int32_t*));
	_static_array__array_statemonitor__indices = new int32_t[10];
	cudaMalloc((void**)&dev_static_array__array_statemonitor__indices, sizeof(int32_t)*10);
	if(!dev_static_array__array_statemonitor__indices)
	{
		printf("ERROR while allocating _static_array__array_statemonitor__indices on device with size %ld\n", sizeof(int32_t)*10);
	}
	cudaMemcpyToSymbol(d_static_array__array_statemonitor__indices, &dev_static_array__array_statemonitor__indices, sizeof(int32_t*));
	_static_array__dynamic_array_spikegeneratorgroup_neuron_index = new int64_t[1000];
	cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t)*1000);
	if(!dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index)
	{
		printf("ERROR while allocating _static_array__dynamic_array_spikegeneratorgroup_neuron_index on device with size %ld\n", sizeof(int64_t)*1000);
	}
	cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_neuron_index, &dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t*));
	_static_array__dynamic_array_spikegeneratorgroup_spike_number = new int64_t[1000];
	cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int64_t)*1000);
	if(!dev_static_array__dynamic_array_spikegeneratorgroup_spike_number)
	{
		printf("ERROR while allocating _static_array__dynamic_array_spikegeneratorgroup_spike_number on device with size %ld\n", sizeof(int64_t)*1000);
	}
	cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_spike_number, &dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int64_t*));
	_static_array__dynamic_array_spikegeneratorgroup_spike_time = new double[1000];
	cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double)*1000);
	if(!dev_static_array__dynamic_array_spikegeneratorgroup_spike_time)
	{
		printf("ERROR while allocating _static_array__dynamic_array_spikegeneratorgroup_spike_time on device with size %ld\n", sizeof(double)*1000);
	}
	cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_spike_time, &dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double*));
	_static_array__dynamic_array_synapses_w = new double[10000];
	cudaMalloc((void**)&dev_static_array__dynamic_array_synapses_w, sizeof(double)*10000);
	if(!dev_static_array__dynamic_array_synapses_w)
	{
		printf("ERROR while allocating _static_array__dynamic_array_synapses_w on device with size %ld\n", sizeof(double)*10000);
	}
	cudaMemcpyToSymbol(d_static_array__dynamic_array_synapses_w, &dev_static_array__dynamic_array_synapses_w, sizeof(double*));

	_dynamic_array_statemonitor_1_v = new thrust::device_vector<double>[_num__array_statemonitor_1__indices];
	_dynamic_array_statemonitor_w = new thrust::device_vector<double>[_num__array_statemonitor__indices];
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_static_array__array_neurongroup_v;
	f_static_array__array_neurongroup_v.open("static_arrays/_static_array__array_neurongroup_v", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_v.is_open())
	{
		f_static_array__array_neurongroup_v.read(reinterpret_cast<char*>(_static_array__array_neurongroup_v), 100*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_v." << endl;
	}
	cudaMemcpy(dev_static_array__array_neurongroup_v, _static_array__array_neurongroup_v, sizeof(double)*100, cudaMemcpyHostToDevice);
	ifstream f_static_array__array_statemonitor_1__indices;
	f_static_array__array_statemonitor_1__indices.open("static_arrays/_static_array__array_statemonitor_1__indices", ios::in | ios::binary);
	if(f_static_array__array_statemonitor_1__indices.is_open())
	{
		f_static_array__array_statemonitor_1__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor_1__indices), 10*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_statemonitor_1__indices." << endl;
	}
	cudaMemcpy(dev_static_array__array_statemonitor_1__indices, _static_array__array_statemonitor_1__indices, sizeof(int32_t)*10, cudaMemcpyHostToDevice);
	ifstream f_static_array__array_statemonitor__indices;
	f_static_array__array_statemonitor__indices.open("static_arrays/_static_array__array_statemonitor__indices", ios::in | ios::binary);
	if(f_static_array__array_statemonitor__indices.is_open())
	{
		f_static_array__array_statemonitor__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor__indices), 10*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_statemonitor__indices." << endl;
	}
	cudaMemcpy(dev_static_array__array_statemonitor__indices, _static_array__array_statemonitor__indices, sizeof(int32_t)*10, cudaMemcpyHostToDevice);
	ifstream f_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
	f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_neuron_index", ios::in | ios::binary);
	if(f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
	{
		f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_neuron_index), 1000*sizeof(int64_t));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_neuron_index." << endl;
	}
	cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, _static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t)*1000, cudaMemcpyHostToDevice);
	ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_number;
	f_static_array__dynamic_array_spikegeneratorgroup_spike_number.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_number", ios::in | ios::binary);
	if(f_static_array__dynamic_array_spikegeneratorgroup_spike_number.is_open())
	{
		f_static_array__dynamic_array_spikegeneratorgroup_spike_number.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_number), 1000*sizeof(int64_t));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_number." << endl;
	}
	cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, _static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int64_t)*1000, cudaMemcpyHostToDevice);
	ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_time;
	f_static_array__dynamic_array_spikegeneratorgroup_spike_time.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_time", ios::in | ios::binary);
	if(f_static_array__dynamic_array_spikegeneratorgroup_spike_time.is_open())
	{
		f_static_array__dynamic_array_spikegeneratorgroup_spike_time.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_time), 1000*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_time." << endl;
	}
	cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, _static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double)*1000, cudaMemcpyHostToDevice);
	ifstream f_static_array__dynamic_array_synapses_w;
	f_static_array__dynamic_array_synapses_w.open("static_arrays/_static_array__dynamic_array_synapses_w", ios::in | ios::binary);
	if(f_static_array__dynamic_array_synapses_w.is_open())
	{
		f_static_array__dynamic_array_synapses_w.read(reinterpret_cast<char*>(_static_array__dynamic_array_synapses_w), 10000*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_synapses_w." << endl;
	}
	cudaMemcpy(dev_static_array__dynamic_array_synapses_w, _static_array__dynamic_array_synapses_w, sizeof(double)*10000, cudaMemcpyHostToDevice);
}	

void _write_arrays()
{
	using namespace brian;

	cudaMemcpy(_array_defaultclock_dt, dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyDeviceToHost);
	ofstream outfile__array_defaultclock_dt;
	outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_-9215759865592636245", ios::binary | ios::out);
	if(outfile__array_defaultclock_dt.is_open())
	{
		outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(double));
		outfile__array_defaultclock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
	}
	cudaMemcpy(_array_defaultclock_t, dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyDeviceToHost);
	ofstream outfile__array_defaultclock_t;
	outfile__array_defaultclock_t.open("results/_array_defaultclock_t_7263079326120112646", ios::binary | ios::out);
	if(outfile__array_defaultclock_t.is_open())
	{
		outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(double));
		outfile__array_defaultclock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_t." << endl;
	}
	cudaMemcpy(_array_defaultclock_timestep, dev_array_defaultclock_timestep, sizeof(uint64_t)*_num__array_defaultclock_timestep, cudaMemcpyDeviceToHost);
	ofstream outfile__array_defaultclock_timestep;
	outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_-8300011050550298960", ios::binary | ios::out);
	if(outfile__array_defaultclock_timestep.is_open())
	{
		outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(uint64_t));
		outfile__array_defaultclock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
	}
	cudaMemcpy(_array_neurongroup__spikespace, dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyDeviceToHost);
	ofstream outfile__array_neurongroup__spikespace;
	outfile__array_neurongroup__spikespace.open("results/_array_neurongroup__spikespace_6291509255835556833", ios::binary | ios::out);
	if(outfile__array_neurongroup__spikespace.is_open())
	{
		outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 101*sizeof(int32_t));
		outfile__array_neurongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
	}
	cudaMemcpy(_array_neurongroup_g, dev_array_neurongroup_g, sizeof(double)*_num__array_neurongroup_g, cudaMemcpyDeviceToHost);
	ofstream outfile__array_neurongroup_g;
	outfile__array_neurongroup_g.open("results/_array_neurongroup_g_-2688036259655650205", ios::binary | ios::out);
	if(outfile__array_neurongroup_g.is_open())
	{
		outfile__array_neurongroup_g.write(reinterpret_cast<char*>(_array_neurongroup_g), 100*sizeof(double));
		outfile__array_neurongroup_g.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_g." << endl;
	}
	cudaMemcpy(_array_neurongroup_i, dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyDeviceToHost);
	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open("results/_array_neurongroup_i_-2688036259655650195", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 100*sizeof(int32_t));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}
	cudaMemcpy(_array_neurongroup_lastspike, dev_array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike, cudaMemcpyDeviceToHost);
	ofstream outfile__array_neurongroup_lastspike;
	outfile__array_neurongroup_lastspike.open("results/_array_neurongroup_lastspike_-2029934132282177282", ios::binary | ios::out);
	if(outfile__array_neurongroup_lastspike.is_open())
	{
		outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), 100*sizeof(double));
		outfile__array_neurongroup_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
	}
	cudaMemcpy(_array_neurongroup_not_refractory, dev_array_neurongroup_not_refractory, sizeof(char)*_num__array_neurongroup_not_refractory, cudaMemcpyDeviceToHost);
	ofstream outfile__array_neurongroup_not_refractory;
	outfile__array_neurongroup_not_refractory.open("results/_array_neurongroup_not_refractory_-8596383477608809440", ios::binary | ios::out);
	if(outfile__array_neurongroup_not_refractory.is_open())
	{
		outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), 100*sizeof(char));
		outfile__array_neurongroup_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
	}
	cudaMemcpy(_array_neurongroup_v, dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyDeviceToHost);
	ofstream outfile__array_neurongroup_v;
	outfile__array_neurongroup_v.open("results/_array_neurongroup_v_-2688036259655650190", ios::binary | ios::out);
	if(outfile__array_neurongroup_v.is_open())
	{
		outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 100*sizeof(double));
		outfile__array_neurongroup_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_v." << endl;
	}
	cudaMemcpy(_array_spikegeneratorgroup__lastindex, dev_array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex, cudaMemcpyDeviceToHost);
	ofstream outfile__array_spikegeneratorgroup__lastindex;
	outfile__array_spikegeneratorgroup__lastindex.open("results/_array_spikegeneratorgroup__lastindex_7651580413293037816", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup__lastindex.is_open())
	{
		outfile__array_spikegeneratorgroup__lastindex.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__lastindex), 1*sizeof(int32_t));
		outfile__array_spikegeneratorgroup__lastindex.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup__lastindex." << endl;
	}
	cudaMemcpy(_array_spikegeneratorgroup__spikespace, dev_array_spikegeneratorgroup__spikespace, sizeof(int32_t)*_num__array_spikegeneratorgroup__spikespace, cudaMemcpyDeviceToHost);
	ofstream outfile__array_spikegeneratorgroup__spikespace;
	outfile__array_spikegeneratorgroup__spikespace.open("results/_array_spikegeneratorgroup__spikespace_-1541437192670537063", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup__spikespace.is_open())
	{
		outfile__array_spikegeneratorgroup__spikespace.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__spikespace), 101*sizeof(int32_t));
		outfile__array_spikegeneratorgroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup__spikespace." << endl;
	}
	cudaMemcpy(_array_spikegeneratorgroup_i, dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyDeviceToHost);
	ofstream outfile__array_spikegeneratorgroup_i;
	outfile__array_spikegeneratorgroup_i.open("results/_array_spikegeneratorgroup_i_-7676153806036632795", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup_i.is_open())
	{
		outfile__array_spikegeneratorgroup_i.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_i), 100*sizeof(int32_t));
		outfile__array_spikegeneratorgroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup_i." << endl;
	}
	cudaMemcpy(_array_spikegeneratorgroup_period, dev_array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period, cudaMemcpyDeviceToHost);
	ofstream outfile__array_spikegeneratorgroup_period;
	outfile__array_spikegeneratorgroup_period.open("results/_array_spikegeneratorgroup_period_-5885060602818549350", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup_period.is_open())
	{
		outfile__array_spikegeneratorgroup_period.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_period), 1*sizeof(double));
		outfile__array_spikegeneratorgroup_period.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup_period." << endl;
	}
	cudaMemcpy(_array_spikemonitor__source_idx, dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyDeviceToHost);
	ofstream outfile__array_spikemonitor__source_idx;
	outfile__array_spikemonitor__source_idx.open("results/_array_spikemonitor__source_idx_-4045852888739339153", ios::binary | ios::out);
	if(outfile__array_spikemonitor__source_idx.is_open())
	{
		outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 100*sizeof(int32_t));
		outfile__array_spikemonitor__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
	}
	cudaMemcpy(_array_spikemonitor_count, dev_array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count, cudaMemcpyDeviceToHost);
	ofstream outfile__array_spikemonitor_count;
	outfile__array_spikemonitor_count.open("results/_array_spikemonitor_count_-3651895352503201284", ios::binary | ios::out);
	if(outfile__array_spikemonitor_count.is_open())
	{
		outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 100*sizeof(int32_t));
		outfile__array_spikemonitor_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
	}
	cudaMemcpy(_array_spikemonitor_N, dev_array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N, cudaMemcpyDeviceToHost);
	ofstream outfile__array_spikemonitor_N;
	outfile__array_spikemonitor_N.open("results/_array_spikemonitor_N_73938390545997659", ios::binary | ios::out);
	if(outfile__array_spikemonitor_N.is_open())
	{
		outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(int32_t));
		outfile__array_spikemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
	}
	cudaMemcpy(_array_statemonitor_1__indices, dev_array_statemonitor_1__indices, sizeof(int32_t)*_num__array_statemonitor_1__indices, cudaMemcpyDeviceToHost);
	ofstream outfile__array_statemonitor_1__indices;
	outfile__array_statemonitor_1__indices.open("results/_array_statemonitor_1__indices_5442903941946222241", ios::binary | ios::out);
	if(outfile__array_statemonitor_1__indices.is_open())
	{
		outfile__array_statemonitor_1__indices.write(reinterpret_cast<char*>(_array_statemonitor_1__indices), 10*sizeof(int32_t));
		outfile__array_statemonitor_1__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor_1__indices." << endl;
	}
	cudaMemcpy(_array_statemonitor_1_N, dev_array_statemonitor_1_N, sizeof(int32_t)*_num__array_statemonitor_1_N, cudaMemcpyDeviceToHost);
	ofstream outfile__array_statemonitor_1_N;
	outfile__array_statemonitor_1_N.open("results/_array_statemonitor_1_N_2048012943032035014", ios::binary | ios::out);
	if(outfile__array_statemonitor_1_N.is_open())
	{
		outfile__array_statemonitor_1_N.write(reinterpret_cast<char*>(_array_statemonitor_1_N), 1*sizeof(int32_t));
		outfile__array_statemonitor_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor_1_N." << endl;
	}
	cudaMemcpy(_array_statemonitor__indices, dev_array_statemonitor__indices, sizeof(int32_t)*_num__array_statemonitor__indices, cudaMemcpyDeviceToHost);
	ofstream outfile__array_statemonitor__indices;
	outfile__array_statemonitor__indices.open("results/_array_statemonitor__indices_6163485638831984707", ios::binary | ios::out);
	if(outfile__array_statemonitor__indices.is_open())
	{
		outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 10*sizeof(int32_t));
		outfile__array_statemonitor__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
	}
	cudaMemcpy(_array_statemonitor_N, dev_array_statemonitor_N, sizeof(int32_t)*_num__array_statemonitor_N, cudaMemcpyDeviceToHost);
	ofstream outfile__array_statemonitor_N;
	outfile__array_statemonitor_N.open("results/_array_statemonitor_N_1126150466128921572", ios::binary | ios::out);
	if(outfile__array_statemonitor_N.is_open())
	{
		outfile__array_statemonitor_N.write(reinterpret_cast<char*>(_array_statemonitor_N), 1*sizeof(int32_t));
		outfile__array_statemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor_N." << endl;
	}
	cudaMemcpy(_array_synapses_1_N, dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyDeviceToHost);
	ofstream outfile__array_synapses_1_N;
	outfile__array_synapses_1_N.open("results/_array_synapses_1_N_-7473518110119523383", ios::binary | ios::out);
	if(outfile__array_synapses_1_N.is_open())
	{
		outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(int32_t));
		outfile__array_synapses_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_1_N." << endl;
	}
	cudaMemcpy(_array_synapses_N, dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyDeviceToHost);
	ofstream outfile__array_synapses_N;
	outfile__array_synapses_N.open("results/_array_synapses_N_-7833853409752232273", ios::binary | ios::out);
	if(outfile__array_synapses_N.is_open())
	{
		outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(int32_t));
		outfile__array_synapses_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N." << endl;
	}

	thrust::host_vector<int32_t> temp_dynamic_array_spikegeneratorgroup_neuron_index = dev_dynamic_array_spikegeneratorgroup_neuron_index;
	ofstream outfile__dynamic_array_spikegeneratorgroup_neuron_index;
	outfile__dynamic_array_spikegeneratorgroup_neuron_index.open("results/_dynamic_array_spikegeneratorgroup_neuron_index_-8147356448785106711", ios::binary | ios::out);
	if(outfile__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
	{
		outfile__dynamic_array_spikegeneratorgroup_neuron_index.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_spikegeneratorgroup_neuron_index[0])), dev_dynamic_array_spikegeneratorgroup_neuron_index.size()*sizeof(int32_t));
		outfile__dynamic_array_spikegeneratorgroup_neuron_index.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_neuron_index." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_spikegeneratorgroup_spike_number = dev_dynamic_array_spikegeneratorgroup_spike_number;
	ofstream outfile__dynamic_array_spikegeneratorgroup_spike_number;
	outfile__dynamic_array_spikegeneratorgroup_spike_number.open("results/_dynamic_array_spikegeneratorgroup_spike_number_6285234774451964247", ios::binary | ios::out);
	if(outfile__dynamic_array_spikegeneratorgroup_spike_number.is_open())
	{
		outfile__dynamic_array_spikegeneratorgroup_spike_number.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_spikegeneratorgroup_spike_number[0])), dev_dynamic_array_spikegeneratorgroup_spike_number.size()*sizeof(int32_t));
		outfile__dynamic_array_spikegeneratorgroup_spike_number.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_number." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_spikegeneratorgroup_spike_time = dev_dynamic_array_spikegeneratorgroup_spike_time;
	ofstream outfile__dynamic_array_spikegeneratorgroup_spike_time;
	outfile__dynamic_array_spikegeneratorgroup_spike_time.open("results/_dynamic_array_spikegeneratorgroup_spike_time_6974015387653485977", ios::binary | ios::out);
	if(outfile__dynamic_array_spikegeneratorgroup_spike_time.is_open())
	{
		outfile__dynamic_array_spikegeneratorgroup_spike_time.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_spikegeneratorgroup_spike_time[0])), dev_dynamic_array_spikegeneratorgroup_spike_time.size()*sizeof(double));
		outfile__dynamic_array_spikegeneratorgroup_spike_time.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_time." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_spikemonitor_i = dev_dynamic_array_spikemonitor_i;
	ofstream outfile__dynamic_array_spikemonitor_i;
	outfile__dynamic_array_spikemonitor_i.open("results/_dynamic_array_spikemonitor_i_3873805716461528078", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_i.is_open())
	{
		outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_spikemonitor_i[0])), dev_dynamic_array_spikemonitor_i.size()*sizeof(int32_t));
		outfile__dynamic_array_spikemonitor_i.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_spikemonitor_t = dev_dynamic_array_spikemonitor_t;
	ofstream outfile__dynamic_array_spikemonitor_t;
	outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t_3873805716461528083", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_t.is_open())
	{
		outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_spikemonitor_t[0])), dev_dynamic_array_spikemonitor_t.size()*sizeof(double));
		outfile__dynamic_array_spikemonitor_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_statemonitor_1_t = dev_dynamic_array_statemonitor_1_t;
	ofstream outfile__dynamic_array_statemonitor_1_t;
	outfile__dynamic_array_statemonitor_1_t.open("results/_dynamic_array_statemonitor_1_t_-8409868653121327110", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_1_t.is_open())
	{
		outfile__dynamic_array_statemonitor_1_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_statemonitor_1_t[0])), dev_dynamic_array_statemonitor_1_t.size()*sizeof(double));
		outfile__dynamic_array_statemonitor_1_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_1_t." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_statemonitor_t = dev_dynamic_array_statemonitor_t;
	ofstream outfile__dynamic_array_statemonitor_t;
	outfile__dynamic_array_statemonitor_t.open("results/_dynamic_array_statemonitor_t_6620044162385838772", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_t.is_open())
	{
		outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_statemonitor_t[0])), dev_dynamic_array_statemonitor_t.size()*sizeof(double));
		outfile__dynamic_array_statemonitor_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_t." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses_1__synaptic_post = dev_dynamic_array_synapses_1__synaptic_post;
	ofstream outfile__dynamic_array_synapses_1__synaptic_post;
	outfile__dynamic_array_synapses_1__synaptic_post.open("results/_dynamic_array_synapses_1__synaptic_post_-4367449856540212009", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
	{
		outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_1__synaptic_post[0])), dev_dynamic_array_synapses_1__synaptic_post.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses_1__synaptic_post.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses_1__synaptic_pre = dev_dynamic_array_synapses_1__synaptic_pre;
	ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
	outfile__dynamic_array_synapses_1__synaptic_pre.open("results/_dynamic_array_synapses_1__synaptic_pre_1368795276670783483", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
	{
		outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_1__synaptic_pre[0])), dev_dynamic_array_synapses_1__synaptic_pre.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses_1__synaptic_pre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_synapses_1_lastupdate = dev_dynamic_array_synapses_1_lastupdate;
	ofstream outfile__dynamic_array_synapses_1_lastupdate;
	outfile__dynamic_array_synapses_1_lastupdate.open("results/_dynamic_array_synapses_1_lastupdate_6875119916677774017", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_lastupdate.is_open())
	{
		outfile__dynamic_array_synapses_1_lastupdate.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_1_lastupdate[0])), dev_dynamic_array_synapses_1_lastupdate.size()*sizeof(double));
		outfile__dynamic_array_synapses_1_lastupdate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_lastupdate." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses_1_N_incoming = dev_dynamic_array_synapses_1_N_incoming;
	ofstream outfile__dynamic_array_synapses_1_N_incoming;
	outfile__dynamic_array_synapses_1_N_incoming.open("results/_dynamic_array_synapses_1_N_incoming_-5364435978754666149", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
	{
		outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_1_N_incoming[0])), dev_dynamic_array_synapses_1_N_incoming.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses_1_N_incoming.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses_1_N_outgoing = dev_dynamic_array_synapses_1_N_outgoing;
	ofstream outfile__dynamic_array_synapses_1_N_outgoing;
	outfile__dynamic_array_synapses_1_N_outgoing.open("results/_dynamic_array_synapses_1_N_outgoing_7721560298971024321", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
	{
		outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_1_N_outgoing[0])), dev_dynamic_array_synapses_1_N_outgoing.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses_1_N_outgoing.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_synapses_1_pre_delay = dev_dynamic_array_synapses_1_pre_delay;
	ofstream outfile__dynamic_array_synapses_1_pre_delay;
	outfile__dynamic_array_synapses_1_pre_delay.open("results/_dynamic_array_synapses_1_pre_delay_6658020171927933066", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_pre_delay.is_open())
	{
		outfile__dynamic_array_synapses_1_pre_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_1_pre_delay[0])), dev_dynamic_array_synapses_1_pre_delay.size()*sizeof(double));
		outfile__dynamic_array_synapses_1_pre_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_pre_delay." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses_1_pre_spiking_synapses = dev_dynamic_array_synapses_1_pre_spiking_synapses;
	ofstream outfile__dynamic_array_synapses_1_pre_spiking_synapses;
	outfile__dynamic_array_synapses_1_pre_spiking_synapses.open("results/_dynamic_array_synapses_1_pre_spiking_synapses_-5096904537968888248", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_pre_spiking_synapses.is_open())
	{
		outfile__dynamic_array_synapses_1_pre_spiking_synapses.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_1_pre_spiking_synapses[0])), dev_dynamic_array_synapses_1_pre_spiking_synapses.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses_1_pre_spiking_synapses.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_pre_spiking_synapses." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_synapses_1_w = dev_dynamic_array_synapses_1_w;
	ofstream outfile__dynamic_array_synapses_1_w;
	outfile__dynamic_array_synapses_1_w.open("results/_dynamic_array_synapses_1_w_-1083981064091387634", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_w.is_open())
	{
		outfile__dynamic_array_synapses_1_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_1_w[0])), dev_dynamic_array_synapses_1_w.size()*sizeof(double));
		outfile__dynamic_array_synapses_1_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_w." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses__synaptic_post = dev_dynamic_array_synapses__synaptic_post;
	ofstream outfile__dynamic_array_synapses__synaptic_post;
	outfile__dynamic_array_synapses__synaptic_post.open("results/_dynamic_array_synapses__synaptic_post_3840486125387374025", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_post.is_open())
	{
		outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses__synaptic_post[0])), dev_dynamic_array_synapses__synaptic_post.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses__synaptic_post.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses__synaptic_pre = dev_dynamic_array_synapses__synaptic_pre;
	ofstream outfile__dynamic_array_synapses__synaptic_pre;
	outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre_5162992210040840425", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
	{
		outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses__synaptic_pre[0])), dev_dynamic_array_synapses__synaptic_pre.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses__synaptic_pre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_synapses_Apost = dev_dynamic_array_synapses_Apost;
	ofstream outfile__dynamic_array_synapses_Apost;
	outfile__dynamic_array_synapses_Apost.open("results/_dynamic_array_synapses_Apost_3034042580529184036", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_Apost.is_open())
	{
		outfile__dynamic_array_synapses_Apost.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_Apost[0])), dev_dynamic_array_synapses_Apost.size()*sizeof(double));
		outfile__dynamic_array_synapses_Apost.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_Apost." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_synapses_Apre = dev_dynamic_array_synapses_Apre;
	ofstream outfile__dynamic_array_synapses_Apre;
	outfile__dynamic_array_synapses_Apre.open("results/_dynamic_array_synapses_Apre_-5633892405563205630", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_Apre.is_open())
	{
		outfile__dynamic_array_synapses_Apre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_Apre[0])), dev_dynamic_array_synapses_Apre.size()*sizeof(double));
		outfile__dynamic_array_synapses_Apre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_Apre." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_synapses_lastupdate = dev_dynamic_array_synapses_lastupdate;
	ofstream outfile__dynamic_array_synapses_lastupdate;
	outfile__dynamic_array_synapses_lastupdate.open("results/_dynamic_array_synapses_lastupdate_562699891839928247", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_lastupdate.is_open())
	{
		outfile__dynamic_array_synapses_lastupdate.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_lastupdate[0])), dev_dynamic_array_synapses_lastupdate.size()*sizeof(double));
		outfile__dynamic_array_synapses_lastupdate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_lastupdate." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses_N_incoming = dev_dynamic_array_synapses_N_incoming;
	ofstream outfile__dynamic_array_synapses_N_incoming;
	outfile__dynamic_array_synapses_N_incoming.open("results/_dynamic_array_synapses_N_incoming_6651214916728133133", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_incoming.is_open())
	{
		outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_N_incoming[0])), dev_dynamic_array_synapses_N_incoming.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses_N_incoming.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses_N_outgoing = dev_dynamic_array_synapses_N_outgoing;
	ofstream outfile__dynamic_array_synapses_N_outgoing;
	outfile__dynamic_array_synapses_N_outgoing.open("results/_dynamic_array_synapses_N_outgoing_-3277140854151949897", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_outgoing.is_open())
	{
		outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_N_outgoing[0])), dev_dynamic_array_synapses_N_outgoing.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses_N_outgoing.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_synapses_post_delay = dev_dynamic_array_synapses_post_delay;
	ofstream outfile__dynamic_array_synapses_post_delay;
	outfile__dynamic_array_synapses_post_delay.open("results/_dynamic_array_synapses_post_delay_-7661039483155161074", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_post_delay.is_open())
	{
		outfile__dynamic_array_synapses_post_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_post_delay[0])), dev_dynamic_array_synapses_post_delay.size()*sizeof(double));
		outfile__dynamic_array_synapses_post_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_post_delay." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses_post_spiking_synapses = dev_dynamic_array_synapses_post_spiking_synapses;
	ofstream outfile__dynamic_array_synapses_post_spiking_synapses;
	outfile__dynamic_array_synapses_post_spiking_synapses.open("results/_dynamic_array_synapses_post_spiking_synapses_-1366635816657035108", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_post_spiking_synapses.is_open())
	{
		outfile__dynamic_array_synapses_post_spiking_synapses.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_post_spiking_synapses[0])), dev_dynamic_array_synapses_post_spiking_synapses.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses_post_spiking_synapses.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_post_spiking_synapses." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_synapses_pre_delay = dev_dynamic_array_synapses_pre_delay;
	ofstream outfile__dynamic_array_synapses_pre_delay;
	outfile__dynamic_array_synapses_pre_delay.open("results/_dynamic_array_synapses_pre_delay_7653745164894875960", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_pre_delay.is_open())
	{
		outfile__dynamic_array_synapses_pre_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_pre_delay[0])), dev_dynamic_array_synapses_pre_delay.size()*sizeof(double));
		outfile__dynamic_array_synapses_pre_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_pre_delay." << endl;
	}
	thrust::host_vector<int32_t> temp_dynamic_array_synapses_pre_spiking_synapses = dev_dynamic_array_synapses_pre_spiking_synapses;
	ofstream outfile__dynamic_array_synapses_pre_spiking_synapses;
	outfile__dynamic_array_synapses_pre_spiking_synapses.open("results/_dynamic_array_synapses_pre_spiking_synapses_5132394361331489466", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_pre_spiking_synapses.is_open())
	{
		outfile__dynamic_array_synapses_pre_spiking_synapses.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_pre_spiking_synapses[0])), dev_dynamic_array_synapses_pre_spiking_synapses.size()*sizeof(int32_t));
		outfile__dynamic_array_synapses_pre_spiking_synapses.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_pre_spiking_synapses." << endl;
	}
	thrust::host_vector<double> temp_dynamic_array_synapses_w = dev_dynamic_array_synapses_w;
	ofstream outfile__dynamic_array_synapses_w;
	outfile__dynamic_array_synapses_w.open("results/_dynamic_array_synapses_w_-2614024316621171204", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_w.is_open())
	{
		outfile__dynamic_array_synapses_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp_dynamic_array_synapses_w[0])), dev_dynamic_array_synapses_w.size()*sizeof(double));
		outfile__dynamic_array_synapses_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_w." << endl;
	}

		ofstream outfile__dynamic_array_statemonitor_1_v;
		outfile__dynamic_array_statemonitor_1_v.open("results/_dynamic_array_statemonitor_1_v_-8409868653121327112", ios::binary | ios::out);
		if(outfile__dynamic_array_statemonitor_1_v.is_open())
		{
			thrust::host_vector<double>* temp_array_dynamic_array_statemonitor_1_v = new thrust::host_vector<double>[_num__array_statemonitor_1__indices];
	        for (int n=0; n<_num__array_statemonitor_1__indices; n++)
	        {
	        	temp_array_dynamic_array_statemonitor_1_v[n] = _dynamic_array_statemonitor_1_v[n];
	        }
	        for(int j = 0; j < temp_array_dynamic_array_statemonitor_1_v[0].size(); j++)
	        {
	        	for(int i = 0; i < _num__array_statemonitor_1__indices; i++)
	        	{
		        	outfile__dynamic_array_statemonitor_1_v.write(reinterpret_cast<char*>(&temp_array_dynamic_array_statemonitor_1_v[i][j]), sizeof(double));
	        	}
	        }
	        outfile__dynamic_array_statemonitor_1_v.close();
		} else
		{
			std::cout << "Error writing output file for _dynamic_array_statemonitor_1_v." << endl;
		}
		ofstream outfile__dynamic_array_statemonitor_w;
		outfile__dynamic_array_statemonitor_w.open("results/_dynamic_array_statemonitor_w_6620044162385838775", ios::binary | ios::out);
		if(outfile__dynamic_array_statemonitor_w.is_open())
		{
			thrust::host_vector<double>* temp_array_dynamic_array_statemonitor_w = new thrust::host_vector<double>[_num__array_statemonitor__indices];
	        for (int n=0; n<_num__array_statemonitor__indices; n++)
	        {
	        	temp_array_dynamic_array_statemonitor_w[n] = _dynamic_array_statemonitor_w[n];
	        }
	        for(int j = 0; j < temp_array_dynamic_array_statemonitor_w[0].size(); j++)
	        {
	        	for(int i = 0; i < _num__array_statemonitor__indices; i++)
	        	{
		        	outfile__dynamic_array_statemonitor_w.write(reinterpret_cast<char*>(&temp_array_dynamic_array_statemonitor_w[i][j]), sizeof(double));
	        	}
	        }
	        outfile__dynamic_array_statemonitor_w.close();
		} else
		{
			std::cout << "Error writing output file for _dynamic_array_statemonitor_w." << endl;
		}
}

__global__ void synapses_post_destroy()
{
	using namespace brian;

	synapses_post.destroy();
}
__global__ void synapses_pre_destroy()
{
	using namespace brian;

	synapses_pre.destroy();
}
__global__ void synapses_1_pre_destroy()
{
	using namespace brian;

	synapses_1_pre.destroy();
}

void _dealloc_arrays()
{
	using namespace brian;


	synapses_post_destroy<<<1,1>>>();
	synapses_pre_destroy<<<1,1>>>();
	synapses_1_pre_destroy<<<1,1>>>();

	dev_dynamic_array_spikegeneratorgroup_neuron_index.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup_neuron_index);
	dev_dynamic_array_spikegeneratorgroup_spike_number.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup_spike_number);
	dev_dynamic_array_spikegeneratorgroup_spike_time.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_spikegeneratorgroup_spike_time);
	dev_dynamic_array_spikemonitor_i.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_i);
	dev_dynamic_array_spikemonitor_t.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_t);
	dev_dynamic_array_statemonitor_1_t.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_statemonitor_1_t);
	dev_dynamic_array_statemonitor_t.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_statemonitor_t);
	dev_dynamic_array_synapses_1__synaptic_post.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_post);
	dev_dynamic_array_synapses_1__synaptic_pre.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_pre);
	dev_dynamic_array_synapses_1_lastupdate.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_lastupdate);
	dev_dynamic_array_synapses_1_N_incoming.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_incoming);
	dev_dynamic_array_synapses_1_N_outgoing.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_outgoing);
	dev_dynamic_array_synapses_1_pre_delay.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_pre_delay);
	dev_dynamic_array_synapses_1_pre_spiking_synapses.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_pre_spiking_synapses);
	dev_dynamic_array_synapses_1_w.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_w);
	dev_dynamic_array_synapses__synaptic_post.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_post);
	dev_dynamic_array_synapses__synaptic_pre.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_pre);
	dev_dynamic_array_synapses_Apost.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_synapses_Apost);
	dev_dynamic_array_synapses_Apre.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_synapses_Apre);
	dev_dynamic_array_synapses_lastupdate.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_synapses_lastupdate);
	dev_dynamic_array_synapses_N_incoming.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_incoming);
	dev_dynamic_array_synapses_N_outgoing.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_outgoing);
	dev_dynamic_array_synapses_post_delay.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_synapses_post_delay);
	dev_dynamic_array_synapses_post_spiking_synapses.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_post_spiking_synapses);
	dev_dynamic_array_synapses_pre_delay.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_synapses_pre_delay);
	dev_dynamic_array_synapses_pre_spiking_synapses.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_pre_spiking_synapses);
	dev_dynamic_array_synapses_w.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_synapses_w);

	if(_array_defaultclock_dt!=0)
	{
		delete [] _array_defaultclock_dt;
		_array_defaultclock_dt = 0;
	}
	if(dev_array_defaultclock_dt!=0)
	{
		cudaFree(dev_array_defaultclock_dt);
		dev_array_defaultclock_dt = 0;
	}
	if(_array_defaultclock_t!=0)
	{
		delete [] _array_defaultclock_t;
		_array_defaultclock_t = 0;
	}
	if(dev_array_defaultclock_t!=0)
	{
		cudaFree(dev_array_defaultclock_t);
		dev_array_defaultclock_t = 0;
	}
	if(_array_defaultclock_timestep!=0)
	{
		delete [] _array_defaultclock_timestep;
		_array_defaultclock_timestep = 0;
	}
	if(dev_array_defaultclock_timestep!=0)
	{
		cudaFree(dev_array_defaultclock_timestep);
		dev_array_defaultclock_timestep = 0;
	}
	if(_array_neurongroup__spikespace!=0)
	{
		delete [] _array_neurongroup__spikespace;
		_array_neurongroup__spikespace = 0;
	}
	if(dev_array_neurongroup__spikespace!=0)
	{
		cudaFree(dev_array_neurongroup__spikespace);
		dev_array_neurongroup__spikespace = 0;
	}
	if(_array_neurongroup_g!=0)
	{
		delete [] _array_neurongroup_g;
		_array_neurongroup_g = 0;
	}
	if(dev_array_neurongroup_g!=0)
	{
		cudaFree(dev_array_neurongroup_g);
		dev_array_neurongroup_g = 0;
	}
	if(_array_neurongroup_i!=0)
	{
		delete [] _array_neurongroup_i;
		_array_neurongroup_i = 0;
	}
	if(dev_array_neurongroup_i!=0)
	{
		cudaFree(dev_array_neurongroup_i);
		dev_array_neurongroup_i = 0;
	}
	if(_array_neurongroup_lastspike!=0)
	{
		delete [] _array_neurongroup_lastspike;
		_array_neurongroup_lastspike = 0;
	}
	if(dev_array_neurongroup_lastspike!=0)
	{
		cudaFree(dev_array_neurongroup_lastspike);
		dev_array_neurongroup_lastspike = 0;
	}
	if(_array_neurongroup_not_refractory!=0)
	{
		delete [] _array_neurongroup_not_refractory;
		_array_neurongroup_not_refractory = 0;
	}
	if(dev_array_neurongroup_not_refractory!=0)
	{
		cudaFree(dev_array_neurongroup_not_refractory);
		dev_array_neurongroup_not_refractory = 0;
	}
	if(_array_neurongroup_v!=0)
	{
		delete [] _array_neurongroup_v;
		_array_neurongroup_v = 0;
	}
	if(dev_array_neurongroup_v!=0)
	{
		cudaFree(dev_array_neurongroup_v);
		dev_array_neurongroup_v = 0;
	}
	if(_array_spikegeneratorgroup__lastindex!=0)
	{
		delete [] _array_spikegeneratorgroup__lastindex;
		_array_spikegeneratorgroup__lastindex = 0;
	}
	if(dev_array_spikegeneratorgroup__lastindex!=0)
	{
		cudaFree(dev_array_spikegeneratorgroup__lastindex);
		dev_array_spikegeneratorgroup__lastindex = 0;
	}
	if(_array_spikegeneratorgroup__spikespace!=0)
	{
		delete [] _array_spikegeneratorgroup__spikespace;
		_array_spikegeneratorgroup__spikespace = 0;
	}
	if(dev_array_spikegeneratorgroup__spikespace!=0)
	{
		cudaFree(dev_array_spikegeneratorgroup__spikespace);
		dev_array_spikegeneratorgroup__spikespace = 0;
	}
	if(_array_spikegeneratorgroup_i!=0)
	{
		delete [] _array_spikegeneratorgroup_i;
		_array_spikegeneratorgroup_i = 0;
	}
	if(dev_array_spikegeneratorgroup_i!=0)
	{
		cudaFree(dev_array_spikegeneratorgroup_i);
		dev_array_spikegeneratorgroup_i = 0;
	}
	if(_array_spikegeneratorgroup_period!=0)
	{
		delete [] _array_spikegeneratorgroup_period;
		_array_spikegeneratorgroup_period = 0;
	}
	if(dev_array_spikegeneratorgroup_period!=0)
	{
		cudaFree(dev_array_spikegeneratorgroup_period);
		dev_array_spikegeneratorgroup_period = 0;
	}
	if(_array_spikemonitor__source_idx!=0)
	{
		delete [] _array_spikemonitor__source_idx;
		_array_spikemonitor__source_idx = 0;
	}
	if(dev_array_spikemonitor__source_idx!=0)
	{
		cudaFree(dev_array_spikemonitor__source_idx);
		dev_array_spikemonitor__source_idx = 0;
	}
	if(_array_spikemonitor_count!=0)
	{
		delete [] _array_spikemonitor_count;
		_array_spikemonitor_count = 0;
	}
	if(dev_array_spikemonitor_count!=0)
	{
		cudaFree(dev_array_spikemonitor_count);
		dev_array_spikemonitor_count = 0;
	}
	if(_array_spikemonitor_N!=0)
	{
		delete [] _array_spikemonitor_N;
		_array_spikemonitor_N = 0;
	}
	if(dev_array_spikemonitor_N!=0)
	{
		cudaFree(dev_array_spikemonitor_N);
		dev_array_spikemonitor_N = 0;
	}
	if(_array_statemonitor_1__indices!=0)
	{
		delete [] _array_statemonitor_1__indices;
		_array_statemonitor_1__indices = 0;
	}
	if(dev_array_statemonitor_1__indices!=0)
	{
		cudaFree(dev_array_statemonitor_1__indices);
		dev_array_statemonitor_1__indices = 0;
	}
	if(_array_statemonitor_1_N!=0)
	{
		delete [] _array_statemonitor_1_N;
		_array_statemonitor_1_N = 0;
	}
	if(dev_array_statemonitor_1_N!=0)
	{
		cudaFree(dev_array_statemonitor_1_N);
		dev_array_statemonitor_1_N = 0;
	}
	if(_array_statemonitor_1_v!=0)
	{
		delete [] _array_statemonitor_1_v;
		_array_statemonitor_1_v = 0;
	}
	if(dev_array_statemonitor_1_v!=0)
	{
		cudaFree(dev_array_statemonitor_1_v);
		dev_array_statemonitor_1_v = 0;
	}
	if(_array_statemonitor__indices!=0)
	{
		delete [] _array_statemonitor__indices;
		_array_statemonitor__indices = 0;
	}
	if(dev_array_statemonitor__indices!=0)
	{
		cudaFree(dev_array_statemonitor__indices);
		dev_array_statemonitor__indices = 0;
	}
	if(_array_statemonitor_N!=0)
	{
		delete [] _array_statemonitor_N;
		_array_statemonitor_N = 0;
	}
	if(dev_array_statemonitor_N!=0)
	{
		cudaFree(dev_array_statemonitor_N);
		dev_array_statemonitor_N = 0;
	}
	if(_array_statemonitor_w!=0)
	{
		delete [] _array_statemonitor_w;
		_array_statemonitor_w = 0;
	}
	if(dev_array_statemonitor_w!=0)
	{
		cudaFree(dev_array_statemonitor_w);
		dev_array_statemonitor_w = 0;
	}
	if(_array_synapses_1_N!=0)
	{
		delete [] _array_synapses_1_N;
		_array_synapses_1_N = 0;
	}
	if(dev_array_synapses_1_N!=0)
	{
		cudaFree(dev_array_synapses_1_N);
		dev_array_synapses_1_N = 0;
	}
	if(_array_synapses_N!=0)
	{
		delete [] _array_synapses_N;
		_array_synapses_N = 0;
	}
	if(dev_array_synapses_N!=0)
	{
		cudaFree(dev_array_synapses_N);
		dev_array_synapses_N = 0;
	}

	for(int i = 0; i < _num__array_statemonitor_1__indices; i++)
	{
		_dynamic_array_statemonitor_1_v[i].clear();
		thrust::device_vector<double>().swap(_dynamic_array_statemonitor_1_v[i]);
	}
	addresses_monitor__dynamic_array_statemonitor_1_v.clear();
	thrust::device_vector<double*>().swap(addresses_monitor__dynamic_array_statemonitor_1_v);
	for(int i = 0; i < _num__array_statemonitor__indices; i++)
	{
		_dynamic_array_statemonitor_w[i].clear();
		thrust::device_vector<double>().swap(_dynamic_array_statemonitor_w[i]);
	}
	addresses_monitor__dynamic_array_statemonitor_w.clear();
	thrust::device_vector<double*>().swap(addresses_monitor__dynamic_array_statemonitor_w);

	// static arrays
	if(_static_array__array_neurongroup_v!=0)
	{
		delete [] _static_array__array_neurongroup_v;
		_static_array__array_neurongroup_v = 0;
	}
	if(_static_array__array_statemonitor_1__indices!=0)
	{
		delete [] _static_array__array_statemonitor_1__indices;
		_static_array__array_statemonitor_1__indices = 0;
	}
	if(_static_array__array_statemonitor__indices!=0)
	{
		delete [] _static_array__array_statemonitor__indices;
		_static_array__array_statemonitor__indices = 0;
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
	if(_static_array__dynamic_array_synapses_w!=0)
	{
		delete [] _static_array__dynamic_array_synapses_w;
		_static_array__dynamic_array_synapses_w = 0;
	}
}

