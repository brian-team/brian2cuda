
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
const int brian::_num__array_neurongroup__spikespace = 10001;

int32_t * brian::_array_neurongroup_i;
int32_t * brian::dev_array_neurongroup_i;
__device__ int32_t * brian::d_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 10000;

double * brian::_array_neurongroup_v;
double * brian::dev_array_neurongroup_v;
__device__ double * brian::d_array_neurongroup_v;
const int brian::_num__array_neurongroup_v = 10000;

int32_t * brian::_array_spikemonitor__source_idx;
int32_t * brian::dev_array_spikemonitor__source_idx;
__device__ int32_t * brian::d_array_spikemonitor__source_idx;
const int brian::_num__array_spikemonitor__source_idx = 10000;

int32_t * brian::_array_spikemonitor_count;
int32_t * brian::dev_array_spikemonitor_count;
__device__ int32_t * brian::d_array_spikemonitor_count;
const int brian::_num__array_spikemonitor_count = 10000;

int32_t * brian::_array_spikemonitor_N;
int32_t * brian::dev_array_spikemonitor_N;
__device__ int32_t * brian::d_array_spikemonitor_N;
const int brian::_num__array_spikemonitor_N = 1;


//////////////// dynamic arrays 1d /////////
std::vector<int32_t> brian::_dynamic_array_spikemonitor_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_i;
std::vector<double> brian::_dynamic_array_spikemonitor_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_t;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////

//////////////// synapses /////////////////

unsigned int brian::num_parallel_blocks;
unsigned int brian::max_threads_per_block;
unsigned int brian::max_shared_mem_size;


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


    // Arrays initialized to 0
			_array_spikemonitor__source_idx = new int32_t[10000];
			for(int i=0; i<10000; i++) _array_spikemonitor__source_idx[i] = 0;
			cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx);
			if(!dev_array_spikemonitor__source_idx)
			{
				printf("ERROR while allocating _array_spikemonitor__source_idx on device with size %ld\n", sizeof(int32_t)*_num__array_spikemonitor__source_idx);
			}
			cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice);
		
			_array_neurongroup__spikespace = new int32_t[10001];
			for(int i=0; i<10001; i++) _array_neurongroup__spikespace[i] = 0;
			cudaMalloc((void**)&dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace);
			if(!dev_array_neurongroup__spikespace)
			{
				printf("ERROR while allocating _array_neurongroup__spikespace on device with size %ld\n", sizeof(int32_t)*_num__array_neurongroup__spikespace);
			}
			cudaMemcpy(dev_array_neurongroup__spikespace, _array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyHostToDevice);
		
			_array_spikemonitor_count = new int32_t[10000];
			for(int i=0; i<10000; i++) _array_spikemonitor_count[i] = 0;
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
		
			_array_neurongroup_i = new int32_t[10000];
			for(int i=0; i<10000; i++) _array_neurongroup_i[i] = 0;
			cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i);
			if(!dev_array_neurongroup_i)
			{
				printf("ERROR while allocating _array_neurongroup_i on device with size %ld\n", sizeof(int32_t)*_num__array_neurongroup_i);
			}
			cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice);
		
			_array_spikemonitor_N = new int32_t[1];
			for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;
			cudaMalloc((void**)&dev_array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N);
			if(!dev_array_spikemonitor_N)
			{
				printf("ERROR while allocating _array_spikemonitor_N on device with size %ld\n", sizeof(int32_t)*_num__array_spikemonitor_N);
			}
			cudaMemcpy(dev_array_spikemonitor_N, _array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N, cudaMemcpyHostToDevice);
		
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
		
			_array_neurongroup_v = new double[10000];
			for(int i=0; i<10000; i++) _array_neurongroup_v[i] = 0;
			cudaMalloc((void**)&dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v);
			if(!dev_array_neurongroup_v)
			{
				printf("ERROR while allocating _array_neurongroup_v on device with size %ld\n", sizeof(double)*_num__array_neurongroup_v);
			}
			cudaMemcpy(dev_array_neurongroup_v, _array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyHostToDevice);
		

	// Arrays initialized to an "arange"
	_array_spikemonitor__source_idx = new int32_t[10000];
	for(int i=0; i<10000; i++) _array_spikemonitor__source_idx[i] = 0 + i;
	cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx);
	if(!dev_array_spikemonitor__source_idx)
	{
		printf("ERROR while allocating _array_spikemonitor__source_idx on device with size %ld\n", sizeof(int32_t)*_num__array_spikemonitor__source_idx);
	}
	cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice);

	_array_neurongroup_i = new int32_t[10000];
	for(int i=0; i<10000; i++) _array_neurongroup_i[i] = 0 + i;
	cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i);
	if(!dev_array_neurongroup_i)
	{
		printf("ERROR while allocating _array_neurongroup_i on device with size %ld\n", sizeof(int32_t)*_num__array_neurongroup_i);
	}
	cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice);


	// static arrays

}

void _load_arrays()
{
	using namespace brian;

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
		outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 10001*sizeof(int32_t));
		outfile__array_neurongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
	}
	cudaMemcpy(_array_neurongroup_i, dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyDeviceToHost);
	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open("results/_array_neurongroup_i_-2688036259655650195", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 10000*sizeof(int32_t));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}
	cudaMemcpy(_array_neurongroup_v, dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyDeviceToHost);
	ofstream outfile__array_neurongroup_v;
	outfile__array_neurongroup_v.open("results/_array_neurongroup_v_-2688036259655650190", ios::binary | ios::out);
	if(outfile__array_neurongroup_v.is_open())
	{
		outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 10000*sizeof(double));
		outfile__array_neurongroup_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_v." << endl;
	}
	cudaMemcpy(_array_spikemonitor__source_idx, dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyDeviceToHost);
	ofstream outfile__array_spikemonitor__source_idx;
	outfile__array_spikemonitor__source_idx.open("results/_array_spikemonitor__source_idx_-4045852888739339153", ios::binary | ios::out);
	if(outfile__array_spikemonitor__source_idx.is_open())
	{
		outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 10000*sizeof(int32_t));
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
		outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 10000*sizeof(int32_t));
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

}


void _dealloc_arrays()
{
	using namespace brian;



	dev_dynamic_array_spikemonitor_i.clear();
	thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_i);
	dev_dynamic_array_spikemonitor_t.clear();
	thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_t);

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


	// static arrays
}

