{% macro cu_file() %}

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/cuda_utils.h"
#include "network.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <utility>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>

size_t brian::used_device_memory = 0;

//////////////// clocks ///////////////////
{% for clock in clocks | sort(attribute='name') %}
Clock brian::{{clock.name}};
{% endfor %}

//////////////// networks /////////////////
{% for net in networks | sort(attribute='name') %}
Network brian::{{net.name}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
{% if not var in dynamic_array_specs %}
{{c_data_type(var.dtype)}} * brian::{{varname}};
{{c_data_type(var.dtype)}} * brian::dev{{varname}};
__device__ {{c_data_type(var.dtype)}} * brian::d{{varname}};
const int brian::_num_{{varname}} = {{var.size}};

{% endif %}
{% endfor %}

//////////////// eventspaces ///////////////
// we dynamically create multiple eventspaces in no_or_const_delay_mode
// for initiating the first spikespace, we need a host pointer
// for choosing the right spikespace, we need a global index variable
{% for var, varname in eventspace_arrays | dictsort(by='value') %}
{{c_data_type(var.dtype)}} * brian::{{varname}};
const int brian::_num_{{varname}} = {{var.size}};
thrust::host_vector<{{c_data_type(var.dtype)}}*> brian::dev{{varname}}(1);
int brian::current_idx{{varname}} = 0;
{% endfor %}

//////////////// dynamic arrays 1d /////////
{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
thrust::host_vector<{{c_data_type(var.dtype)}}> brian::{{varname}};
thrust::device_vector<{{c_data_type(var.dtype)}}> brian::dev{{varname}};
{% endfor %}

//////////////// dynamic arrays 2d /////////
{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
thrust::device_vector<{{c_data_type(var.dtype)}}*> brian::addresses_monitor_{{varname}};
thrust::device_vector<{{c_data_type(var.dtype)}}>* brian::{{varname}};
{% endfor %}

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
{# arrays that are initialized from static data are already declared #}
{% if not (name in array_specs.values() or name in dynamic_array_specs.values() or name in dynamic_array_2d_specs.values())%}
{{dtype_spec}} * brian::{{name}};
{{dtype_spec}} * brian::dev{{name}};
__device__ {{dtype_spec}} * brian::d{{name}};
const int brian::_num_{{name}} = {{N}};
{% endif %}
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses | sort(attribute='name') %}
// {{S.name}}
Synapses<double> brian::{{S.name}}({{S.source|length}}, {{S.target|length}});
int32_t {{S.name}}_source_start_index;
int32_t {{S.name}}_source_stop_index;
bool brian::{{S.name}}_multiple_pre_post = false;
{% for path in S._pathways | sort(attribute='name') %}
// {{path.name}}
__device__ unsigned int* brian::{{path.name}}_size_by_pre;
__device__ unsigned int* brian::{{path.name}}_size_by_bundle_id;
__device__ unsigned int* brian::{{path.name}}_global_bundle_id_start_idx_by_pre;
unsigned int brian::{{path.name}}_max_bundle_size = 0;
unsigned int brian::{{path.name}}_mean_bundle_size = 0;
unsigned int brian::{{path.name}}_max_size = 0;
__device__ unsigned int* brian::{{path.name}}_unique_delay_size_by_pre;
unsigned int brian::{{path.name}}_max_num_bundles = 0;
__device__ int32_t** brian::{{path.name}}_synapses_id_by_pre;
__device__ int32_t** brian::{{path.name}}_synapses_id_by_bundle_id;
__device__ unsigned int** brian::{{path.name}}_unique_delay_start_idx_by_pre;
__device__ unsigned int** brian::{{path.name}}_unique_delay_by_pre;
__device__ SynapticPathway<double> brian::{{path.name}};
int brian::{{path.name}}_eventspace_idx = 0;
unsigned int brian::{{path.name}}_delay;
bool brian::{{path.name}}_scalar_delay;
{% endfor %}
{% endfor %}

unsigned int brian::num_parallel_blocks;
unsigned int brian::max_threads_per_block;
unsigned int brian::max_threads_per_sm;
unsigned int brian::max_shared_mem_size;
unsigned int brian::num_threads_per_warp;

{% for S in synapses | sort(attribute='name') %}
{% for path in S._pathways | sort(attribute='name') %}
__global__ void {{path.name}}_init(
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

	{{path.name}}.init(Nsource, Ntarget, delays, sources, targets, dt, start, stop);
}
{% endfor %}
{% endfor %}

// timers for profiling
{% for codeobj in active_objects %}
timer_type brian::{{codeobj}}_timer_start;
timer_type brian::{{codeobj}}_timer_stop;
{% endfor %}
timer_type brian::random_number_generation_timer_start;
timer_type brian::random_number_generation_timer_stop;

// profiling infos
{% for codeobj in active_objects %}
double brian::{{codeobj}}_profiling_info = 0.0;
{% endfor %}
double brian::random_number_generation_profiling_info = 0.0;

//////////////random numbers//////////////////
curandGenerator_t brian::curand_generator;
{% for co in codeobj_with_rand | sort(attribute='name') %}
randomNumber_t* brian::dev_{{co.name}}_rand;
randomNumber_t* brian::dev_{{co.name}}_rand_allocator;
__device__ randomNumber_t* brian::_array_{{co.name}}_rand;
{% endfor %}
{% for co in codeobj_with_randn | sort(attribute='name') %}
randomNumber_t* brian::dev_{{co.name}}_randn;
randomNumber_t* brian::dev_{{co.name}}_randn_allocator;
__device__ randomNumber_t* brian::_array_{{co.name}}_randn;
{% endfor %}

void _init_arrays()
{
	using namespace brian;

	std::clock_t start_timer = std::clock();

	CUDA_CHECK_MEMORY();
	size_t used_device_memory_start = used_device_memory;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	

	{% if num_parallel_blocks %}
	num_parallel_blocks = {{num_parallel_blocks}};
	{% else %}
	num_parallel_blocks = props.multiProcessorCount * {{sm_multiplier}};
	{% endif %}
	printf("objects cu num par blocks %d\n", num_parallel_blocks);
	max_threads_per_block = props.maxThreadsPerBlock;
	max_threads_per_sm = props.maxThreadsPerMultiProcessor;
	max_shared_mem_size = props.sharedMemPerBlock;
	num_threads_per_warp = props.warpSize;
	
	curandCreateGenerator(&curand_generator, {{curand_generator_type}});
	{% if curand_generator_ordering %}
	curandSetGeneratorOrdering(curand_generator, {{curand_generator_ordering}});
	{% endif %}
	// These random seeds might be overwritten in main.cu
	curandSetPseudoRandomGeneratorSeed(curand_generator, time(0));

	{% for S in synapses | sort(attribute='name') %}
	{% for path in S._pathways | sort(attribute='name') %}
	{{path.name}}_init<<<1,1>>>(
			{{path.source|length}},
			{{path.target|length}},
			thrust::raw_pointer_cast(&dev{{dynamic_array_specs[path.variables['delay']]}}[0]),
			thrust::raw_pointer_cast(&dev{{dynamic_array_specs[path.synapse_sources]}}[0]),
			thrust::raw_pointer_cast(&dev{{dynamic_array_specs[path.synapse_targets]}}[0]),
			0,	//was dt, maybe irrelevant?
			{{path.source.start}},
			{{path.source.stop}}
			);
	{% endfor %}
	{% endfor %}

	{% if profile and profile != 'blocking' %}
	// Create cudaEvents for profiling
	{% for codeobj in active_objects %}
	cudaEventCreate(&{{codeobj}}_timer_start);
	cudaEventCreate(&{{codeobj}}_timer_stop);
	{% endfor %}
	cudaEventCreate(&random_number_generation_timer_start);
	cudaEventCreate(&random_number_generation_timer_stop);
	{% endif %}

    // Arrays initialized to 0
	{% for var in zero_arrays | sort(attribute='name') %}
		{% if not (var in dynamic_array_specs or var in eventspace_arrays) %}
			{% set varname = array_specs[var] %}
			{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
			for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = 0;
			cudaMalloc((void**)&dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
			if(!dev{{varname}})
			{
				printf("ERROR while allocating {{varname}} on device with size %ld\n", sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
			}
			cudaMemcpy(dev{{varname}}, {{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyHostToDevice);
		
		{% endif %}
		{% if (var in dynamic_array_specs) %}
			{% set varname = array_specs[var] %}
			_dynamic{{varname}}.resize({{var.size}});
			dev_dynamic{{varname}}.resize({{var.size}});
		        for(int i=0; i<{{var.size}}; i++)
			{
				_dynamic{{varname}}[i] = 0;
				dev_dynamic{{varname}}[i] = 0;
			}
		
		{% endif %}
	{% endfor %}

	// Arrays initialized to an "arange"
	{% for var, start in arange_arrays %}
	{% set varname = array_specs[var] %}
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = {{start}} + i;
	cudaMalloc((void**)&dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
	if(!dev{{varname}})
	{
		printf("ERROR while allocating {{varname}} on device with size %ld\n", sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
	}
	cudaMemcpy(dev{{varname}}, {{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyHostToDevice);

	{% endfor %}

	// static arrays
	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	{% if (name in dynamic_array_specs.values())  %}
	{{name}}.resize({{N}});
	dev{{name}}.resize({{N}});
	{% else %}
	{{name}} = new {{dtype_spec}}[{{N}}];
	cudaMalloc((void**)&dev{{name}}, sizeof({{dtype_spec}})*{{N}});
	if(!dev{{name}})
	{
		printf("ERROR while allocating {{name}} on device with size %ld\n", sizeof({{dtype_spec}})*{{N}});
	}
	cudaMemcpyToSymbol(d{{name}}, &dev{{name}}, sizeof({{dtype_spec}}*));
	{% endif %}
	{% endfor %}

	{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
	{{varname}} = new thrust::device_vector<{{c_data_type(var.dtype)}}>[_num__array_{{var.owner.name}}__indices];
	{% endfor %}

	// eventspace_arrays
	{% for var, varname in eventspace_arrays | dictsort(by='value') %}
	cudaMalloc((void**)&dev{{varname}}[0], sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	{% endfor %}

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

	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	ifstream f{{name}};
	f{{name}}.open("static_arrays/{{name}}", ios::in | ios::binary);
	if(f{{name}}.is_open())
	{
	    {% if name in dynamic_array_specs.values() %}
	    f{{name}}.read(reinterpret_cast<char*>(&{{name}}[0]), {{N}}*sizeof({{dtype_spec}}));
	    {% else %}
		f{{name}}.read(reinterpret_cast<char*>({{name}}), {{N}}*sizeof({{dtype_spec}}));
		{% endif %}
	} else
	{
		std::cout << "Error opening static array {{name}}." << endl;
	}
	{% if not (name in dynamic_array_specs.values()) %}
	cudaMemcpy(dev{{name}}, {{name}}, sizeof({{dtype_spec}})*{{N}}, cudaMemcpyHostToDevice);
	{% else %}
    for(int i=0; i<{{N}}; i++)
    {
        dev{{name}}[i] = {{name}}[i];
    }
	{% endif %}
	{% endfor %}
}	

void _write_arrays()
{
	using namespace brian;

	{% for var, varname in array_specs | dictsort(by='value') %}
	{% if not (var in dynamic_array_specs or var in dynamic_array_2d_specs or var in static_array_specs) %}
	cudaMemcpy({{varname}}, dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyDeviceToHost);
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("{{get_array_filename(var) | replace('\\', '\\\\')}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
		outfile_{{varname}}.write(reinterpret_cast<char*>({{varname}}), {{var.size}}*sizeof({{c_data_type(var.dtype)}}));
		outfile_{{varname}}.close();
	} else
	{
		std::cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endif %}
	{% endfor %}

	{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
	{% if not var in multisynaptic_idx_vars and not var.name == 'delay' %}
	{{varname}} = dev{{varname}};
	{% endif %}
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("{{get_array_filename(var) | replace('\\', '\\\\')}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
		outfile_{{varname}}.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&{{varname}}[0])), {{varname}}.size()*sizeof({{c_data_type(var.dtype)}}));
		outfile_{{varname}}.close();
	} else
	{
		std::cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endfor %}

	{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
		ofstream outfile_{{varname}};
		outfile_{{varname}}.open("{{get_array_filename(var) | replace('\\', '\\\\')}}", ios::binary | ios::out);
		if(outfile_{{varname}}.is_open())
		{
			thrust::host_vector<{{c_data_type(var.dtype)}}>* temp_array{{varname}} = new thrust::host_vector<{{c_data_type(var.dtype)}}>[_num__array_{{var.owner.name}}__indices];
	        for (int n=0; n<_num__array_{{var.owner.name}}__indices; n++)
	        {
	        	temp_array{{varname}}[n] = {{varname}}[n];
	        }
	        for(int j = 0; j < temp_array{{varname}}[0].size(); j++)
	        {
	        	for(int i = 0; i < _num__array_{{var.owner.name}}__indices; i++)
	        	{
		        	outfile_{{varname}}.write(reinterpret_cast<char*>(&temp_array{{varname}}[i][j]), sizeof({{c_data_type(var.dtype)}}));
	        	}
	        }
	        outfile_{{varname}}.close();
		} else
		{
			std::cout << "Error writing output file for {{varname}}." << endl;
		}
		{% endfor %}

	// Write profiling info to disk
	ofstream outfile_profiling_info;
	outfile_profiling_info.open("results/profiling_info.txt", ios::out);
	if(outfile_profiling_info.is_open())
	{
	{% for obj in active_objects %}
	outfile_profiling_info << "{{obj}}\t" << {{obj}}_profiling_info << std::endl;
	{% endfor %}
	outfile_profiling_info << "random_number_generation\t" << random_number_generation_profiling_info << std::endl;
	outfile_profiling_info.close();
	} else
	{
	    std::cout << "Error writing profiling info to file." << std::endl;
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

{% for S in synapses | sort(attribute='name') %}
{% for path in S._pathways | sort(attribute='name') %}
__global__ void {{path.name}}_destroy()
{
	using namespace brian;

	{{path.name}}.destroy();
}
{% endfor %}
{% endfor %}

void _dealloc_arrays()
{
	using namespace brian;

	{% if profile and profile != 'blocking' %}
	// Destroy cudaEvents for profiling
	{% for codeobj in active_objects %}
	cudaEventDestroy({{codeobj}}_timer_start);
	cudaEventDestroy({{codeobj}}_timer_stop);
	{% endfor %}
	cudaEventDestroy(random_number_generation_timer_start);
	cudaEventDestroy(random_number_generation_timer_stop);
	{% endif %}

	{% for co in codeobj_with_rand | sort(attribute='name') %}
	cudaFree(dev_{{co.name}}_rand_allocator);
	{% endfor %}
	{% for co in codeobj_with_randn | sort(attribute='name') %}
	cudaFree(dev_{{co.name}}_randn_allocator);
	{% endfor %}

	{% for S in synapses | sort(attribute='name') %}
	{% for path in S._pathways | sort(attribute='name') %}
	{{path.name}}_destroy<<<1,1>>>();
	{% endfor %}
	{% endfor %}

	{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
	dev{{varname}}.clear();
	thrust::device_vector<{{c_data_type(var.dtype)}}>().swap(dev{{varname}});
	{{varname}}.clear();
	thrust::host_vector<{{c_data_type(var.dtype)}}>().swap({{varname}});
	{% endfor %}

	{% for var, varname in array_specs | dictsort(by='value') %}
	{% if not var in dynamic_array_specs %}
	if({{varname}}!=0)
	{
		delete [] {{varname}};
		{{varname}} = 0;
	}
	if(dev{{varname}}!=0)
	{
		cudaFree(dev{{varname}});
		dev{{varname}} = 0;
	}
	{% endif %}
	{% endfor %}

	{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
	for(int i = 0; i < _num__array_{{var.owner.name}}__indices; i++)
	{
		{{varname}}[i].clear();
		thrust::device_vector<{{c_data_type(var.dtype)}}>().swap({{varname}}[i]);
	}
	addresses_monitor_{{varname}}.clear();
	thrust::device_vector<{{c_data_type(var.dtype)}}*>().swap(addresses_monitor_{{varname}});
	{% endfor %}

	// static arrays
	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	{% if not (name in dynamic_array_specs.values()) %}
	if({{name}}!=0)
	{
		delete [] {{name}};
		{{name}} = 0;
	}
	{% endif %}
	{% endfor %}

}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}
#include <ctime>
// typedefs need to be outside the include guards to
// be visible to all files including objects.h
{% if not profile or profile == 'blocking' %}
typedef std::clock_t timer_type;
{% else %}
typedef cudaEvent_t timer_type;
{% endif %}
typedef {{curand_float_type}} randomNumber_t;  // random number type

#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "network.h"

#include <thrust/device_vector.h>
#include <curand.h>

namespace brian {

extern size_t used_device_memory;

//////////////// clocks ///////////////////
{% for clock in clocks %}
extern Clock {{clock.name}};
{% endfor %}

//////////////// networks /////////////////
{% for net in networks %}
extern Network {{net.name}};
{% endfor %}

//////////////// dynamic arrays ///////////
{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
extern thrust::host_vector<{{c_data_type(var.dtype)}}> {{varname}};
extern thrust::device_vector<{{c_data_type(var.dtype)}}> dev{{varname}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
{% if not var in dynamic_array_specs %}
extern {{c_data_type(var.dtype)}} * {{varname}};
extern {{c_data_type(var.dtype)}} * dev{{varname}};
extern __device__ {{c_data_type(var.dtype)}} *d{{varname}};
extern const int _num_{{varname}};
{% endif %}
{% endfor %}

//////////////// eventspaces ///////////////
{% for var, varname in eventspace_arrays | dictsort(by='value') %}
extern {{c_data_type(var.dtype)}} * {{varname}};
extern thrust::host_vector<{{c_data_type(var.dtype)}}*> dev{{varname}};
extern const int _num_{{varname}};
extern int current_idx{{varname}};
{% endfor %}

//////////////// dynamic arrays 2d /////////
{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
extern thrust::device_vector<{{c_data_type(var.dtype)}}*> addresses_monitor_{{varname}};
extern thrust::device_vector<{{c_data_type(var.dtype)}}>* {{varname}};
{% endfor %}

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
{# arrays that are initialized from static data are already declared #}
{% if not (name in array_specs.values() or name in dynamic_array_specs.values() or name in dynamic_array_2d_specs.values())%}
extern {{dtype_spec}} *{{name}};
extern {{dtype_spec}} *dev{{name}};
extern __device__ {{dtype_spec}} *d{{name}};
extern const int _num_{{name}};
{% endif %}
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses | sort(attribute='name') %}
// {{S.name}}
extern Synapses<double> {{S.name}};
extern bool {{S.name}}_multiple_pre_post;
{% for path in S._pathways | sort(attribute='name') %}
extern __device__ unsigned int* {{path.name}}_size_by_pre;
extern __device__ unsigned int* {{path.name}}_size_by_bundle_id;
extern __device__ unsigned int* {{path.name}}_global_bundle_id_start_idx_by_pre;
extern unsigned int {{path.name}}_max_bundle_size;
extern unsigned int {{path.name}}_mean_bundle_size;
extern unsigned int {{path.name}}_max_size;
extern __device__ unsigned int* {{path.name}}_unique_delay_size_by_pre;
extern unsigned int {{path.name}}_max_num_bundles;
extern __device__ int32_t** {{path.name}}_synapses_id_by_pre;
extern __device__ int32_t** {{path.name}}_synapses_id_by_bundle_id;
extern __device__ unsigned int** {{path.name}}_unique_delay_start_idx_by_pre;
extern __device__ unsigned int** {{path.name}}_unique_delay_by_pre;
extern __device__ SynapticPathway<double> {{path.name}};
extern int {{path.name}}_eventspace_idx;
extern unsigned int {{path.name}}_delay;
extern bool {{path.name}}_scalar_delay;
{% endfor %}
{% endfor %}

// timers for profiling
{% for codeobj in active_objects %}
extern timer_type {{codeobj}}_timer_start;
extern timer_type {{codeobj}}_timer_stop;
{% endfor %}
extern timer_type random_number_generation_timer_start;
extern timer_type random_number_generation_timer_stop;

// profiling infos
{% for codeobj in active_objects %}
extern double {{codeobj}}_profiling_info;
{% endfor %}
extern double random_number_generation_profiling_info;

//////////////// random numbers /////////////////
extern curandGenerator_t curand_generator;

{% for co in codeobj_with_rand | sort(attribute='name') %}
extern randomNumber_t* dev_{{co.name}}_rand;
extern randomNumber_t* dev_{{co.name}}_rand_allocator;
extern __device__ randomNumber_t* _array_{{co.name}}_rand;
{% endfor %}
{% for co in codeobj_with_randn | sort(attribute='name') %}
extern randomNumber_t* dev_{{co.name}}_randn;
extern randomNumber_t* dev_{{co.name}}_randn_allocator;
extern __device__ randomNumber_t* _array_{{co.name}}_randn;
{% endfor %}

//CUDA
extern unsigned int num_parallel_blocks;
extern unsigned int max_threads_per_block;
extern unsigned int max_threads_per_sm;
extern unsigned int max_shared_mem_size;
extern unsigned int num_threads_per_warp;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


{% endmacro %}
