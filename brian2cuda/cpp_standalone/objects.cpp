
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<vector>
#include<iostream>
#include<fstream>

//////////////// networks /////////////////
Network brian::magicnetwork;

//////////////// arrays ///////////////////
double * brian::_array_defaultclock_dt;
const int brian::_num__array_defaultclock_dt = 1;
double * brian::_array_defaultclock_t;
const int brian::_num__array_defaultclock_t = 1;
uint64_t * brian::_array_defaultclock_timestep;
const int brian::_num__array_defaultclock_timestep = 1;
int32_t * brian::_array_neurongroup__spikespace;
const int brian::_num__array_neurongroup__spikespace = 101;
double * brian::_array_neurongroup_g;
const int brian::_num__array_neurongroup_g = 100;
int32_t * brian::_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 100;
double * brian::_array_neurongroup_lastspike;
const int brian::_num__array_neurongroup_lastspike = 100;
char * brian::_array_neurongroup_not_refractory;
const int brian::_num__array_neurongroup_not_refractory = 100;
double * brian::_array_neurongroup_v;
const int brian::_num__array_neurongroup_v = 100;
int32_t * brian::_array_spikegeneratorgroup__lastindex;
const int brian::_num__array_spikegeneratorgroup__lastindex = 1;
int32_t * brian::_array_spikegeneratorgroup__spikespace;
const int brian::_num__array_spikegeneratorgroup__spikespace = 101;
int32_t * brian::_array_spikegeneratorgroup_i;
const int brian::_num__array_spikegeneratorgroup_i = 100;
double * brian::_array_spikegeneratorgroup_period;
const int brian::_num__array_spikegeneratorgroup_period = 1;
int32_t * brian::_array_spikemonitor__source_idx;
const int brian::_num__array_spikemonitor__source_idx = 100;
int32_t * brian::_array_spikemonitor_count;
const int brian::_num__array_spikemonitor_count = 100;
int32_t * brian::_array_spikemonitor_N;
const int brian::_num__array_spikemonitor_N = 1;
int32_t * brian::_array_statemonitor_1__indices;
const int brian::_num__array_statemonitor_1__indices = 10;
int32_t * brian::_array_statemonitor_1_N;
const int brian::_num__array_statemonitor_1_N = 1;
double * brian::_array_statemonitor_1_v;
const int brian::_num__array_statemonitor_1_v = (0, 10);
int32_t * brian::_array_statemonitor__indices;
const int brian::_num__array_statemonitor__indices = 10;
int32_t * brian::_array_statemonitor_N;
const int brian::_num__array_statemonitor_N = 1;
double * brian::_array_statemonitor_w;
const int brian::_num__array_statemonitor_w = (0, 10);
int32_t * brian::_array_synapses_1_N;
const int brian::_num__array_synapses_1_N = 1;
int32_t * brian::_array_synapses_N;
const int brian::_num__array_synapses_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<int32_t> brian::_dynamic_array_spikegeneratorgroup_neuron_index;
std::vector<int32_t> brian::_dynamic_array_spikegeneratorgroup_spike_number;
std::vector<double> brian::_dynamic_array_spikegeneratorgroup_spike_time;
std::vector<int32_t> brian::_dynamic_array_spikemonitor_i;
std::vector<double> brian::_dynamic_array_spikemonitor_t;
std::vector<double> brian::_dynamic_array_statemonitor_1_t;
std::vector<double> brian::_dynamic_array_statemonitor_t;
std::vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_post;
std::vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_pre;
std::vector<double> brian::_dynamic_array_synapses_1_lastupdate;
std::vector<int32_t> brian::_dynamic_array_synapses_1_N_incoming;
std::vector<int32_t> brian::_dynamic_array_synapses_1_N_outgoing;
std::vector<double> brian::_dynamic_array_synapses_1_pre_delay;
std::vector<int32_t> brian::_dynamic_array_synapses_1_pre_spiking_synapses;
std::vector<double> brian::_dynamic_array_synapses_1_w;
std::vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
std::vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
std::vector<double> brian::_dynamic_array_synapses_Apost;
std::vector<double> brian::_dynamic_array_synapses_Apre;
std::vector<double> brian::_dynamic_array_synapses_lastupdate;
std::vector<int32_t> brian::_dynamic_array_synapses_N_incoming;
std::vector<int32_t> brian::_dynamic_array_synapses_N_outgoing;
std::vector<double> brian::_dynamic_array_synapses_post_delay;
std::vector<int32_t> brian::_dynamic_array_synapses_post_spiking_synapses;
std::vector<double> brian::_dynamic_array_synapses_pre_delay;
std::vector<int32_t> brian::_dynamic_array_synapses_pre_spiking_synapses;
std::vector<double> brian::_dynamic_array_synapses_w;

//////////////// dynamic arrays 2d /////////
DynamicArray2D<double> brian::_dynamic_array_statemonitor_1_v;
DynamicArray2D<double> brian::_dynamic_array_statemonitor_w;

/////////////// static arrays /////////////
double * brian::_static_array__array_neurongroup_v;
const int brian::_num__static_array__array_neurongroup_v = 100;
int32_t * brian::_static_array__array_statemonitor_1__indices;
const int brian::_num__static_array__array_statemonitor_1__indices = 10;
int32_t * brian::_static_array__array_statemonitor__indices;
const int brian::_num__static_array__array_statemonitor__indices = 10;
int64_t * brian::_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_neuron_index = 1000;
int64_t * brian::_static_array__dynamic_array_spikegeneratorgroup_spike_number;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_spike_number = 1000;
double * brian::_static_array__dynamic_array_spikegeneratorgroup_spike_time;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_spike_time = 1000;
double * brian::_static_array__dynamic_array_synapses_w;
const int brian::_num__static_array__dynamic_array_synapses_w = 10000;

//////////////// synapses /////////////////
// synapses
SynapticPathway<double> brian::synapses_post(
		_dynamic_array_synapses_post_delay,
		_dynamic_array_synapses__synaptic_post,
		0, 100);
SynapticPathway<double> brian::synapses_pre(
		_dynamic_array_synapses_pre_delay,
		_dynamic_array_synapses__synaptic_pre,
		0, 100);
// synapses_1
SynapticPathway<double> brian::synapses_1_pre(
		_dynamic_array_synapses_1_pre_delay,
		_dynamic_array_synapses_1__synaptic_pre,
		0, 100);

//////////////// clocks ///////////////////
Clock brian::defaultclock;  // attributes will be set in run.cpp

// Profiling information for each code object
double brian::neurongroup_resetter_codeobject_profiling_info = 0.0;
double brian::neurongroup_stateupdater_codeobject_profiling_info = 0.0;
double brian::neurongroup_thresholder_codeobject_profiling_info = 0.0;
double brian::spikegeneratorgroup_codeobject_profiling_info = 0.0;
double brian::spikemonitor_codeobject_profiling_info = 0.0;
double brian::statemonitor_1_codeobject_profiling_info = 0.0;
double brian::statemonitor_codeobject_profiling_info = 0.0;
double brian::synapses_1_group_variable_set_conditional_codeobject_profiling_info = 0.0;
double brian::synapses_1_pre_codeobject_profiling_info = 0.0;
double brian::synapses_1_pre_initialise_queue_profiling_info = 0.0;
double brian::synapses_1_pre_push_spikes_profiling_info = 0.0;
double brian::synapses_1_synapses_create_codeobject_profiling_info = 0.0;
double brian::synapses_group_variable_set_conditional_codeobject_profiling_info = 0.0;
double brian::synapses_post_codeobject_profiling_info = 0.0;
double brian::synapses_post_initialise_queue_profiling_info = 0.0;
double brian::synapses_post_push_spikes_profiling_info = 0.0;
double brian::synapses_pre_codeobject_profiling_info = 0.0;
double brian::synapses_pre_initialise_queue_profiling_info = 0.0;
double brian::synapses_pre_push_spikes_profiling_info = 0.0;
double brian::synapses_synapses_create_codeobject_profiling_info = 0.0;

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_statemonitor__indices = new int32_t[10];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<10; i++) _array_statemonitor__indices[i] = 0;
	_array_statemonitor_1__indices = new int32_t[10];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<10; i++) _array_statemonitor_1__indices[i] = 0;
	_array_spikegeneratorgroup__lastindex = new int32_t[1];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<1; i++) _array_spikegeneratorgroup__lastindex[i] = 0;
	_array_spikemonitor__source_idx = new int32_t[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0;
	_array_neurongroup__spikespace = new int32_t[101];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<101; i++) _array_neurongroup__spikespace[i] = 0;
	_array_spikegeneratorgroup__spikespace = new int32_t[101];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<101; i++) _array_spikegeneratorgroup__spikespace[i] = 0;
	_array_spikemonitor_count = new int32_t[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_spikemonitor_count[i] = 0;
	_array_defaultclock_dt = new double[1];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;
	_array_neurongroup_g = new double[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_neurongroup_g[i] = 0;
	_array_neurongroup_i = new int32_t[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_neurongroup_i[i] = 0;
	_array_spikegeneratorgroup_i = new int32_t[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_spikegeneratorgroup_i[i] = 0;
	_array_neurongroup_lastspike = new double[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_neurongroup_lastspike[i] = 0;
	_array_synapses_N = new int32_t[1];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<1; i++) _array_synapses_N[i] = 0;
	_array_synapses_1_N = new int32_t[1];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;
	_array_spikemonitor_N = new int32_t[1];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;
	_array_statemonitor_N = new int32_t[1];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<1; i++) _array_statemonitor_N[i] = 0;
	_array_statemonitor_1_N = new int32_t[1];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<1; i++) _array_statemonitor_1_N[i] = 0;
	_array_neurongroup_not_refractory = new char[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_neurongroup_not_refractory[i] = 0;
	_array_spikegeneratorgroup_period = new double[1];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<1; i++) _array_spikegeneratorgroup_period[i] = 0;
	_array_defaultclock_t = new double[1];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;
	_array_defaultclock_timestep = new uint64_t[1];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;
	_array_neurongroup_v = new double[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_neurongroup_v[i] = 0;

	// Arrays initialized to an "arange"
	_array_spikemonitor__source_idx = new int32_t[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0 + i;
	_array_neurongroup_i = new int32_t[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_neurongroup_i[i] = 0 + i;
	_array_spikegeneratorgroup_i = new int32_t[100];
	#pragma omp parallel for schedule(static)
	for(int i=0; i<100; i++) _array_spikegeneratorgroup_i[i] = 0 + i;

	// static arrays
	_static_array__array_neurongroup_v = new double[100];
	_static_array__array_statemonitor_1__indices = new int32_t[10];
	_static_array__array_statemonitor__indices = new int32_t[10];
	_static_array__dynamic_array_spikegeneratorgroup_neuron_index = new int64_t[1000];
	_static_array__dynamic_array_spikegeneratorgroup_spike_number = new int64_t[1000];
	_static_array__dynamic_array_spikegeneratorgroup_spike_time = new double[1000];
	_static_array__dynamic_array_synapses_w = new double[10000];
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
	ifstream f_static_array__array_statemonitor_1__indices;
	f_static_array__array_statemonitor_1__indices.open("static_arrays/_static_array__array_statemonitor_1__indices", ios::in | ios::binary);
	if(f_static_array__array_statemonitor_1__indices.is_open())
	{
		f_static_array__array_statemonitor_1__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor_1__indices), 10*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_statemonitor_1__indices." << endl;
	}
	ifstream f_static_array__array_statemonitor__indices;
	f_static_array__array_statemonitor__indices.open("static_arrays/_static_array__array_statemonitor__indices", ios::in | ios::binary);
	if(f_static_array__array_statemonitor__indices.is_open())
	{
		f_static_array__array_statemonitor__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor__indices), 10*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_statemonitor__indices." << endl;
	}
	ifstream f_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
	f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_neuron_index", ios::in | ios::binary);
	if(f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
	{
		f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_neuron_index), 1000*sizeof(int64_t));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_neuron_index." << endl;
	}
	ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_number;
	f_static_array__dynamic_array_spikegeneratorgroup_spike_number.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_number", ios::in | ios::binary);
	if(f_static_array__dynamic_array_spikegeneratorgroup_spike_number.is_open())
	{
		f_static_array__dynamic_array_spikegeneratorgroup_spike_number.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_number), 1000*sizeof(int64_t));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_number." << endl;
	}
	ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_time;
	f_static_array__dynamic_array_spikegeneratorgroup_spike_time.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_time", ios::in | ios::binary);
	if(f_static_array__dynamic_array_spikegeneratorgroup_spike_time.is_open())
	{
		f_static_array__dynamic_array_spikegeneratorgroup_spike_time.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_time), 1000*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_time." << endl;
	}
	ifstream f_static_array__dynamic_array_synapses_w;
	f_static_array__dynamic_array_synapses_w.open("static_arrays/_static_array__dynamic_array_synapses_w", ios::in | ios::binary);
	if(f_static_array__dynamic_array_synapses_w.is_open())
	{
		f_static_array__dynamic_array_synapses_w.read(reinterpret_cast<char*>(_static_array__dynamic_array_synapses_w), 10000*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_synapses_w." << endl;
	}
}

void _write_arrays()
{
	using namespace brian;

	ofstream outfile__array_defaultclock_dt;
	outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_-9215759865592636245", ios::binary | ios::out);
	if(outfile__array_defaultclock_dt.is_open())
	{
		outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
		outfile__array_defaultclock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
	}
	ofstream outfile__array_defaultclock_t;
	outfile__array_defaultclock_t.open("results/_array_defaultclock_t_7263079326120112646", ios::binary | ios::out);
	if(outfile__array_defaultclock_t.is_open())
	{
		outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
		outfile__array_defaultclock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_t." << endl;
	}
	ofstream outfile__array_defaultclock_timestep;
	outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_-8300011050550298960", ios::binary | ios::out);
	if(outfile__array_defaultclock_timestep.is_open())
	{
		outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
		outfile__array_defaultclock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
	}
	ofstream outfile__array_neurongroup__spikespace;
	outfile__array_neurongroup__spikespace.open("results/_array_neurongroup__spikespace_6291509255835556833", ios::binary | ios::out);
	if(outfile__array_neurongroup__spikespace.is_open())
	{
		outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 101*sizeof(_array_neurongroup__spikespace[0]));
		outfile__array_neurongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_g;
	outfile__array_neurongroup_g.open("results/_array_neurongroup_g_-2688036259655650205", ios::binary | ios::out);
	if(outfile__array_neurongroup_g.is_open())
	{
		outfile__array_neurongroup_g.write(reinterpret_cast<char*>(_array_neurongroup_g), 100*sizeof(_array_neurongroup_g[0]));
		outfile__array_neurongroup_g.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_g." << endl;
	}
	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open("results/_array_neurongroup_i_-2688036259655650195", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 100*sizeof(_array_neurongroup_i[0]));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}
	ofstream outfile__array_neurongroup_lastspike;
	outfile__array_neurongroup_lastspike.open("results/_array_neurongroup_lastspike_-2029934132282177282", ios::binary | ios::out);
	if(outfile__array_neurongroup_lastspike.is_open())
	{
		outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), 100*sizeof(_array_neurongroup_lastspike[0]));
		outfile__array_neurongroup_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
	}
	ofstream outfile__array_neurongroup_not_refractory;
	outfile__array_neurongroup_not_refractory.open("results/_array_neurongroup_not_refractory_-8596383477608809440", ios::binary | ios::out);
	if(outfile__array_neurongroup_not_refractory.is_open())
	{
		outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), 100*sizeof(_array_neurongroup_not_refractory[0]));
		outfile__array_neurongroup_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
	}
	ofstream outfile__array_neurongroup_v;
	outfile__array_neurongroup_v.open("results/_array_neurongroup_v_-2688036259655650190", ios::binary | ios::out);
	if(outfile__array_neurongroup_v.is_open())
	{
		outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 100*sizeof(_array_neurongroup_v[0]));
		outfile__array_neurongroup_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_v." << endl;
	}
	ofstream outfile__array_spikegeneratorgroup__lastindex;
	outfile__array_spikegeneratorgroup__lastindex.open("results/_array_spikegeneratorgroup__lastindex_7651580413293037816", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup__lastindex.is_open())
	{
		outfile__array_spikegeneratorgroup__lastindex.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__lastindex), 1*sizeof(_array_spikegeneratorgroup__lastindex[0]));
		outfile__array_spikegeneratorgroup__lastindex.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup__lastindex." << endl;
	}
	ofstream outfile__array_spikegeneratorgroup__spikespace;
	outfile__array_spikegeneratorgroup__spikespace.open("results/_array_spikegeneratorgroup__spikespace_-1541437192670537063", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup__spikespace.is_open())
	{
		outfile__array_spikegeneratorgroup__spikespace.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__spikespace), 101*sizeof(_array_spikegeneratorgroup__spikespace[0]));
		outfile__array_spikegeneratorgroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup__spikespace." << endl;
	}
	ofstream outfile__array_spikegeneratorgroup_i;
	outfile__array_spikegeneratorgroup_i.open("results/_array_spikegeneratorgroup_i_-7676153806036632795", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup_i.is_open())
	{
		outfile__array_spikegeneratorgroup_i.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_i), 100*sizeof(_array_spikegeneratorgroup_i[0]));
		outfile__array_spikegeneratorgroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup_i." << endl;
	}
	ofstream outfile__array_spikegeneratorgroup_period;
	outfile__array_spikegeneratorgroup_period.open("results/_array_spikegeneratorgroup_period_-5885060602818549350", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup_period.is_open())
	{
		outfile__array_spikegeneratorgroup_period.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_period), 1*sizeof(_array_spikegeneratorgroup_period[0]));
		outfile__array_spikegeneratorgroup_period.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup_period." << endl;
	}
	ofstream outfile__array_spikemonitor__source_idx;
	outfile__array_spikemonitor__source_idx.open("results/_array_spikemonitor__source_idx_-4045852888739339153", ios::binary | ios::out);
	if(outfile__array_spikemonitor__source_idx.is_open())
	{
		outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 100*sizeof(_array_spikemonitor__source_idx[0]));
		outfile__array_spikemonitor__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
	}
	ofstream outfile__array_spikemonitor_count;
	outfile__array_spikemonitor_count.open("results/_array_spikemonitor_count_-3651895352503201284", ios::binary | ios::out);
	if(outfile__array_spikemonitor_count.is_open())
	{
		outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 100*sizeof(_array_spikemonitor_count[0]));
		outfile__array_spikemonitor_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
	}
	ofstream outfile__array_spikemonitor_N;
	outfile__array_spikemonitor_N.open("results/_array_spikemonitor_N_73938390545997659", ios::binary | ios::out);
	if(outfile__array_spikemonitor_N.is_open())
	{
		outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(_array_spikemonitor_N[0]));
		outfile__array_spikemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
	}
	ofstream outfile__array_statemonitor_1__indices;
	outfile__array_statemonitor_1__indices.open("results/_array_statemonitor_1__indices_5442903941946222241", ios::binary | ios::out);
	if(outfile__array_statemonitor_1__indices.is_open())
	{
		outfile__array_statemonitor_1__indices.write(reinterpret_cast<char*>(_array_statemonitor_1__indices), 10*sizeof(_array_statemonitor_1__indices[0]));
		outfile__array_statemonitor_1__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor_1__indices." << endl;
	}
	ofstream outfile__array_statemonitor_1_N;
	outfile__array_statemonitor_1_N.open("results/_array_statemonitor_1_N_2048012943032035014", ios::binary | ios::out);
	if(outfile__array_statemonitor_1_N.is_open())
	{
		outfile__array_statemonitor_1_N.write(reinterpret_cast<char*>(_array_statemonitor_1_N), 1*sizeof(_array_statemonitor_1_N[0]));
		outfile__array_statemonitor_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor_1_N." << endl;
	}
	ofstream outfile__array_statemonitor__indices;
	outfile__array_statemonitor__indices.open("results/_array_statemonitor__indices_6163485638831984707", ios::binary | ios::out);
	if(outfile__array_statemonitor__indices.is_open())
	{
		outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 10*sizeof(_array_statemonitor__indices[0]));
		outfile__array_statemonitor__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
	}
	ofstream outfile__array_statemonitor_N;
	outfile__array_statemonitor_N.open("results/_array_statemonitor_N_1126150466128921572", ios::binary | ios::out);
	if(outfile__array_statemonitor_N.is_open())
	{
		outfile__array_statemonitor_N.write(reinterpret_cast<char*>(_array_statemonitor_N), 1*sizeof(_array_statemonitor_N[0]));
		outfile__array_statemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor_N." << endl;
	}
	ofstream outfile__array_synapses_1_N;
	outfile__array_synapses_1_N.open("results/_array_synapses_1_N_-7473518110119523383", ios::binary | ios::out);
	if(outfile__array_synapses_1_N.is_open())
	{
		outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(_array_synapses_1_N[0]));
		outfile__array_synapses_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_1_N." << endl;
	}
	ofstream outfile__array_synapses_N;
	outfile__array_synapses_N.open("results/_array_synapses_N_-7833853409752232273", ios::binary | ios::out);
	if(outfile__array_synapses_N.is_open())
	{
		outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(_array_synapses_N[0]));
		outfile__array_synapses_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N." << endl;
	}

	ofstream outfile__dynamic_array_spikegeneratorgroup_neuron_index;
	outfile__dynamic_array_spikegeneratorgroup_neuron_index.open("results/_dynamic_array_spikegeneratorgroup_neuron_index_-8147356448785106711", ios::binary | ios::out);
	if(outfile__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
	{
        if (! _dynamic_array_spikegeneratorgroup_neuron_index.empty() )
			outfile__dynamic_array_spikegeneratorgroup_neuron_index.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup_neuron_index[0]), _dynamic_array_spikegeneratorgroup_neuron_index.size()*sizeof(_dynamic_array_spikegeneratorgroup_neuron_index[0]));
		outfile__dynamic_array_spikegeneratorgroup_neuron_index.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_neuron_index." << endl;
	}
	ofstream outfile__dynamic_array_spikegeneratorgroup_spike_number;
	outfile__dynamic_array_spikegeneratorgroup_spike_number.open("results/_dynamic_array_spikegeneratorgroup_spike_number_6285234774451964247", ios::binary | ios::out);
	if(outfile__dynamic_array_spikegeneratorgroup_spike_number.is_open())
	{
        if (! _dynamic_array_spikegeneratorgroup_spike_number.empty() )
			outfile__dynamic_array_spikegeneratorgroup_spike_number.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup_spike_number[0]), _dynamic_array_spikegeneratorgroup_spike_number.size()*sizeof(_dynamic_array_spikegeneratorgroup_spike_number[0]));
		outfile__dynamic_array_spikegeneratorgroup_spike_number.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_number." << endl;
	}
	ofstream outfile__dynamic_array_spikegeneratorgroup_spike_time;
	outfile__dynamic_array_spikegeneratorgroup_spike_time.open("results/_dynamic_array_spikegeneratorgroup_spike_time_6974015387653485977", ios::binary | ios::out);
	if(outfile__dynamic_array_spikegeneratorgroup_spike_time.is_open())
	{
        if (! _dynamic_array_spikegeneratorgroup_spike_time.empty() )
			outfile__dynamic_array_spikegeneratorgroup_spike_time.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup_spike_time[0]), _dynamic_array_spikegeneratorgroup_spike_time.size()*sizeof(_dynamic_array_spikegeneratorgroup_spike_time[0]));
		outfile__dynamic_array_spikegeneratorgroup_spike_time.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_time." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_i;
	outfile__dynamic_array_spikemonitor_i.open("results/_dynamic_array_spikemonitor_i_3873805716461528078", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_i.is_open())
	{
        if (! _dynamic_array_spikemonitor_i.empty() )
			outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_i[0]), _dynamic_array_spikemonitor_i.size()*sizeof(_dynamic_array_spikemonitor_i[0]));
		outfile__dynamic_array_spikemonitor_i.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_t;
	outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t_3873805716461528083", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_t.is_open())
	{
        if (! _dynamic_array_spikemonitor_t.empty() )
			outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_t[0]), _dynamic_array_spikemonitor_t.size()*sizeof(_dynamic_array_spikemonitor_t[0]));
		outfile__dynamic_array_spikemonitor_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_statemonitor_1_t;
	outfile__dynamic_array_statemonitor_1_t.open("results/_dynamic_array_statemonitor_1_t_-8409868653121327110", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_1_t.is_open())
	{
        if (! _dynamic_array_statemonitor_1_t.empty() )
			outfile__dynamic_array_statemonitor_1_t.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_1_t[0]), _dynamic_array_statemonitor_1_t.size()*sizeof(_dynamic_array_statemonitor_1_t[0]));
		outfile__dynamic_array_statemonitor_1_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_1_t." << endl;
	}
	ofstream outfile__dynamic_array_statemonitor_t;
	outfile__dynamic_array_statemonitor_t.open("results/_dynamic_array_statemonitor_t_6620044162385838772", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_t.is_open())
	{
        if (! _dynamic_array_statemonitor_t.empty() )
			outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_t[0]), _dynamic_array_statemonitor_t.size()*sizeof(_dynamic_array_statemonitor_t[0]));
		outfile__dynamic_array_statemonitor_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1__synaptic_post;
	outfile__dynamic_array_synapses_1__synaptic_post.open("results/_dynamic_array_synapses_1__synaptic_post_-4367449856540212009", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_1__synaptic_post.empty() )
			outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_post[0]), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(_dynamic_array_synapses_1__synaptic_post[0]));
		outfile__dynamic_array_synapses_1__synaptic_post.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
	outfile__dynamic_array_synapses_1__synaptic_pre.open("results/_dynamic_array_synapses_1__synaptic_pre_1368795276670783483", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_1__synaptic_pre.empty() )
			outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_pre[0]), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(_dynamic_array_synapses_1__synaptic_pre[0]));
		outfile__dynamic_array_synapses_1__synaptic_pre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_lastupdate;
	outfile__dynamic_array_synapses_1_lastupdate.open("results/_dynamic_array_synapses_1_lastupdate_6875119916677774017", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_lastupdate.is_open())
	{
        if (! _dynamic_array_synapses_1_lastupdate.empty() )
			outfile__dynamic_array_synapses_1_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_lastupdate[0]), _dynamic_array_synapses_1_lastupdate.size()*sizeof(_dynamic_array_synapses_1_lastupdate[0]));
		outfile__dynamic_array_synapses_1_lastupdate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_N_incoming;
	outfile__dynamic_array_synapses_1_N_incoming.open("results/_dynamic_array_synapses_1_N_incoming_-5364435978754666149", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_1_N_incoming.empty() )
			outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_incoming[0]), _dynamic_array_synapses_1_N_incoming.size()*sizeof(_dynamic_array_synapses_1_N_incoming[0]));
		outfile__dynamic_array_synapses_1_N_incoming.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_N_outgoing;
	outfile__dynamic_array_synapses_1_N_outgoing.open("results/_dynamic_array_synapses_1_N_outgoing_7721560298971024321", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_1_N_outgoing.empty() )
			outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_outgoing[0]), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(_dynamic_array_synapses_1_N_outgoing[0]));
		outfile__dynamic_array_synapses_1_N_outgoing.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_pre_delay;
	outfile__dynamic_array_synapses_1_pre_delay.open("results/_dynamic_array_synapses_1_pre_delay_6658020171927933066", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_pre_delay.is_open())
	{
        if (! _dynamic_array_synapses_1_pre_delay.empty() )
			outfile__dynamic_array_synapses_1_pre_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_pre_delay[0]), _dynamic_array_synapses_1_pre_delay.size()*sizeof(_dynamic_array_synapses_1_pre_delay[0]));
		outfile__dynamic_array_synapses_1_pre_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_pre_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_pre_spiking_synapses;
	outfile__dynamic_array_synapses_1_pre_spiking_synapses.open("results/_dynamic_array_synapses_1_pre_spiking_synapses_-5096904537968888248", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_pre_spiking_synapses.is_open())
	{
        if (! _dynamic_array_synapses_1_pre_spiking_synapses.empty() )
			outfile__dynamic_array_synapses_1_pre_spiking_synapses.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_pre_spiking_synapses[0]), _dynamic_array_synapses_1_pre_spiking_synapses.size()*sizeof(_dynamic_array_synapses_1_pre_spiking_synapses[0]));
		outfile__dynamic_array_synapses_1_pre_spiking_synapses.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_pre_spiking_synapses." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_w;
	outfile__dynamic_array_synapses_1_w.open("results/_dynamic_array_synapses_1_w_-1083981064091387634", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_w.is_open())
	{
        if (! _dynamic_array_synapses_1_w.empty() )
			outfile__dynamic_array_synapses_1_w.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_w[0]), _dynamic_array_synapses_1_w.size()*sizeof(_dynamic_array_synapses_1_w[0]));
		outfile__dynamic_array_synapses_1_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_w." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_post;
	outfile__dynamic_array_synapses__synaptic_post.open("results/_dynamic_array_synapses__synaptic_post_3840486125387374025", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses__synaptic_post.empty() )
			outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_post[0]), _dynamic_array_synapses__synaptic_post.size()*sizeof(_dynamic_array_synapses__synaptic_post[0]));
		outfile__dynamic_array_synapses__synaptic_post.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_pre;
	outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre_5162992210040840425", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses__synaptic_pre.empty() )
			outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_pre[0]), _dynamic_array_synapses__synaptic_pre.size()*sizeof(_dynamic_array_synapses__synaptic_pre[0]));
		outfile__dynamic_array_synapses__synaptic_pre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_Apost;
	outfile__dynamic_array_synapses_Apost.open("results/_dynamic_array_synapses_Apost_3034042580529184036", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_Apost.is_open())
	{
        if (! _dynamic_array_synapses_Apost.empty() )
			outfile__dynamic_array_synapses_Apost.write(reinterpret_cast<char*>(&_dynamic_array_synapses_Apost[0]), _dynamic_array_synapses_Apost.size()*sizeof(_dynamic_array_synapses_Apost[0]));
		outfile__dynamic_array_synapses_Apost.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_Apost." << endl;
	}
	ofstream outfile__dynamic_array_synapses_Apre;
	outfile__dynamic_array_synapses_Apre.open("results/_dynamic_array_synapses_Apre_-5633892405563205630", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_Apre.is_open())
	{
        if (! _dynamic_array_synapses_Apre.empty() )
			outfile__dynamic_array_synapses_Apre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_Apre[0]), _dynamic_array_synapses_Apre.size()*sizeof(_dynamic_array_synapses_Apre[0]));
		outfile__dynamic_array_synapses_Apre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_Apre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_lastupdate;
	outfile__dynamic_array_synapses_lastupdate.open("results/_dynamic_array_synapses_lastupdate_562699891839928247", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_lastupdate.is_open())
	{
        if (! _dynamic_array_synapses_lastupdate.empty() )
			outfile__dynamic_array_synapses_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_lastupdate[0]), _dynamic_array_synapses_lastupdate.size()*sizeof(_dynamic_array_synapses_lastupdate[0]));
		outfile__dynamic_array_synapses_lastupdate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_synapses_N_incoming;
	outfile__dynamic_array_synapses_N_incoming.open("results/_dynamic_array_synapses_N_incoming_6651214916728133133", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_N_incoming.empty() )
			outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_incoming[0]), _dynamic_array_synapses_N_incoming.size()*sizeof(_dynamic_array_synapses_N_incoming[0]));
		outfile__dynamic_array_synapses_N_incoming.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_N_outgoing;
	outfile__dynamic_array_synapses_N_outgoing.open("results/_dynamic_array_synapses_N_outgoing_-3277140854151949897", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_N_outgoing.empty() )
			outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_outgoing[0]), _dynamic_array_synapses_N_outgoing.size()*sizeof(_dynamic_array_synapses_N_outgoing[0]));
		outfile__dynamic_array_synapses_N_outgoing.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_post_delay;
	outfile__dynamic_array_synapses_post_delay.open("results/_dynamic_array_synapses_post_delay_-7661039483155161074", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_post_delay.is_open())
	{
        if (! _dynamic_array_synapses_post_delay.empty() )
			outfile__dynamic_array_synapses_post_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_post_delay[0]), _dynamic_array_synapses_post_delay.size()*sizeof(_dynamic_array_synapses_post_delay[0]));
		outfile__dynamic_array_synapses_post_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_post_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_post_spiking_synapses;
	outfile__dynamic_array_synapses_post_spiking_synapses.open("results/_dynamic_array_synapses_post_spiking_synapses_-1366635816657035108", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_post_spiking_synapses.is_open())
	{
        if (! _dynamic_array_synapses_post_spiking_synapses.empty() )
			outfile__dynamic_array_synapses_post_spiking_synapses.write(reinterpret_cast<char*>(&_dynamic_array_synapses_post_spiking_synapses[0]), _dynamic_array_synapses_post_spiking_synapses.size()*sizeof(_dynamic_array_synapses_post_spiking_synapses[0]));
		outfile__dynamic_array_synapses_post_spiking_synapses.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_post_spiking_synapses." << endl;
	}
	ofstream outfile__dynamic_array_synapses_pre_delay;
	outfile__dynamic_array_synapses_pre_delay.open("results/_dynamic_array_synapses_pre_delay_7653745164894875960", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_pre_delay.is_open())
	{
        if (! _dynamic_array_synapses_pre_delay.empty() )
			outfile__dynamic_array_synapses_pre_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_pre_delay[0]), _dynamic_array_synapses_pre_delay.size()*sizeof(_dynamic_array_synapses_pre_delay[0]));
		outfile__dynamic_array_synapses_pre_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_pre_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_pre_spiking_synapses;
	outfile__dynamic_array_synapses_pre_spiking_synapses.open("results/_dynamic_array_synapses_pre_spiking_synapses_5132394361331489466", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_pre_spiking_synapses.is_open())
	{
        if (! _dynamic_array_synapses_pre_spiking_synapses.empty() )
			outfile__dynamic_array_synapses_pre_spiking_synapses.write(reinterpret_cast<char*>(&_dynamic_array_synapses_pre_spiking_synapses[0]), _dynamic_array_synapses_pre_spiking_synapses.size()*sizeof(_dynamic_array_synapses_pre_spiking_synapses[0]));
		outfile__dynamic_array_synapses_pre_spiking_synapses.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_pre_spiking_synapses." << endl;
	}
	ofstream outfile__dynamic_array_synapses_w;
	outfile__dynamic_array_synapses_w.open("results/_dynamic_array_synapses_w_-2614024316621171204", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_w.is_open())
	{
        if (! _dynamic_array_synapses_w.empty() )
			outfile__dynamic_array_synapses_w.write(reinterpret_cast<char*>(&_dynamic_array_synapses_w[0]), _dynamic_array_synapses_w.size()*sizeof(_dynamic_array_synapses_w[0]));
		outfile__dynamic_array_synapses_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_w." << endl;
	}

	ofstream outfile__dynamic_array_statemonitor_1_v;
	outfile__dynamic_array_statemonitor_1_v.open("results/_dynamic_array_statemonitor_1_v_-8409868653121327112", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_1_v.is_open())
	{
        for (int n=0; n<_dynamic_array_statemonitor_1_v.n; n++)
        {
            if (! _dynamic_array_statemonitor_1_v(n).empty())
                outfile__dynamic_array_statemonitor_1_v.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_1_v(n, 0)), _dynamic_array_statemonitor_1_v.m*sizeof(_dynamic_array_statemonitor_1_v(0, 0)));
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
        for (int n=0; n<_dynamic_array_statemonitor_w.n; n++)
        {
            if (! _dynamic_array_statemonitor_w(n).empty())
                outfile__dynamic_array_statemonitor_w.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_w(n, 0)), _dynamic_array_statemonitor_w.m*sizeof(_dynamic_array_statemonitor_w(0, 0)));
        }
        outfile__dynamic_array_statemonitor_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_w." << endl;
	}

	// Write profiling info to disk
	ofstream outfile_profiling_info;
	outfile_profiling_info.open("results/profiling_info.txt", ios::out);
	if(outfile_profiling_info.is_open())
	{
	outfile_profiling_info << "neurongroup_resetter_codeobject\t" << neurongroup_resetter_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "neurongroup_stateupdater_codeobject\t" << neurongroup_stateupdater_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "neurongroup_thresholder_codeobject\t" << neurongroup_thresholder_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "spikegeneratorgroup_codeobject\t" << spikegeneratorgroup_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "spikemonitor_codeobject\t" << spikemonitor_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "statemonitor_1_codeobject\t" << statemonitor_1_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "statemonitor_codeobject\t" << statemonitor_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_1_group_variable_set_conditional_codeobject\t" << synapses_1_group_variable_set_conditional_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_1_pre_codeobject\t" << synapses_1_pre_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_1_pre_initialise_queue\t" << synapses_1_pre_initialise_queue_profiling_info << std::endl;
	outfile_profiling_info << "synapses_1_pre_push_spikes\t" << synapses_1_pre_push_spikes_profiling_info << std::endl;
	outfile_profiling_info << "synapses_1_synapses_create_codeobject\t" << synapses_1_synapses_create_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_group_variable_set_conditional_codeobject\t" << synapses_group_variable_set_conditional_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_post_codeobject\t" << synapses_post_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_post_initialise_queue\t" << synapses_post_initialise_queue_profiling_info << std::endl;
	outfile_profiling_info << "synapses_post_push_spikes\t" << synapses_post_push_spikes_profiling_info << std::endl;
	outfile_profiling_info << "synapses_pre_codeobject\t" << synapses_pre_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_pre_initialise_queue\t" << synapses_pre_initialise_queue_profiling_info << std::endl;
	outfile_profiling_info << "synapses_pre_push_spikes\t" << synapses_pre_push_spikes_profiling_info << std::endl;
	outfile_profiling_info << "synapses_synapses_create_codeobject\t" << synapses_synapses_create_codeobject_profiling_info << std::endl;
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

void _dealloc_arrays()
{
	using namespace brian;


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

