
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>
#include<iostream>
#include<fstream>

namespace brian {

std::vector< rk_state* > _mersenne_twister_states;

//////////////// networks /////////////////

//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_neurongroup_1__spikespace;
const int _num__array_neurongroup_1__spikespace = 101;
double * _array_neurongroup_1_g_eKC_eKC;
const int _num__array_neurongroup_1_g_eKC_eKC = 100;
double * _array_neurongroup_1_g_iKC_eKC;
const int _num__array_neurongroup_1_g_iKC_eKC = 100;
double * _array_neurongroup_1_h;
const int _num__array_neurongroup_1_h = 100;
int32_t * _array_neurongroup_1_i;
const int _num__array_neurongroup_1_i = 100;
double * _array_neurongroup_1_lastspike;
const int _num__array_neurongroup_1_lastspike = 100;
double * _array_neurongroup_1_m;
const int _num__array_neurongroup_1_m = 100;
double * _array_neurongroup_1_n;
const int _num__array_neurongroup_1_n = 100;
char * _array_neurongroup_1_not_refractory;
const int _num__array_neurongroup_1_not_refractory = 100;
double * _array_neurongroup_1_V;
const int _num__array_neurongroup_1_V = 100;
int32_t * _array_neurongroup__spikespace;
const int _num__array_neurongroup__spikespace = 2501;
double * _array_neurongroup_g_PN_iKC;
const int _num__array_neurongroup_g_PN_iKC = 2500;
double * _array_neurongroup_h;
const int _num__array_neurongroup_h = 2500;
int32_t * _array_neurongroup_i;
const int _num__array_neurongroup_i = 2500;
double * _array_neurongroup_lastspike;
const int _num__array_neurongroup_lastspike = 2500;
double * _array_neurongroup_m;
const int _num__array_neurongroup_m = 2500;
double * _array_neurongroup_n;
const int _num__array_neurongroup_n = 2500;
char * _array_neurongroup_not_refractory;
const int _num__array_neurongroup_not_refractory = 2500;
double * _array_neurongroup_V;
const int _num__array_neurongroup_V = 2500;
int32_t * _array_spikegeneratorgroup__lastindex;
const int _num__array_spikegeneratorgroup__lastindex = 1;
int32_t * _array_spikegeneratorgroup__period_bins;
const int _num__array_spikegeneratorgroup__period_bins = 1;
int32_t * _array_spikegeneratorgroup__spikespace;
const int _num__array_spikegeneratorgroup__spikespace = 101;
int32_t * _array_spikegeneratorgroup_i;
const int _num__array_spikegeneratorgroup_i = 100;
double * _array_spikegeneratorgroup_period;
const int _num__array_spikegeneratorgroup_period = 1;
int32_t * _array_spikemonitor_1__source_idx;
const int _num__array_spikemonitor_1__source_idx = 2500;
int32_t * _array_spikemonitor_1_count;
const int _num__array_spikemonitor_1_count = 2500;
int32_t * _array_spikemonitor_1_N;
const int _num__array_spikemonitor_1_N = 1;
int32_t * _array_spikemonitor_2__source_idx;
const int _num__array_spikemonitor_2__source_idx = 100;
int32_t * _array_spikemonitor_2_count;
const int _num__array_spikemonitor_2_count = 100;
int32_t * _array_spikemonitor_2_N;
const int _num__array_spikemonitor_2_N = 1;
int32_t * _array_spikemonitor__source_idx;
const int _num__array_spikemonitor__source_idx = 100;
int32_t * _array_spikemonitor_count;
const int _num__array_spikemonitor_count = 100;
int32_t * _array_spikemonitor_N;
const int _num__array_spikemonitor_N = 1;
int32_t * _array_synapses_1_N;
const int _num__array_synapses_1_N = 1;
int32_t * _array_synapses_2_N;
const int _num__array_synapses_2_N = 1;
int32_t * _array_synapses_N;
const int _num__array_synapses_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<int32_t> _dynamic_array_spikegeneratorgroup__timebins;
std::vector<int32_t> _dynamic_array_spikegeneratorgroup_neuron_index;
std::vector<int32_t> _dynamic_array_spikegeneratorgroup_spike_number;
std::vector<double> _dynamic_array_spikegeneratorgroup_spike_time;
std::vector<int32_t> _dynamic_array_spikemonitor_1_i;
std::vector<double> _dynamic_array_spikemonitor_1_t;
std::vector<int32_t> _dynamic_array_spikemonitor_2_i;
std::vector<double> _dynamic_array_spikemonitor_2_t;
std::vector<int32_t> _dynamic_array_spikemonitor_i;
std::vector<double> _dynamic_array_spikemonitor_t;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
std::vector<double> _dynamic_array_synapses_1_Apost;
std::vector<double> _dynamic_array_synapses_1_Apre;
std::vector<double> _dynamic_array_synapses_1_delay;
std::vector<double> _dynamic_array_synapses_1_delay_1;
std::vector<double> _dynamic_array_synapses_1_g_raw;
std::vector<double> _dynamic_array_synapses_1_lastupdate;
std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
std::vector<int32_t> _dynamic_array_synapses_2__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_2__synaptic_pre;
std::vector<double> _dynamic_array_synapses_2_delay;
std::vector<int32_t> _dynamic_array_synapses_2_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_2_N_outgoing;
std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
std::vector<double> _dynamic_array_synapses_delay;
std::vector<int32_t> _dynamic_array_synapses_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_N_outgoing;
std::vector<double> _dynamic_array_synapses_weight;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
int32_t * _static_array__dynamic_array_spikegeneratorgroup__timebins;
const int _num__static_array__dynamic_array_spikegeneratorgroup__timebins = 19676;
int64_t * _static_array__dynamic_array_spikegeneratorgroup_neuron_index;
const int _num__static_array__dynamic_array_spikegeneratorgroup_neuron_index = 19676;
int64_t * _static_array__dynamic_array_spikegeneratorgroup_spike_number;
const int _num__static_array__dynamic_array_spikegeneratorgroup_spike_number = 19676;
double * _static_array__dynamic_array_spikegeneratorgroup_spike_time;
const int _num__static_array__dynamic_array_spikegeneratorgroup_spike_time = 19676;

//////////////// synapses /////////////////
// synapses
SynapticPathway synapses_pre(
		_dynamic_array_synapses__synaptic_pre,
		0, 100);
// synapses_1
SynapticPathway synapses_1_post(
		_dynamic_array_synapses_1__synaptic_post,
		0, 100);
SynapticPathway synapses_1_pre(
		_dynamic_array_synapses_1__synaptic_pre,
		0, 2500);
// synapses_2
SynapticPathway synapses_2_pre(
		_dynamic_array_synapses_2__synaptic_pre,
		0, 100);

//////////////// clocks ///////////////////
Clock defaultclock;  // attributes will be set in run.cpp

}

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_defaultclock_dt = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

	_array_defaultclock_t = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

	_array_defaultclock_timestep = new int64_t[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

	_array_neurongroup_1__spikespace = new int32_t[101];
    
	for(int i=0; i<101; i++) _array_neurongroup_1__spikespace[i] = 0;

	_array_neurongroup_1_g_eKC_eKC = new double[100];
    
	for(int i=0; i<100; i++) _array_neurongroup_1_g_eKC_eKC[i] = 0;

	_array_neurongroup_1_g_iKC_eKC = new double[100];
    
	for(int i=0; i<100; i++) _array_neurongroup_1_g_iKC_eKC[i] = 0;

	_array_neurongroup_1_h = new double[100];
    
	for(int i=0; i<100; i++) _array_neurongroup_1_h[i] = 0;

	_array_neurongroup_1_i = new int32_t[100];
    
	for(int i=0; i<100; i++) _array_neurongroup_1_i[i] = 0;

	_array_neurongroup_1_lastspike = new double[100];
    
	for(int i=0; i<100; i++) _array_neurongroup_1_lastspike[i] = 0;

	_array_neurongroup_1_m = new double[100];
    
	for(int i=0; i<100; i++) _array_neurongroup_1_m[i] = 0;

	_array_neurongroup_1_n = new double[100];
    
	for(int i=0; i<100; i++) _array_neurongroup_1_n[i] = 0;

	_array_neurongroup_1_not_refractory = new char[100];
    
	for(int i=0; i<100; i++) _array_neurongroup_1_not_refractory[i] = 0;

	_array_neurongroup_1_V = new double[100];
    
	for(int i=0; i<100; i++) _array_neurongroup_1_V[i] = 0;

	_array_neurongroup__spikespace = new int32_t[2501];
    
	for(int i=0; i<2501; i++) _array_neurongroup__spikespace[i] = 0;

	_array_neurongroup_g_PN_iKC = new double[2500];
    
	for(int i=0; i<2500; i++) _array_neurongroup_g_PN_iKC[i] = 0;

	_array_neurongroup_h = new double[2500];
    
	for(int i=0; i<2500; i++) _array_neurongroup_h[i] = 0;

	_array_neurongroup_i = new int32_t[2500];
    
	for(int i=0; i<2500; i++) _array_neurongroup_i[i] = 0;

	_array_neurongroup_lastspike = new double[2500];
    
	for(int i=0; i<2500; i++) _array_neurongroup_lastspike[i] = 0;

	_array_neurongroup_m = new double[2500];
    
	for(int i=0; i<2500; i++) _array_neurongroup_m[i] = 0;

	_array_neurongroup_n = new double[2500];
    
	for(int i=0; i<2500; i++) _array_neurongroup_n[i] = 0;

	_array_neurongroup_not_refractory = new char[2500];
    
	for(int i=0; i<2500; i++) _array_neurongroup_not_refractory[i] = 0;

	_array_neurongroup_V = new double[2500];
    
	for(int i=0; i<2500; i++) _array_neurongroup_V[i] = 0;

	_array_spikegeneratorgroup__lastindex = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikegeneratorgroup__lastindex[i] = 0;

	_array_spikegeneratorgroup__period_bins = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikegeneratorgroup__period_bins[i] = 0;

	_array_spikegeneratorgroup__spikespace = new int32_t[101];
    
	for(int i=0; i<101; i++) _array_spikegeneratorgroup__spikespace[i] = 0;

	_array_spikegeneratorgroup_i = new int32_t[100];
    
	for(int i=0; i<100; i++) _array_spikegeneratorgroup_i[i] = 0;

	_array_spikegeneratorgroup_period = new double[1];
    
	for(int i=0; i<1; i++) _array_spikegeneratorgroup_period[i] = 0;

	_array_spikemonitor_1__source_idx = new int32_t[2500];
    
	for(int i=0; i<2500; i++) _array_spikemonitor_1__source_idx[i] = 0;

	_array_spikemonitor_1_count = new int32_t[2500];
    
	for(int i=0; i<2500; i++) _array_spikemonitor_1_count[i] = 0;

	_array_spikemonitor_1_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikemonitor_1_N[i] = 0;

	_array_spikemonitor_2__source_idx = new int32_t[100];
    
	for(int i=0; i<100; i++) _array_spikemonitor_2__source_idx[i] = 0;

	_array_spikemonitor_2_count = new int32_t[100];
    
	for(int i=0; i<100; i++) _array_spikemonitor_2_count[i] = 0;

	_array_spikemonitor_2_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikemonitor_2_N[i] = 0;

	_array_spikemonitor__source_idx = new int32_t[100];
    
	for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0;

	_array_spikemonitor_count = new int32_t[100];
    
	for(int i=0; i<100; i++) _array_spikemonitor_count[i] = 0;

	_array_spikemonitor_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;

	_array_synapses_1_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;

	_array_synapses_2_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_2_N[i] = 0;

	_array_synapses_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_N[i] = 0;

	_dynamic_array_spikegeneratorgroup__timebins.resize(19676);
    
	for(int i=0; i<19676; i++) _dynamic_array_spikegeneratorgroup__timebins[i] = 0;

	_dynamic_array_synapses_1_delay.resize(1);
    
	for(int i=0; i<1; i++) _dynamic_array_synapses_1_delay[i] = 0;

	_dynamic_array_synapses_2_delay.resize(1);
    
	for(int i=0; i<1; i++) _dynamic_array_synapses_2_delay[i] = 0;


	// Arrays initialized to an "arange"
	_array_neurongroup_1_i = new int32_t[100];
    
	for(int i=0; i<100; i++) _array_neurongroup_1_i[i] = 0 + i;

	_array_neurongroup_i = new int32_t[2500];
    
	for(int i=0; i<2500; i++) _array_neurongroup_i[i] = 0 + i;

	_array_spikegeneratorgroup_i = new int32_t[100];
    
	for(int i=0; i<100; i++) _array_spikegeneratorgroup_i[i] = 0 + i;

	_array_spikemonitor_1__source_idx = new int32_t[2500];
    
	for(int i=0; i<2500; i++) _array_spikemonitor_1__source_idx[i] = 0 + i;

	_array_spikemonitor_2__source_idx = new int32_t[100];
    
	for(int i=0; i<100; i++) _array_spikemonitor_2__source_idx[i] = 0 + i;

	_array_spikemonitor__source_idx = new int32_t[100];
    
	for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0 + i;


	// static arrays
	_static_array__dynamic_array_spikegeneratorgroup__timebins = new int32_t[19676];
	_static_array__dynamic_array_spikegeneratorgroup_neuron_index = new int64_t[19676];
	_static_array__dynamic_array_spikegeneratorgroup_spike_number = new int64_t[19676];
	_static_array__dynamic_array_spikegeneratorgroup_spike_time = new double[19676];

	// Random number generator states
	for (int i=0; i<1; i++)
	    _mersenne_twister_states.push_back(new rk_state());
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_static_array__dynamic_array_spikegeneratorgroup__timebins;
	f_static_array__dynamic_array_spikegeneratorgroup__timebins.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup__timebins", ios::in | ios::binary);
	if(f_static_array__dynamic_array_spikegeneratorgroup__timebins.is_open())
	{
		f_static_array__dynamic_array_spikegeneratorgroup__timebins.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup__timebins), 19676*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup__timebins." << endl;
	}
	ifstream f_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
	f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_neuron_index", ios::in | ios::binary);
	if(f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
	{
		f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_neuron_index), 19676*sizeof(int64_t));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_neuron_index." << endl;
	}
	ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_number;
	f_static_array__dynamic_array_spikegeneratorgroup_spike_number.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_number", ios::in | ios::binary);
	if(f_static_array__dynamic_array_spikegeneratorgroup_spike_number.is_open())
	{
		f_static_array__dynamic_array_spikegeneratorgroup_spike_number.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_number), 19676*sizeof(int64_t));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_number." << endl;
	}
	ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_time;
	f_static_array__dynamic_array_spikegeneratorgroup_spike_time.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_time", ios::in | ios::binary);
	if(f_static_array__dynamic_array_spikegeneratorgroup_spike_time.is_open())
	{
		f_static_array__dynamic_array_spikegeneratorgroup_spike_time.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_time), 19676*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_time." << endl;
	}
}

void _write_arrays()
{
	using namespace brian;

	ofstream outfile__array_defaultclock_dt;
	outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_-4715622247378976283", ios::binary | ios::out);
	if(outfile__array_defaultclock_dt.is_open())
	{
		outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
		outfile__array_defaultclock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
	}
	ofstream outfile__array_defaultclock_t;
	outfile__array_defaultclock_t.open("results/_array_defaultclock_t_-942942453034905548", ios::binary | ios::out);
	if(outfile__array_defaultclock_t.is_open())
	{
		outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
		outfile__array_defaultclock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_t." << endl;
	}
	ofstream outfile__array_defaultclock_timestep;
	outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_1624256667711379119", ios::binary | ios::out);
	if(outfile__array_defaultclock_timestep.is_open())
	{
		outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
		outfile__array_defaultclock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
	}
	ofstream outfile__array_neurongroup_1__spikespace;
	outfile__array_neurongroup_1__spikespace.open("results/_array_neurongroup_1__spikespace_9190998307066982524", ios::binary | ios::out);
	if(outfile__array_neurongroup_1__spikespace.is_open())
	{
		outfile__array_neurongroup_1__spikespace.write(reinterpret_cast<char*>(_array_neurongroup_1__spikespace), 101*sizeof(_array_neurongroup_1__spikespace[0]));
		outfile__array_neurongroup_1__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_1_g_eKC_eKC;
	outfile__array_neurongroup_1_g_eKC_eKC.open("results/_array_neurongroup_1_g_eKC_eKC_-2642734152970358722", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_g_eKC_eKC.is_open())
	{
		outfile__array_neurongroup_1_g_eKC_eKC.write(reinterpret_cast<char*>(_array_neurongroup_1_g_eKC_eKC), 100*sizeof(_array_neurongroup_1_g_eKC_eKC[0]));
		outfile__array_neurongroup_1_g_eKC_eKC.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_g_eKC_eKC." << endl;
	}
	ofstream outfile__array_neurongroup_1_g_iKC_eKC;
	outfile__array_neurongroup_1_g_iKC_eKC.open("results/_array_neurongroup_1_g_iKC_eKC_-2612150487629841522", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_g_iKC_eKC.is_open())
	{
		outfile__array_neurongroup_1_g_iKC_eKC.write(reinterpret_cast<char*>(_array_neurongroup_1_g_iKC_eKC), 100*sizeof(_array_neurongroup_1_g_iKC_eKC[0]));
		outfile__array_neurongroup_1_g_iKC_eKC.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_g_iKC_eKC." << endl;
	}
	ofstream outfile__array_neurongroup_1_h;
	outfile__array_neurongroup_1_h.open("results/_array_neurongroup_1_h_-5709179245434172730", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_h.is_open())
	{
		outfile__array_neurongroup_1_h.write(reinterpret_cast<char*>(_array_neurongroup_1_h), 100*sizeof(_array_neurongroup_1_h[0]));
		outfile__array_neurongroup_1_h.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_h." << endl;
	}
	ofstream outfile__array_neurongroup_1_i;
	outfile__array_neurongroup_1_i.open("results/_array_neurongroup_1_i_-7245080336651614241", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_i.is_open())
	{
		outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 100*sizeof(_array_neurongroup_1_i[0]));
		outfile__array_neurongroup_1_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
	}
	ofstream outfile__array_neurongroup_1_lastspike;
	outfile__array_neurongroup_1_lastspike.open("results/_array_neurongroup_1_lastspike_5266518080362794635", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_lastspike.is_open())
	{
		outfile__array_neurongroup_1_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_1_lastspike), 100*sizeof(_array_neurongroup_1_lastspike[0]));
		outfile__array_neurongroup_1_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_lastspike." << endl;
	}
	ofstream outfile__array_neurongroup_1_m;
	outfile__array_neurongroup_1_m.open("results/_array_neurongroup_1_m_1584225977104821452", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_m.is_open())
	{
		outfile__array_neurongroup_1_m.write(reinterpret_cast<char*>(_array_neurongroup_1_m), 100*sizeof(_array_neurongroup_1_m[0]));
		outfile__array_neurongroup_1_m.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_m." << endl;
	}
	ofstream outfile__array_neurongroup_1_n;
	outfile__array_neurongroup_1_n.open("results/_array_neurongroup_1_n_2884843497568471020", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_n.is_open())
	{
		outfile__array_neurongroup_1_n.write(reinterpret_cast<char*>(_array_neurongroup_1_n), 100*sizeof(_array_neurongroup_1_n[0]));
		outfile__array_neurongroup_1_n.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_n." << endl;
	}
	ofstream outfile__array_neurongroup_1_not_refractory;
	outfile__array_neurongroup_1_not_refractory.open("results/_array_neurongroup_1_not_refractory_3785485727548187697", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_not_refractory.is_open())
	{
		outfile__array_neurongroup_1_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_1_not_refractory), 100*sizeof(_array_neurongroup_1_not_refractory[0]));
		outfile__array_neurongroup_1_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_not_refractory." << endl;
	}
	ofstream outfile__array_neurongroup_1_V;
	outfile__array_neurongroup_1_V.open("results/_array_neurongroup_1_V_-6699657252984638200", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_V.is_open())
	{
		outfile__array_neurongroup_1_V.write(reinterpret_cast<char*>(_array_neurongroup_1_V), 100*sizeof(_array_neurongroup_1_V[0]));
		outfile__array_neurongroup_1_V.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_V." << endl;
	}
	ofstream outfile__array_neurongroup__spikespace;
	outfile__array_neurongroup__spikespace.open("results/_array_neurongroup__spikespace_8790907868525549454", ios::binary | ios::out);
	if(outfile__array_neurongroup__spikespace.is_open())
	{
		outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 2501*sizeof(_array_neurongroup__spikespace[0]));
		outfile__array_neurongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_g_PN_iKC;
	outfile__array_neurongroup_g_PN_iKC.open("results/_array_neurongroup_g_PN_iKC_7475736948780694548", ios::binary | ios::out);
	if(outfile__array_neurongroup_g_PN_iKC.is_open())
	{
		outfile__array_neurongroup_g_PN_iKC.write(reinterpret_cast<char*>(_array_neurongroup_g_PN_iKC), 2500*sizeof(_array_neurongroup_g_PN_iKC[0]));
		outfile__array_neurongroup_g_PN_iKC.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_g_PN_iKC." << endl;
	}
	ofstream outfile__array_neurongroup_h;
	outfile__array_neurongroup_h.open("results/_array_neurongroup_h_9037143158809171154", ios::binary | ios::out);
	if(outfile__array_neurongroup_h.is_open())
	{
		outfile__array_neurongroup_h.write(reinterpret_cast<char*>(_array_neurongroup_h), 2500*sizeof(_array_neurongroup_h[0]));
		outfile__array_neurongroup_h.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_h." << endl;
	}
	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open("results/_array_neurongroup_i_532635648569835609", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 2500*sizeof(_array_neurongroup_i[0]));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}
	ofstream outfile__array_neurongroup_lastspike;
	outfile__array_neurongroup_lastspike.open("results/_array_neurongroup_lastspike_-6308789563404776624", ios::binary | ios::out);
	if(outfile__array_neurongroup_lastspike.is_open())
	{
		outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), 2500*sizeof(_array_neurongroup_lastspike[0]));
		outfile__array_neurongroup_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
	}
	ofstream outfile__array_neurongroup_m;
	outfile__array_neurongroup_m.open("results/_array_neurongroup_m_1239499265801404614", ios::binary | ios::out);
	if(outfile__array_neurongroup_m.is_open())
	{
		outfile__array_neurongroup_m.write(reinterpret_cast<char*>(_array_neurongroup_m), 2500*sizeof(_array_neurongroup_m[0]));
		outfile__array_neurongroup_m.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_m." << endl;
	}
	ofstream outfile__array_neurongroup_n;
	outfile__array_neurongroup_n.open("results/_array_neurongroup_n_-2031658838389855829", ios::binary | ios::out);
	if(outfile__array_neurongroup_n.is_open())
	{
		outfile__array_neurongroup_n.write(reinterpret_cast<char*>(_array_neurongroup_n), 2500*sizeof(_array_neurongroup_n[0]));
		outfile__array_neurongroup_n.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_n." << endl;
	}
	ofstream outfile__array_neurongroup_not_refractory;
	outfile__array_neurongroup_not_refractory.open("results/_array_neurongroup_not_refractory_5237735263788100149", ios::binary | ios::out);
	if(outfile__array_neurongroup_not_refractory.is_open())
	{
		outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), 2500*sizeof(_array_neurongroup_not_refractory[0]));
		outfile__array_neurongroup_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
	}
	ofstream outfile__array_neurongroup_V;
	outfile__array_neurongroup_V.open("results/_array_neurongroup_V_7017721009805895108", ios::binary | ios::out);
	if(outfile__array_neurongroup_V.is_open())
	{
		outfile__array_neurongroup_V.write(reinterpret_cast<char*>(_array_neurongroup_V), 2500*sizeof(_array_neurongroup_V[0]));
		outfile__array_neurongroup_V.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_V." << endl;
	}
	ofstream outfile__array_spikegeneratorgroup__lastindex;
	outfile__array_spikegeneratorgroup__lastindex.open("results/_array_spikegeneratorgroup__lastindex_-2092603496930772822", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup__lastindex.is_open())
	{
		outfile__array_spikegeneratorgroup__lastindex.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__lastindex), 1*sizeof(_array_spikegeneratorgroup__lastindex[0]));
		outfile__array_spikegeneratorgroup__lastindex.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup__lastindex." << endl;
	}
	ofstream outfile__array_spikegeneratorgroup__period_bins;
	outfile__array_spikegeneratorgroup__period_bins.open("results/_array_spikegeneratorgroup__period_bins_-6142051961911285310", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup__period_bins.is_open())
	{
		outfile__array_spikegeneratorgroup__period_bins.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__period_bins), 1*sizeof(_array_spikegeneratorgroup__period_bins[0]));
		outfile__array_spikegeneratorgroup__period_bins.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup__period_bins." << endl;
	}
	ofstream outfile__array_spikegeneratorgroup__spikespace;
	outfile__array_spikegeneratorgroup__spikespace.open("results/_array_spikegeneratorgroup__spikespace_-4858010840100393062", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup__spikespace.is_open())
	{
		outfile__array_spikegeneratorgroup__spikespace.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__spikespace), 101*sizeof(_array_spikegeneratorgroup__spikespace[0]));
		outfile__array_spikegeneratorgroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup__spikespace." << endl;
	}
	ofstream outfile__array_spikegeneratorgroup_i;
	outfile__array_spikegeneratorgroup_i.open("results/_array_spikegeneratorgroup_i_5023119052171209669", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup_i.is_open())
	{
		outfile__array_spikegeneratorgroup_i.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_i), 100*sizeof(_array_spikegeneratorgroup_i[0]));
		outfile__array_spikegeneratorgroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup_i." << endl;
	}
	ofstream outfile__array_spikegeneratorgroup_period;
	outfile__array_spikegeneratorgroup_period.open("results/_array_spikegeneratorgroup_period_8187017143679110142", ios::binary | ios::out);
	if(outfile__array_spikegeneratorgroup_period.is_open())
	{
		outfile__array_spikegeneratorgroup_period.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_period), 1*sizeof(_array_spikegeneratorgroup_period[0]));
		outfile__array_spikegeneratorgroup_period.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikegeneratorgroup_period." << endl;
	}
	ofstream outfile__array_spikemonitor_1__source_idx;
	outfile__array_spikemonitor_1__source_idx.open("results/_array_spikemonitor_1__source_idx_4311227457789481753", ios::binary | ios::out);
	if(outfile__array_spikemonitor_1__source_idx.is_open())
	{
		outfile__array_spikemonitor_1__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_1__source_idx), 2500*sizeof(_array_spikemonitor_1__source_idx[0]));
		outfile__array_spikemonitor_1__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_1__source_idx." << endl;
	}
	ofstream outfile__array_spikemonitor_1_count;
	outfile__array_spikemonitor_1_count.open("results/_array_spikemonitor_1_count_5632793696640609457", ios::binary | ios::out);
	if(outfile__array_spikemonitor_1_count.is_open())
	{
		outfile__array_spikemonitor_1_count.write(reinterpret_cast<char*>(_array_spikemonitor_1_count), 2500*sizeof(_array_spikemonitor_1_count[0]));
		outfile__array_spikemonitor_1_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_1_count." << endl;
	}
	ofstream outfile__array_spikemonitor_1_N;
	outfile__array_spikemonitor_1_N.open("results/_array_spikemonitor_1_N_2858339697764412581", ios::binary | ios::out);
	if(outfile__array_spikemonitor_1_N.is_open())
	{
		outfile__array_spikemonitor_1_N.write(reinterpret_cast<char*>(_array_spikemonitor_1_N), 1*sizeof(_array_spikemonitor_1_N[0]));
		outfile__array_spikemonitor_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_1_N." << endl;
	}
	ofstream outfile__array_spikemonitor_2__source_idx;
	outfile__array_spikemonitor_2__source_idx.open("results/_array_spikemonitor_2__source_idx_6493219805939906996", ios::binary | ios::out);
	if(outfile__array_spikemonitor_2__source_idx.is_open())
	{
		outfile__array_spikemonitor_2__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_2__source_idx), 100*sizeof(_array_spikemonitor_2__source_idx[0]));
		outfile__array_spikemonitor_2__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_2__source_idx." << endl;
	}
	ofstream outfile__array_spikemonitor_2_count;
	outfile__array_spikemonitor_2_count.open("results/_array_spikemonitor_2_count_-9042356216020750687", ios::binary | ios::out);
	if(outfile__array_spikemonitor_2_count.is_open())
	{
		outfile__array_spikemonitor_2_count.write(reinterpret_cast<char*>(_array_spikemonitor_2_count), 100*sizeof(_array_spikemonitor_2_count[0]));
		outfile__array_spikemonitor_2_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_2_count." << endl;
	}
	ofstream outfile__array_spikemonitor_2_N;
	outfile__array_spikemonitor_2_N.open("results/_array_spikemonitor_2_N_6528715863422346178", ios::binary | ios::out);
	if(outfile__array_spikemonitor_2_N.is_open())
	{
		outfile__array_spikemonitor_2_N.write(reinterpret_cast<char*>(_array_spikemonitor_2_N), 1*sizeof(_array_spikemonitor_2_N[0]));
		outfile__array_spikemonitor_2_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_2_N." << endl;
	}
	ofstream outfile__array_spikemonitor__source_idx;
	outfile__array_spikemonitor__source_idx.open("results/_array_spikemonitor__source_idx_-7381243524531015710", ios::binary | ios::out);
	if(outfile__array_spikemonitor__source_idx.is_open())
	{
		outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 100*sizeof(_array_spikemonitor__source_idx[0]));
		outfile__array_spikemonitor__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
	}
	ofstream outfile__array_spikemonitor_count;
	outfile__array_spikemonitor_count.open("results/_array_spikemonitor_count_8256918035359257624", ios::binary | ios::out);
	if(outfile__array_spikemonitor_count.is_open())
	{
		outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 100*sizeof(_array_spikemonitor_count[0]));
		outfile__array_spikemonitor_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
	}
	ofstream outfile__array_spikemonitor_N;
	outfile__array_spikemonitor_N.open("results/_array_spikemonitor_N_2177318049605994951", ios::binary | ios::out);
	if(outfile__array_spikemonitor_N.is_open())
	{
		outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(_array_spikemonitor_N[0]));
		outfile__array_spikemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
	}
	ofstream outfile__array_synapses_1_N;
	outfile__array_synapses_1_N.open("results/_array_synapses_1_N_2532369240704119479", ios::binary | ios::out);
	if(outfile__array_synapses_1_N.is_open())
	{
		outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(_array_synapses_1_N[0]));
		outfile__array_synapses_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_1_N." << endl;
	}
	ofstream outfile__array_synapses_2_N;
	outfile__array_synapses_2_N.open("results/_array_synapses_2_N_15395925652631311", ios::binary | ios::out);
	if(outfile__array_synapses_2_N.is_open())
	{
		outfile__array_synapses_2_N.write(reinterpret_cast<char*>(_array_synapses_2_N), 1*sizeof(_array_synapses_2_N[0]));
		outfile__array_synapses_2_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_2_N." << endl;
	}
	ofstream outfile__array_synapses_N;
	outfile__array_synapses_N.open("results/_array_synapses_N_8308290687023610208", ios::binary | ios::out);
	if(outfile__array_synapses_N.is_open())
	{
		outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(_array_synapses_N[0]));
		outfile__array_synapses_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N." << endl;
	}

	ofstream outfile__dynamic_array_spikegeneratorgroup__timebins;
	outfile__dynamic_array_spikegeneratorgroup__timebins.open("results/_dynamic_array_spikegeneratorgroup__timebins_-5271953684463739277", ios::binary | ios::out);
	if(outfile__dynamic_array_spikegeneratorgroup__timebins.is_open())
	{
        if (! _dynamic_array_spikegeneratorgroup__timebins.empty() )
        {
			outfile__dynamic_array_spikegeneratorgroup__timebins.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup__timebins[0]), _dynamic_array_spikegeneratorgroup__timebins.size()*sizeof(_dynamic_array_spikegeneratorgroup__timebins[0]));
		    outfile__dynamic_array_spikegeneratorgroup__timebins.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup__timebins." << endl;
	}
	ofstream outfile__dynamic_array_spikegeneratorgroup_neuron_index;
	outfile__dynamic_array_spikegeneratorgroup_neuron_index.open("results/_dynamic_array_spikegeneratorgroup_neuron_index_-3828194948906838247", ios::binary | ios::out);
	if(outfile__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
	{
        if (! _dynamic_array_spikegeneratorgroup_neuron_index.empty() )
        {
			outfile__dynamic_array_spikegeneratorgroup_neuron_index.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup_neuron_index[0]), _dynamic_array_spikegeneratorgroup_neuron_index.size()*sizeof(_dynamic_array_spikegeneratorgroup_neuron_index[0]));
		    outfile__dynamic_array_spikegeneratorgroup_neuron_index.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_neuron_index." << endl;
	}
	ofstream outfile__dynamic_array_spikegeneratorgroup_spike_number;
	outfile__dynamic_array_spikegeneratorgroup_spike_number.open("results/_dynamic_array_spikegeneratorgroup_spike_number_-7962737313149232762", ios::binary | ios::out);
	if(outfile__dynamic_array_spikegeneratorgroup_spike_number.is_open())
	{
        if (! _dynamic_array_spikegeneratorgroup_spike_number.empty() )
        {
			outfile__dynamic_array_spikegeneratorgroup_spike_number.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup_spike_number[0]), _dynamic_array_spikegeneratorgroup_spike_number.size()*sizeof(_dynamic_array_spikegeneratorgroup_spike_number[0]));
		    outfile__dynamic_array_spikegeneratorgroup_spike_number.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_number." << endl;
	}
	ofstream outfile__dynamic_array_spikegeneratorgroup_spike_time;
	outfile__dynamic_array_spikegeneratorgroup_spike_time.open("results/_dynamic_array_spikegeneratorgroup_spike_time_-5239418278040634268", ios::binary | ios::out);
	if(outfile__dynamic_array_spikegeneratorgroup_spike_time.is_open())
	{
        if (! _dynamic_array_spikegeneratorgroup_spike_time.empty() )
        {
			outfile__dynamic_array_spikegeneratorgroup_spike_time.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup_spike_time[0]), _dynamic_array_spikegeneratorgroup_spike_time.size()*sizeof(_dynamic_array_spikegeneratorgroup_spike_time[0]));
		    outfile__dynamic_array_spikegeneratorgroup_spike_time.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_time." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_1_i;
	outfile__dynamic_array_spikemonitor_1_i.open("results/_dynamic_array_spikemonitor_1_i_7751520737792369972", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_1_i.is_open())
	{
        if (! _dynamic_array_spikemonitor_1_i.empty() )
        {
			outfile__dynamic_array_spikemonitor_1_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_1_i[0]), _dynamic_array_spikemonitor_1_i.size()*sizeof(_dynamic_array_spikemonitor_1_i[0]));
		    outfile__dynamic_array_spikemonitor_1_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_1_t;
	outfile__dynamic_array_spikemonitor_1_t.open("results/_dynamic_array_spikemonitor_1_t_4114818300371855858", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_1_t.is_open())
	{
        if (! _dynamic_array_spikemonitor_1_t.empty() )
        {
			outfile__dynamic_array_spikemonitor_1_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_1_t[0]), _dynamic_array_spikemonitor_1_t.size()*sizeof(_dynamic_array_spikemonitor_1_t[0]));
		    outfile__dynamic_array_spikemonitor_1_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_t." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_2_i;
	outfile__dynamic_array_spikemonitor_2_i.open("results/_dynamic_array_spikemonitor_2_i_-1322732478480285391", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_2_i.is_open())
	{
        if (! _dynamic_array_spikemonitor_2_i.empty() )
        {
			outfile__dynamic_array_spikemonitor_2_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_2_i[0]), _dynamic_array_spikemonitor_2_i.size()*sizeof(_dynamic_array_spikemonitor_2_i[0]));
		    outfile__dynamic_array_spikemonitor_2_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_2_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_2_t;
	outfile__dynamic_array_spikemonitor_2_t.open("results/_dynamic_array_spikemonitor_2_t_6446990258431135936", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_2_t.is_open())
	{
        if (! _dynamic_array_spikemonitor_2_t.empty() )
        {
			outfile__dynamic_array_spikemonitor_2_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_2_t[0]), _dynamic_array_spikemonitor_2_t.size()*sizeof(_dynamic_array_spikemonitor_2_t[0]));
		    outfile__dynamic_array_spikemonitor_2_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_2_t." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_i;
	outfile__dynamic_array_spikemonitor_i.open("results/_dynamic_array_spikemonitor_i_-2405956570836330415", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_i.is_open())
	{
        if (! _dynamic_array_spikemonitor_i.empty() )
        {
			outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_i[0]), _dynamic_array_spikemonitor_i.size()*sizeof(_dynamic_array_spikemonitor_i[0]));
		    outfile__dynamic_array_spikemonitor_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_t;
	outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t_2230804304852047519", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_t.is_open())
	{
        if (! _dynamic_array_spikemonitor_t.empty() )
        {
			outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_t[0]), _dynamic_array_spikemonitor_t.size()*sizeof(_dynamic_array_spikemonitor_t[0]));
		    outfile__dynamic_array_spikemonitor_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1__synaptic_post;
	outfile__dynamic_array_synapses_1__synaptic_post.open("results/_dynamic_array_synapses_1__synaptic_post_-1260052851024091814", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_1__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_post[0]), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(_dynamic_array_synapses_1__synaptic_post[0]));
		    outfile__dynamic_array_synapses_1__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
	outfile__dynamic_array_synapses_1__synaptic_pre.open("results/_dynamic_array_synapses_1__synaptic_pre_-3279311761150895792", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_1__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_pre[0]), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(_dynamic_array_synapses_1__synaptic_pre[0]));
		    outfile__dynamic_array_synapses_1__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_Apost;
	outfile__dynamic_array_synapses_1_Apost.open("results/_dynamic_array_synapses_1_Apost_-943692019737572824", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_Apost.is_open())
	{
        if (! _dynamic_array_synapses_1_Apost.empty() )
        {
			outfile__dynamic_array_synapses_1_Apost.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_Apost[0]), _dynamic_array_synapses_1_Apost.size()*sizeof(_dynamic_array_synapses_1_Apost[0]));
		    outfile__dynamic_array_synapses_1_Apost.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_Apost." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_Apre;
	outfile__dynamic_array_synapses_1_Apre.open("results/_dynamic_array_synapses_1_Apre_-1106472975864171241", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_Apre.is_open())
	{
        if (! _dynamic_array_synapses_1_Apre.empty() )
        {
			outfile__dynamic_array_synapses_1_Apre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_Apre[0]), _dynamic_array_synapses_1_Apre.size()*sizeof(_dynamic_array_synapses_1_Apre[0]));
		    outfile__dynamic_array_synapses_1_Apre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_Apre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_delay;
	outfile__dynamic_array_synapses_1_delay.open("results/_dynamic_array_synapses_1_delay_7763448962795292474", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_delay.is_open())
	{
        if (! _dynamic_array_synapses_1_delay.empty() )
        {
			outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_delay[0]), _dynamic_array_synapses_1_delay.size()*sizeof(_dynamic_array_synapses_1_delay[0]));
		    outfile__dynamic_array_synapses_1_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_delay_1;
	outfile__dynamic_array_synapses_1_delay_1.open("results/_dynamic_array_synapses_1_delay_1_-5765490777359899456", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_delay_1.is_open())
	{
        if (! _dynamic_array_synapses_1_delay_1.empty() )
        {
			outfile__dynamic_array_synapses_1_delay_1.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_delay_1[0]), _dynamic_array_synapses_1_delay_1.size()*sizeof(_dynamic_array_synapses_1_delay_1[0]));
		    outfile__dynamic_array_synapses_1_delay_1.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_delay_1." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_g_raw;
	outfile__dynamic_array_synapses_1_g_raw.open("results/_dynamic_array_synapses_1_g_raw_-8555754931006574664", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_g_raw.is_open())
	{
        if (! _dynamic_array_synapses_1_g_raw.empty() )
        {
			outfile__dynamic_array_synapses_1_g_raw.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_g_raw[0]), _dynamic_array_synapses_1_g_raw.size()*sizeof(_dynamic_array_synapses_1_g_raw[0]));
		    outfile__dynamic_array_synapses_1_g_raw.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_g_raw." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_lastupdate;
	outfile__dynamic_array_synapses_1_lastupdate.open("results/_dynamic_array_synapses_1_lastupdate_2483364739843424404", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_lastupdate.is_open())
	{
        if (! _dynamic_array_synapses_1_lastupdate.empty() )
        {
			outfile__dynamic_array_synapses_1_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_lastupdate[0]), _dynamic_array_synapses_1_lastupdate.size()*sizeof(_dynamic_array_synapses_1_lastupdate[0]));
		    outfile__dynamic_array_synapses_1_lastupdate.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_N_incoming;
	outfile__dynamic_array_synapses_1_N_incoming.open("results/_dynamic_array_synapses_1_N_incoming_3716960530702479588", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_1_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_incoming[0]), _dynamic_array_synapses_1_N_incoming.size()*sizeof(_dynamic_array_synapses_1_N_incoming[0]));
		    outfile__dynamic_array_synapses_1_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_N_outgoing;
	outfile__dynamic_array_synapses_1_N_outgoing.open("results/_dynamic_array_synapses_1_N_outgoing_6301477802276829413", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_1_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_outgoing[0]), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(_dynamic_array_synapses_1_N_outgoing[0]));
		    outfile__dynamic_array_synapses_1_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2__synaptic_post;
	outfile__dynamic_array_synapses_2__synaptic_post.open("results/_dynamic_array_synapses_2__synaptic_post_-7878296401867179188", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_2__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses_2__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2__synaptic_post[0]), _dynamic_array_synapses_2__synaptic_post.size()*sizeof(_dynamic_array_synapses_2__synaptic_post[0]));
		    outfile__dynamic_array_synapses_2__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2__synaptic_pre;
	outfile__dynamic_array_synapses_2__synaptic_pre.open("results/_dynamic_array_synapses_2__synaptic_pre_6346010521677370995", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_2__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses_2__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2__synaptic_pre[0]), _dynamic_array_synapses_2__synaptic_pre.size()*sizeof(_dynamic_array_synapses_2__synaptic_pre[0]));
		    outfile__dynamic_array_synapses_2__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_delay;
	outfile__dynamic_array_synapses_2_delay.open("results/_dynamic_array_synapses_2_delay_-418772450127157232", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_delay.is_open())
	{
        if (! _dynamic_array_synapses_2_delay.empty() )
        {
			outfile__dynamic_array_synapses_2_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_delay[0]), _dynamic_array_synapses_2_delay.size()*sizeof(_dynamic_array_synapses_2_delay[0]));
		    outfile__dynamic_array_synapses_2_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_N_incoming;
	outfile__dynamic_array_synapses_2_N_incoming.open("results/_dynamic_array_synapses_2_N_incoming_-7629994045094012121", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_2_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_2_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_N_incoming[0]), _dynamic_array_synapses_2_N_incoming.size()*sizeof(_dynamic_array_synapses_2_N_incoming[0]));
		    outfile__dynamic_array_synapses_2_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_N_outgoing;
	outfile__dynamic_array_synapses_2_N_outgoing.open("results/_dynamic_array_synapses_2_N_outgoing_-1162906691673147932", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_2_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_2_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_N_outgoing[0]), _dynamic_array_synapses_2_N_outgoing.size()*sizeof(_dynamic_array_synapses_2_N_outgoing[0]));
		    outfile__dynamic_array_synapses_2_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_post;
	outfile__dynamic_array_synapses__synaptic_post.open("results/_dynamic_array_synapses__synaptic_post_-2371895981297589966", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_post[0]), _dynamic_array_synapses__synaptic_post.size()*sizeof(_dynamic_array_synapses__synaptic_post[0]));
		    outfile__dynamic_array_synapses__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_pre;
	outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre_-2345947451451094597", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_pre[0]), _dynamic_array_synapses__synaptic_pre.size()*sizeof(_dynamic_array_synapses__synaptic_pre[0]));
		    outfile__dynamic_array_synapses__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_delay;
	outfile__dynamic_array_synapses_delay.open("results/_dynamic_array_synapses_delay_-4543666643619482562", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_delay.is_open())
	{
        if (! _dynamic_array_synapses_delay.empty() )
        {
			outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_delay[0]), _dynamic_array_synapses_delay.size()*sizeof(_dynamic_array_synapses_delay[0]));
		    outfile__dynamic_array_synapses_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_N_incoming;
	outfile__dynamic_array_synapses_N_incoming.open("results/_dynamic_array_synapses_N_incoming_8741384138218738586", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_incoming[0]), _dynamic_array_synapses_N_incoming.size()*sizeof(_dynamic_array_synapses_N_incoming[0]));
		    outfile__dynamic_array_synapses_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_N_outgoing;
	outfile__dynamic_array_synapses_N_outgoing.open("results/_dynamic_array_synapses_N_outgoing_2744852400496361443", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_outgoing[0]), _dynamic_array_synapses_N_outgoing.size()*sizeof(_dynamic_array_synapses_N_outgoing[0]));
		    outfile__dynamic_array_synapses_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_weight;
	outfile__dynamic_array_synapses_weight.open("results/_dynamic_array_synapses_weight_3175780951192449195", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_weight.is_open())
	{
        if (! _dynamic_array_synapses_weight.empty() )
        {
			outfile__dynamic_array_synapses_weight.write(reinterpret_cast<char*>(&_dynamic_array_synapses_weight[0]), _dynamic_array_synapses_weight.size()*sizeof(_dynamic_array_synapses_weight[0]));
		    outfile__dynamic_array_synapses_weight.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_weight." << endl;
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
	if(_static_array__dynamic_array_spikegeneratorgroup__timebins!=0)
	{
		delete [] _static_array__dynamic_array_spikegeneratorgroup__timebins;
		_static_array__dynamic_array_spikegeneratorgroup__timebins = 0;
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
}

