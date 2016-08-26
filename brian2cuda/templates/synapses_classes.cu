{% macro cu_file() %}
{% endmacro %}

{% macro h_file() %}

#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>

#include "brianlib/spikequeue.h"

template<class scalar> class Synapses;
template<class scalar> class SynapticPathway;

template <class scalar>
class SynapticPathway
{
public:
	int Nsource;
	int Ntarget;
	scalar* dev_delay;
	int32_t* dev_sources;
	int32_t* dev_targets;
	
	// first and last index of source NeuronGroup in SynapticPathway
	// important for Subgroups (syntax NeuronGroup(N=4000,...)[:3200])
	unsigned int spikes_start;
	unsigned int spikes_stop;

	scalar dt;
	CSpikeQueue<scalar>* queue;
	bool no_or_const_delay_mode;
	unsigned int which_spikespace;

	//our real constructor
	__device__ void init(int _Nsource, int _Ntarget, scalar* d_delay, int32_t* _sources,
				int32_t* _targets, scalar _dt, int _spikes_start, int _spikes_stop)
	{
		Nsource = _Nsource;
		Ntarget = _Ntarget;
		dev_delay = d_delay;
		dev_sources = _sources;
		dev_targets = _targets;
		dt = _dt;
		spikes_start = _spikes_start;
		spikes_stop = _spikes_stop;
		queue = new CSpikeQueue<scalar>;
		which_spikespace = 0;
    	};

	//our real destructor
	__device__ void destroy()
	{
		queue->destroy();
		delete queue;
	}
};

template <class scalar>
class Synapses
{
public:
    int _N_value;
    inline double _N() { return _N_value;};
	int Nsource;
	int Ntarget;
	std::vector< std::vector<int> > _pre_synaptic;
	std::vector< std::vector<int> > _post_synaptic;

	Synapses(int _Nsource, int _Ntarget)
		: Nsource(_Nsource), Ntarget(_Ntarget)
	{
		for(int i=0; i<Nsource; i++)
			_pre_synaptic.push_back(std::vector<int>());
		for(int i=0; i<Ntarget; i++)
			_post_synaptic.push_back(std::vector<int>());
		_N_value = 0;
	};
};

#endif

{% endmacro %}
