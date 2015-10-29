
#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>
#include <omp.h>

#include "brianlib/spikequeue.h"

template<class scalar> class SynapticPathway;

template <class scalar>
class SynapticPathway
{
public:
	int Nsource, Ntarget, _nb_threads;
	std::vector<scalar> &delay;
	std::vector<int> &sources;
	std::vector<int> all_peek;
	std::vector< CSpikeQueue<scalar> * > queue;
	SynapticPathway(std::vector<scalar>& _delay, std::vector<int> &_sources,
					int _spikes_start, int _spikes_stop)
		: delay(_delay), sources(_sources)
	{
	   _nb_threads = 4;

	   for (int _idx=0; _idx < _nb_threads; _idx++)
	       queue.push_back(new CSpikeQueue<scalar>(_spikes_start, _spikes_stop));
    };

	~SynapticPathway()
	{
		for (int _idx=0; _idx < _nb_threads; _idx++)
			delete(queue[_idx]);
	}

	void push(int *spikes, unsigned int nspikes)
    {
    	queue[omp_get_thread_num()]->push(spikes, nspikes);
    }

	void advance()
    {
    	queue[omp_get_thread_num()]->advance();
    }

	vector<int32_t>* peek()
    {
    	#pragma omp for schedule(static) ordered
		for(int _thread=0; _thread < 4; _thread++)
		{
			#pragma omp ordered
			{
    			if (_thread == 0)
					all_peek.clear();
				all_peek.insert(all_peek.end(), queue[_thread]->peek()->begin(), queue[_thread]->peek()->end());
    		}
    	}
   
    	return &all_peek;
    }

    void prepare(int n_source, int n_target, scalar *real_delays, unsigned int n_delays,
                 int *sources, unsigned int n_synapses, double _dt)
    {
        Nsource = n_source;
        Ntarget = n_target;
    	#pragma omp parallel
    	{
            unsigned int length;
            if (omp_get_thread_num() == _nb_threads - 1) 
                length = n_synapses - (unsigned int) omp_get_thread_num()*(n_synapses/_nb_threads);
            else
                length = (unsigned int) n_synapses/_nb_threads;

            unsigned int padding  = omp_get_thread_num()*(n_synapses/_nb_threads);

            queue[omp_get_thread_num()]->openmp_padding = padding;
            if (n_delays > 1)
    		    queue[omp_get_thread_num()]->prepare(&real_delays[padding], length, &sources[padding], length, _dt);
    		else
    		    queue[omp_get_thread_num()]->prepare(&real_delays[0], 1, &sources[padding], length, _dt);
    	}
    }

};

#endif

