
#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>
#include <omp.h>

#include "brianlib/spikequeue.h"

class SynapticPathway
{
public:
	int Nsource, Ntarget, _nb_threads;
	std::vector<int> &sources;
	std::vector<int> all_peek;
	std::vector< CSpikeQueue * > queue;
	SynapticPathway(std::vector<int> &_sources, int _spikes_start, int _spikes_stop)
		: sources(_sources)
	{
	   _nb_threads = 2;

	   for (int _idx=0; _idx < _nb_threads; _idx++)
	       queue.push_back(new CSpikeQueue(_spikes_start, _spikes_stop));
    };

	~SynapticPathway()
	{
		for (int _idx=0; _idx < _nb_threads; _idx++)
			delete(queue[_idx]);
	}

	void push(int *spikes, int nspikes)
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
		for(int _thread=0; _thread < 2; _thread++)
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

    template <typename scalar> void prepare(int n_source, int n_target, scalar *real_delays, int n_delays,
                 int *sources, int n_synapses, double _dt)
    {
        Nsource = n_source;
        Ntarget = n_target;
    	#pragma omp parallel
    	{
            int length;
            if (omp_get_thread_num() == _nb_threads - 1) 
                length = n_synapses - (int)omp_get_thread_num()*(n_synapses/_nb_threads);
            else
                length = (int) n_synapses/_nb_threads;

            int padding  = omp_get_thread_num()*(n_synapses/_nb_threads);

            queue[omp_get_thread_num()]->openmp_padding = padding;
            if (n_delays > 1)
    		    queue[omp_get_thread_num()]->prepare(&real_delays[padding], length, &sources[padding], length, _dt);
    		else if (n_delays == 1)
    		    queue[omp_get_thread_num()]->prepare(&real_delays[0], 1, &sources[padding], length, _dt);
    		else  // no synapses
    		    queue[omp_get_thread_num()]->prepare((scalar *)NULL, 0, &sources[padding], length, _dt);
    	}
    }

};

#endif

