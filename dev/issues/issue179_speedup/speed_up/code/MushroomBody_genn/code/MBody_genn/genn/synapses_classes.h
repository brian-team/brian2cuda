
#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>


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
	   _nb_threads = 1;

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
    	queue[0]->push(spikes, nspikes);
    }

	void advance()
    {
    	queue[0]->advance();
    }

	vector<int32_t>* peek()
    {
    	
		for(int _thread=0; _thread < 1; _thread++)
		{
			
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
    	
    	{
            int length;
            if (0 == _nb_threads - 1) 
                length = n_synapses - (int)0*(n_synapses/_nb_threads);
            else
                length = (int) n_synapses/_nb_threads;

            int padding  = 0*(n_synapses/_nb_threads);

            queue[0]->openmp_padding = padding;
            if (n_delays > 1)
    		    queue[0]->prepare(&real_delays[padding], length, &sources[padding], length, _dt);
    		else if (n_delays == 1)
    		    queue[0]->prepare(&real_delays[0], 1, &sources[padding], length, _dt);
    		else  // no synapses
    		    queue[0]->prepare((scalar *)NULL, 0, &sources[padding], length, _dt);
    	}
    }

};

#endif

