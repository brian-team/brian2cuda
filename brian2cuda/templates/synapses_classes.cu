{% macro cu_file() %}
{% endmacro %}

{% macro h_file() %}

#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>

#include "brianlib/spikequeue.h"

class SynapticPathway
{
public:
    int32_t* dev_sources;
    int32_t* dev_targets;

    // first and last index in source NeuronGroup corresponding to Subgroup in SynapticPathway
    // important for Subgroups created with syntax: NeuronGroup(N=4000,...)[:3200]
    int32_t spikes_start;
    int32_t spikes_stop;

    double dt;
    CudaSpikeQueue* queue;
    bool no_or_const_delay_mode;

    //our real constructor
    __device__ void init(
            int32_t* _sources,
            int32_t* _targets,
            double _dt,
            int32_t _spikes_start,
            int32_t _spikes_stop)
    {
        dev_sources = _sources;
        dev_targets = _targets;
        dt = _dt;
        spikes_start = _spikes_start;
        spikes_stop = _spikes_stop;
        queue = new CudaSpikeQueue;
    };

    //our real destructor
    __device__ void destroy()
    {
        queue->destroy();
        delete queue;
    }
};
#endif

{% endmacro %}
