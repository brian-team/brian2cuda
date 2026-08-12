{% macro cu_file() %}
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/spikequeue.h"
#include "brianlib/cuda_utils.h"

__device__ void SynapticPathway::init(
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
}

__device__ void SynapticPathway::destroy()
{
    queue->destroy();
    delete queue;
}

{% for S in synapses | sort(attribute='name') %}
{% for path in S._pathways | sort(attribute='name') %}
__global__ void {{path.name}}_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop)
{
    using namespace brian;
    {{path.name}}.init(sources, targets, dt, source_start, source_stop);
}

__global__ void {{path.name}}_destroy()
{
    using namespace brian;
    {{path.name}}.destroy();
}
{% endfor %}
{% endfor %}

{% endmacro %}

{% macro h_file() %}

#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include <stdint.h>

class CudaSpikeQueue;

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

    __device__ void init(
            int32_t* _sources,
            int32_t* _targets,
            double _dt,
            int32_t _spikes_start,
            int32_t _spikes_stop);

    __device__ void destroy();
};

{% for S in synapses | sort(attribute='name') %}
{% for path in S._pathways | sort(attribute='name') %}
__global__ void {{path.name}}_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop);
__global__ void {{path.name}}_destroy();
{% endfor %}
{% endfor %}

#endif

{% endmacro %}
