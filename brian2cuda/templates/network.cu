{% macro cu_file() %}

#include "brianlib/cuda_utils.h"
#include "objects.h"
#include "network.h"
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <utility>
#include <stdio.h>
#include <assert.h>

#define Clock_epsilon 1e-14

double Network::_last_run_time = 0.0;
double Network::_last_run_completed_fraction = 0.0;

Network::Network()
{
    t = 0.0;
}

void Network::clear()
{
    objects.clear();
    cuda_events.clear();
    profiling_infos.clear();
}

void Network::add(Clock *clock, codeobj_func func, timer_type *timer_start, timer_type *timer_stop, double *profiling_time)
{
#if defined(_MSC_VER) && (_MSC_VER>=1700)
    objects.push_back(std::make_pair(std::move(clock), std::move(func)));
    cuda_events.push_back(std::make_pair(std::move(timer_start), std::move(timer_stop)));
#else
    objects.push_back(std::make_pair(clock, func));
    cuda_events.push_back(std::make_pair(timer_start, timer_stop));
#endif
    profiling_infos.push_back(profiling_time);
}

void Network::run(const double duration, void (*report_func)(const double, const double, const double, const double), const double report_period)
{
    std::clock_t start, current;
    const double t_start = t;
    const double t_end = t + duration;
    double next_report_time = report_period;
    // compute the set of clocks
    compute_clocks();
    // set interval for all clocks

    for(std::set<Clock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
        (*i)->set_interval(t, t_end);

    start = std::clock();
    if (report_func)
    {
        report_func(0.0, 0.0, t_start, duration);
    }

    Clock* clock = next_clocks();
    double elapsed_realtime;
    bool did_break_early = false;

    {% if profile %}
    assert(profiling_infos.size() == objects.size());
    bool first_cycle = true;
    {% endif %}
    while(clock && clock->running())
    {
        t = clock->t[0];

        for(int i=0; i<objects.size(); i++)
        {
            if (report_func)
            {
                current = std::clock();
                const double elapsed = (double)(current - start) / CLOCKS_PER_SEC;
                if (elapsed > next_report_time)
                {
                    report_func(elapsed, (clock->t[0]-t_start)/duration, t_start, duration);
                    next_report_time += report_period;
                }
            }
            Clock *obj_clock = objects[i].first;
            // Only execute the object if it uses the right clock for this step
            if (curclocks.find(obj_clock) != curclocks.end())
            {
                codeobj_func func = objects[i].second;
                if (func)  // code objects can be NULL in cases where we store just the clock
                {
                    {# Since cudaEvents run asynchornously on the GPU, we calculate elapsed times
                       for the codeobjects just before they are run again (which resets the events). #}
                    {% if profile %}
                    // collect profiling infos from last execution of the current codeobject
                    if (!first_cycle)
                    {
                        {% if profile == 'blocking' %}
                        *(profiling_infos[i]) += (double)(*cuda_events[i].second - *cuda_events[i].first)/CLOCKS_PER_SEC;
                        {% else %}
                        cudaError_t kernel_status = cudaEventQuery(*cuda_events[i].second);
                        if (kernel_status == cudaSuccess)
                        {
                            float elapsed_gpu_time;
                            CUDA_SAFE_CALL(
                                    cudaEventElapsedTime(&elapsed_gpu_time, *cuda_events[i].first, *cuda_events[i].second)
                                    );
                            *(profiling_infos[i]) += elapsed_gpu_time / 1000;
                        }
                        else if (kernel_status == cudaErrorNotReady)
                        {
                            printf("WARNING kernels in %i. active codeobject took longer than one clock cycle. "
                                    "Can't take into account GPU time spend in this kernel call. If this warning appears "
                                    "often, the profiling results will be unreliable. Consider using the "
                                    "profile='blocking' option in network_run().\n", i);
                        }
                        else
                        {
                            printf("ERROR caught in %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(kernel_status));
                        }
                        {% endif %}{# profile == 'blocking #}
                    }
                    {% endif %}{# profile #}
                    // run codeobject
                    func();
                }
            }
        }
        for(std::set<Clock*>::iterator i=curclocks.begin(); i!=curclocks.end(); i++)
            (*i)->tick();
        clock = next_clocks();

        // Advance index for circular eventspace vector (for no_or_const_delay_mode)
        {% for var, varname in eventspace_arrays | dictsort(by='value') %}
        brian::current_idx{{varname}} = (brian::current_idx{{varname}} + 1) % brian::dev{{varname}}.size();
        {% endfor %}

        current = std::clock();
        elapsed_realtime = (double)(current - start)/CLOCKS_PER_SEC;

        {% if maximum_run_time is not none %}
        if(elapsed_realtime>{{maximum_run_time}})
        {
            did_break_early = true;
            break;
        }
        {% endif %}

        {% if profile %}
        first_cycle = false;
        {% endif %}
    }

    if(!did_break_early) t = t_end;

    _last_run_time = elapsed_realtime;
    if(duration>0)
    {
        _last_run_completed_fraction = (t-t_start)/duration;
    } else {
        _last_run_completed_fraction = 1.0;
    }
    if (report_func)
    {
        report_func(elapsed_realtime, 1.0, t_start, duration);
    }

    {% if profile %}
    // Add profiled times from last cycle
    for(int i=0; i<objects.size(); i++)
    {
        codeobj_func func = objects[i].second;
        if (func)  // func can be NULL
        {
            {% if profile == 'blocking' %}
            *(profiling_infos[i]) += (double)(*cuda_events[i].second - *cuda_events[i].first)/CLOCKS_PER_SEC;
            {% else %}
            cudaError_t kernel_status = cudaEventQuery(*cuda_events[i].second);
            if (kernel_status == cudaSuccess)
            {
                float elapsed_gpu_time;
                CUDA_SAFE_CALL(
                        cudaEventElapsedTime(&elapsed_gpu_time, *cuda_events[i].first, *cuda_events[i].second)
                        );
                *(profiling_infos[i]) += elapsed_gpu_time / 1000;
            }
            else if (kernel_status == cudaErrorNotReady)
            {
                printf("WARNING kernels in %i. active codeobject took longer than one clock cycle. "
                        "Can't take into account GPU time spend in this kernel call. If this warning appears "
                        "often, the profiling results will be unreliable. Consider using the "
                        "profile='blocking' option in network_run().\n", i);
            }
            else
            {
                printf("ERROR caught in %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(kernel_status));
            }
            {% endif %}
        }
    }
    {% endif %}
}

void Network::compute_clocks()
{
    clocks.clear();
    for(int i=0; i<objects.size(); i++)
    {
        Clock *clock = objects[i].first;
        clocks.insert(clock);
    }
}

Clock* Network::next_clocks()
{
    // find minclock, clock with smallest t value
    Clock *minclock = *clocks.begin();
    if (!minclock) // empty list of clocks
        return NULL;

    for(std::set<Clock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
    {
        Clock *clock = *i;
        if(clock->t[0]<minclock->t[0])
            minclock = clock;
    }
    // find set of equal clocks
    curclocks.clear();

    double t = minclock->t[0];
    for(std::set<Clock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
    {
        Clock *clock = *i;
        double s = clock->t[0];
        if(s==t || fabs(s-t)<=Clock_epsilon)
            curclocks.insert(clock);
    }
    return minclock;
}

{% endmacro %}

{% macro h_file() %}

#ifndef _BRIAN_NETWORK_H
#define _BRIAN_NETWORK_H

#include <vector>
#include <utility>
#include <set>
#include <ctime>
#include "brianlib/clocks.h"

typedef void (*codeobj_func)();

class Network
{
    std::set<Clock*> clocks, curclocks;
    void compute_clocks();
    Clock* next_clocks();
public:
    std::vector< std::pair< Clock*, codeobj_func > > objects;
    double t;
    static double _last_run_time;
    static double _last_run_completed_fraction;
    std::vector< std::pair< timer_type*, timer_type* > > cuda_events;
    std::vector< double* > profiling_infos;

    Network();
    void clear();
    void add(Clock *clock, codeobj_func func, timer_type* timer_start, timer_type *timer_stop, double *profiling_time);
    void run(const double duration, void (*report_func)(const double, const double, const double, const double), const double report_period);
};

#endif

{% endmacro %}
