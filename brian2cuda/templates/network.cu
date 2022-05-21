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
{% if parallelize %}
cudaStream_t custom_stream[{{num_stream}}];
{% endif %}

Network::Network()
{
    t = 0.0;
    {% if parallelize %}
    for(int i=0;i<{{num_stream}};i++){
        CUDA_SAFE_CALL(cudaStreamCreate(&(custom_stream[i])));
    }
    {% endif %}
}

void Network::clear()
{
    objects.clear();
}

// TODO have to makr change in objects - make it a tuple
// make decision which bject has which stream
void Network::add(Clock *clock, codeobj_func func, int group_num)
{
#if defined(_MSC_VER) && (_MSC_VER>=1700)
    objects.push_back(std::make_tuple(std::move(clock), std::move(func), std::move(group_num)));
#else
    objects.push_back(std::make_tuple(clock, func, group_num));
#endif
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
    //TODO here
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
            // TODO tuple of clock and function
            //Clock *obj_clock = objects[i].first;
            Clock *obj_clock = std::get<0>(objects[i]);
            int group_int = std::get<2>(objects[i]);
            // Only execute the object if it uses the right clock for this step
            if (curclocks.find(obj_clock) != curclocks.end())
            {
               // function -> whixh is in templates like common_group.cu
               // sort the code object - waiting mechanism between groups
               // cudaEvent or cudaSynchronise
                //codeobj_func func = objects[i].second;
                codeobj_func func = std::get<1>(objects[i]);
                int func_group_int = std::get<2>(objects[i]);
                if (func)  // code objects can be NULL in cases where we store just the clock
                {
                      func_groups[func_group_int].push_back(func);
                      //func_groups.push_back(std::make_pair(func_group_int,func));
                    //func();
                    // [[func1,func2,func3],[func4...]]
                }
            }
        }

        // get maximum in objects.cu array

        // go through each list of func group - 2 loops
        for(int i=0; i<func_groups.size(); i++){
            for(int j=0; j<func_groups[i].size(); j++){
                codeobj_func func = func_groups[i][j];
                func(custom_stream[j]);
            }
            // reset the func group for that sub stream
            cudaDeviceSynchronize();
            func_groups[i].resize(0);
        }

        for(std::set<Clock*>::iterator i=curclocks.begin(); i!=curclocks.end(); i++)
            (*i)->tick();
        clock = next_clocks();

        // Advance index for circular eventspace vector (for no_or_const_delay_mode)
        {% for var, varname in eventspace_arrays | dictsort(by='value') %}
        {% if varname in spikegenerator_eventspaces %}
        brian::previous_idx{{varname}} = brian::current_idx{{varname}};
        {% endif %}
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
}

void Network::compute_clocks()
{
    clocks.clear();
    for(int i=0; i<objects.size(); i++)
    {
        Clock *clock = std::get<0>(objects[i]);
        // Clock *clock = std::get<0>()objects[i].first;
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

typedef void (*codeobj_func)(cudaStream_t);

class Network
{
    std::set<Clock*> clocks, curclocks;
    void compute_clocks();
    Clock* next_clocks();
public:
// TODO vectory of tuples having clock , codeobj_func and stread integer
    std::vector< std::tuple< Clock*, codeobj_func, int > > objects;
    //std::vector< std::pair< Clock*, codeobj_func > > objects;
    std::vector<std::vector<codeobj_func >> func_groups = std::vector<std::vector<codeobj_func >>({{num_stream}});
    //std::vector<std::pair< int, codeobj_func >> func_groups;
    double t;
    static double _last_run_time;
    static double _last_run_completed_fraction;
    int num_streams;
    {% if parallelize %}
    cudaStream_t custom_stream[{{num_stream}}];
    {% endif %}

    Network();
    void clear();
    void add(Clock *clock, codeobj_func func, int num_streams);
    void run(const double duration, void (*report_func)(const double, const double, const double, const double), const double report_period);
};

#endif

{% endmacro %}
