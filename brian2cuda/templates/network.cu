{% macro cu_file() %}

#include "brianlib/cuda_utils.h"
#include "objects.h"
#include "network.h"
#include <stdlib.h>
#include <iostream>
#include <chrono>
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
}

void Network::add(BaseClock* clock, codeobj_func func)
{
#if defined(_MSC_VER) && (_MSC_VER>=1700)
    objects.push_back(std::make_pair(std::move(clock), std::move(func)));
#else
    objects.push_back(std::make_pair(clock, func));
#endif
}

void Network::run(const double duration, void (*report_func)(const double, const double, const double, const double), const double report_period)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, current;
    const double t_start = t;
    const double t_end = t + duration;
    double next_report_time = report_period;
    // compute the set of clocks
    compute_clocks();
    // set interval for all clocks

    for(std::set<BaseClock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
        (*i)->set_interval(t, t_end);

    start = std::chrono::high_resolution_clock::now();
    if (report_func)
    {
        report_func(0.0, 0.0, t_start, duration);
    }

    BaseClock* clock = next_clocks();
    double elapsed_realtime;
    bool did_break_early = false;

    while(clock && clock->running())
    {
        t = clock->t[0];

        for(int i=0; i<objects.size(); i++)
        {
            if (report_func)
            {
                current = std::chrono::high_resolution_clock::now();
                const double elapsed = std::chrono::duration<double>(current - start).count();
                if (elapsed > next_report_time)
                {
                    report_func(elapsed, (clock->t[0]-t_start)/duration, t_start, duration);
                    next_report_time += report_period;
                }
            }
            BaseClock *obj_clock = objects[i].first;
            // Only execute the object if it uses the right clock for this step
            if (curclocks.find(obj_clock) != curclocks.end())
            {
                codeobj_func func = objects[i].second;
                if (func)  // code objects can be NULL in cases where we store just the clock
                {
                    func();
                }
            }
        }
        for(std::set<BaseClock*>::iterator i=curclocks.begin(); i!=curclocks.end(); i++)
            (*i)->tick();
        clock = next_clocks();

        // Advance index for circular eventspace vector (for no_or_const_delay_mode)
        {% for var, varname in eventspace_arrays | dictsort(by='value') %}
        {% if varname in spikegenerator_eventspaces %}
        brian::previous_idx{{varname}} = brian::current_idx{{varname}};
        {% endif %}
        brian::current_idx{{varname}} = (brian::current_idx{{varname}} + 1) % brian::dev{{varname}}.size();
        {% endfor %}

        current = std::chrono::high_resolution_clock::now();
        elapsed_realtime = std::chrono::duration<double>(current - start).count();

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
        BaseClock *clock = objects[i].first;
        clocks.insert(clock);
    }
}

BaseClock* Network::next_clocks()
{
    // find minclock, clock with smallest t value
    BaseClock *minclock = *clocks.begin();
    if (!minclock) // empty list of clocks
        return NULL;

    for(std::set<BaseClock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
    {
        BaseClock *clock = *i;
        if(clock->t[0]<minclock->t[0])
            minclock = clock;
    }
    // find set of equal clocks
    curclocks.clear();

    double t = minclock->t[0];
    for(std::set<BaseClock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
    {
        BaseClock *clock = *i;
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
    std::set<BaseClock*> clocks, curclocks;
    void compute_clocks();
    BaseClock* next_clocks();
public:
    std::vector< std::pair< BaseClock*, codeobj_func > > objects;
    double t;
    static double _last_run_time;
    static double _last_run_completed_fraction;

    Network();
    void clear();
    void add(BaseClock *clock, codeobj_func func);
    void run(const double duration, void (*report_func)(const double, const double, const double, const double), const double report_period);
};

#endif

{% endmacro %}
