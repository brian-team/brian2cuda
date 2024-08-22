
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

    Network();
    void clear();
    void add(Clock *clock, codeobj_func func);
    void run(const double duration, void (*report_func)(const double, const double, const double, const double), const double report_period);
};

#endif

