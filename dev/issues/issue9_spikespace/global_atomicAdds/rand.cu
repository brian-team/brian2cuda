
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "network.h"
#include<iostream>
#include<fstream>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void _run_random_number_generation()
{
	using namespace brian;

	float mean = 0.0;
	float std_deviation = 1.0;
	
	unsigned int needed_random_numbers;

}
