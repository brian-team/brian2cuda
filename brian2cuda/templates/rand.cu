{% macro cu_file() %}

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/cuda_utils.h"
#include "network.h"
#include <curand.h>
#include <ctime>

void _run_random_number_generation()
{
    using namespace brian;

    // how many random numbers we want to create at once (tradeoff memory usage <-> generation overhead)
    static double mb_per_obj = 50;  // MB per codeobject and rand / randn
    static int floats_per_obj = (mb_per_obj * 1024.0 * 1024.0) / sizeof(randomNumber_t);

    // Get the number of needed random numbers per clock cycle, the generation interval, and the number generated per curand call.
    {% for co in codeobj_with_rand %}
    static int num_per_cycle_rand_{{co.name}};
    static int rand_interval_{{co.name}};
    static int num_per_gen_rand_{{co.name}};
    static int idx_rand_{{co.name}};
    static int rand_floats_per_obj_{{co.name}};
    {% endfor %}
    {% for co in codeobj_with_randn %}
    static int num_per_cycle_randn_{{co.name}};
    static int randn_interval_{{co.name}};
    static int num_per_gen_randn_{{co.name}};
    static int idx_randn_{{co.name}};
    static int randn_floats_per_obj_{{co.name}};
    {% endfor %}

    // Allocate device memory
    static bool first_run = true;
    if (first_run)
    {

        // check that we have enough memory available
        size_t free_byte ;
        size_t total_byte ;
        CUDA_SAFE_CALL(
                cudaMemGetInfo(&free_byte, &total_byte)
                );
        size_t num_free_floats = free_byte / sizeof(randomNumber_t);


        {% for co in codeobj_with_rand | sort(attribute='name') %}
        {% if co.template_name == 'synapses' %}
        {% set N = '_array_' + co.owner.name + '_N[0]' %}
        {% else %}
        {% set N = co.owner._N %}
        {% endif %}
        // Get the number of needed random numbers per clock cycle, the generation interval, and the number generated per curand call.
        num_per_cycle_rand_{{co.name}} = {{N}} * {{co.rand_calls}};
        rand_floats_per_obj_{{co.name}} = floats_per_obj;
        if (floats_per_obj < num_per_cycle_rand_{{co.name}})
            rand_floats_per_obj_{{co.name}} = num_per_cycle_rand_{{co.name}};
        rand_interval_{{co.name}} = (int)(rand_floats_per_obj_{{co.name}} / num_per_cycle_rand_{{co.name}});
        num_per_gen_rand_{{co.name}} = num_per_cycle_rand_{{co.name}} * rand_interval_{{co.name}};
        idx_rand_{{co.name}} = rand_interval_{{co.name}};

        if (rand_interval_{{co.name}} > {{co.owner.clock.name}}.i_end)
        {
            // create max as many random numbers as will be needed in the entire simulation
            num_per_gen_rand_{{co.name}} = num_per_cycle_rand_{{co.name}} * {{co.owner.clock.name}}.i_end;
        }
        if (num_per_gen_rand_{{co.name}} % 2 != 0)
        {
            // curandGenerateNormal requires an even number for pseudorandom generators
            num_per_gen_rand_{{co.name}} = num_per_gen_rand_{{co.name}} + 1;
        }

        // make sure that we don't use more memory then available
        // TODO: this checks per codeobject the number of generated floats against total available floats. But we neet to check all generated floats against available floats.
        while (num_free_floats < num_per_gen_rand_{{co.name}})
        {
            printf("INFO not enough memory available to generate %i random numbers for {{co.name}}, reducing the buffer size\n", num_free_floats);
            if (num_per_gen_rand_{{co.name}} < num_per_cycle_rand_{{co.name}})
            {
                if (num_free_floats < num_per_cycle_rand_{{co.name}})
                {
                    printf("ERROR not enough memory to generate random numbers for {{co.name}} %s:%d\n", __FILE__, __LINE__);
                    _dealloc_arrays();
                    exit(1);
                }
                else
                {
                    num_per_gen_rand_{{co.name}} = num_per_cycle_rand_{{co.name}};
                    break;
                }
            }
            num_per_gen_rand_{{co.name}} /= 2;
        }
        printf("INFO generating %i rand every %i clock cycles for {{co.name}}\n", num_per_gen_rand_{{co.name}}, rand_interval_{{co.name}});

        CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_{{co.name}}_rand_allocator, sizeof(randomNumber_t)*num_per_gen_rand_{{co.name}})
                );
        {% endfor %}


        {% for co in codeobj_with_randn | sort(attribute='name') %}
        {% if co.template_name == 'synapses' %}
        {% set N = '_array_' + co.owner.name + '_N[0]' %}
        {% else %}
        {% set N = co.owner._N %}
        {% endif %}
        // Get the number of needed random numbers per clock cycle, the generation interval, and the number generated per curand call.
        num_per_cycle_randn_{{co.name}} = {{N}} * {{co.randn_calls}};
        randn_floats_per_obj_{{co.name}} = floats_per_obj;
        if (floats_per_obj < num_per_cycle_randn_{{co.name}})
            randn_floats_per_obj_{{co.name}} = num_per_cycle_randn_{{co.name}};
        randn_interval_{{co.name}} = (int)(randn_floats_per_obj_{{co.name}} / num_per_cycle_randn_{{co.name}});
        num_per_gen_randn_{{co.name}} = num_per_cycle_randn_{{co.name}} * randn_interval_{{co.name}};
        idx_randn_{{co.name}} = randn_interval_{{co.name}};

        if (randn_interval_{{co.name}} > {{co.owner.clock.name}}.i_end)
        {
            // create max as many random numbers as will be needed in the entire simulation
            num_per_gen_randn_{{co.name}} = num_per_cycle_randn_{{co.name}} * {{co.owner.clock.name}}.i_end;
        }
        if (num_per_gen_randn_{{co.name}} % 2 != 0)
        {
            // curandGenerateNormal requires an even number for pseudorandom generators
            num_per_gen_randn_{{co.name}} = num_per_gen_randn_{{co.name}} + 1;
        }

        // make sure that we don't use more memory then available
        // TODO: this checks per codeobject the number of generated floats against total available floats. But we neet to check all generated floats against available floats.
        while (num_free_floats < num_per_gen_randn_{{co.name}})
        {
            printf("INFO not enough memory available to generate %i random numbers for {{co.name}}, reducing the buffer size\n", num_free_floats);
            if (num_per_gen_randn_{{co.name}} < num_per_cycle_randn_{{co.name}})
            {
                if (num_free_floats < num_per_cycle_randn_{{co.name}})
                {
                    printf("ERROR not enough memory to generate random numbers for {{co.name}} %s:%d\n", __FILE__, __LINE__);
                    _dealloc_arrays();
                    exit(1);
                }
                else
                {
                    num_per_gen_randn_{{co.name}} = num_per_cycle_randn_{{co.name}};
                    break;
                }
            }
            num_per_gen_randn_{{co.name}} /= 2;
        }
        printf("INFO generating %i randn every %i clock cycles for {{co.name}}\n", num_per_gen_randn_{{co.name}}, randn_interval_{{co.name}});

        CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_{{co.name}}_randn_allocator, sizeof(randomNumber_t)*num_per_gen_randn_{{co.name}})
                );
        {% endfor %}

        first_run = false;
    }

    // Generate random numbers
    {% for co in codeobj_with_rand %}
    if (idx_rand_{{co.name}} == rand_interval_{{co.name}})
    {
        {% if curand_float_type == 'float' %}
        curandGenerateUniform(curand_generator, dev_{{co.name}}_rand_allocator, num_per_gen_rand_{{co.name}});
        {% else %}
        curandGenerateUniformDouble(curand_generator, dev_{{co.name}}_rand_allocator, num_per_gen_rand_{{co.name}});
        {% endif %}
        dev_{{co.name}}_rand = &dev_{{co.name}}_rand_allocator[0];
        idx_rand_{{co.name}} = 1;
    }
    else
    {
        // move device pointer to next numbers
        dev_{{co.name}}_rand += num_per_cycle_rand_{{co.name}};
        idx_rand_{{co.name}} += 1;
    }
    assert(dev_{{co.name}}_rand < dev_{{co.name}}_rand_allocator + num_per_gen_rand_{{co.name}});
    {% endfor %}

    {% for co in codeobj_with_randn %}
    if (idx_randn_{{co.name}} == randn_interval_{{co.name}})
    {
        {% if curand_float_type == 'float' %}
        curandGenerateNormal(curand_generator, dev_{{co.name}}_randn_allocator, num_per_gen_randn_{{co.name}}, 0, 1);
        {% else %}
        curandGenerateNormalDouble(curand_generator, dev_{{co.name}}_randn_allocator, num_per_gen_randn_{{co.name}}, 0, 1);
        {% endif %}
        dev_{{co.name}}_randn = &dev_{{co.name}}_randn_allocator[0];
        idx_randn_{{co.name}} = 1;
    }
    else
    {
        // move device pointer to next numbers
        dev_{{co.name}}_randn += num_per_cycle_randn_{{co.name}};
        idx_randn_{{co.name}} += 1;
    }
    assert(dev_{{co.name}}_randn < dev_{{co.name}}_randn_allocator + num_per_gen_randn_{{co.name}});
    {% endfor %}
}
{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_RAND_H
#define _BRIAN_RAND_H

#include <curand.h>

void _run_random_number_generation();

#endif


{% endmacro %}
