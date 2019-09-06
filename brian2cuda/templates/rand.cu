{% macro cu_file() %}

#include "objects.h"
#include "rand.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/cuda_utils.h"
#include "network.h"
#include <curand.h>
#include <ctime>
#include <curand_kernel.h>

using namespace brian;

// TODO make this a class member function
namespace {


    {% for co in codeobj_with_binomial | sort(attribute='name') %}
    __global__ void init_curand_states_{{co.name}}(int N)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N)
        {
            // Each thread gets the same seed, a different sequence number and
            // no offset
            // TODO: different seed and 0 sequence number is much faster, with
            // less security for independent sequences
            //curand_init(curand_seed + idx, 0, 0,
            curand_init(d_curand_seed, idx, 0,
                    &d_{{co.name}}_curand_states[idx]);
        }
    }
    {% endfor %}
}


// need a function pointer for Network::add(), can't pass a pointer to a class
// method, which is of different type
void _run_random_number_buffer()
{
    // random_number_buffer is a RandomNumberBuffer instanced, declared in objects.cu
    random_number_buffer.next_time_step();
}

{#
// TODO move stuff from objects.cu
void RandomNumberBuffer::RandomNumberBuffer()
{

    // Random seeds might be overwritten in main.cu
    unsigned long long seed = time(0);

    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_curand_seed,
                sizeof(unsigned long long))
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_curand_seed, &seed,
                sizeof(unsigned long long), cudaMemcpyHostToDevice)
            );

    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_curand_seed, &dev_curand_seed,
                sizeof(unsigned long long*))
            );


    curandCreateGenerator(&curand_generator, {{curand_generator_type}});
    {% if curand_generator_ordering %}
    curandSetGeneratorOrdering(curand_generator, {{curand_generator_ordering}});
    {% endif %}
    curandSetPseudoRandomGeneratorSeed(curand_generator, seed);

}
#}

void RandomNumberBuffer::init()
{
    // check that we have enough memory available
    size_t free_byte;
    size_t total_byte;
    CUDA_SAFE_CALL(
            cudaMemGetInfo(&free_byte, &total_byte)
            );
    size_t num_free_floats = free_byte / sizeof(randomNumber_t);

    // number of time steps each codeobject is executed during current Network::run() call
    // XXX: we are assuming here that this function is only run in the first time step of a Network::run()
    {% for co in codeobj_with_rand_or_randn | sort(attribute='name') %}
    int64_t num_steps_this_run_{{co.name}} = {{co.owner.clock.name}}.i_end - {{co.owner.clock.name}}.t[0];
    {% endfor %}

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

    // create max as many random numbers as will be needed during the current Network.run() call
    if ((int64_t)rand_interval_{{co.name}} > num_steps_this_run_{{co.name}})
    {
        // NOTE: if the conditional is true, we can savely cast num_steps_this_run_{{co.name}} to int
        num_per_gen_rand_{{co.name}} = num_per_cycle_rand_{{co.name}} * (int)num_steps_this_run_{{co.name}};
        assert((int64_t)num_per_cycle_rand_{{co.name}} * num_steps_this_run_{{co.name}} == num_per_gen_rand_{{co.name}});
    }

    // curandGenerateNormal requires an even number for pseudorandom generators
    if (num_per_gen_rand_{{co.name}} % 2 != 0)
    {
        num_per_gen_rand_{{co.name}} = num_per_gen_rand_{{co.name}} + 1;
    }

    // make sure that we don't use more memory then available
    // this checks per codeobject the number of generated floats against total available floats
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

    // create max as many random numbers as will be needed during the current Network.run() call
    if ((int64_t)randn_interval_{{co.name}} > num_steps_this_run_{{co.name}})
    {
        // NOTE: if the conditional is true, we can savely cast num_steps_this_run_{{co.name}} to int
        num_per_gen_randn_{{co.name}} = num_per_cycle_randn_{{co.name}} * (int)num_steps_this_run_{{co.name}};
        assert((int64_t)num_per_cycle_randn_{{co.name}} * num_steps_this_run_{{co.name}} == num_per_gen_randn_{{co.name}});
    }

    // curandGenerateNormal requires an even number for pseudorandom generators
    if (num_per_gen_randn_{{co.name}} % 2 != 0)
    {
        num_per_gen_randn_{{co.name}} = num_per_gen_randn_{{co.name}} + 1;
    }

    // make sure that we don't use more memory then available
    // this checks per codeobject the number of generated floats against total available floats
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

    // now check if the total number of generated floats fit into available memory
    int total_num_generated_floats = 0;
    {% for co in codeobj_with_rand %}
    total_num_generated_floats += num_per_gen_rand_{{co.name}};
    {% endfor %}
    {% for co in codeobj_with_randn %}
    total_num_generated_floats += num_per_gen_randn_{{co.name}};
    {% endfor %}
    if (num_free_floats < total_num_generated_floats)
    {
        // TODO: find a way to deal with this? E.g. looping over buffers sorted
        // by buffer size and reducing them until it fits.
        printf("MEMORY ERROR: Trying to generate more random numbers than fit "
               "into available memory. Please report this as an issue on "
               "GitHub: https://github.com/brian-team/brian2cuda/issues/new");
        _dealloc_arrays();
        exit(1);
    }

    // allocate globabl memory for curand device api states
    {% for co in codeobj_with_binomial | sort(attribute='name') %}
    {% if co.template_name == 'synapses' %}
    {% set N = '_array_' + co.owner.name + '_N[0]' %}
    {% else %}
    {% set N = co.owner._N %}
    {% endif %}
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_{{co.name}}_curand_states,
                sizeof(curandState) * {{N}})
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_{{co.name}}_curand_states,
                &dev_{{co.name}}_curand_states, sizeof(curandState*))
            );
    {% endfor %}

    // set curand device api states
    set_curand_device_api_states();
}


void RandomNumberBuffer::set_curand_device_api_states()
{
    {% for co in codeobj_with_binomial | sort(attribute='name') %}
    {% if co.template_name == 'synapses' %}
    {% set N = '_array_' + co.owner.name + '_N[0]' %}
    {% else %}
    {% set N = co.owner._N %}
    {% endif %}
    int num_threads, num_blocks;
    num_threads = max_threads_per_block;
    num_blocks = {{N}} / max_threads_per_block + 1;
    if ({{N}} < num_threads)
        num_threads = {{N}};
    init_curand_states_{{co.name}}<<<num_blocks, num_threads>>>({{N}});
    {% endfor %}
}


void RandomNumberBuffer::set_seed(unsigned long long seed)
{
    CUDA_CHECK_ERROR("before set seed");
    CUDA_SAFE_CALL(
            curandSetPseudoRandomGeneratorSeed(curand_generator, seed)
            );

    CUDA_CHECK_ERROR("after set seed");
    // generator offset needs to be reset to its default (=0)
    CUDA_SAFE_CALL(
            curandSetGeneratorOffset(curand_generator, 0ULL)
            );

    // reinit the buffers, dt might have changed or the num_steps_this_run_{}
    // need to free memory for init() to work
    // TODO: could be solved more efficiently:
    //      have one buffer object per codeobject and check per codeobject if
    //      dt has changed or if num_steps_this_run_ was used previously to
    //      generate less random numbers! -> issue?
    CUDA_CHECK_ERROR("after offset");
    {% for co in codeobj_with_rand | sort(attribute='name') %}
    CUDA_SAFE_CALL(
            cudaFree(dev_{{co.name}}_rand_allocator)
            );
    {% endfor %}

    {% for co in codeobj_with_randn | sort(attribute='name') %}
    CUDA_SAFE_CALL(
            cudaFree(dev_{{co.name}}_randn_allocator)
            );
    {% endfor %}

    // don't call init() here already since the network clocks might not be set
    // up yet, call init() only once network started running
    needs_init = true;

    // set seed for curand device api calls
    // don't set the same seed for host api and device api random states, just in case
    seed += 1;
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_curand_seed, &seed,
                sizeof(unsigned long long), cudaMemcpyHostToDevice)
            );

    // update curand device api states with new seed
    set_curand_device_api_states();
}


void RandomNumberBuffer::refill_uniform_numbers(
        randomNumber_t* dev_rand_allocator,
        randomNumber_t* &dev_rand,
        int num_per_gen_rand,
        int &idx_rand)
{
    // generate uniform distributed random numbers and reset buffer index

    {% if curand_float_type == 'float' %}
    curandGenerateUniform(curand_generator, dev_rand_allocator, num_per_gen_rand);
    {% else %}
    curandGenerateUniformDouble(curand_generator, dev_rand_allocator, num_per_gen_rand);
    {% endif %}
    // before: XXX dev_rand = &dev_rand_allocator[0];
    dev_rand = dev_rand_allocator;
    idx_rand = 1;
}


void RandomNumberBuffer::refill_normal_numbers(
        randomNumber_t* dev_randn_allocator,
        randomNumber_t* &dev_randn,
        int num_per_gen_randn,
        int &idx_randn)
{
    // generate normal distributed random numbers and reset buffer index

    {% if curand_float_type == 'float' %}
    curandGenerateNormal(curand_generator, dev_randn_allocator, num_per_gen_randn, 0, 1);
    {% else %}
    curandGenerateNormalDouble(curand_generator, dev_randn_allocator, num_per_gen_randn, 0, 1);
    {% endif %}
    // before: XXX dev_randn = &dev_randn_allocator[0];
    dev_randn = dev_randn_allocator;
    idx_randn = 1;
}


void RandomNumberBuffer::next_time_step()
{
    if (needs_init)
    {
        init();
        needs_init = false;
    }

    {% for co in codeobj_with_rand %}
    // uniform numbers for {{co.name}}
    if (idx_rand_{{co.name}} == rand_interval_{{co.name}})
    {
        refill_uniform_numbers(
                dev_{{co.name}}_rand_allocator,
                dev_{{co.name}}_rand,
                num_per_gen_rand_{{co.name}},
                idx_rand_{{co.name}});
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
    // normal numbers for {{co.name}}
    if (idx_randn_{{co.name}} == randn_interval_{{co.name}})
    {
        refill_normal_numbers(
                dev_{{co.name}}_randn_allocator,
                dev_{{co.name}}_randn,
                num_per_gen_randn_{{co.name}},
                idx_randn_{{co.name}});
    }
    else
    {
        // move device pointer to next numbers
        dev_{{co.name}}_randn += num_per_cycle_randn_{{co.name}};
        idx_randn_{{co.name}} += 1;
    }
    if (dev_{{co.name}}_randn < dev_{{co.name}}_randn_allocator + num_per_gen_randn_{{co.name}})
        printf("dev_randn %u, dev_randn_allocator %u, num_per_gen_randn %d\n",
                dev_{{co.name}}_randn, dev_{{co.name}}_randn_allocator,
                num_per_gen_randn_{{co.name}});
    else
        printf("ERROR: dev_randn %u, dev_randn_allocator %u, num_per_gen_randn %d\n",
                dev_{{co.name}}_randn, dev_{{co.name}}_randn_allocator,
                num_per_gen_randn_{{co.name}});
    assert(dev_{{co.name}}_randn < dev_{{co.name}}_randn_allocator + num_per_gen_randn_{{co.name}});
    {% endfor %}
}
{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_RAND_H
#define _BRIAN_RAND_H

#include <curand.h>

void _run_random_number_buffer();

class RandomNumberBuffer
{
    bool needs_init = true;

    // how many random numbers we want to create at once (tradeoff memory usage <-> generation overhead)
    double mb_per_obj = 50;  // MB per codeobject and rand / randn
    int floats_per_obj = (mb_per_obj * 1024.0 * 1024.0) / sizeof(randomNumber_t);

    // The number of needed random numbers per clock cycle, the generation interval, and the number generated per curand call.
    //
    // needed random numbers per clock cycle
    // int num_per_cycle_rand_{};
    //
    // number of time steps after which buffer needs to be refilled
    // int rand_interval_{};
    //
    // buffer size
    // int num_per_gen_rand_{};
    //
    // number of time steps since last buffer refill
    // int idx_rand_{};
    //
    // maximum number of random numbers fitting given allocated memory
    // int rand_floats_per_obj_{};


    // uniform distributed random numbers (rand)

    {% for co in codeobj_with_rand %}
    // {{co.name}}
    int num_per_cycle_rand_{{co.name}};
    int rand_interval_{{co.name}};
    int num_per_gen_rand_{{co.name}};
    int idx_rand_{{co.name}};
    int rand_floats_per_obj_{{co.name}};

    {% endfor %}

    // normal distributed random numbers (randn)

    {% for co in codeobj_with_randn %}
    // {{co.name}}
    int num_per_cycle_randn_{{co.name}};
    int randn_interval_{{co.name}};
    int num_per_gen_randn_{{co.name}};
    int idx_randn_{{co.name}};
    int randn_floats_per_obj_{{co.name}};

    {% endfor %}

    void init();
    void set_curand_device_api_states();
    void refill_uniform_numbers(randomNumber_t*, randomNumber_t*&, int, int&);
    void refill_normal_numbers(randomNumber_t*, randomNumber_t*&, int, int&);

public:
    //Network();
    void next_time_step();
    void set_seed(unsigned long long);
};

#endif

{% endmacro %}
