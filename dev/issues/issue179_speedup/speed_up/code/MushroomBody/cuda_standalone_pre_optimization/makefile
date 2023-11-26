PROGRAM = main

SRCS = main.cu code_objects/synapses_synapses_create_generator_codeobject.cu code_objects/synapses_1_synapses_create_generator_codeobject.cu code_objects/synapses_2_synapses_create_generator_codeobject.cu code_objects/synapses_group_variable_set_conditional_codeobject.cu code_objects/synapses_1_group_variable_set_conditional_codeobject.cu code_objects/synapses_1_group_variable_set_conditional_codeobject_1.cu code_objects/neurongroup_1_stateupdater_codeobject.cu code_objects/neurongroup_stateupdater_codeobject.cu code_objects/neurongroup_1_thresholder_codeobject.cu code_objects/neurongroup_thresholder_codeobject.cu code_objects/spikegeneratorgroup_codeobject.cu code_objects/spikemonitor_codeobject.cu code_objects/spikemonitor_1_codeobject.cu code_objects/spikemonitor_2_codeobject.cu code_objects/synapses_1_pre_initialise_queue.cu code_objects/synapses_1_pre_codeobject.cu code_objects/synapses_1_pre_push_spikes.cu code_objects/synapses_2_pre_initialise_queue.cu code_objects/synapses_2_pre_codeobject.cu code_objects/synapses_2_pre_push_spikes.cu code_objects/synapses_pre_initialise_queue.cu code_objects/synapses_pre_codeobject.cu code_objects/synapses_pre_push_spikes.cu code_objects/synapses_1_post_initialise_queue.cu code_objects/synapses_1_post_codeobject.cu code_objects/synapses_1_post_push_spikes.cu objects.cu network.cu synapses_classes.cu run.cu rand.cu
H_SRCS = code_objects/synapses_synapses_create_generator_codeobject.h code_objects/synapses_1_synapses_create_generator_codeobject.h code_objects/synapses_2_synapses_create_generator_codeobject.h code_objects/synapses_group_variable_set_conditional_codeobject.h code_objects/synapses_1_group_variable_set_conditional_codeobject.h code_objects/synapses_1_group_variable_set_conditional_codeobject_1.h code_objects/neurongroup_1_stateupdater_codeobject.h code_objects/neurongroup_stateupdater_codeobject.h code_objects/neurongroup_1_thresholder_codeobject.h code_objects/neurongroup_thresholder_codeobject.h code_objects/spikegeneratorgroup_codeobject.h code_objects/spikemonitor_codeobject.h code_objects/spikemonitor_1_codeobject.h code_objects/spikemonitor_2_codeobject.h code_objects/synapses_1_pre_initialise_queue.h code_objects/synapses_1_pre_codeobject.h code_objects/synapses_1_pre_push_spikes.h code_objects/synapses_2_pre_initialise_queue.h code_objects/synapses_2_pre_codeobject.h code_objects/synapses_2_pre_push_spikes.h code_objects/synapses_pre_initialise_queue.h code_objects/synapses_pre_codeobject.h code_objects/synapses_pre_push_spikes.h code_objects/synapses_1_post_initialise_queue.h code_objects/synapses_1_post_codeobject.h code_objects/synapses_1_post_push_spikes.h objects.h network.h synapses_classes.h run.h rand.h brianlib/cuda_utils.h brianlib/common_math.h brianlib/stdint_compat.h brianlib/dynamic_array.h brianlib/cudaVector.h brianlib/clocks.h brianlib/spikequeue.h brianlib/curand_buffer.h
OBJS = ${SRCS:.cu=.o}
OBJS := ${OBJS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
NVCC = @/usr/local/cuda-11.2/bin/nvcc
NVCCFLAGS = -I. -std=c++11 -arch=sm_75 -w -use_fast_math -g -DDEBUG -G -DTHRUST_DEBUG -Xcompiler "-I/cognition/home/subora/miniconda3/envs/b2c/include -w -O3 -ffast-math -fno-finite-math-only -march=native"
LFLAGS = -lcurand -I. -arch=sm_75 -L/cognition/home/subora/miniconda3/envs/b2c/lib -Xlinker -R/cognition/home/subora/miniconda3/envs/b2c/lib -w -g -G

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS)
	$(NVCC) $(LFLAGS) $(OBJS) -o $(PROGRAM)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@
