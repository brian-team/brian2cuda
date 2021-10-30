PROGRAM = main

SRCS = code_objects/before_run_synapses_1_post_push_spikes.cu code_objects/before_run_synapses_1_pre_push_spikes.cu code_objects/before_run_synapses_2_pre_push_spikes.cu code_objects/before_run_synapses_pre_push_spikes.cu code_objects/neurongroup_1_stateupdater_codeobject.cu code_objects/neurongroup_1_thresholder_codeobject.cu code_objects/neurongroup_stateupdater_codeobject.cu code_objects/neurongroup_thresholder_codeobject.cu code_objects/spikegeneratorgroup_codeobject.cu code_objects/spikemonitor_1_codeobject.cu code_objects/spikemonitor_2_codeobject.cu code_objects/spikemonitor_codeobject.cu code_objects/synapses_1_group_variable_set_conditional_codeobject.cu code_objects/synapses_1_group_variable_set_conditional_codeobject_1.cu code_objects/synapses_1_post_codeobject.cu code_objects/synapses_1_post_push_spikes.cu code_objects/synapses_1_pre_codeobject.cu code_objects/synapses_1_pre_push_spikes.cu code_objects/synapses_1_synapses_create_generator_codeobject.cu code_objects/synapses_2_pre_codeobject.cu code_objects/synapses_2_pre_push_spikes.cu code_objects/synapses_2_synapses_create_generator_codeobject.cu code_objects/synapses_group_variable_set_conditional_codeobject.cu code_objects/synapses_pre_codeobject.cu code_objects/synapses_pre_push_spikes.cu code_objects/synapses_synapses_create_generator_codeobject.cu main.cu network.cu objects.cu rand.cu run.cu synapses_classes.cu
H_SRCS = brianlib/clocks.h brianlib/common_math.h brianlib/cudaVector.h brianlib/cuda_utils.h brianlib/curand_buffer.h brianlib/dynamic_array.h brianlib/spikequeue.h brianlib/stdint_compat.h code_objects/before_run_synapses_1_post_push_spikes.h code_objects/before_run_synapses_1_pre_push_spikes.h code_objects/before_run_synapses_2_pre_push_spikes.h code_objects/before_run_synapses_pre_push_spikes.h code_objects/neurongroup_1_stateupdater_codeobject.h code_objects/neurongroup_1_thresholder_codeobject.h code_objects/neurongroup_stateupdater_codeobject.h code_objects/neurongroup_thresholder_codeobject.h code_objects/spikegeneratorgroup_codeobject.h code_objects/spikemonitor_1_codeobject.h code_objects/spikemonitor_2_codeobject.h code_objects/spikemonitor_codeobject.h code_objects/synapses_1_group_variable_set_conditional_codeobject.h code_objects/synapses_1_group_variable_set_conditional_codeobject_1.h code_objects/synapses_1_post_codeobject.h code_objects/synapses_1_post_push_spikes.h code_objects/synapses_1_pre_codeobject.h code_objects/synapses_1_pre_push_spikes.h code_objects/synapses_1_synapses_create_generator_codeobject.h code_objects/synapses_2_pre_codeobject.h code_objects/synapses_2_pre_push_spikes.h code_objects/synapses_2_synapses_create_generator_codeobject.h code_objects/synapses_group_variable_set_conditional_codeobject.h code_objects/synapses_pre_codeobject.h code_objects/synapses_pre_push_spikes.h code_objects/synapses_synapses_create_generator_codeobject.h network.h objects.h rand.h run.h synapses_classes.h
OBJS = ${SRCS:.cu=.o}
OBJS := ${OBJS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
NVCC = @/cognition/home/local/cuda/cuda-11.2/bin/nvcc
NVCCFLAGS = -I. -std=c++11 -arch=sm_61 -w -use_fast_math -g -DDEBUG -G -DTHRUST_DEBUG -Xcompiler "-I/cognition/home/subora/miniconda3/envs/b2c/include -w -O3 -ffast-math -fno-finite-math-only -march=native -std=c++11"
LFLAGS = -lcurand -I. -arch=sm_61 -L/cognition/home/subora/miniconda3/envs/b2c/lib -Xlinker -R/cognition/home/subora/miniconda3/envs/b2c/lib -w -g -G

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS)
	$(NVCC) $(LFLAGS) $(OBJS) -o $(PROGRAM)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@
