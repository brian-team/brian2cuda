PROGRAM = main
SRCS = objects.cpp main.cpp code_objects/neurongroup_resetter_codeobject.cpp code_objects/synapses_pre_push_spikes.cpp code_objects/synapses_post_codeobject.cpp code_objects/neurongroup_thresholder_codeobject.cpp code_objects/spikemonitor_codeobject.cpp code_objects/synapses_1_pre_push_spikes.cpp code_objects/spikegeneratorgroup_codeobject.cpp code_objects/synapses_post_initialise_queue.cpp code_objects/statemonitor_1_codeobject.cpp code_objects/synapses_pre_codeobject.cpp code_objects/synapses_1_pre_codeobject.cpp code_objects/statemonitor_codeobject.cpp code_objects/synapses_1_group_variable_set_conditional_codeobject.cpp code_objects/synapses_1_synapses_create_codeobject.cpp code_objects/synapses_post_push_spikes.cpp code_objects/synapses_1_pre_initialise_queue.cpp code_objects/synapses_synapses_create_codeobject.cpp code_objects/neurongroup_stateupdater_codeobject.cpp code_objects/synapses_pre_initialise_queue.cpp code_objects/synapses_group_variable_set_conditional_codeobject.cpp network.cpp synapses_classes.cpp run.cpp
H_SRCS = objects.h code_objects/neurongroup_resetter_codeobject.h code_objects/synapses_pre_push_spikes.h code_objects/synapses_post_codeobject.h code_objects/neurongroup_thresholder_codeobject.h code_objects/spikemonitor_codeobject.h code_objects/synapses_1_pre_push_spikes.h code_objects/spikegeneratorgroup_codeobject.h code_objects/synapses_post_initialise_queue.h code_objects/statemonitor_1_codeobject.h code_objects/synapses_pre_codeobject.h code_objects/synapses_1_pre_codeobject.h code_objects/statemonitor_codeobject.h code_objects/synapses_1_group_variable_set_conditional_codeobject.h code_objects/synapses_1_synapses_create_codeobject.h code_objects/synapses_post_push_spikes.h code_objects/synapses_1_pre_initialise_queue.h code_objects/synapses_synapses_create_codeobject.h code_objects/neurongroup_stateupdater_codeobject.h code_objects/synapses_pre_initialise_queue.h code_objects/synapses_group_variable_set_conditional_codeobject.h network.h synapses_classes.h run.h brianlib/common_math.h brianlib/dynamic_array.h brianlib/clocks.h
OBJS = ${SRCS:.cpp=.o}
CC = @g++
DEBUG = -g
OPTIMISATIONS = -w -O3
CFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I. -fopenmp
LFLAGS = -fopenmp 
DEPS = make.deps

all: executable

# Adds debug flags
debug: CFLAGS += $(DEBUG)
debug: LFLAGS += $(DEBUG)
debug: executable

# Adds -march=native to optimisations, which only works for recent gcc versions
native: OPTIMISATIONS += -march=native
native: executable

.PHONY: all debug native executable clean

executable: $(OBJS) $(DEPS)
	$(CC) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

make.deps: $(SRCS) $(H_SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > make.deps
	
ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@