GENERATED_CODE_DIR	:=magicnetwork_model_CODE
CXXFLAGS		+=-std=c++11 -Wno-write-strings -I. -Ibrianlib/randomkit -w -O3 -ffast-math -fno-finite-math-only -march=native -std=c++11
LDFLAGS			+=-L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR) 

.PHONY: all clean generated_code

all: main

main: main.cpp  code_objects/spikemonitor_codeobject.cpp  code_objects/spikegeneratorgroup_codeobject.cpp  code_objects/synapses_synapses_create_generator_codeobject.cpp  code_objects/spikemonitor_2_codeobject.cpp  code_objects/synapses_1_group_variable_set_conditional_codeobject_1.cpp  code_objects/synapses_1_synapses_create_generator_codeobject.cpp  code_objects/synapses_pre_push_spikes.cpp  code_objects/synapses_1_post_push_spikes.cpp  code_objects/synapses_1_pre_push_spikes.cpp  code_objects/synapses_group_variable_set_conditional_codeobject.cpp  code_objects/synapses_2_synapses_create_generator_codeobject.cpp  code_objects/synapses_2_pre_push_spikes.cpp  code_objects/spikemonitor_1_codeobject.cpp  objects.cpp  code_objects/synapses_1_group_variable_set_conditional_codeobject.cpp  brianlib/randomkit/randomkit.cc generated_code
	$(CXX) $(CXXFLAGS)  main.cpp  code_objects/spikemonitor_codeobject.cpp  code_objects/spikegeneratorgroup_codeobject.cpp  code_objects/synapses_synapses_create_generator_codeobject.cpp  code_objects/spikemonitor_2_codeobject.cpp  code_objects/synapses_1_group_variable_set_conditional_codeobject_1.cpp  code_objects/synapses_1_synapses_create_generator_codeobject.cpp  code_objects/synapses_pre_push_spikes.cpp  code_objects/synapses_1_post_push_spikes.cpp  code_objects/synapses_1_pre_push_spikes.cpp  code_objects/synapses_group_variable_set_conditional_codeobject.cpp  code_objects/synapses_2_synapses_create_generator_codeobject.cpp  code_objects/synapses_2_pre_push_spikes.cpp  code_objects/spikemonitor_1_codeobject.cpp  objects.cpp  code_objects/synapses_1_group_variable_set_conditional_codeobject.cpp  brianlib/randomkit/randomkit.cc -o main $(LDFLAGS)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)