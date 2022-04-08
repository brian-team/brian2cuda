OBJECTS := customUpdate.o neuronUpdate.o synapseUpdate.o init.o runner.o 
CUDA_PATH ?=/usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
NVCCFLAGS := -x cu -arch sm_61 -std=c++11 --compiler-options "-fPIC -Wno-return-type-c-linkage" -Xcudafe "--diag_suppress=extern_entity_treated_as_static" 
LINKFLAGS := --shared -arch sm_61

DEPS := $(OBJECTS:.o=.d)

.PHONY: all clean

all: librunner.so

librunner.so: $(OBJECTS)
	@$(NVCC) $(LINKFLAGS) -o $@ $(OBJECTS)

-include $(DEPS)

%.d: %.cc
	@$(NVCC) -M $(NVCCFLAGS) $< 1> $@

%.o: %.cc %.d
	@$(NVCC) -dc $(NVCCFLAGS) $<

%.d: ;

clean:
	@rm -f $(OBJECTS) $(DEPS) librunner.so
