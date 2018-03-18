PROGRAM = main
SRCS_CU  = $(wildcard *.cu */*.cu)
OBJS_CU  = ${SRCS_CU:.cu=.o}
CC = @nvcc
DEBUG = -g -G -DTHRUST_DEBUG
OPTIMISATIONFLAGS = {{ nvcc_optimization_flags }} -Xcompiler "{{ cpp_compiler_flags }}"
NVCCFLAGS = {% if disable_warnings %}-w{% endif %} {% if disable_asserts %}-DNDEBUG{% endif %} -I. -std=c++11
GPU_ARCH_FLAGS = {{ gpu_arch_flags }}
LFLAGS = -lcurand

# Add optimisation flags
all: NVCCFLAGS += $(OPTIMISATIONFLAGS)
all: executable

# Add debug flags
debug: GPU_ARCH_FLAGS += $(DEBUG)
debug: executable

.PHONY: all debug executable clean

executable: $(OBJS_CPP) $(OBJS_CU)
	$(CC) $(GPU_ARCH_FLAGS) $(LFLAGS) $(OBJS_CU) -o $(PROGRAM)

clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) $(PROGRAM) results/*

%.o : %.cu
	$(CC) $(GPU_ARCH_FLAGS) $(NVCCFLAGS) -dc $< -o $@

