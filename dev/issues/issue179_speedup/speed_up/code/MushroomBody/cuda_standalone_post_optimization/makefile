PROGRAM = main

SRCS = main.cu code_objects/synapses_synapses_create_generator_codeobject.cu code_objects/synapses_2_synapses_create_generator_codeobject.cu code_objects/synapses_1_synapses_create_generator_codeobject.cu  
H_SRCS = code_objects/synapses_synapses_create_generator_codeobject.h code_objects/synapses_2_synapses_create_generator_codeobject.h code_objects/synapses_1_synapses_create_generator_codeobject.h brianlib/cuda_utils.h brianlib/common_math.h brianlib/stdint_compat.h brianlib/dynamic_array.h brianlib/cudaVector.h brianlib/clocks.h brianlib/spikequeue.h brianlib/curand_buffer.h
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
