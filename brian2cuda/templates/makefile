PROGRAM = main

SRCS = {{source_files}}
H_SRCS = {{header_files}}
OBJS = ${SRCS:.cu=.o}
OBJS := ${OBJS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
NVCC = @{{ nvcc_path }} -ccbin $(CXX)
NVCCFLAGS = -I. -std=c++11 {{gpu_arch_flags}} {{nvcc_compiler_flags}} {{compiler_debug_flags}} -Xcompiler "{{cpp_compiler_flags}}"
LFLAGS = -lcurand -I. {{gpu_arch_flags}} {{cpp_linker_flags}} {{linker_debug_flags}}

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS)
	$(NVCC) $(LFLAGS) $(OBJS) -o $(PROGRAM)

clean:
	{{rm_cmd}}

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@
