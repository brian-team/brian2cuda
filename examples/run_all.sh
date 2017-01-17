#!/bin/bash

function run_all_cpp {
	for CPP_MODEL in *cpp.py; do
		printf "Running $CPP_MODEL ...\n"
		( PYTHONPATH=../frozen_repos/brian2 python $CPP_MODEL > /tmp/tmp_$CPP_MODEL 2>&1 && printf "\n... done with $CPP_MODEL\n\n" ) &
	done
}

function run_all_cuda {
	for CUDA_MODEL in *cuda.py; do
		printf "Running $CUDA_MODEL ...\n"
		PYTHONPATH=..:../frozen_repos/brian2 python $CUDA_MODEL
		printf "\n... done with $CUDA_MODEL \n\n"
	done
}

function print_cpp_output {
	for CPP_MODEL in *cpp.py; do
		printf "\nOutput from $CPP_MODEL:\n"
		cat /tmp/tmp_$CPP_MODEL
	done
}


if [ $# -eq 0 ]; then # run all
	run_all_cpp
	run_all_cuda
	wait
	print_cpp_output
	./combine_plots.sh # create result_figures with all resulting plots
elif [ $# -eq 1 ]; then
	case "$1" in
		cpp) # only run cpp scripts
			run_all_cpp
			wait
			print_cpp_output
			;;
		cuda)
			run_all_cuda
			;;
	esac
else
	echo $"Usage: $0 [cpp | cuda]"
	exit 1
fi
