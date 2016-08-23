#!/bin/bash

for CPP_MODEL in *cpp.py; do 
	printf "Running $CPP_MODEL ...\n"
	( PYTHONPATH=..:../../moritzaugustin_brian2 python $CPP_MODEL > /tmp/tmp_$CPP_MODEL 2>&1 && printf "\n... done with $CPP_MODEL\n\n" ) &
done


for CUDA_MODEL in *cuda.py; do 
	printf "Running $CUDA_MODEL ...\n"
	PYTHONPATH=..:../../moritzaugustin_brian2 python $CUDA_MODEL
	printf "\n... done with $CUDA_MODEL \n\n"
done


wait

for CPP_MODEL in *cpp.py; do 
	printf "\nOutput from $CPP_MODEL:\n"
	cat /tmp/tmp_$CPP_MODEL
done


# create result_figures with all resulting plots
./combine_plots.sh

