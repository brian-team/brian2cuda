#!/bin/bash

script_names=(COBAHH.py COBAHHUncoupled.py COBAHHPseudocoupled80.py COBAHHPseudocoupled1000.py MushroomBody.py STDPCUDA.py STDPCUDAHeterogeneousDelays.py STDPCUDAHomogeneousDelays.py BrunelHakimHeterogDelays.py BrunelHakimHeterogDelaysNarrowDistr.py BrunelHakimHomogDelays.py )

modes=(cpp_standalone cuda_standalone)
output_file=results_$(date '+%Y_%m_%d__%H_%M_%S').txt
echo "The output is stored in $output_file"

for script_name in "${script_names[@]}"
do
  for mode in "${modes[@]}"
  do
    echo "Running $script_name in $mode"
    python "$script_name" "$mode" >> "$output_file" 2>&1
  done
done