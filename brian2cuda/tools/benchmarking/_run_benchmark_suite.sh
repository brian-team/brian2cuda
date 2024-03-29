#!/bin/bash
# This script assumes it is called from where it is stored in in the brian2cuda
# repository and it uses the brian2cuda and brian2 of that repository.

# $1 logfile name, should be `full.log`
# $2 logfile directory, this is the task name with timestemp

logfile_name=$1
results_dir=$2
shift 2
benchmark_suite_args=$@

mkdir -p "$results_dir"

logfile="$results_dir/$logfile_name"

start_time=`date +%s`

# Set GeNN environment variales such that they always use the GeNN from the
# frozen_repos/genn submodule
b2c_dir=$(git rev-parse --show-toplevel)
init_genn_sh="$b2c_dir/frozen_repos/init_genn.sh"
source $init_genn_sh

# Can't pipe the `source` command to tee (blocks variable export), hence echo here
echo -e "\nINFO: sourced $init_genn_sh\n  CUDA_PATH=$CUDA_PATH\n  GENN_PATH=$GENN_PATH\n" \
    | tee -a "$logfile"

benchmark_suite_cmd="python run_benchmark_suite.py $benchmark_suite_args"
echo "$benchmark_suite_cmd" | tee -a "$logfile"
PYTHONPATH="../../..:../../../frozen_repos/brian2:$PYTHONPATH" \
    $test_suite_cmd 2>&1 | tee -a "$logfile"
PYTHONPATH="../../..:../../../frozen_repos/brian2:../../../frozen_repos/brian2genn:$PYTHONPATH" \
    $benchmark_suite_cmd 2>&1 | tee -a "$logfile"

runtime=$(( $(date +%s) - $start_time ))
min=$(( $runtime / 60 ))
remaining_sec=$(( $runtime - $min * 60 ))
hour=$(( $min / 60 ))
remaining_min=$(( $min - $hour * 60 ))
echo -e "Benchmark suite took $hour h $remaining_min m $remaining_sec s." | tee -a "$logfile"
