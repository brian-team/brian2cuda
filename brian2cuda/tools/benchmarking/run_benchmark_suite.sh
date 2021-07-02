#!/bin/bash
# $1:    path to brian2cuda git repository which should be run as
# $2:    entire path to logfile
# $3:    path to conda.sh
# $4:    conda env name
# $5...: the rest is passed as args to run_test_suite.py
echo "Running benchmark suite"
b2c_dir="$1"
logfile="$2"
path_conda_sh="$3"
conda_env="$4"
shift 4
benchmark_suite_args="$@"

# deletes the brian2cuda directory
function cleanup {
    cd
    rm -rf "$b2c_dir"
    echo "Script EXIT: Deleted tmp brian2cuda directory $b2c_dir" | tee -a "$logfile"
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

# where to save log files
logdir="$(dirname $logfile)"
run_name="$(basename $logfile)"

# activate bashrc (for conda activation and CUDA paths)
. "$path_conda_sh"
conda activate "$conda_env"

# CUDA
if test -f ~/.init_cuda.sh; then
    . ~/.init_cuda.sh
else
    . /cognition/home/local/.init_cuda.sh
fi

# XXX: needs to cd into the tools directory for PYTHONPATH setup to work
cd "$b2c_dir"/brian2cuda/tools

mkdir -p "$logdir"

logfile="$logdir/$run_name"

start_time=`date +%s`

echo "Current Directory ${PWD}"

PYTHONPATH="../..:../../frozen_repos/brian2genn:$PYTHONPATH"
echo "The python path ${PYTHONPATH}"

cd ../..

echo "The current directory of execution ${PWD}"
python dev/benchmarks/run_manuscript_runtime_vs_N_benchmarks.py $benchmark_suite_args 2>&1 | tee -a "$logfile"

runtime=$(( $(date +%s) - $start_time ))
min=$(( $runtime / 60 ))
remaining_sec=$(( $runtime - $min * 60 ))
hour=$(( $min / 60 ))
remaining_min=$(( $min - $hour * 60 ))
echo -e "Test suite took $hour h $remaining_min m $remaining_sec s." | tee -a "$logfile"