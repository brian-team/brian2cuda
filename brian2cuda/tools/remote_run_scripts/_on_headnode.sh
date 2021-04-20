#!/bin/bash
# $1:    path to brian2cuda git repository which should be run as
# $2:    entire path to logfile
# $3:    path to conda.sh
# $4:    conda env name
# $5...: the rest is passed as args to run_test_suite.py
b2c_dir="$1"
logfile="$2"
path_conda_sh="$3"
conda_env="$4"
shift 4
test_suite_args="$@"

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
bash _run_test_suite.sh "$run_name" "$logdir" "$test_suite_args"
