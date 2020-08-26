#!/bin/bash

b2c_dir=$1  # path to brian2cuda git repository which should be run as
logfile=$2  # entire path to logfile

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
. ~/anaconda3/etc/profile.d/conda.sh
conda activate b2c

# CUDA
. ~/.init_cuda.sh

# XXX: needs to cd into the tools directory for PYTHONPATH setup to work
cd "$b2c_dir"/brian2cuda/tools
bash _run_test_suite.sh "$run_name" "$logdir"
