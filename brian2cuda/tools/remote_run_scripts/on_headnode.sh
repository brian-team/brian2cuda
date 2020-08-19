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

mkdir -p logdir

# get code state
cd "$b2c_dir"
b2c_branch="$(git rev-parse --abbrev-ref HEAD)"
b2c_commit="$(git rev-parse HEAD)"
b2c_diff="$(git diff)"
cd frozen_repos/brian2
brian2_branch="$(git rev-parse --abbrev-ref HEAD)"
brian2_commit="$(git rev-parse HEAD)"
brian2_diff="$(git diff)"

# create logfile
if [ -f $logfile ]; then
    echo "ERROR: logfile $logfile already exists. Overwriting..."
fi

cat > "$logfile" <<EOL
brian2CUDA test suite run
name: $run_name
run directory: $b2c_dir

b2c branch: $b2c_branch
b2c commit: $b2c_commit
--- b2c-diff start ---
$b2c_diff
--- b2c-diff end ---

brian2 branch: $brian2_branch
brian2 commit: $brian2_commit
--- brian2-diff start ---
$brian2_diff
--- brian2-diff end ---

EOL

# activate bashrc (for conda activation and CUDA paths)
. ~/anaconda3/etc/profile.d/conda.sh
conda activate b2c

# CUDA
. ~/.init_cuda.sh

# XXX: needs to cd into the tools directory for PYTHONPATH setup to work
cd "$b2c_dir"/brian2cuda/tools
bash _run_test_suite.sh "$run_name" "$logdir"
