#!/bin/bash
# Don't call this file directly, it is used for submitting test suite and
# benchmark scripts to the grid engine (see e.g. run_test_suite_on_cluster.sh)
# It will set the conda environment, initialize CUDA variables and make sure
# the previously copied brian2cuda repository is deleted after the run.

# $1:    path to bash script that will be submitted to grid engine, relative to
#        brian2cuda repository path ($2).
#        Example: brian2cuda/tools/test_suite/_run_test_suite.sh
# $2:    path to brian2cuda git repository which should be run as
# $3:    entire path to logfile
# $4:    path to conda.sh
# $5:    conda env name
# $6...: the rest is passed as args to run_test_suite.py
echo "On headnode"
bash_script="$1"
b2c_dir="$2"
logfile="$3"
path_conda_sh="$4"
conda_env="$5"
keep_remote_repo="$6"
shift 6
test_suite_args="$@"

# deletes the brian2cuda directory
function cleanup {
    cd
    if [ $keep_remote_repo -eq 1 ]; then
        rm -rf "$b2c_dir"
        echo "Script EXIT: Deleted tmp brian2cuda directory $b2c_dir" | tee -a "$logfile"
    else
        echo "Script EXIT: tmp brian2cuda directory was not deleted" \
             "(-k|--keep-remote-dir option). If you don't need it, delete it:" \
             "$b2c_dir" | tee -a "$logfile"
    fi
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

# Get directory and name of logfile. For test suite script, the logfile has the
# timestemp and the directory is where all different logfiles are stored. For
# benchmark scripts, the directory has the timestemp and the logfile is always
# `full.log`
logfile_dir="$(dirname $logfile)"
logfile_name="$(basename $logfile)"

# activate bashrc (for conda activation and CUDA paths)
. "$path_conda_sh"
conda activate "$conda_env"

# CUDA
if test -f ~/.init_cuda.sh; then
    . ~/.init_cuda.sh
else
    . /cognition/home/local/.init_cuda.sh
fi

# Set GeNN environment variales such that they always use the GeNN from the
# frozen_repos/genn submodule
. "$b2c_dir/frozen_repos/init_genn.sh"

script_path="$b2c_dir/$bash_script"
script_folder="$(dirname $script_path)"
script_name="$(basename $script_path)"
# XXX: needs to cd into the tools directory for PYTHONPATH setup to work
echo "About to call script in $script_folder"
cd "$script_folder"
bash "$script_name" "$logfile_name" "$logfile_dir" "$test_suite_args"
