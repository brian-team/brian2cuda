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
# $6:    delete temore directory after tests are run or not
# $7...: the rest is passed as args to run_test_suite.py
bash_script="$1"
b2c_dir="$2"
logfile="$3"
path_conda_sh="$4"
conda_env="$5"
keep_remote_repo="$6"
parallel="$7"
shift 7
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
source "$path_conda_sh"
conda activate "$conda_env"

# CUDA
if test -f ~/.init_cuda.sh; then
    source ~/.init_cuda.sh
else
    source /cognition/home/local/.init_cuda.sh
fi

if [ ! "$parallel" -eq 0 ]; then
    # Source grid engine commands for qmake/qrsh
    source /opt/ge/default/common/settings.sh
    # Set tmpdir in home directory, where different compute nodes have access
    # to (needed for parallel compilation / execution)
    export TMPDIR="$b2c_dir/tmp"
    mkdir $TMPDIR
    echo "Set TMPDIR=$TMPDIR"
fi

script_path="$b2c_dir/$bash_script"
script_folder="$(dirname $script_path)"
script_name="$(basename $script_path)"
# XXX: needs to cd into the tools directory for PYTHONPATH setup to work
cd "$script_folder"
bash "$script_name" "$logfile_name" "$logfile_dir" "$test_suite_args"
