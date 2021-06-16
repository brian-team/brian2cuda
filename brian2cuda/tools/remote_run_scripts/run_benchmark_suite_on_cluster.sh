#!/bin/bash

# show the manual for this script
usage=$(cat << END
usage: $0 <options> -- <run_manuscript_runtime_vs_N_benchmarks.py arguments>

with <options>:
    -h|--help                 Show usage message
    -n|--name <string>        Name for benchmark suite log file
    -g|--gpu <K40|RTX2080>    Which GPU to run on
    -c|--cores <int>          Number of CPU cores to request (-binding linear:<>)
    -r|--remote <head-node>   Remote machine url or name (if configured in ~/.ssh/config)
    -l|--log-dir              Remote path to directory where logfiles will be stored.
    -s|--remote-conda-sh    Remote path to conda.sh
    -e|--remote-conda-env   Conda environment name with brian2cuda on remote
END
)

echo_usage() {
    echo "$usage"
}

# DEFAULTS
# remote machine name
remote="cluster"
# default task name
benchmark_suite_task_name=noname
# -l cuda=$test_suite_gpu
benchmark_suite_gpu=1
# number of cores, with 2 threads per core
benchmark_suite_cores=2
# path to conda.sh on remote
path_conda_sh_remote="~/anaconda3/etc/profile.d/conda.sh"
# conda environment for brian2cuda on the remote
conda_env_remote="b2c"
# where to store the logfile on the remote
benchmark_suite_remote_dir="~/projects/brian2cuda/benchmark-suite"

# Load configuration file
script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$script_path/_load_remote_config.sh" ~/.brian2cuda-remote-dev.conf

# long args seperated by comma, short args not
# colon after arg indicates that an option is expected (kwarg)
short_args=hn:g:c:r:l:s:e:
long_args=help,name:,gpu:,cores:,remote:,log-dir:,remote-conda-sh:,remote-conda-env:
opts=$(getopt --options $short_args --long $long_args --name "$0" -- "$@")
if [ "$?" -ne 0 ]; then
    echo_usage
    exit 1
fi

eval set -- "$opts"

# parse arguments
while true; do
    case "$1" in
        -h | --help)
            echo_usage
            exit 0
            ;;
        -n | --name )
            benchmark_suite_task_name="$2"
            shift 2
            ;;
        -g | --gpu )
            gpu="$2"
            if [ "$gpu" != "RTX2080" ] && [ "$gpu" != "K40" ]; then
                echo_usage
                echo -e "\n$0: error: invalid argument $gpu for $1"
                exit 1
            fi
            benchmark_suite_gpu="\"1($gpu)\""
            shift 2
            ;;
        -c | --cores )
            benchmark_suite_cores="$2"
            shift 2
            ;;
        -r | --remote )
            remote="$2"
            shift 2
            ;;
        -l | --log-dir )
            benchmark_suite_remote_dir="$2"
            shift 2
            ;;
        -s | --remote-conda-sh )
            path_conda_sh_remote="$2"
            shift 2
            ;;
        -e | --remote-conda-env )
            conda_env_remote="$2"
            shift 2
            ;;
        -- )
            # $@ has all arguments after --
            shift
            break
            ;;
        * )
            echo_usage
            exit 1
            ;;
    esac
done

# all args after --
benchmark_suite_args=$@

benchmark_suite_remote_logdir="$benchmark_suite_remote_dir/results"

# get path to this git repo (brian2cuda)
local_b2c_dir=$(git rev-parse --show-toplevel)

# create tmp folder name for brian2cuda code (in $HOME)
remote_b2c_dir="$benchmark_suite_remote_dir/brian2cuda-synced-repos/$run_name"
# folder name for qsub .o and .e logs
remote_ge_log_dir="$benchmark_suite_remote_dir/ge-logs"

### Create logdirs on remote
ssh "$remote" "mkdir -p $benchmark_suite_remote_logdir $remote_b2c_dir $remote_ge_log_dir"

echo "Copying brian2cuda repository to server"
### Copy brian2cuda repo over to remote
rsync -avzz \
    --exclude '*.o' \
    --exclude 'tags' \
    --exclude 'examples' \
    --exclude '.eggs'\
    --exclude '.git' \
    "$local_b2c_dir"/ "$remote:$remote_b2c_dir"

echo $remote_b2c_dir/brian2cuda/tools/run_benchmark_suite.sh
# submit test suite script through qsub on cluster headnote
ssh $remote "source /opt/ge/default/common/settings.sh && \
    qsub \
    -wd $remote_ge_log_dir \
    -q cognition-all.q \
    -l cuda=$benchmark_suite_gpu \
    -N $qsub_name \
    -binding linear:$benchmark_suite_cores \
    $remote_b2c_dir/brian2cuda/tools/run_benchmark_suite.sh \
    $remote_b2c_dir $remote_logfile $benchmark_suite_args $path_conda_sh_remote $conda_env_remote"
    # $1: b2c_dir, $2: logfile, $3 remote conda.sh $4 remote conda env (run_benchmark_suite.sh)