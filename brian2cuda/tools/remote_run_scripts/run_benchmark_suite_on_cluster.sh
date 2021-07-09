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
# -l cuda=$benchmark_suite_gpu (default RTX2080)
benchmark_suite_gpu="\"1(RTX2080)\""
# number of cores, with 2 threads per core
benchmark_suite_cores=2
# path to conda.sh on remote
path_conda_sh_remote="~/anaconda3/etc/profile.d/conda.sh"
# conda environment for brian2cuda on the remote
conda_env_remote="b2c"
# where to store the logfile on the remote
benchmark_suite_remote_dir="~/projects/brian2cuda/benchmark-suite"
# result directory
benchmark_result_dir="$benchmark_suite_remote_dir/results"

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

run_name="$benchmark_suite_task_name\_$(date +%y-%m-%d_%T)"
remote_dir="$benchmark_result_dir/$run_name"
remote_logfile="$remote_dir/full.log"
qsub_name=${run_name//_/__}
qsub_name=b2c-${qsub_name//:/_}

# all args after --
benchmark_suite_args="-d $remote_dir $@"

# Check that benchmark_suite_args are valid arguments for
# run_manuscript_runtime_vs_N_benchmarks.py
dry_run_output=$(python ../benchmarking/run_manuscript_runtime_vs_N_benchmarks.py --dry-run $benchmark_suite_args 2>&1)
if [ $? -ne 0 ]; then
    echo_usage
    echo -e "$0: error: invalid <run_manuscript_runtime_vs_N_benchmarks.py arguments>\n"
    echo "$dry_run_output"
    exit 1
fi

# get path to this git repo (brian2cuda)
local_b2c_dir=$(git rev-parse --show-toplevel)

# create tmp folder name for brian2cuda code (in $HOME)
remote_b2c_dir="$benchmark_suite_remote_dir/brian2cuda-synced-repos/$run_name"
# folder name for qsub .o and .e logs
remote_ge_log_dir="$benchmark_suite_remote_dir/ge-logs"

### Create logdirs on remote
ssh "$remote" "mkdir -p $remote_dir $remote_b2c_dir $remote_ge_log_dir $benchmark_result_dir"

### Copy brian2cuda repo over to remote
rsync -avzz \
    --exclude '*.o' \
    --exclude 'tags' \
    --exclude 'examples' \
    --exclude 'dev' \
    --exclude '.eggs'\
    --exclude 'worktrees' \
    "$local_b2c_dir"/ "$remote:$remote_b2c_dir"

bash_script=brian2cuda/tools/benchmarking/_run_benchmark_suite.sh
# submit test suite script through qsub on cluster headnote
ssh $remote "source /opt/ge/default/common/settings.sh && \
    qsub \
    -wd $remote_ge_log_dir \
    -q cognition-all.q \
    -l cuda=$benchmark_suite_gpu \
    -N $qsub_name \
    -binding linear:$benchmark_suite_cores \
    $remote_b2c_dir/brian2cuda/tools/remote_run_scripts/_on_headnode.sh \
    $bash_script $remote_b2c_dir $remote_logfile $path_conda_sh_remote $conda_env_remote $benchmark_suite_args"
    # $1: bash_script $2: b2c_dir, $3: logfile, $4 remote conda.sh $5 remote conda env (_run_benchmark_suite.sh)
