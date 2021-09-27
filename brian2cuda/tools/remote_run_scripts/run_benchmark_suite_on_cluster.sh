#!/bin/bash

# show the manual for this script
usage=$(cat << END
usage: $0 <options> -- <run_benchmark_suite.py arguments>

with <options>:
    -h|--help                 Show usage message
    -n|--name <string>        Name for benchmark suite log file
    -g|--gpu <K40|RTX2080>    Which GPU to run on
    -c|--cores <int>          Number of CPU cores to request (-binding linear:<>)
    -r|--remote <head-node>   Remote machine url or name (if configured in ~/.ssh/config)
    -l|--log-dir              Remote path to directory where logfiles will be stored.
    -s|--remote-conda-sh      Remote path to conda.sh
    -e|--remote-conda-env     Conda environment name with brian2cuda on remote
    -k|--keep-remote-repo     Don't delete remote repository after run
    -t|--copy-tools-dir       Copy (rsync) local brian2cuda/tools directory to remote
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
# keep remote (false by default)
keep_remote_repo=1
# copy local brian2cuda/tools directory for remote (use this when you are
# modifying benchmark scripts and want to test them without commiting yet)
copy_tools_dir=1

# Load configuration file
script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$script_path/_load_remote_config.sh" ~/.brian2cuda-remote-dev.conf

# long args seperated by comma, short args not
# colon after arg indicates that an option is expected (kwarg)
short_args=hn:g:c:r:l:s:e:kt
long_args=help,name:,gpu:,cores:,remote:,log-dir:,remote-conda-sh:,remote-conda-env:,keep-remote-repo,copy-tools-dir
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
        -k | --keep-remote-dir )
            keep_remote_repo=0
            shift 1
            ;;
        -t | --copy-tools-dir )
            copy_tools_dir=0
            shift 1
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

run_name="$benchmark_suite_task_name"_"$(date +%y-%m-%d_%H-%M-%S)"
remote_dir="$benchmark_result_dir/$run_name"
remote_logfile="$remote_dir/full.log"
qsub_name=${run_name//_/__}
qsub_name=b2c-benchmarks-${qsub_name//:/_}

# all args after --
benchmark_suite_args="-d $remote_dir $@"

echo -e "\nINFO: Running benchmarks script with options $benchmark_suite_args"

# Check that benchmark_suite_args are valid arguments for
# run_benchmark_suite.py
dry_run_output=$(python ../benchmarking/run_benchmark_suite.py --dry-run $benchmark_suite_args 2>&1)
if [ $? -ne 0 ]; then
    echo_usage
    echo -e "$0: error: invalid <run_benchmark_suite.py arguments>\n"
    echo "$dry_run_output"
    exit 1
fi

# get path to this git repo (brian2cuda)
local_b2c_dir=$(git rev-parse --show-toplevel)

# create tmp folder name for brian2cuda code (in $HOME)
remote_b2c_dir="$benchmark_suite_remote_dir/brian2cuda-synced-repos/$run_name"
# folder name for qsub .o and .e logs
remote_ge_log_dir="$benchmark_suite_remote_dir/ge-logs"

echo -e "\nINFO: Setting up remote brian2cuda repository..."

# Setup remote brian2cuda repository by cloning from GitHub
ssh "$remote" "/bin/bash" << EOF
    # Create remote directories
    mkdir -p $remote_dir $remote_b2c_dir $remote_ge_log_dir $benchmark_result_dir

    # Clone brian2cuda from GitHub
    git clone https://github.com/brian-team/brian2cuda.git $remote_b2c_dir

    # Make bare repository (you can push to it)
    cd $remote_b2c_dir
    git config receive.denyCurrentBranch updateInstead
EOF

local_commit="$(git rev-parse HEAD)"
echo -e "\nINFO: Pushing local commit $local_commit to remote repository..."

# Add remote repository as git remote (allows pushing from local repo)
cd $local_b2c_dir
git remote add remote_repo $remote:$remote_b2c_dir

### Push local commit to remote repo
git push remote_repo

# Remove remote from local repo
git remote remove remote_repo

# Checkout correct remote branch and initialize submodules
echo -e "\nINFO: Checking out remote branch and intializing submodules..."
ssh "$remote" "/bin/bash" << EOF
    # Checkout correct commit on remote
    cd $remote_b2c_dir
    git checkout $local_commit

    # Update submodules and apply diff files if present
    cd frozen_repos

    bash submodules_update.sh
EOF

# Copy at least local run_benchmark_suite.py to remote such that the locally
# chosen benchmark configuration is run on the remote (without having to commit
# it). If `-t | --copy-tools-dir` is set, copy the entire brian2cuda/tools dir.
if [ $copy_tools_dir -eq 0 ]; then
    echo -e "\nINFO: Copying local tools directory to remote..."
    # -m: --prune-empty-dirs
    rsync -avzzm --include="*/" --include="*.sh" --include="*.py" --exclude="*" \
        $local_b2c_dir/brian2cuda/tools/ $remote:$remote_b2c_dir/brian2cuda/tools
else
    echo -e "\nINFO: Copying local 'run_benchmark_suite.py' file to remote..."
    scp \
        $local_b2c_dir/brian2cuda/tools/benchmarking/run_benchmark_suite.py \
        $remote:$remote_b2c_dir/brian2cuda/tools/benchmarking/
fi

echo -e "\nINFO: Submitting script to grid engine to run benchmarks..."
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
            $bash_script \
            $remote_b2c_dir \
            $remote_logfile \
            $path_conda_sh_remote \
            $conda_env_remote \
            $keep_remote_repo \
            0 \
            $benchmark_suite_args"
            # $1: bash_script $2: b2c_dir, $3: logfile, $4: remote conda.sh
            # $5: remote conda env $6: bool for keeping tmp remote b2c dir
            # $7: bool for running in parallel (here false)
            # $@ (the rest): arguments passed to benchmark script
            # (_on_headnode.sh)

echo
echo "INFO: Grid engine logs are stored on remote at $remote_ge_log_dir/$run_name"
echo "INFO: Benchmark results are stored on remote at $benchmark_result_dir/$run_name"
if [ $keep_remote_repo -eq 0 ]; then
    echo "WARNING: The remote brian2cuda repository will not be deleted " \
         "due to the '-k | --keep-remote-dir' option. Make sure you delete " \
         "it once you don't need it anymore! It is at $remote_b2c_dir"
fi
