#!/bin/bash

usage=$(cat << END
usage: $0 <options> -- <run_test_suite.py arguments>

with <options>:
    -h|--help                 Show usage message
    -n|--name <string>        Name for test suite run (default: 'noname'). This
                              name will also be used for the pytest cache
                              directory. That means when running the test suite
                              multiple times with the same --name, pytest
                              options --last-failed or --failed-first can be
                              used.
    -s|--suffix <string>      Name suffix for the logfile (<name>_<suffix>_<timestemp>)
    -g|--gpu <K40|RTX2080>    Which GPU to run on
    -c|--cores <int>          Number of CPU cores to request (-binding linear:<>)
    -r|--remote <head-node>   Remote machine url or name (if configured in ~/.ssh/config)
    -l|--log-dir              Remote path to directory where logfiles will be stored.
    -s|--remote-conda-sh      Remote path to conda.sh
    -e|--remote-conda-env     Conda environment name with brian2cuda on remote
    -k|--keep-remote-repo     Don't delete remote repository after run
END
)

echo_usage() {
    echo "$usage"
}

# DEFAULTS
# remote machine name
remote="cluster"
# default task name
test_suite_task_name=noname
# -l cuda=$test_suite_gpu
test_suite_gpu=1
# number of cores, with 2 threads per core
test_suite_cores=2
# path to conda.sh on remote
path_conda_sh_remote="~/anaconda3/etc/profile.d/conda.sh"
# conda environment for brian2cuda on the remote
conda_env_remote="b2c"
# where to store the logfile on the remote
test_suite_remote_dir="~/projects/brian2cuda/test-suite"
# keep remote (false by default)
keep_remote_repo=1
# $suffix unset by default

# Load configuration file
script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$script_path/_load_remote_config.sh" ~/.brian2cuda-remote-dev.conf

# long args seperated by comma, short args not
# colon after arg indicates that an option is expected (kwarg)
short_args=hn:g:c:r:l:s:e:k
long_args=help,name:,gpu:,cores:,remote:,log-dir:,remote-conda-sh:,remote-conda-env:,keep-remote-repo
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
            test_suite_task_name="$2"
            shift 2
            ;;
        -s | --suffix )
            suffix="$2"
            shift 2
            ;;
        -g | --gpu )
            gpu="$2"
            if [ "$gpu" != "RTX2080" ] && [ "$gpu" != "K40" ]; then
                echo_usage
                echo -e "\n$0: error: invalid argument $gpu for $1"
                exit 1
            fi
            test_suite_gpu="\"1($gpu)\""
            shift 2
            ;;
        -c | --cores )
            test_suite_cores="$2"
            shift 2
            ;;
        -r | --remote )
            remote="$2"
            shift 2
            ;;
        -l | --log-dir )
            test_suite_remote_dir="$2"
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

# all args after -- are passed to run_test_suite.py (collected in $@)
pytest_cache_dir="$test_suite_remote_dir"/.pytest_caches
test_suite_args="--notify-slack --cache-dir $pytest_cache_dir/$test_suite_task_name $@"

test_suite_remote_logdir="$test_suite_remote_dir/results"

# Check that test_suite_args are valid arguments for run_test_suite.py
dry_run_output=$(python ../test_suite/run_test_suite.py --dry-run $test_suite_args 2>&1)
if [ $? -ne 0 ]; then
    echo_usage
    echo -e "$0: error: invalid <run_test_suite.py arguments>\n"
    echo "$dry_run_output"
    exit 1
fi

# get path to this git repo (brian2cuda)
local_b2c_dir=$(git rev-parse --show-toplevel)

if [ -n "$suffix" ]; then
    run_name="$test_suite_task_name"_"$suffix"_"$(date +%y-%m-%d_%H-%M-%S)"
else
    run_name="$test_suite_task_name"_"$(date +%y-%m-%d_%H-%M-%S)"
fi

local_logfile="/tmp/$run_name.log"
remote_logfile="$test_suite_remote_logdir/$run_name.log"
qsub_name=${run_name//_/__}
qsub_name=b2c-tests-${qsub_name//:/_}

# create tmp folder name for brian2cuda code (in $HOME)
remote_b2c_dir="$test_suite_remote_dir/brian2cuda-synced-repos/$run_name"
# folder name for qsub .o and .e logs
remote_ge_log_dir="$test_suite_remote_dir/ge-logs"


### Create git diffs locally
# else excluding tracked files in rsync will mess  diffs up

# get code state
cd "$local_b2c_dir"
b2c_branch="$(git rev-parse --abbrev-ref HEAD)"
b2c_commit="$(git rev-parse HEAD)"
b2c_diff="$(git diff)"
cd frozen_repos/brian2
brian2_branch="$(git rev-parse --abbrev-ref HEAD)"
brian2_commit="$(git rev-parse HEAD)"
brian2_diff="$(git diff)"


cat > "$local_logfile" <<EOL
brian2CUDA test suite run
name: $run_name
run directory: $local_b2c_dir

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


### Create logdirs on remote
ssh "$remote" \
    "mkdir -p \
        $test_suite_remote_logdir \
        $remote_b2c_dir \
        $remote_ge_log_dir \
        $pytest_cache_dir"

### Copy local logfile with diff over to remote
rsync -avzz "$local_logfile" "$remote:$remote_logfile"


### Copy brian2cuda repo over to remote
rsync -avzz \
    --exclude '*.o' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'tags' \
    --exclude 'examples' \
    --exclude 'dev'\
    --exclude '.eggs'\
    --exclude '.git' \
    --exclude 'worktrees' \
    "$local_b2c_dir"/ "$remote:$remote_b2c_dir"


bash_script=brian2cuda/tools/test_suite/_run_test_suite.sh
# submit test suite script through qsub on cluster headnote
ssh $remote "source /opt/ge/default/common/settings.sh && \
    qsub \
        -wd $remote_ge_log_dir \
        -q cognition-all.q \
        -l cuda=$test_suite_gpu \
        -N $qsub_name \
        -binding linear:$test_suite_cores \
        $remote_b2c_dir/brian2cuda/tools/remote_run_scripts/_on_headnode.sh \
            $bash_script \
            $remote_b2c_dir \
            $remote_logfile \
            $path_conda_sh_remote \
            $conda_env_remote \
            $keep_remote_repo \
            $test_suite_args"
            # $1: bash_script $2: b2c_dir, $3: logfile, $4 remote conda.sh
            # $5 remote conda env $6 bool for keeping tmp remote b2c dir
            # $@ (the rest): arguments passed to test suite script
            # (_on_headnode.sh)


if [ $keep_remote_repo -eq 0 ]; then
    echo -e "\nThe copied brian2cuda directory on the remote will not be deleted" \
            "due to the -k|--keep-remote-dir option. Make sure you delete it once" \
            "you don't need it anymore!"
fi
