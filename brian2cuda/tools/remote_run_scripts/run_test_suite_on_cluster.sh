#!/bin/bash

usage=$(cat << END
usage: $0 <options> -- <run_test_suite.py arguments>

with <options>:
    -h|--help                 Show usage message
    -n|--name <string>        Name for test suite log file
    -g|--gpu <K40|RTX2080>    Which GPU to run on
    -c|--cores <int>          Number of CPU cores to request (-binding linear:<>)
END
)

echo_usage() {
    echo "$usage"
}

# defaults
task_name=noname
cuda=1  # empty, will use -l cuda=1 (without specifying a GPU)
cores=2 # number of cores, with 2 threads per core

# long args seperated by comma, short args not
# colon after arg indicates that an option is expected (kwarg)
short_args=hn:g:c:
long_args=help,name:,gpu:,cores
opts=$(getopt --options $short_args --long $long_args --name "$0" -- "$@")
eval set -- "$opts"

# parse arguments
while true; do
    case "$1" in
        -h | --help)
            echo_usage
            exit 0
            ;;
        -n | --name )
            task_name="$2"
            shift 2
            ;;
        -g | --gpu )
            gpu="$2"
            if [ "$gpu" != "RTX2080" ] && [ "$gpu" != "K40" ]; then
                echo_usage
                echo -e "\n$0: error: invalid argument $gpu for $1"
                exit 1
            fi
            cuda="\"1($gpu)\""
            shift 2
            ;;
        -c | --cores )
            cores="$2"
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
test_suite_args=$@

# Check that test_suite_args are valid arguments for run_test_suite.py
source /etc/profile.d/conda.sh
conda activate b2c
dry_run_output=$(python ../run_test_suite.py --dry-run $test_suite_args 2>&1)
if [ $? -ne 0 ]; then
    echo_usage
    echo -e "$0: error: invalid <run_test_suite.py arguments>\n"
    echo "$dry_run_output"
    exit 1
fi

# remote machine name
remote="cluster"
# where to store the logfile on the remote
remote_logdir="~/projects/brian2cuda/test-suite/results"
# get path to this git repo (brian2cuda)
local_b2c_dir=$(git rev-parse --show-toplevel)

run_name="$(date +%y-%m-%d_%T)_$task_name"
local_logfile="/tmp/$run_name.log"
remote_logfile="$remote_logdir/$run_name.log"
qsub_name=${run_name//_/__}
qsub_name=b2c-tests__${qsub_name//:/_}

# create tmp folder name for brian2cuda code (in $HOME)
remote_b2c_dir="~/projects/brian2cuda/test-suite/brian2cuda-synced-repos/$run_name"


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


### Create logdir on remote
ssh "$remote" "mkdir -p $remote_logdir"

### Copy local logfile with diff over to remote
rsync -avzz "$local_logfile" "$remote:$remote_logfile"


### Copy brian2cuda repo over to remote
rsync -avzz \
    --exclude '*.o' \
    --exclude 'tags' \
    --exclude 'examples' \
    --exclude 'dev'\
    --exclude '.eggs'\
    --exclude '.git' \
    "$local_b2c_dir"/ "$remote:$remote_b2c_dir"


# submit test suite script through qsub on cluster headnote
ssh $remote "source /opt/ge/default/common/settings.sh && \
    qsub \
    -cwd \
    -q cognition-all.q \
    -l cuda=$cuda \
    -N $qsub_name \
    -binding linear:$cores \
    $remote_b2c_dir/brian2cuda/tools/remote_run_scripts/on_headnode.sh \
    $remote_b2c_dir $remote_logfile $test_suite_args
    "
    # $1: b2c_dir, # $2: logfile (on_headnode.sh)
