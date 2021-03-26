#!/bin/bash
# $1: task name (optional, default: noname)
# $2: GPU name (optional, default: RTX2080)

task_name=${1:-noname}
# TODO: default should be none and choose any GPU
gpu=${2:-RTX2080}

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
ssh "$remote" "mkdir -p $logdir"

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
    -l cuda=\"1($gpu)\" \
    -N $qsub_name \
    -binding linear:2 \
    $remote_b2c_dir/brian2cuda/tools/remote_run_scripts/on_headnode.sh \
    $remote_b2c_dir $remote_logfile
    "
    # $1: b2c_dir, # $2: logfile (on_headnode.sh)
