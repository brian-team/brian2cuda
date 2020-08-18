#!/bin/bash
# $1: task name (optional, default: noname)

task_name=${1:-noname}

# remote machine name
remote="cluster"
# where to store the logfile on the remote
logdir="~/projects/brian2cuda/test-suite/results"
# get path to this git repo (brian2cuda)
local_folder=$(git rev-parse --show-toplevel)

run_name="$(date +%y-%m-%d_%T)_$task_name"
logfile="$logdir/$run_name.log"
qsub_name=${run_name//_/__}
qsub_name=b2c-tests__${qsub_name//:/_}

# create tmp folder name for brian2cuda code (in $HOME)
remote_folder="~/projects/brian2cuda/test-suite/brian2cuda-synced-repos/$run_name"
#remote_folder="/cognition/home/denis/projects/brian2cuda/test-suite/$run_name"

# if excluding tracked files, the git diff on the remote is messed up
# TODO: create the git diff locally instead ...
#rsync -avzz --exclude 'dev' --exclude 'examples' \

rsync -avzz --exclude '*.o' \
    "$local_folder"/ "$remote:$remote_folder"

# This works for the git diff but excludes any new and still untracked files...
## copy brian2cuda directory to remote server
#tracked=$local_folder/.tracked_files
#cd $local_folder
#git ls-tree -r HEAD --name-only > $tracked
## add git directory such that git commands work remotely
#echo ".git" >> $tracked
##num_files=$(wc -l < $tracked)
#rsync -avzzr --files-from="$tracked" \
#    "$local_folder"/ "$remote:$remote_folder"
## --info=progress2
#cd - > /dev/null


# submit test suite script through qsub on cluster headnote
ssh $remote "source /opt/ge/default/common/settings.sh && \
    qsub \
    -cwd \
    -q cognition-all.q \
    -l cuda=1 \
    -l h=cognition13 \
    -N $qsub_name \
    -pe cognition.pe 4 \
    $remote_folder/brian2cuda/tools/remote_run_scripts/on_headnode.sh \
    $remote_folder $logfile
    "
    # $1: b2c_dir, # $2: logfile (on_headnode.sh)
