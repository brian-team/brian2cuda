#!/bin/bash
# arguments:
# $1: additional name for log_file
# $2: commit hash or branch name to check out after cloning
# $3: number of cores used for parallel compilation (make -j $3)

# the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# the temp directory used, within $DIR
WORK_DIR=`mktemp -d -p "$DIR"`
# the directory of the brian2cuda repository
GIT_DIR="$(dirname "$(dirname "`pwd`")")"
# the file for loggint the output
TIME=`date +"%y.%m.%d_%H:%M"`
LOG_FILE="$DIR/logfile_tests_$1_$TIME.txt"

# deletes the temp directory
function cleanup {
    rm -rf "$WORK_DIR"
    echo "Deleted temp working directory $WORK_DIR"
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

if [ -z "$3" ]
then
    J=12
else
    J=$3
fi

source activate dev_b2c

git clone "$GIT_DIR" "$WORK_DIR"
cd "$WORK_DIR"
git submodule update --init frozen_repos/brian2
if [ -z "$2" ]
then
    echo "using tip of master"
else
    git checkout $2
    echo "checked out $2"
fi
git rev-parse --abbrev-ref HEAD 2>&1 | tee -a "LOG_FILE"
git rev-parse HEAD 2>&1 | tee -a "LOG_FILE"
cd "brian2cuda/tools"
PYTHONPATH="../..:../../frozen_repos/brian2:$PYTHONPATH" python run_test_suite.py --fail-not-implemented -j"$J" 2>&1 | tee -a "$LOG_FILE"

