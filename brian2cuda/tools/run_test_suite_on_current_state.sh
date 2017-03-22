#!/bin/bash
# arguments:
# $1: additional name for log_file
# $2: number of cores used for parallel compilation (make -j $2)

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

if [ -z "$2" ]
then
	J=12
else
	J=$2
fi
	

source activate b2c_test

git clone "$GIT_DIR" "$WORK_DIR"
cd "$WORK_DIR"
git submodule update --init frozen_repos/brian2
git rev-parse --abbrev-ref HEAD 2>&1 | tee -a "LOG_FILE"
git rev-parse HEAD 2>&1 | tee -a "LOG_FILE"
cd "brian2cuda/tools"
PYTHONPATH="$PYTHONPATH:../..:../../frozen_repos/brian2" python run_test_suite.py --fail-not-implemented -j"$J" 2>&1 | tee -a "$LOG_FILE"

