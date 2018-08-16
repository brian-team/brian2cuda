#!/bin/bash
# arguments:
# $1: additional name for log_file
# $2: commit hash or branch name to check out after cloning
# $3: float32 or float64 or both (default)
# $4: number of cores used for parallel compilation (make -j $4)

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

if [ -z "$4" ]
then
    J=12
else
    J=$4
fi

if [ -z "$3" ]
then
    FLOAT_DTYPE="float64 float32"
elif [ "$3" = "float32" ]
then
    FLOAT_DTYPE="float32"
elif [ "$3" = "float64" ]
then
    FLOAT_DTYPE="float64"
elif [ "$3" = "both" ]
then
    FLOAT_DTYPE="float64 float32"
else
    echo "ERROR, the third argument needs to be from {'float32'|'float64'|'both'}"
    exit 1
fi

source activate dev_b2c

git clone "$GIT_DIR" "$WORK_DIR"
cd "$WORK_DIR"
if [ -z "$2" ]
then
    echo "using tip of master" | tee -a "$LOG_FILE"
else
    git checkout $2 | tee -a "$LOG_FILE"
    #echo "checked out $2" | tee -a "$LOG_FILE"
fi
git rev-parse --abbrev-ref HEAD 2>&1 | tee -a "$LOG_FILE"
git rev-parse HEAD 2>&1 | tee -a "$LOG_FILE"
git submodule update --init frozen_repos/brian2 | tee -a "$LOG_FILE"
cd frozen_repos/brian2
#git checkout "float32_support" | tee -a "$LOG_FILE"
#echo "Checked out brian2 version:" | tee -a "$LOG_FILE"
git rev-parse --abbrev-ref HEAD 2>&1 | tee -a "$LOG_FILE"
git rev-parse HEAD 2>&1 | tee -a "$LOG_FILE"
git apply ../brian2.diff | tee -a "$LOG_FILE"
echo "Applied brian2.diff" | tee -a "$LOG_FILE"
cd ../../brian2cuda/tools
PYTHONPATH="../..:../../frozen_repos/brian2:$PYTHONPATH" python run_test_suite.py --float-dtype $FLOAT_DTYPE --fail-not-implemented -j"$J" 2>&1 | tee -a "$LOG_FILE"

