# Don't use this script directly, use `run_test_suite.sh` instead!
# This script assumes it is called from where it is stored in in the brian2cuda
# repository and it uses the brian2cuda and brian2 of that repository.

# $1 task name
# $2 logdir

task_name=$1
logdir=$2
shift 2
test_suite_args=$@

mkdir -p "$logdir"

logfile="$logdir/$task_name"

start_time=`date +%s`

test_suite_cmd="python run_test_suite.py $test_suite_args"
echo "$test_suite_cmd" | tee -a "$logfile"
PYTHONPATH="../../..:../../../frozen_repos/brian2:$PYTHONPATH" \
    $test_suite_cmd 2>&1 | tee -a "$logfile"

runtime=$(( $(date +%s) - $start_time ))
min=$(( $runtime / 60 ))
remaining_sec=$(( $runtime - $min * 60 ))
hour=$(( $min / 60 ))
remaining_min=$(( $min - $hour * 60 ))
echo -e "Test suite took $hour h $remaining_min m $remaining_sec s." | tee -a "$logfile"
