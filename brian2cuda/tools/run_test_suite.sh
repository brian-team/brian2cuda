# Run test suite with current repository (sets PYTHONPATH).
# $1 task name (optional, default: noname)
# $2 logdir (optional, default: test_suit_logs)

task_name=${1:-noname}       # default: noname
logdir=${2:-test_suite_logs}  # default: test_suit_logs

# add timestemp to name
task_name="$(date +%y-%m-%d_%T)"__"$task_name".log

bash _run_test_suite.sh "$task_name" "$logdir"
