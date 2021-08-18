#!/bin/bash
# Run test suite with current repository (sets PYTHONPATH).

usage=$(cat << END
usage: $0 <options> -- <run_test_suite.py arguments>

with <options>:
    -h|--help                 Show usage message
    -n|--name <string>        Name for test suite run (default: 'noname')
    -l|--log-dir              Directory for test suite logs (default: 'test_suite_logs')
END
)

echo_usage() {
    echo "$usage"
}

# DEFAULTS
task_name=noname
logdir=test_suite_logs

# long args seperated by comma, short args not
# colon after arg indicates that an option is expected (kwarg)
short_args=hn:l:
long_args=help,name:,log-dir:
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
            task_name="$2"
            shift 2
            ;;
        -l | --log-dir )
            logdir="$2"
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

# add timestemp to name
task_name="$task_name"_"$(date +%y-%m-%d_%H-%M-%S)".log

bash _run_test_suite.sh "$task_name" "$logdir" "$test_suite_args"
