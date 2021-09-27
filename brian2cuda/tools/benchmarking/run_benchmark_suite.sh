#!/bin/bash
usage=$(cat << END
usage: $0 <options> -- <run_test_suite.py arguments>

with <options>:
    -h|--help                 Show usage message
    -n|--name <string>        Name for test suite run (default: 'noname')
    -l|--log-dir              Directory for test suite logs (default: 'test_suite_logs')
    -t|--testing              Make a test run without nvprof / slack notifications.
END
)

echo_usage() {
    echo "$usage"
}

echo "Running _run_benchmark_suite.sh on $HOSTNAME"

# DEFAULTS
benchmark_suite_task_name=noname
benchmark_result_dir="results"

# long args seperated by comma, short args not
# colon after arg indicates that an option is expected (kwarg)
short_args=hn:l:t
long_args=help,name:,log-dir:,testing
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
            benchmark_suite_task_name="$2"
            shift 2
            ;;
        -l | --log-dir )
            benchmark_result_dir="$2"
            shift 2
            ;;
        -t | --testing )
            testing=0
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

# all args after --
test_suite_args=$@

if [ -n "$testing" ]; then
    test_suite_args+=" --no-nvprof --no-slack"
    benchmark_suite_task_name+="-testing"
fi

# add timestemp to name
benchmark_suite_task_name="$benchmark_suite_task_name"_"$(date +%y-%m-%d_%H-%M-%S)"
logfile_name="full.log"
results_dir="$benchmark_result_dir/$benchmark_suite_task_name"

# Set GeNN environment variales such that they always use the GeNN from the
# frozen_repos/genn submodule
source ../../../frozen_repos/init_genn.sh

bash _run_benchmark_suite.sh "$logfile_name" "$results_dir" -d "$results_dir" "$test_suite_args"
