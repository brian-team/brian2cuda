# Run benchmark suite with current repository (sets PYTHONPATH).
# $1 task name (optional, default: noname)
# $2 logdir (optional, default: test_suit_logs)

run_name=${1:-noname}       # default: noname
benchmarks_result_dir=${2:-results}  # default: results

# add timestemp to name
run_name="$run_name\_$(date +%y-%m-%d_%T)"
logfile_name="full.log"
results_dir="$benchmarks_result_dir/$run_name"

# Set GeNN environment variales such that they always use the GeNN from the
# frozen_repos/genn submodule
source ../../../frozen_repos/init_genn.sh

bash _run_benchmark_suite.sh "$logfile_name" "$results_dir" -d "$results_dir"
