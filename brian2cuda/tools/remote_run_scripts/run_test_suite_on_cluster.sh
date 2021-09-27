#!/bin/bash

usage=$(cat << END
usage: $0 <options> -- <run_test_suite.py arguments>

with <options>:
    -h|--help                   Show usage message
    -n|--name <string>          Name for test suite run (default: 'noname'). This
                                name will also be used for the pytest cache
                                directory. That means when running the test suite
                                multiple times with the same --name, pytest
                                options --last-failed or --failed-first can be
                                used.
    -s|--suffix <string>        Name suffix for the logfile
                                (<name>_<suffix>_<timestemp>)
    -g|--gpu <GTX1080|RTX2080>  Which GPU to run on
    -p|--parallel <n_tasks>     Submit all tests in parallel using '2*<n_cores>'
                                pytest-xdist workers, which each compile in
                                parallel with <n_tasks> qmake jobs.
    -H|--host                   Which compute node to run on (e.g. cognition13)
    -c|--cores <n_cores>        Number of CPU cores to request. If '<n_tasks> == 0'
                                (default), this will request 2*<n_cores> via
                                '-binding linear:<n_cores>'. If '<n_tasks> > 0',
                                this will request 2*<n_cores> threads via
                                '-pe cognition.pe 2*<n_cores>'.
    -r|--remote <head-node>     Remote machine url or name (if configured in
                                ~/.ssh/config)
    -l|--log-dir                Remote path to directory where logfiles will be stored.
    -S|--remote-conda-sh        Remote path to conda.sh
    -E|--remote-conda-env       Conda environment name with brian2cuda on remote
    -k|--keep-remote-repo       Don't delete remote repository after run
    -a|--after <job-id>         Hold this job until 'job-id' is finished
END
)

echo_usage() {
    echo "$usage"
}

# DEFAULTS
# remote machine name
remote="cluster"
# default task name
test_suite_task_name=noname
# -l cuda=1,gputype=$gputype
gputype="RTX2080"
# number of cores, with 2 threads per core
test_suite_cores=2
# path to conda.sh on remote
path_conda_sh_remote="~/anaconda3/etc/profile.d/conda.sh"
# conda environment for brian2cuda on the remote
conda_env_remote="b2c"
# where to store the logfile on the remote
test_suite_remote_dir="~/projects/brian2cuda/test-suite"
# keep remote (false by default)
keep_remote_repo=1
# By default, don't use parallel jobs (0)
parallel=0
# If given, hold this job until $hold_job_id is finished
hold_job_id=""
# $suffix unset by default
# $remote_host unset by default

# Load configuration file
script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$script_path/_load_remote_config.sh" ~/.brian2cuda-remote-dev.conf

# long args seperated by comma, short args not
# colon after arg indicates that an option is expected (kwarg)
short_args=hn:s:g:p:H:c:r:l:S:E:ka:
long_args=help,name:,suffix:,gpu:,parallel:,host:,cores:,remote:,log-dir:,remote-conda-sh:,remote-conda-env:,keep-remote-repo,after:
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
            test_suite_task_name="$2"
            shift 2
            ;;
        -s | --suffix )
            suffix="$2"
            shift 2
            ;;
        -g | --gpu )
            gputype="$2"
            if [ "$gputype" != "RTX2080" ] && [ "$gputype" != "GTX1080" ]; then
                echo_usage
                echo -e "\n$0: error: invalid argument $gputype for $1"
                exit 1
            fi
            shift 2
            ;;
        -p | --parallel )
            parallel="$2"
            shift 2
            ;;
        -H | --host )
            remote_host="$2"
            shift 2
            ;;
        -c | --cores )
            test_suite_cores="$2"
            shift 2
            ;;
        -r | --remote )
            remote="$2"
            shift 2
            ;;
        -l | --log-dir )
            test_suite_remote_dir="$2"
            shift 2
            ;;
        -S | --remote-conda-sh )
            path_conda_sh_remote="$2"
            shift 2
            ;;
        -E | --remote-conda-env )
            conda_env_remote="$2"
            shift 2
            ;;
        -k | --keep-remote-dir )
            keep_remote_repo=0
            shift 1
            ;;
        -a | --after )
            hold_job_id="-hold_jid $2"
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

pytest_cache_dir="$test_suite_remote_dir"/.pytest_caches
test_suite_args="--notify-slack --cache-dir $pytest_cache_dir/$test_suite_task_name"
if [ ! "$parallel" -eq 0 ]; then
    # parallel run -> the ./main binaries request the GPU in run_test_suite.py
    test_suite_args+=" --grid-engine \
        --test-parallel \
        --grid-engine-gpu $gputype \
        --jobs $parallel"
    # cognition06 is not a submit host
    grid_engine_ressources="h=!cognition06"
    # Submit the _on_headnode.sh script with 2*<n_cores> threads in a parallel
    # environment, these are the pytest-xdist workers
    parallel_ressources="-pe cognition.pe $((2*$test_suite_cores))"
    if [ "$parallel" -eq 1 ]; then
        # TODO: delete this if nvcc is available on all nodes without cuda=0
        # Not using qmake but make on the node itself, needs nvcc
        grid_engine_ressources+=",cuda=0"
        echo
    fi
else
    # not running in parallel -> submit entire test suite script to GPU node
    grid_engine_ressources="cuda=1,gputype=$gputype"
    #parallel_ressources="-binding linear:$test_suite_cores"
    parallel_ressources="-pe cognition.pe $((2*$test_suite_cores))"
fi
if [ -n "$remote_host" ]; then
    # add host ressource
    grid_engine_ressources+=",h=$remote_host"
fi
if [ -n "$grid_engine_ressources" ]; then
    grid_engine_ressources="-l $grid_engine_ressources"
fi


# all args after -- are passed to run_test_suite.py (collected in $@)
test_suite_args+=" $@"

test_suite_remote_logdir="$test_suite_remote_dir/results"

# Check that test_suite_args are valid arguments for run_test_suite.py
echo "Performing pytest dry run locally ..."
dry_run_cmd="python ../test_suite/run_test_suite.py --dry-run $test_suite_args 2>&1"
echo $dry_run_cmd
dry_run_output=$(eval $dry_run_cmd)
if [ $? -ne 0 ]; then
    echo_usage
    echo -e "$0: error: invalid <run_test_suite.py arguments>\n"
    echo "$dry_run_output"
    exit 1
else
    echo ... OK
    echo
fi

# get path to this git repo (brian2cuda)
local_b2c_dir=$(git rev-parse --show-toplevel)

if [ -n "$suffix" ]; then
    run_name="$test_suite_task_name"_"$suffix"_"$(date +%y-%m-%d_%H-%M-%S)"
else
    run_name="$test_suite_task_name"_"$(date +%y-%m-%d_%H-%M-%S)"
fi

local_logfile="/tmp/$run_name.log"
remote_logfile="$test_suite_remote_logdir/$run_name.log"
qsub_name=${run_name//_/__}
qsub_name=b2c-tests-${qsub_name//:/_}

# create tmp folder name for brian2cuda code (in $HOME)
remote_b2c_dir="$test_suite_remote_dir/brian2cuda-synced-repos/$run_name"
# folder name for qsub .o and .e logs
remote_ge_log_dir="$test_suite_remote_dir/ge-logs"


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


### Create logdirs on remote
ssh "$remote" \
    "mkdir -p \
        $test_suite_remote_logdir \
        $remote_b2c_dir \
        $remote_ge_log_dir \
        $pytest_cache_dir"

### Copy local logfile with diff over to remote
rsync -avzz "$local_logfile" "$remote:$remote_logfile"


### Copy brian2cuda repo over to remote
rsync -avzz \
    --exclude '*.o' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'tags' \
    --exclude 'examples' \
    --exclude 'dev'\
    --exclude '.eggs'\
    --exclude '.git' \
    --exclude 'worktrees' \
    --exclude 'cpp_standalone_runs' \
    --exclude 'cuda_standalone_runs' \
    --exclude 'genn_runs' \
    --exclude 'GeNNworkspace' \
    --exclude 'output' \
    --exclude 'results' \
    --exclude 'benchmark_results' \
    "$local_b2c_dir"/ "$remote:$remote_b2c_dir"

bash_script=brian2cuda/tools/test_suite/_run_test_suite.sh
# submit test suite script through qsub on cluster headnote
ssh $remote "source /opt/ge/default/common/settings.sh && \
    qsub \
        -wd $remote_ge_log_dir \
        -q cognition-all.q \
        $grid_engine_ressources \
        -N $qsub_name \
        $parallel_ressources \
        $hold_job_id \
        $remote_b2c_dir/brian2cuda/tools/remote_run_scripts/_on_headnode.sh \
            $bash_script \
            $remote_b2c_dir \
            $remote_logfile \
            $path_conda_sh_remote \
            $conda_env_remote \
            $keep_remote_repo \
            $parallel \
            $test_suite_args"
            # $1: bash_script $2: b2c_dir, $3: logfile, $4: remote conda.sh
            # $5: remote conda env $6: bool for keeping tmp remote b2c dir
            # $7: bool for running in parallel
            # $@ (the rest): arguments passed to benchmark script
            # (_on_headnode.sh)


if [ $keep_remote_repo -eq 0 ]; then
    echo -e "\nThe copied brian2cuda directory on the remote will not be deleted" \
            "due to the -k|--keep-remote-dir option. Make sure you delete it once" \
            "you don't need it anymore!"
fi
