#!/bin/bash
# This script copies the python file into the cluster and executes it

# show the manual for this script
usage=$(cat << END
usage: $0 <options> -- <argument for running a python script in cluster>

with <options>:
    -h|--help                 Show usage message
    -n|--name <string>        Name for python file
END
)

echo_usage() {
    echo "$usage"
}

# DEFAULTS
# remote machine name
remote="cluster"
make_target_dir_relative_to="$HOME"
python_task_name="noname"
python_script_name"noname"

# Load configuration file
script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$script_path/_load_remote_config.sh" ~/.brian2cuda-remote-dev.conf

# long args seperated by comma, short args not
# colon after arg indicates that an option is expected (kwarg)
short_args=hn:
long_args=help,name:
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
            python_script_name="$2"
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

remote_home=$(ssh $remote 'echo $HOME')
local_dir=$(pwd)
relative_remote_dir=$(realpath --relative-to=$make_target_dir_relative_to $local_dir)
remote_dir=$remote_home/$relative_remote_dir

### Copy current folder over to remote
rsync -avvzz \
    --rsync-path="mkdir -p $remote_dir && rsync"\
    "$local_dir"/"$python_script_name" "$remote:$remote_dir"

echo "Uploaded $python_script_name to remote directory $remote_dir"

path_conda_sh_remote="~/miniconda3/etc/profile.d/conda.sh"
# conda environment for brian2cuda on the remote
conda_env_remote="b2c"

ssh -p 1234 localhost "/bin/bash" << EOF
    source $path_conda_sh_remote
    conda activate $conda_env_remote
    python $remote_dir/$python_script_name
EOF