#!/usr/bin/env bash

# DEFAULTS that can be changed via ~/.brian2cuda-remote-dev.conf
# (there is a template in this directory)

# remote machine name
remote="cluster"
# path to conda.sh on remote
path_conda_sh_remote="~/anaconda3/etc/profile.d/conda.sh"
# where to store the logfile on the remote
benchmark_suite_remote_dir="~/projects/brian2cuda/benchmark-suite"
# what grip port to use on the remote
remote_grip_port=6420

# Load configuration file
script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$script_path/_load_remote_config.sh" ~/.brian2cuda-remote-dev.conf

control_socket=/tmp/socket-grip-%C

# Kill `grip` on head node when exiting this script
function cleanup {
    echo
    echo Making sure ssh tunnel is closed...
    ssh -S $control_socket -O exit $remote
    echo Killing any running grip instances on remote head node...
    ssh $remote "killall grip"
}

# Register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

{
ssh $remote "/bin/bash" << EOF
    source $path_conda_sh_remote
    cd $benchmark_suite_remote_dir/results
    if conda activate grip; then
        echo Killing any running instance of grip on remote...
        killall grip
        echo Starting grip on remote...
        grip $remote_grip_port
    else
        echo No conda environment with name grip found. Please create it and \
             install grip in that environment.
    fi
EOF
} &

sleep 2

echo Starting ssh tunnel to connect to remote grip server...
echo "  grip server is running on remote port $remote_grip_port and tunneled to local port 6420"
echo "  You can access it via your browser at http://localhost:6420"
echo
ssh -Y -N -M -S $control_socket -L localhost:6420:localhost:$remote_grip_port $remote
