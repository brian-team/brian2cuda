#!/bin/bash
# Execute make on $remote
# All paramters ($@) are passed to the remote make command

# DEFAULTS
# remote machine name
remote="cluster"
make_target_dir_relative_to="$HOME"

# Load configuration file
script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$script_path/_load_remote_config.sh" ~/.brian2cuda-remote-dev.conf

relative_remote_dir=$(realpath --relative-to=$make_target_dir_relative_to $(pwd))

# Run
ssh -p 1234 -i ~/.ssh/id_internal localhost "source /cognition/home/local/.init_cuda.sh \
    && cd \$HOME/$relative_remote_dir \
    && make -j $@"
