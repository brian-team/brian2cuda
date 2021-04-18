#!/bin/bash
# Execute `main` binary on $remote
# All paramters ($@) are passed to the remote `main` command

# DEFAULTS
# remote machine name
remote="cluster"

make_target_dir_relative_to="$HOME"

# Load configuration file
source "${BASH_SOURCE%/*}/_load_remote_config.sh" ~/.brian2cuda-remote-dev.conf

relative_remote_dir=$(realpath --relative-to=$make_target_dir_relative_to $(pwd))

ssh -p 1234 -i ~/.ssh/id_internal localhost "cd "'$HOME'"/$relative_remote_dir && ./main $@"
