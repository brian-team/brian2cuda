#!/bin/bash

# source this file to set environmental variables for genn:
# `source ./init_genn.sh`

# If CUDA_PATH is not set, set it to /usr/local/cuda
if [[ -z "${CUDA_PATH}" ]]; then
    export CUDA_PATH=/usr/local/cuda
fi

export GENN_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/genn" && pwd )"
export PATH=$PATH:$GENN_PATH/lib/bin
