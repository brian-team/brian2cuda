#!/bin/bash

# source this file to set environmental variables for genn:
# `source ./init_genn.sh`

# If CUDA_PATH is not set, set it to /usr/local/cuda
if [[ -z "${CUDA_PATH}" ]]; then
    export CUDA_PATH=/usr/local/cuda
    echo "Setting CUDA_PATH=$CUDA_PATH"
fi

export GENN_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/genn" && pwd )"
echo "Setting GENN_PATH=$GENN_PATH"
export PATH=$GENN_PATH/lib/bin:$PATH
