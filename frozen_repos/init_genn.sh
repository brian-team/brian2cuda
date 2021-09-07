#!/bin/bash

# source this file to set environmental variables for genn:
# `source ./init_genn.sh`

# If CUDA_PATH is not set, set it to /usr/local/cuda
#if [[ -z "${CUDA_PATH}" ]]; then
#    export CUDA_PATH=/usr/local/cuda
#    echo "Setting CUDA_PATH=$CUDA_PATH"
#fi

#export CXX=/usr/bin/g++-7
##export CC=/usr/bin/gcc-7
#
#GENN_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/genn" && pwd )"
#echo "Updating PATH with GeNN/bin directory $GENN_PATH/bin"
#export PATH="$GENN_PATH/bin:$PATH"
#
#
#
# TODO: Instead of having to set CPLUS_INCLUDE_PATH, the conda
# environment should set -isystem (which it does, just not for the c++ headers)
export CXX_VERSION="$("$CXX" -dumpversion)"
export CXX_DIR="$(dirname "$CXX")"
export CXX_INCLUDE=$CXX_DIR/../x86_64-conda-linux-gnu/include/c++/$CXX_VERSION
#export CPATH=$CXX_INCLUDE:$CPATH
export CPLUS_INCLUDE_PATH=$CXX_INCLUDE:$CPLUS_INCLUDE_PATH
echo "Adding $CXX_INCLUDE"
GENN_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/genn" && pwd )"
echo "Updating PATH with GeNN/bin directory $GENN_PATH/bin"
export PATH="$GENN_PATH/bin:$PATH"
