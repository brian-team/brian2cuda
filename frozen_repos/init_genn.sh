#!/bin/bash

# source this file to set environmental variables for genn:
# `source ./init_genn.sh`

GENN_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/genn" && pwd )"
echo "Updating PATH with GeNN/bin directory $GENN_PATH/bin"
export PATH="$GENN_PATH/bin:$PATH"
