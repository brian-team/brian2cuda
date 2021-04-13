#!/bin/sh
# Load remote paramters stored in $1
config_file="${BASH_SOURCE%/*}/$1"

if test -f "$config_file"; then
    source $config_file
    # Get all lines that are not blanc or start with comment (#)
    loaded=$(grep -v -E '^\s*$|^#' "$config_file")
    # These are the sourced variables, print them
    if [[ -n $loaded ]]; then
        echo "The following remote parameters were set in $config_file"
        echo "$loaded"
        echo
    fi
fi
