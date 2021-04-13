#!/bin/bash
# Remove all files passed as arguments both, locally and on the cluster.
# Accepts multiple arguments and wildcards (just use tab completion locally)

for file in "$@"
do
    ssh cluster "rm -v ~/projects/brian2cuda/test-suite/results/$file"
    rm -v $file
done
