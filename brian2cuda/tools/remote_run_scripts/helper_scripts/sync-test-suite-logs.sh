#!/bin/bash
# Sync all test suite logs from cluster to here

rsync -vt cluster:~/projects/brian2cuda/test-suite/results/* .
