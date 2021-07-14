The scripts here run the brian2cuda benchmark suite.

1. Run locally
  To run the benchmarks on your local GPU, use `run_benchmark_suite.sh`. Don't use
  the Python file `run_benchmark_suite.py` directly. The shell script will take
  care of logging git state and logfiles before calling `run_benchmark_suite.py`.
  This will use the brian2, brian2cuda, brian2genn and genn packages that are
  available in this local git repository (you need to initialize the
  submodules correctly for this to work). If you modify the code during benchmark run, it will effect your benchmarks.

2. Make copy, then run locally
  TODO: Create script `run_benchmark_suite_on_current_state.sh` for this.

3. Run on a remote cluster
  To run the benchmark suite on the Sprekelerlab cluster, use
  `../remote_run_scripts/run_benchmark_suite_on_cluster.sh` instead, it will take
  care of creating your repository on the remote.
