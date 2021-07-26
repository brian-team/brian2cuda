The scripts here run the brian2 and brian2cuda test suite.

1. Run locally
  To run the test suite on your local GPU, use `run_test_suite.sh`. Don't use
  the Python file `run_test_suite.py` directly. The shell script will take
  care of logging git state and logfiles before calling `run_test_suite.py`.
  This will use the brian2 and brian2cuda packages that are available in your
  local Python environment. If you installed in developer mode, you can't
  modify the source code during test suite run, otherwise it will change your
  test results.

  If you want to make a copy of the current repository before running the test
  suite, such that you can continue development, use
  `run_test_suite_on_current_state.sh` instead.
  TODO: This script is old and is not using the same result structure as the other
  sctipts. It should be updated to use `_run_test_suite.sh`.

2. Run on a remote cluster
  To run the test suite on the Sprekelerlab cluster, use
  `../remote_run_scripts/run_test_suite_on_cluster.sh` instead, it will take
  care of copying your local repository to the remote.
