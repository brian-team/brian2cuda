# Collection of helpful tools for brian2CUDA development

## Test suite scripts
### Run locally
To run the test suite locally, run
```bash
cd test_suite
bash run_test_suite.sh <task-name> <log-dir>
```
The script needs to be run from the `./test_suite` directory.
This will run the full test suite and store a log file of the format
`<timestemp>__<task-name>` in the `log_dir` folder. Arguments are optional,
defaults are `task-name=noname` and `log_dir=test_suite_logs`.

This will use the brian2CUDA and brian2 installation from this repository (by
setting `PYTHONPATH`). If you are modifying files in this repository while the
test suite is running, this will affect the test suite run. If you want to
continue development and run the test suite on a clone of the current state of
the repository, run
```bash
bash run_test_suite_on_current_state.sh
```
The arguments are different, check the bash script for information. This
script will use the latest commit to run the tests on. Uncommited changes will
not be in the cloned repository.

See `test_suite/README.md` for more details.

TODO: Make this script and `run_test_suite.sh` less redundant and using the
same parameters.

### Run test suite through qsub on the cluster
```bash
cd remote_run_scripts
bash run_test_suite_on_cluster.sh
```
This is the newest and cleanest implementation. It runs the test suite on the
cluster using the queue system. And it uses `rsync` to copy the repo over,
therefore uncommited changes are also tested. The results end up in
`/cognition/home/denis/projects/brian2cuda/test-suite`.

See `remote_run_scripts/README.md` for more details.

TODO: Update README with changes made when Sudeshna joined ->
remote_run_scripts with config file etc.

## Benchmark scripts
See `benchmarking/README.md` for more details.
### Run locally
To run the benchmarks locally, run
```bash
cd benchmarking
# Modify python file `run_benchmark_suite.py` to set the benchmark configuration
# Then run this bash script
bash run_benchmark_suite.sh
```

### Run benchmark suite through qsub on cluster
```bash
cd remote_run_scripts
bash run_benchmark_suite_on_cluster.sh
```
See `remote_run_scripts/README.md` for more details.

