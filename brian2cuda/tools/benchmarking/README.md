## Benchmarking

To run benchmarks, use the `run_benchmark_suite.sh` script by executing it from this directory. This script will make sure that the `brian2cuda` in this repository and the `brian2` in the `brian2cuda/frozen_repos/brian2` repository are used by modifying `PYTHONPATH` accordingly. If you haven't checked out the submodule in `brian2cuda/frozen_repos/brian2` yet, please do so via this command:
```
cd /path/to/brian2cuda/frozen_repos
bash update_brian2.sh
```
This will check out the correct brian2 version and apply the patch stored in `brian2cuda/frozen_repos/brian2.diff`.


Run
```
bash run_benchmark_suite.sh --help
```
to see the benchark options.

This script will execute the `run_benchmark_suite.py` file. In this file you
can modify the benchmarks and the configurations for which the benchmarks
should run:
- Modify the `configurations` list to set the configurations, which are
  defined in `brian2cuda/tests/features/cuda_configuration.py`. All
  configuration classes have the `BenchmarkConfiguration` class as parent.
  This class takes care of all time measurements during the benchmarks. The 
- Modify the `speed_tests` list to set what benchmarks to run and for which
  network sizes. The benchmark classes are defined in
  `brian2cuda/tests/features/speed.py`.

If you don't want to use the brian2cuda and brian2 from this repository but some other versions you have installed, you can also execute the Python file `run_benchmark_suite.py` file directly (see the `--help` options for the parameters).
  
### Examples

To run all benchmarks for all configurations defined in `run_benchmark_suite.py`, execute:
```
bash run_benchmark_suite.sh --name my-benchmark-run
```
This will store the benchmark results in
`results/my-benchmark-run_<date_time>`.
To run the same configuration but with profiling of different parts of the
simulation (which has an effect on overall performance), run:
```
bash run_benchmark_suite.sh --name my-benchmark-run-profiled -- --profile
```
This will store the results in
`results/my-benchmark-run-profiled_<date_time>`.


### Running benchmarks on a remote computer

If you have a GPU on a remote computer but the repository checked out on your
local computer, there are multiple helper scripts to help you execute
benchmarks or the test suite on the remote machine. They are located at
`../remote_run_scripts`, check out the README file there.
