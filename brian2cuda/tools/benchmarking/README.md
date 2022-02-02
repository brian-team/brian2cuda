## Benchmarking

To run benchmarks on your local computer, use the `run_benchmark_suite.sh` script. Run
```
bash run_benchmark_suite.sh --help
```
to see the options.

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
