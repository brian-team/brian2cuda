## Benchmarking

You can either run multiple benchmarks at once via the `run_benchmark_suite.sh` script or
run individual simulations by executing the scripts in the `brian2cuda/examples` (see
below for instructions).

### Running multiple benchmarks at once

The `run_benchmark_suite.sh` script will make sure that the `brian2cuda` in this
repository and the `brian2` and `brian2genn` in the `brian2cuda/frozen_repos` folder are
used by modifying `PYTHONPATH` accordingly.

If you haven't initialized and checked out the submodules in `brian2cuda/frozen_repos`
yet, please do so via this command:
```bash
cd /path/to/brian2cuda/frozen_repos
git submodule update --init
bash update_brian2.sh
# If you want run also Brian2GeNN benchmarks
bash update_brian2genn.sh
source init_genn.sh
```
This will check out the correct Brian2 / Brian2GeNN version and apply the
patches stored in `brian2cuda/frozen_repos/brian2.diff` and
`brian2cuda/frozen_repos/brian2genn.diff`. And the `init_genn.sh` script sets the
environment variable needed for GeNN to work (needs to be set again when starting a new
environment).

In addition to Brian's dependencies, running the benchmark suite requires the `matplotlib` and `pandas` packages.


Run
```bash
bash run_benchmark_suite.sh --help
```
to see the benchmark options.

This script will execute the `run_benchmark_suite.py` file. In this file you
can modify the benchmarks and the configurations for which the benchmarks
should run:
- Modify the `configurations` list to set the configurations, which are
  defined in `brian2cuda/tests/features/cuda_configuration.py`. All
  configuration classes have the `BenchmarkConfiguration` class as parent.
  This class takes care of all time measurements during the benchmarks.
- Modify the `speed_tests` list to set what benchmarks to run and for which
  network sizes. The benchmark classes are defined in
  `brian2cuda/tests/features/speed.py`.

If you don't want to use the Brian2CUDA and Brian2 from this repository but some other versions you have installed, you can also execute the Python file `run_benchmark_suite.py` file directly (see the `--help` options for the parameters).

### Examples

To run all benchmarks for all configurations defined in `run_benchmark_suite.py`, execute:
```
bash run_benchmark_suite.sh --name my-benchmark-run
```
This will store the benchmark results in
`results/my-benchmark-run_<date_time>`.
To run the same configuration but with profiling of different parts of the
simulation (which has an effect on overall performance), run:
```bash
bash run_benchmark_suite.sh --name my-benchmark-run-profiled -- --profile
```
This will store the results in
`results/my-benchmark-run-profiled_<date_time>`.

### Running individual examples
To run single instances of an example model with a specific size and configuration, you
can use the Python files in the `brian2cuda/examples` directory of this repository.
If you haven't done so yet, please first follow the installation instructions on the main
repositories README page (the example scripts don't automatically set the correct
PYTHONPATH for you).

Each of the scripts has a number of optional command line arguments to specify the
configuration. To get a list of all supported arguments, run the script with `--help`.
Some of the scripts can be used to run different variants of a model (e.g. with and
without synaptic connections). See below for a list of all benchmarked models and their
corresponding script calls:

<dl>
  <dt>HH benchmark I: Hodgkin-Huxley type neurons without synapses</dt>
  <dd>

  `python cobahh.py --scenario uncoupled`
  </dd>
  <dt>HH benchmark II: Hodgkin-Huxley type neurons with static synapses</dt>
  <dd>

  `python cobahh.py --scenario pseudocoupled-1000`
  </dd>
  <dt>LIF benchmark I: Noisy integrate-and-fire neurons with homogeneous synaptic delays</dt>
  <dd>

  `python brunelhakim.py --no-heterog-delays`
  </dd>
  <dt>LIF benchmark I: Noisy integrate-and-fire neurons with heterogeneous synaptic delays</dt>
  <dd>

  `python brunelhakim.py --heterog-delays`
  </dd>
  <dt>STDP benchmark I: Dynamic synapses with spike-timing dependent plasticity and homogeneous synaptic delays
  </dt>
  <dd>

  `python stdp.py --delays homogeneous`
  </dd>
  <dt>STDP benchmark II: Dynamic synapses with spike-timing dependent plasticity and heterogeneous synaptic delays
  </dt>
  <dd>

  `python stdp.py --delays heterogeneous`
  </dd>
  <dt>
  Mushroom body benchmark: Complex model with multiple neuronal populations, spike-timing dependent plasticity and noise
  </dt>
  <dd>

  `python mushroombody.py`
  </dd>
</dl>


## Running benchmarks on a remote computer

If you have a GPU on a remote computer but the repository checked out on your
local computer, there are multiple helper scripts to help you execute
benchmarks or the test suite on the remote machine. They are located at
`../remote_run_scripts`, check out the README file there.
