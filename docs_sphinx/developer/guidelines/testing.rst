Testing
=======

Brian2CUDA uses the same testing framework as Brian2: `pytest`__, invoked
through Brian's ``test()`` function for standalone devices and through ``pytest``
directly for CUDA-specific tests.

__ https://docs.pytest.org/

Running the test suite
----------------------

The main correctness check is Brian2's test suite run against the
``cuda_standalone`` device. This executes the tests in ``brian2/tests`` and
``brian2cuda/tests``. For CUDA standalone, the test runner also sweeps a set of
preference combinations (atomic operations, synapse bundle mode, delay handling,
and related options).

Use Brian's test function to run only the standalone tests::

    import brian2
    brian2.test([], test_standalone='cuda_standalone')

For a full run with logging of git state, use the wrapper script from the
repository::

    cd brian2cuda/tools/test_suite
    bash run_test_suite.sh --name my-run

Logs are written to ``test_suite_logs/``. Pass arguments after ``--`` to
``run_test_suite.py``. For example, ``-k test_functions`` selects a subset and
``-d`` enables debug output from the standalone build.

The script uses the ``brian2`` and ``brian2cuda`` packages from the current
Python environment. If the package is installed in editable mode, changes to
source files during a run will affect the results. To test a fixed snapshot
while continuing development, use ``run_test_suite_on_current_state.sh``.

To run only the brian2cuda pytest directory::

    pip install -e ".[test]"
    pytest brian2cuda/tests

CI uses a lighter smoke test that checks code generation and compilation
without a GPU::

    python brian2cuda/test_platform.py --platform linux

Writing tests
-------------

Brian2CUDA tests live in ``brian2cuda/tests/``. They follow the same conventions
as Brian2: test functions are named ``test_...``, use ``numpy.testing`` helpers
for array comparisons, and use ``pytest.raises`` for expected errors.

For tests that should run on ``cuda_standalone``, use the ``cuda_standalone`` and
``standalone_only`` markers in the same way as Brian2's ``cpp_standalone`` tests.
See `Brian2's testing documentation
<https://brian2.readthedocs.io/en/stable/developer/guidelines/testing.html>`_
for the full table of pytest markers and when to call ``device.build()``.

CUDA-specific tests
~~~~~~~~~~~~~~~~~~~

The tests under ``brian2cuda/tests/`` cover behaviour that the generic Brian2
suite does not exercise in detail:

``test_cuda_standalone.py``
    Device lifecycle: multiple builds, ``store``/``restore``, profiling flags,
    array caching, debug builds, and directory cleanup.

``test_synaptic_propagations.py``
    Spike queues, circular event buffers, heterogeneous delays, and synaptic
    effect modes (including bundle and atomic preferences).

``test_random_number_generation.py``
    cuRAND integration: ``rand``, ``randn``, Poisson and binomial draws, fixed
    and random seeds, and RNG use in synapse dynamics.

``test_cuda_generator.py``
    Code generation: default function implementations, compiler arguments for
    user-defined functions, and atomic parallelisation choices.

``test_cpp_cuda_consistency.py``
    Numerical agreement between ``cpp_standalone`` and ``cuda_standalone`` for
    the STDP example and selected monitor configurations.

``test_stateupdaters.py``, ``test_neurongroup.py``, ``test_monitor.py``
    State updaters with noise, host copy semantics, and monitor threading.

``test_network_multiple_runs.py``
    Multiple ``run()`` calls with scalar and heterogeneous synaptic delays.

``test_functions.py``, ``test_spikegenerator.py``, ``test_stringtools.py``,
``test_gpu_detection.py``, ``test_profiling.py``
    User-defined functions, spike generator edge cases, string/codegen helpers,
    GPU detection preferences, and profiling API.

Feature tests
-------------

Brian2's feature-test framework runs small, focused scenarios (monitors, input
groups, synapse types) across device configurations. The helper script
``brian2cuda/tools/feature_tests/run_feature_tests.py`` runs these against
``CUDAStandaloneConfiguration``. Edit the script to change which configurations
or feature tests are included.

Benchmarks
----------

Performance work uses two entry points: standalone example scripts in
``examples/``, and benchmark classes in ``brian2cuda/tests/features/speed.py``
driven by ``brian2cuda/tools/benchmarking/run_benchmark_suite.sh``.

Example scripts
~~~~~~~~~~~~~~~

Each script in ``examples/`` is a self-contained model with command-line options
for device, network size, profiling, and CUDA preferences (via ``examples/utils.py``).
Run with ``--help`` to see available arguments::

    cd examples
    python mushroombody.py --help
    python stdp.py --delays heterogeneous

Benchmark suite
~~~~~~~~~~~~~~~

``run_benchmark_suite.sh`` sets ``PYTHONPATH`` so the repository's ``brian2cuda``
and the pinned ``brian2`` (and optionally ``brian2genn``) under
``frozen_repos/`` are used. Initialise submodules there before a full comparison
run. See ``brian2cuda/tools/benchmarking/README.md``.

Edit ``run_benchmark_suite.py`` to choose:

* ``configurations`` — device and preference sets from
  ``brian2cuda/tests/features/cuda_configuration.py``
* ``speed_tests`` — benchmark classes and network sizes from ``speed.py``

Then run::

    cd brian2cuda/tools/benchmarking
    bash run_benchmark_suite.sh --name my-benchmark-run

Results are stored under ``results/my-benchmark-run_<timestamp>/``. Add
``-- --profile`` to record per-phase timings inside the generated C++ code.

Common benchmarks
~~~~~~~~~~~~~~~~~

The models below are used most often for regression and publication figures.
Each has a matching class in ``speed.py`` and usually a script in
``examples/``.

COBAHH (``cobahh.py``)
    Conductance-based Hodgkin–Huxley neurons. Variants range from uncoupled
    neurons (pure integration cost, no synapses) through fully coupled networks
    to *pseudocoupled* setups with fixed synapse counts per neuron and near-zero
    weights. Pseudocoupled models isolate synapse storage and update overhead
    without changing dynamics.

Brunel–Hakim (``brunelhakim.py``)
    Noisy integrate-and-fire network with external drive. Homogeneous delays
    (all synapses at 2 ms) versus heterogeneous delays (uniform or narrow
    distributions around the same mean). Heterogeneous delays stress the spike
    queue and delay buffer.

STDP (``stdp.py``)
    Spike-timing-dependent plasticity after Song, Miller and Abbott. Poisson
    input neurons connect to a postsynaptic population. Synaptic weights evolve
    on pre- and post-spike events. The benchmark scales with the number of
    synapses. Variants add homogeneous or heterogeneous delays, random
    connectivity, and runs with postsynaptic effects disabled.

Mushroom body (``mushroombody.py``)
    Multi-population model adapted from the Brian2GeNN benchmarks: antennal lobe,
    mushroom body Kenyon cells, and lateral horn with Traub–Miles neurons, STDP,
    and structured input patterns. Unlike the single-population STDP benchmark,
    it mixes several neuron groups, connection types, and plasticity rules in one
    network. It is a realistic integration test for codegen and runtime.

Other speed-test classes in ``speed.py`` include Brian2's standard dense/sparse
synapse benchmarks, CUBA, and state-monitor read patterns (coalesced versus
uncoalesced). See the class docstrings and ``n_range`` fields for the network
sizes used on recent hardware.
