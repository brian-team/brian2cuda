Generated project layout
========================

.. contents::
   :local:
   :depth: 1

Brian2CUDA reuses Brian2's code generation pipeline and the C++ standalone
project layout. The main visible differences are ``.cu`` sources instead of
``.cpp``, an extra ``rand.*`` pair for cuRAND, the CUDA ``brianlib`` copy, and
an ``nvcc`` makefile. For the shared pipeline, see Brian2's `code generation
<https://brian2.readthedocs.io/en/stable/developer/codegen.html>`_ and
`standalone implementation
<https://brian2.readthedocs.io/en/stable/developer/standalone.html>`_
developer pages.

Even a tiny model produces many files. This page uses a small example to show
**which files appear and what each one is for**. It does not walk through the
generated source line by line.

A minimal example
-----------------

The following script is enough to produce a complete standalone project::

    from brian2 import *
    import brian2cuda
    set_device('cuda_standalone', directory='simple_project')

    G = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1',
                    threshold='v > 1', reset='v = 0')
    run(10*ms)

After ``device.build()`` (or the first ``run`` with the default build
settings), ``simple_project/`` contains the sources below. Names under
``code_objects/`` include the Brian object name (here ``neurongroup``) and the
template (``stateupdater``, ``thresholder``, ``resetter``).

Project tree
------------

A NeuronGroup-only project looks like this (headers listed next to their
``.cu`` files)::

    simple_project/
    ├── main.cu
    ├── objects.cu / objects.h
    ├── network.cu / network.h
    ├── run.cu / run.h
    ├── rand.cu / rand.h
    ├── synapses_classes.cu / synapses_classes.h
    ├── makefile                  # win_makefile on MSVC
    ├── code_objects/
    │   ├── neurongroup_stateupdater_codeobject.cu / .h
    │   ├── neurongroup_thresholder_codeobject.cu / .h
    │   └── neurongroup_resetter_codeobject.cu / .h
    ├── brianlib/                 # copied support library
    ├── results/                  # created for runtime output
    └── static_arrays/            # binary seeds from Python assignments

Adding monitors, synapses, or variable assignments before ``run`` adds more
files under ``code_objects/`` (and may fill ``static_arrays/``). The top-level
basenames stay the same.

What each part does
-------------------

``main.cu``
    Entry point. Selects the GPU, parses command-line options such as
    ``--results_dir``, applies run arguments, and calls ``brian_start``, the
    network run, and ``brian_end``.

``objects.h`` / ``objects.cu``
    Declarations and definitions for host arrays, ``DeviceBuffer`` objects,
    clocks, and networks, plus init / load / write / dealloc helpers. Most
    generated code includes ``objects.h``. See :doc:`dynamic_array/index` for
    how dynamic arrays are represented here.

``network.h`` / ``network.cu``
    Clocked ``Network::run`` loop that schedules code objects over time.

``run.h`` / ``run.cu``
    ``brian_start`` / ``brian_end`` and one function per Brian ``run()`` call.

``rand.h`` / ``rand.cu``
    cuRAND host buffers and related setup for this project. Always generated.
    Code objects that never call ``rand`` / ``randn`` / related functions do
    not include ``rand.h``. See :doc:`random_numbers`.

``synapses_classes.h`` / ``synapses_classes.cu``
    Device types for synaptic pathways. Always written. Without ``Synapses``
    the files are nearly empty.

``code_objects/*.cu`` / ``*.h``
    One translation unit per code object: host wrapper plus device kernels for
    that step (state update, threshold, synaptic push, monitor write, and so
    on). Optional ``before_run_*`` / ``after_run_*`` pairs appear when those
    blocks are non-empty.

``brianlib/``
    Fixed support library copied from the Brian2CUDA package. Headers such as
    ``cuda_utils.h`` and ``device_buffer.h`` are included from generated code.
    A few ``.cu`` files compile once per project. See :doc:`../brianlib/index`.

``makefile`` / ``win_makefile``
    Build rules for ``nvcc`` (Unix ``make``, or ``nmake`` on MSVC). The default
    binary name is ``main``.

``results/``
    Directory for runtime binary dumps and related output (for example host
    WARNING/ERROR lines in ``cuda_log.txt`` when CUDA logging is enabled).
    Passed to the binary as ``--results_dir``.

``static_arrays/``
    Binary files for concrete array initialisations from Python. The same
    mechanism as in C++ standalone.

How this is produced
--------------------

``CudaStandaloneDevice.build`` in ``device.py`` creates the directories, writes
each source through ``CUDAWriter``, and copies ``brianlib``. Writer entries of
the form ``name.*`` expand to ``name.cu`` and ``name.h``. Per–code-object files
come from Jinja templates under ``brian2cuda/templates/`` with extension
``.cu`` (see ``CUDAStandaloneCodeObject`` in ``codeobject.py``).

The include graph and why Thrust or cuRAND stay out of most translation units
are described in :doc:`index`.
