Installation
============

Requirements
------------
.. TODO check minimal compute capability support
- `NVIDIA CUDA GPU`_ with compute capability 3.5 or larger
- `CUDA Toolkit`_ with ``nvcc`` compiler and compiled ``deviceQuery`` binary.
  - A compiled `deviceQuery` binary is only included and automatically detected
    in full CUDA toolkit installations. If it is not included in your
    installation, you can compile it manually, see ...
- `Python`_ version 3.6 or larger
- `Brian2`_: Each Brian2CUDA version is compatible with a specific Brian2
  version. The correct Brian2 version is installed during the Brian2CUDA
  installation.

We recommend installing Brian2CUDA in a separate Python environment, either
using a "virtual environment" or a "conda environment". If you are unfamiliar
with that, check out the `Brian2 installation instructions`_.

.. _NVIDIA CUDA GPU: https://developer.nvidia.com/cuda-gpus
.. _CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
.. _Python: https://www.python.org/
.. _Brian2: https://briansimulator.org/
.. _Brian2 installation instructions: https://brian2.readthedocs.io/en/2.5.0.3/introduction/install.html


.. _standard_install:
Standard install
----------------

To install Brian2CUDA with a compatible Brian2 version, use ``pip``::

   python -m pip install brian2cuda

.. TODO create pip package, with brian2 dependency, make sure this works
.. TODO Make conda package and add instructions here (see nice brian2 docs)


.. _update_install:
Updating an existing installation
---------------------------------

Use the install command together with the ``--upgrade`` option::

   python -m pip install --upgrade brian2cuda

This will also update the installed Brian2 version if required.


.. _development_install:
Development install
-------------------
When you encounter a problem in BrianCUDA, we will sometimes ask you to install
Brian2CUDA's latest development version, which includes changes that were included
after its last release.

We regularly upload the latest development version of Brian2CUDA to PyPI's test
server. You can install it via::

    python -m pip install --upgrade --pre -i https://test.pypi.org/simple/ Brian2CUDA

Note that this requires that you already have a compatible Brian2 version and
all of its dependencies installed.

If you have ``git`` installed, you can also install directly from github::

    python -m pip install git+https://github.com/brian-team/brian2cuda.git

If you want to either contribute to Brian's development or regularly test its
latest development version, you can directly clone the git repository at github
(https://github.com/brian-team/brian2cuda) and then run ``pip install -e .``,
to install Brian2CUDA in "development mode". As long as the compatible Brian2
version doesn't change, updating the git repository is in general enough to
keep up with changes in the code, i.e. it is not necessary to install it again.
If the compatible Brian2 versions changes though, you need to manually update
Brian2.


.. _cuda_configuration:
Configuring the CUDA backend
----------------------------

Brian2CUDA tries to detect your CUDA installation and uses the GPU with highest
compute capability by default. To query information about available GPUs,
``nvidia-smi`` (installed with NVIDIA drivers) and ``deviceQuery`` binaries are
used. Some CUDA installations come without ``deviceQuery`` binary, in which
case it needs to be compiled manually.

This section explains how you can manually set which CUDA installation or GPU
to use, how to manually compile ``deviceQuery`` if it is missing and how to
cross-compile GPU code on systems without GPU access (e.g. during remote
development).

Manually compiling ``deviceQuery``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO: Download `CUDA Samples`_, compile ``deviceQuery``, set
`devices.cuda_standalone.cuda_backend.device_query_path`

TODO: Don't use deviceQuery:
- `devices.cuda_standalone.cuda_backend.detect_gpus`

.. _`CUDA Samples`: https://github.com/NVIDIA/cuda-samples/tree/master/Samples

Manually specifying the CUDA installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you installed the `CUDA toolkit`_ in a non-standard location or if you have
a system with multiple CUDA installations, you may need to manually specify the
installation directory.

Brian2CUDA tries to detect your CUDA installation in the following order:

    1. Use Brian2CUDA preference `devices.cuda_standalone.cuda_backend.cuda_path`
    2. Use ``CUDA_PATH`` environment variable
    3. Use location of ``nvcc`` to detect CUDA installation folder (needs ``nvcc`` binary in ``PATH``)
    4. Use standard location ``/usr/local/cuda``
    5. Use standard location ``/opt/cuda``

If you set the path manually via the 1. or 2. option, specify the parant path
to ``bin/nvcc`` (e.g. ``/usr/local/cuda`` if ``nvcc`` is in ``/usr/local/cuda/bin/nvcc``).

.. TODO Do we need this? Check cluster
.. Depending on your system configuration, you may also need to set the
.. ``LD_LIBRARY_PATH`` environment variable to ``$CUDA_PATH/lib64``.

Manually selecting a GPU to use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `devices.cuda_standalone.cuda_backend.gpu_id`


Cross-compiling on systems without GPU access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `devices.cuda_standalone.cuda_backend.detect_gpus`
- `devices.cuda_standalone.cuda_backend.compute_capability`
- `devices.cuda_standalone.cuda_backend.cuda_runtime_version`


.. _testing_brian2cuda_install:
Testing your installation
-------------------------
You can run a short example script to see if your installation and
configuration were successful::

    import brian2cuda
    brian2cuda.example_run()


.. _testing_brian2cuda:
Running the Brian2CUDA test suit
--------------------------------

If you have the pytest_ testing utility installed, you can run Brian2CUDA's test
suite::

    import brian2cuda
    brian2cuda.test()

.. TODO Let known issue tests fail
This runs all standalone-comatible tests from the Brian2 test suite and
additional Brian2CUDA tests (see the `Brian2 developer documentation on
testing`_ for more details). The test suite should end with "OK", showing a
number of skipped tests but no errors or failures. If you want to run
individual tests instead of the entire test suite (e.g. during development),
check out the `Brian2CUDA tools directory`_.

.. _pytest: https://docs.pytest.org/en/stable/
.. _Brian2 developer documentation on testing: https://brian2.readthedocs.io/en/stable/developer/guidelines/testing.html
.. _Brian2CUDA tools directory: https://github.com/brian-team/brian2cuda/tree/master/brian2cuda/tools
