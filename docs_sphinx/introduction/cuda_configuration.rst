Configuring the CUDA backend
============================

Brian2CUDA tries to detect your CUDA installation and uses the GPU with highest
compute capability by default. To query information about available GPUs,
``nvidia-smi`` (installed alongside NVIDIA display drivers) is used.
For older driver versions (``< 510.39.01``), ``nvidia-smi`` doesn't support querying the
GPU compute capabilities and some additional setup might be required.

This section explains how you can manually set which CUDA installation or GPU
to use, how to cross-compile Brian2CUDA projects on systems without GPU access (e.g.
during remote development) and what to do when the compute capability detection fails.

.. contents::
    :local:
    :depth: 1

Manually specifying the CUDA installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you installed the `CUDA toolkit`_ in a non-standard location or if you have
a system with multiple CUDA installations, you may need to manually specify the
installation directory.

Brian2CUDA tries to detect your CUDA installation in the following order:

    1. Use Brian2CUDA preference `devices.cuda_standalone.cuda_backend.cuda_path`
    2. Use ``CUDA_PATH`` environment variable
    3. Use location of ``nvcc`` to detect CUDA installation folder (needs ``nvcc``
       binary in ``PATH``)
    4. Use standard location ``/usr/local/cuda``
    5. Use standard location ``/opt/cuda``

If you set the path manually via the 1. or 2. option, specify the parent path
to the ``nvcc`` binary (e.g. ``/usr/local/cuda`` if ``nvcc`` is in
``/usr/local/cuda/bin/nvcc``).

Depending on your system configuration, you may also need to set the
``LD_LIBRARY_PATH`` environment variable to ``$CUDA_PATH/lib64``.

Manually selecting a GPU to use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On systems with multiple GPUs, Brian2CUDA uses the first GPU with highest compute
capability as returned by ``nvidia-smi``. If you want to manually choose a GPU you can
do so via Brian2CUDA preference `devices.cuda_standalone.cuda_backend.gpu_id`.

.. note::
   You can limit the visibility of NVIDIA GPUs by setting the environment variable
   ``CUDA_VISIBLE_DEVICES``. This also limits the GPUs visible to Brian2CUDA. That means
   Brian2CUDA's `devices.cuda_standalone.cuda_backend.gpu_id` preference will index only
   those GPUs that are visible. E.g. if you run a Brian2CUDA script with
   ``prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0`` on a system with two GPUs
   via ``CUDA_VISIBLE_DEVICES=1 python your-brian2cuda-script.py``, the simulation would
   run on the second GPU (with ID ``1``, visible to Brian2CUDA as ID ``0``).


Cross-compiling on systems without GPU access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On systems without GPU, Brian2CUDA will fail before code generation by default (since it
tries to detect the compute capability of the available GPUs and the CUDA runtime
version). If you want to compile your code on a system without GPUs, you can disable
automatic GPU detection and manually set the compute capability and runtime version. To
do so, set the following preferences::

   prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
   prefs.devices.cuda_standalone.cuda_backend.compute_capability = <compute_capability>
   prefs.devices.cuda_standalone.cuda_backend.runtime_version = <runtime_version>

See `devices.cuda_standalone.cuda_backend.detect_gpus`,
`devices.cuda_standalone.cuda_backend.compute_capability` and
`devices.cuda_standalone.cuda_backend.cuda_runtime_version`.


Detecting GPU compute capability on systems with outdated NVIDIA drivers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use ``nvidia-smi`` to query the compute capability of GPUs during automatic GPU
selection. On older driver versions (``< 510.39.01``, these are driver versions shipped
with CUDA toolkit ``< 11.6``), this was not supported. For those versions, we use the
``deviceQuery`` tool from the `CUDA samples`_, which is by default installed with the
CUDA Toolkit under ``extras/demo_suite/deviceQuery`` in the CUDA installation directory.
For some custom CUDA installations, the CUDA samples are not included, in which case
Brian2CUDA's GPU detection fails. In that case, you have three options. Do one of the
following:

1. Update your NVIDIA driver
2. Download the `CUDA samples`_ to a folder of your choice and compile ``deviceQuery``
   manually::

      git clone https://github.com/NVIDIA/cuda-samples.git
      cd cuda-samples/Samples/1_Utilities/deviceQuery
      make
      # Run deviceQuery to test it
      ./deviceQuery

   Now set Brian2CUDA preference
   `devices.cuda_standalone.cuda_backend.device_query_path` to point to your
   ``deviceQuery`` binary.
3. Disable automatic GPU detection and manually provide the GPU ID and compute
   capability (you can find the compute capability of your GPU on
   https://developer.nvidia.com/cuda-gpus)::

      prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
      prefs.devices.cuda_standalone.cuda_backend.compute_capability = <compute_capability>

   See `devices.cuda_standalone.cuda_backend.detect_gpus` and
   `devices.cuda_standalone.cuda_backend.compute_capability`.

.. _`CUDA samples`: https://github.com/NVIDIA/cuda-samples/tree/master/Samples
.. _`CUDA toolkit`: https://developer.nvidia.com/cuda-toolkit