brianlib
========

.. contents::
   :local:
   :depth: 1

``brianlib`` is the C++/CUDA support library shipped with every CUDA standalone
project. ``device.py`` copies the whole directory into the generated build tree
and adds each ``.h`` to the header list and each ``.cu`` to the makefile
sources. The same files are used for all models. Only the generated code
objects and ``objects.cu`` change between projects.

The library serves two roles. Headers such as ``cuda_utils.h`` and
``device_buffer.h`` are included from generated ``.cu`` files. A few
``.cu`` files hold implementations that would be too expensive to parse or
instantiate in every translation unit if they lived in headers. Which headers
each code object includes is described in :doc:`../standalone/index`.

Public headers
--------------

``cuda_utils.h``
    CUDA error-checking macros (``CUDA_SAFE_CALL``, ``CUDA_CHECK_ERROR``,
    ``CUDA_CHECK_MEMORY``, ``THRUST_CHECK_ERROR``). The header depends only on
    ``cuda_runtime.h``. ``CUDA_CHECK_MEMORY`` reads ``brian::used_device_memory``
    through an ``extern`` declaration. ``objects.cu`` defines the variable.
    Generated code includes this file widely, so it must not pull in
    ``objects.h``, Thrust, or cuRAND. Error paths use ``B2C_LOG_ERROR`` from
    ``logging.h``.

``logging.h``
    Compile-time logging macros (``B2C_LOG_ERROR``, ``B2C_LOG_WARN``,
    ``B2C_LOG_INFO``, ``B2C_LOG_DEBUG``) gated by ``-DB2C_LOG_LEVEL``. Device
    code emits with ``printf``. Host code calls ``brian::b2c_log_message``. See
    :doc:`../guidelines/logging`.

``device_buffer.h``
    Resizable device storage for dynamic arrays. The header is Thrust-free. See
    :doc:`../standalone/dynamic_array/index`.

``curand_buffer.h``
    Host-side cuRAND buffer used during synapse creation. Declares
    ``CurandBuffer`` and forward-declares ``curandGenerator_t`` so including
    the header does not parse ``<curand.h>``.

``curand_utils.h``
    cuRAND status strings and ``CUDA_SAFE_CALL`` overloads for ``curandStatus_t``.
    Include this from translation units that call host cuRAND APIs, for example
    ``curand_buffer.cu`` or ``objects.cu``.

``clocks.h``
    Host clock types (``Clock``, ``EventClock``) shared between ``objects.h``
    and generated code objects.

``common_math.h``
    ``INFINITY``, ``NAN``, and ``M_PI`` definitions for MSVC.

``stdint_compat.h``
    Fixed-width integer typedefs used across generated CUDA code.

``dynamic_array.h``
    Host ``DynamicArray2D`` for two-dimensional synapse connectivity tables
    built during ``before_run``. See :doc:`../standalone/dynamic_array/index`.

``cudaVector.h``
    Device-side growable array used inside the spike propagation queue.

``spikequeue.h``
    ``CudaSpikeQueue`` and related device logic for heterogeneous synaptic delays
    and bundle-based propagation.

``host_algorithms.h``
    Host ``sort_by_key`` and ``unique_by_key`` helpers. Spike-push templates call
    these on ``std::vector`` data during ``before_run`` instead of pulling Thrust
    into ``objects.cu``.

