Standalone implementation
=========================

.. contents::
   :local:
   :depth: 1

This document describes implementation details of the ``cuda_standalone`` device.
For the general standalone mechanism in Brian2, see `standalone mode
<https://brian2.readthedocs.io/en/stable/user/computation.html#standalone-code-generation>`_
and `standalone implementation
<https://brian2.readthedocs.io/en/stable/developer/standalone.html>`_.

``CudaStandaloneDevice`` subclasses ``CPPStandaloneDevice``. Array cache,
command-line arguments, and project layout follow the C++ standalone device
unchanged. The CUDA-specific additions are code generation into ``.cu`` files,
the ``brianlib`` support library copied into each project, and compilation
with ``nvcc``.

Compilation
-----------

Brian2CUDA generates one ``.cu`` file per code object, so a change to part of
a model recompiles only the affected files. With ``nvcc`` separate compilation,
header parsing is not shared across translation units: a header included from
many files is parsed again in each of them.

Compile time therefore depends more on the include graph than on the amount of
generated code in a file. A short code object that pulls in Thrust, cuRAND, or
large internal headers can compile as slowly as a much longer file with lean
includes.

Include design
--------------

The headers seen by most generated code form a thin public surface.
``objects.h`` declares host ``std::vector`` arrays, ``DeviceBuffer`` for device
storage, and helper functions. It does not include Thrust or cuRAND.
``cuda_utils.h`` provides CUDA error checking and depends only on
``cuda_runtime.h``. See :doc:`../brianlib/index` for the full ``brianlib`` layout.

Thrust and cuRAND are confined to a few ``brianlib`` files that the project
links against but that code objects do not include, for example
``device_buffer.cu``, ``thrust_algorithms.cu``, and ``host_algorithms.h``.

Heavy third-party headers are added only where they are needed. Random number
generation lives in generated ``rand.h`` and ``rand.cu``. Because ``objects.h``
does not include ``rand.h``, code objects without ``rand()`` or ``randn()``
never parse cuRAND. Code objects that do use random numbers register
``"rand.h"`` through ``compiler_kwds["headers"]`` in ``codeobject.py`` or
``cuda_generator.py``. Resizable device arrays follow the same pattern.
``DeviceBuffer`` keeps Thrust out of ``objects.cu`` via a PImpl in a separate
translation unit. That layout is described in :doc:`dynamic_array/index`.

.. toctree::
   :maxdepth: 2

   dynamic_array/index
