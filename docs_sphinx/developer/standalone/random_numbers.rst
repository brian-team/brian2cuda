Random number generation
========================

.. contents::
   :local:
   :depth: 1

Brian2CUDA uses NVIDIA cuRAND for ``rand``, ``randn``, ``poisson``, and
``binomial`` under ``cuda_standalone``. There are two generation paths. The
**host API** fills GPU buffers from the host and kernels only read them. The
**device API** keeps a pool of ``curandState`` values and draws numbers
on the fly inside kernels. Which path is used depends on the Brian function
and on whether Poisson ``lambda`` is a scalar or a vector.

Which path is used
------------------

``rand()`` and ``randn()``
    Host API buffers, refilled in ``rand.cu``. Kernels expand to macros that
    index ``_ptr_array_<codeobj>_rand`` / ``_randn``.

``poisson(lambda)`` with scalar ``lambda``
    Host API buffers, one buffer per distinct scalar ``lambda`` on that code
    object. The generated call is rewritten to pass a buffer pointer into an
    overloaded ``_poisson``.

``poisson(lambda)`` with vectorized ``lambda``
    Device API. Each thread loads ``d_curand_states[idx]``, calls
    ``curand_poisson``, and stores the state back.

``binomial(...)``
    Device API (same ``curandState`` pool). The implementation is a
    ``__host__ __device__`` function registered in ``binomial.py``.

Synapse creation (``synapses_create_generator``)
    Fully host-side. Connectivity code uses ``CurandBuffer`` from
    ``brianlib`` and ``_host_rand`` / ``_host_randn`` / ``_host_poisson``.
    It does not use the every-tick ``RandomNumberBuffer`` path.

Code generation
---------------

CUDA implementations of the default functions live in ``cuda_generator.py``
(and again on ``CUDAStandaloneCodeObject`` for ``rand`` / ``randn``). Each
implementation can declare required headers through
``compiler_kwds["headers"]``. Those headers are emitted for that code object
only via ``user_headers`` in ``common_group.cu``. They are not part of the
shared include block, so a state updater without RNG never includes
``rand.h`` or ``<curand.h>``.

After CUDA source exists, ``prepare_codeobj_code_for_rng`` in ``device.py``
scans the generated file and:

* counts ``_rand``, ``_randn``, and ``_poisson`` appearances
* classifies each Poisson ``lambda`` as scalar or vectorized
* rewrites host-API call indices so successive draws in one timestep use
  disjoint buffer slices (``_vectorisation_idx + i * _N``)
* rewrites scalar Poisson calls to pass buffer pointers
* sets ``needs_curand_states`` when binomial or vectorized Poisson is present

For the ``synapses`` template the text appears twice (homogeneous and
heterogeneous delay branches), so the counted call rates are halved. The
``synapses_create_generator`` template is skipped entirely.

The device keeps registries of code objects that need host-API buffers and
of those that need device-API states. These drive the Jinja context for
``templates/rand.cu``.

Generated ``rand.h`` and ``rand.cu``
------------------------------------

``generate_rand_source`` always writes ``rand.h`` and ``rand.cu`` into the
standalone project. The header declares ``RandomNumberBuffer``, the buffer
pointers, the host ``curandGenerator_t``, and the device ``curandState``
pool. The ``.cu`` file defines them and implements refill and seeding.

``objects.h`` does not include ``rand.h``. Translation units that need RNG
include it themselves: opt-in code objects through ``compiler_kwds``, plus
``main.cu``, ``run.cu``, ``objects.cu``, and
``synapses_create_generator.cu``. That isolation is part of the lean include
design described in :doc:`index`.

Host API buffers
----------------

For every-tick code objects, ``RandomNumberBuffer::init`` sizes each buffer
from the owner ``N``, the number of RNG calls per step, free device memory,
and the remaining run length. Each network clock runs
``_run_random_number_buffer`` before other code objects on that clock. That
call advances or refills the buffers with ``curandGenerateUniform``,
``curandGenerateNormal``, or ``curandGeneratePoisson``.

One-shot code objects (for example ``G.v = 'rand()'``) do not keep a
long-lived buffer. ``generate_codeobj_source`` injects host code that
allocates, fills, and frees a temporary buffer around that kernel launch.

Device API states
-----------------

Code objects with ``needs_curand_states`` share ``dev_curand_states`` /
``d_curand_states``. States are initialized with ``curand_init`` using a
shared seed and distinct sequence numbers. The pool size tracks the largest
owner among device-API objects and can grow after synapse creation through
``ensure_enough_curand_states`` when synaptic initializers need RNG before
``N`` is known.

Synapse creation
----------------

``synapses_create_generator.cu`` treats ``_ptr_array_<codeobj>_rand`` as a
``CurandBuffer`` whose ``operator[]`` ignores the index and returns the next
host-side number. ``CurandBuffer`` is declared in ``curand_buffer.h`` without
including ``<curand.h>``. The implementation in ``curand_buffer.cu`` uses the
host API, copies batches to the host, and streams them through the index
operator. See :doc:`../brianlib/index`.

Seeding
-------

Brian's ``seed(x)`` is queued like other standalone commands. In
``generate_main_source``, ``None`` becomes a random ``uint64`` value, and the
generated ``main.cu`` calls ``random_number_buffer.set_seed(...)``.

``set_seed`` updates the host generator, writes ``seed + 1`` into
``dev_curand_seed`` (so host and device streams are offset), and reinitializes
device states. Without a user ``seed()``, ``_init_arrays`` in ``objects.cu``
seeds from ``time(0)``.

Generator type and optional ordering come from
``devices.cuda_standalone.random_number_generator_type`` and
``random_number_generator_ordering``. The float type of uniform and normal
buffers follows ``core.default_float_dtype``.

After each ``run()``, generated code calls
``random_number_buffer.run_finished()`` so the next run reinitializes
buffers.

Preferences and further reading
-------------------------------

Relevant preferences are defined in ``cuda_prefs.py``. Tests for classification,
rewriting, and seeding live in ``brian2cuda/tests/test_random_number_generation.py``.
Additional design notes are on the project wiki page
`Random number generation
<https://github.com/brian-team/brian2cuda/wiki/Random-number-generation>`_.
