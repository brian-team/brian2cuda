Dynamic arrays
==============

.. contents::
   :local:
   :depth: 1

Brian2's `DynamicArrayVariable` objects can change size during a simulation.
Spike monitors append spike times. Synapses can grow their index arrays during
``before_run``. In ``cpp_standalone``, host and device copies are both
``std::vector``. In ``cuda_standalone``, host data stays on the CPU and device
data lives in GPU memory, so the two sides use different container types.

Host-side storage
-----------------

One-dimensional dynamic arrays are ``std::vector<T>`` declared in
``objects.cu``. Generated names carry a ``_dynamic_array_`` prefix, for example
``_dynamic_array_spikemonitor_t``. The matching device object is named
``dev`` plus that full name (``dev_dynamic_array_spikemonitor_t``).

Host vectors are the source of truth before a run and after results are read
back. During ``before_run``, Python may fill them through the array cache.
Host code in templates calls ``push_back``, ``resize``, and similar STL
methods directly. Before kernels access the data, templates call
``dev….copy_from_host(...)`` on the matching ``DeviceBuffer``. After a run,
``copy_to_host`` downloads results for Python.

Two-dimensional connectivity tables built while creating synapses use
``DynamicArray2D<T>`` from ``dynamic_array.h``. This is a host-only structure:
an outer ``std::vector`` of inner ``std::vector`` pointers, resized along the
first dimension. It is filled during synapse creation on the host and is not
mirrored as a single GPU array.

Device-side storage
-------------------

``DeviceBuffer`` is a resizable, type-erased byte buffer. Kernels and
generated host code call its methods directly: ``resize``, ``clear``,
``copy_from_host``, ``copy_to_host``, and ``data_as<T>()``.See :doc:`pimpl` and
:doc:`type_erasure`.

Fixed-size arrays follow a different pattern: separate host pointer, device
pointer, and ``__device__`` symbol. Dynamic arrays do not use global
``dev_array_*`` pointers that can go stale after ``resize``. The
``DeviceBuffer`` refreshes its cached raw pointer on every reallocation.

Other device containers
-----------------------

Not every resizable structure in ``brianlib`` is a Brian dynamic array.

``cudaVector`` (``cudaVector.h``) is a growable array allocated with
``malloc`` inside device code. ``CudaSpikeQueue`` uses it to record spikes
while propagating through bundles. It exists only on the device, is not paired
with a host ``std::vector``, and is unrelated to ``DynamicArrayVariable``.

Delay eventspaces keep a host ``std::vector`` of raw device pointers
(``std::vector<T*>``). Extra queues are allocated with ``cudaMalloc`` when
more delay slots are needed.

Subgroup spike monitors use a separate ``DeviceBuffer``
(``_dev_<source>_eventspace``) as a filtered copy of the source eventspace.
Templates ``resize`` it and pass ``data_as<int32_t>()`` into the filter
kernel.

State monitors over two-dimensional dynamic arrays store one ``DeviceBuffer``
per recorded row, plus an ``addresses_monitor_*`` ``DeviceBuffer`` of row
pointers. The state-monitor template uploads those addresses with
``copy_from_host``. The host still uses ``std::vector`` for the flat row data
when copying results back.

.. toctree::
   :maxdepth: 1

   pimpl
   type_erasure
