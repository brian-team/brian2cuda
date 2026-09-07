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
``_dynamic_array_spikemonitor_t``. Codegen and copy helpers strip that prefix
via ``array_basename()`` in ``codeobject.py``.

Host vectors are the source of truth before a run and after results are read
back. During ``before_run``, Python may fill them through the array cache. Host
code in templates calls ``push_back``, ``resize``, and similar STL methods
directly. Before kernels access the data, ``copy_host_to_dev_array_*`` uploads
the vector contents to the matching ``DeviceBuffer``.

Two-dimensional connectivity tables built while creating synapses use
``DynamicArray2D<T>`` from ``dynamic_array.h``. This is a host-only structure:
an outer ``std::vector`` of inner ``std::vector`` pointers, resized along the
first dimension. It is filled during synapse creation on the host and is not
mirrored as a single GPU array.

Device-side storage
-------------------

The device mirror of each one-dimensional dynamic array is a ``DeviceBuffer``
named ``dev`` plus the full array name (for example
``dev_dynamic_array_spikemonitor_t``). ``DeviceBuffer`` is a resizable,
type-erased byte buffer. Kernels obtain typed pointers with ``data_as<T>()``.
See :doc:`pimpl` and :doc:`type_erasure`.

``objects.cu`` generates paired helpers for every dynamic array:

* ``copy_host_to_dev_array_<basename>()`` — upload after host-side changes
* ``copy_dev_to_host_array_<basename>()`` — download for Python after a run

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

Eventspaces keep a host ``std::vector`` of raw device pointers
(``std::vector<T*> dev_eventspace``). ``expand_eventspace*`` allocates new GPU
buffers with ``cudaMalloc`` when more delay queues are needed. This is pointer
management, not element storage in a ``DeviceBuffer``.

State monitors over two-dimensional dynamic arrays store one ``DeviceBuffer`` per
recorded row on the device, plus a ``DeviceBuffer`` of row addresses uploaded
with ``upload_monitor_row_addresses``. The host still uses ``std::vector`` for
the flat row data when copying results back.

.. toctree::
   :maxdepth: 1

   pimpl
   type_erasure
