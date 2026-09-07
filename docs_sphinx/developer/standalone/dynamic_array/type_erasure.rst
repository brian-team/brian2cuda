Type erasure
============

PImpl keeps Thrust out of public headers. Type erasure keeps Thrust
instantiation cheap inside ``device_buffer.cu``. See :doc:`index` for the
host/device split that ``DeviceBuffer`` implements.

Why type erasure
----------------

A typed ``DeviceBuffer<T>`` would compile a separate
``thrust::device_vector<T>`` for every dtype in the model. ``nvcc`` would run
the same heavy template work many times in one file. Type erasure stores raw
bytes plus an ``elem_size`` instead of parameterising the class on ``T``, so the
project instantiates ``thrust::device_vector<char>`` once regardless of how
many ``double``, ``int32_t``, or pointer arrays exist.

Principle and effect
--------------------

The buffer owns ``n`` logical elements as ``n * elem_size`` bytes. Construction
records ``sizeof(T)``. ``data_as<T>()`` casts the cached raw pointer at the use
site. Host-side data stays in ``std::vector<T>``. ``copy_from_host`` and
``copy_to_host`` move bytes with ``cudaMemcpy``.

For generated code this means: construct with the dtype size, read through
``data_as`` in kernels, and do not rely on separate global ``dev_array_*``
pointers that can go stale after ``resize``. The buffer refreshes its pointer
after every reallocation.

Usage
-----

Construction in ``objects.cu``::

    DeviceBuffer dev_dynamic_array_foo(sizeof(double));

Kernel or device template code::

    double* arr = dev_dynamic_array_foo.data_as<double>();
    int n = static_cast<int>(dev_dynamic_array_foo.size());

See :doc:`pimpl` for where the byte storage lives.

Further reading
---------------

* `Type erasure — Wikipedia <https://en.wikipedia.org/wiki/Type_erasure>`_
* `Thrust device_vector documentation <https://nvidia.github.io/cccl/thrust/api/classthrust_1_1device__vector.html>`_
