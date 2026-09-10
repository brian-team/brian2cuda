PImpl
=====

``DeviceBuffer`` hides its Thrust-backed implementation behind a private
``Impl`` struct. See :doc:`index` for how host ``std::vector`` and device
``DeviceBuffer`` pair up in generated code.

Why PImpl
---------

``nvcc`` parses every header a translation unit includes. Thrust headers are
large, and ``thrust::device_vector`` triggers further template instantiation in
the same file. If ``device_buffer.h`` exposed that type, ``objects.cu`` and every
other file holding a ``DeviceBuffer`` would parse and instantiate Thrust again,
which largely cancels the benefit of keeping ``objects.h`` lean.

The `PImpl idiom <https://en.cppreference.com/w/cpp/language/pimpl>`_ puts the
heavy implementation in a ``.cu`` file and leaves the header with only a stable
public interface. Callers depend on ``resize``, ``clear``, ``copy_from_host``,
``copy_to_host``, ``set_elem_size``, and ``data_as``. They never see
``thrust::device_vector``. Thrust parsing and container instantiation for
device storage happen only in ``device_buffer.cu``.

In practice, ``device_buffer.h`` forward-declares a private ``Impl`` struct and
holds ``std::unique_ptr<Impl>``. ``device_buffer.cu`` defines ``Impl`` and
stores ``thrust::device_vector<char>``. The destructor is declared in the header
and defined in the ``.cu`` file, which is required when ``Impl`` is incomplete
in the header. After ``resize`` or ``clear``, the header caches a raw device
pointer so kernels can use ``data_as`` without touching Thrust.

Using ``DeviceBuffer``
----------------------

Generated ``objects.cu`` declares one ``DeviceBuffer`` per dynamic array (and
related buffers such as monitor row addresses or subgroup eventspaces).
Templates include ``device_buffer.h`` and call methods on those objects
directly, for example::

    // host -> device after growing a host std::vector
    dev_dynamic_array_synapses_delay.copy_from_host(
            _dynamic_array_synapses_delay.data(),
            _dynamic_array_synapses_delay.size());

    // grow on the device, then pass a typed pointer into a kernel
    dev_dynamic_array_spikemonitor_t.resize(new_size);
    double* t = dev_dynamic_array_spikemonitor_t.data_as<double>();

For the ``brianlib`` headers see :doc:`../../brianlib/index`. Typed access and
``set_elem_size`` are described in :doc:`type_erasure`.

Further reading
---------------

* `PImpl — cppreference.com <https://en.cppreference.com/w/cpp/language/pimpl>`_
* `Pointer to implementation — Wikipedia <https://en.wikipedia.org/wiki/Pointer_to_implementation>`_
* `GotW #100: Minimize Compilation Dependencies — Herb Sutter <https://herbsutter.com/gotw/_100/>`_
