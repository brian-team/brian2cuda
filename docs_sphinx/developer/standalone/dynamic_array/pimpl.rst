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
public interface. Callers depend on ``resize``, ``copy_from_host``, and
``clear``. They never see ``thrust::device_vector``. Thrust parsing and
container instantiation for device storage happen only in ``device_buffer.cu``.

In practice, ``device_buffer.h`` forward-declares a private ``Impl`` struct and
holds ``std::unique_ptr<Impl>``. ``device_buffer.cu`` defines ``Impl`` and
stores ``thrust::device_vector<char>``. The destructor is declared in the header
and defined in the ``.cu`` file, which is required when ``Impl`` is incomplete
in the header.

Using ``DeviceBuffer``
----------------------

Generated ``objects.cu`` declares one ``DeviceBuffer`` per dynamic array and
generates copy helpers that call into the buffer. Templates and kernels include
``device_buffer.h`` only. They obtain typed pointers through ``data_as`` (see
:doc:`type_erasure`). For the ``brianlib`` headers see :doc:`../../brianlib/index`.

Generated array names carry a ``_dynamic_array_`` prefix. ``array_basename()``
in ``codeobject.py`` strips it so helpers get names like
``copy_host_to_dev_array_synapses_delay``.

Further reading
---------------

* `PImpl — cppreference.com <https://en.cppreference.com/w/cpp/language/pimpl>`_
* `Pointer to implementation — Wikipedia <https://en.wikipedia.org/wiki/Pointer_to_implementation>`_
* `GotW #100: Minimize Compilation Dependencies — Herb Sutter <https://herbsutter.com/gotw/_100/>`_
