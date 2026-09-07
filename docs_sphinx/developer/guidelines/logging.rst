Logging
=======

For logging in Brian itself, see `Brian2's logging guidelines
<https://brian2.readthedocs.io/en/stable/developer/guidelines/logging.html>`_
and the user documentation on `Brian2 logging
<https://brian2.readthedocs.io/en/stable/advanced/logging.html>`_.

Standalone CUDA code runs in a separate process from Python. Brian2CUDA therefore
adds compile-time logging macros for generated C++/CUDA code. They follow
Brian2's console log level, strip messages below
that level at build time, and copy host WARNING/ERROR back to the Python logger
after ``device.run()`` via files.

Python logging
--------------

Python code in Brian2CUDA should use Brian2's ``get_logger`` with a
``brian2cuda.`` module name::

    from brian2.utils.logger import get_logger

    logger = get_logger(__name__)

    logger.debug('A debug message', 'CudaCodeGenerator')
    logger.warn('A warning message', once=True)

``brian2cuda/utils/logger.py`` also provides ``suppress_brian2_logs()`` for
test runs. It hides most ``brian2.*`` log hierarchies while keeping
``brian2.devices.cuda_standalone`` and ``brian2.codegen.generators.cuda_generator``.

Log level
---------

Brian2CUDA reuses the console level from ``BrianLogger``. Codegen passes it to
``nvcc`` as ``-DB2C_LOG_LEVEL``. The numeric values match Python ``logging``:
``B2C_LOG_LEVEL_DEBUG`` (10), ``B2C_LOG_LEVEL_INFO`` (20),
``B2C_LOG_LEVEL_WARNING`` (30), ``B2C_LOG_LEVEL_ERROR`` (40). The default is
WARNING.

``get_codegen_log_level``, ``nvcc_log_flags``, and ``update_log_flags_stamp`` in
``brian2cuda/utils/logger.py`` handle the Python side. When the level changes,
``device.py`` cleans the standalone project so old object files are not reused.

Set the level before building the standalone project::

    from brian2.utils.logger import BrianLogger

    BrianLogger.log_level_debug()
    set_device('cuda_standalone')

The level cannot be changed inside the compiled binary without a rebuild.

Macros
------

CUDA logging macros are defined in ``brianlib/logging.h``. In templates and
``brianlib`` code, log via::

    #include "brianlib/logging.h"

    B2C_LOG_ERROR("An error message: %s", detail);
    B2C_LOG_WARN("A warning message");
    B2C_LOG_INFO("An info message: %g", value);
    B2C_LOG_DEBUG("A debug message: %s", name.c_str());

Arguments follow ``printf`` conventions. Each line is prefixed with
``[brian2cuda][LEVEL]``. On the device the macros call ``printf``, on the host
they call ``brian::b2c_log_message``.

Use the level-specific macros rather than ``B2C_LOG_EMIT``. For code that
exists only for logging (extra variables, loops, ``ostringstream``), wrap it in::

    #if B2C_LOG_LEVEL <= B2C_LOG_LEVEL_DEBUG
    ...
    #endif

