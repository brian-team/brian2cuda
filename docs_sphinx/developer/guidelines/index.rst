Coding guidelines
=================

The basic principles of developing Brian2CUDA are the same as for Brian2:

1. For the user, the emphasis is on making the package flexible, readable and
   easy to use.
2. For the developer, the emphasis is on keeping the package maintainable by a
   small number of people. Prefer stable, well maintained open source packages
   (and Brian2 itself) over reinventing shared infrastructure.

For workflow, style, object representation, defensive programming, and
docstring conventions, follow Brian2's `coding guidelines
<https://brian2.readthedocs.io/en/stable/developer/guidelines/index.html>`_.
Brian2CUDA-specific notes are limited. The pages below cover logging and
testing. A few additional points for CUDA work:

* Keep Thrust, cuRAND, and other heavy headers out of files that many
  translation units include. The standalone include design and ``brianlib``
  layout document the current split (:doc:`../standalone/index`,
  :doc:`../brianlib/index`).
* Use Brian2's ``get_logger`` with a ``brian2cuda.`` module name in Python.
  Generated standalone code uses compile-time ``B2C_LOG_*`` macros instead of
  the Python logger (:doc:`logging`).
* Prefer the existing test and benchmark entry points over ad hoc scripts when
  checking correctness or performance (:doc:`testing`).

.. toctree::
   :maxdepth: 2

   logging
   testing
