Brian2CUDA documentation
========================

Brian2CUDA is an extension of the spiking neural network simulator `Brian2`_, which
allows flexible model definitions in Python. Brian2CUDA uses the code generation system
from Brian2 to generate simulation code in C++/CUDA, which is executed on NVIDIA
graphics processing units (GPUs).

By adding two lines of code to your Brian2 imports, you can execute your
simulations on a GPU::

   from brian2 import *
   import brian2cuda
   set_device("cuda_standalone")


.. Ref GitHub
.. Ref examples


.. implements a `Brian2 standalone device`_ that generates C++/CUDA code and runs
.. simulations on NVIDIA graphics processing units (GPUs).



🚧🚧🚧 This documentation is currently under construction  🚧🚧🚧

It will soon document the ins and outs of `Brian2CUDA
<https://github.com/brian-team/brian2cuda>`_ - a Brian2 extension to simulate
spiking neural networks on GPUs!

.. _Brian2: https://brian2.readthedocs.io/en/stable/index.html
.. _Brian2 standalone device: https://brian2.readthedocs.io/en/stable/user/computation.html#standalone-code-generation

.. toctree::
   :maxdepth: 2
   :titlesonly:

   introduction/index
   introduction/preferences


.. toctree::
   :maxdepth: 1
   :titlesonly:

   Reference documentation <reference/brian2cuda>
   examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
