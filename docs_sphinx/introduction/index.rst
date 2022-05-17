Introduction
============

Brian2CUDA is a Python package for simulating spiking neural networks on
graphics processing units (GPUs). It is an extension of the spiking neural
network simulator `Brian2`_, which allows flexible model definitions in Python.
Brian2CUDA uses the code generation system from Brian2 to generate simulation
code in C++/CUDA, which is then executed on NVIDIA GPUs.

To use Brian2CUDA, add the following two lines of code to your Brian2 imports.
This will execute your simulations on a GPU::

    from brian2 import *
    import brian2cuda
    set_device("cuda_standalone")

For more details on the code generation process and settings, read the `Brian2
standalone device documentation`_.

Getting help and reporting bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need help with Brian2CUDA, please use the `Brian2 discourse forum`_. If you think
you found a bug, please report it in our `issue tracker on GitHub`_.

Citing Brian2CUDA
~~~~~~~~~~~~~~~~~
If you use Brian2CUDA in your work, please cite:

    Alevi, D., Stimberg, M., Sprekeler, H., Obermayer, K., & Augustin, M. (2022).
    Brian2CUDA: flexible and efficient simulation of spiking neural network models on GPUs.
    Frontiers in Neuroinformatics. https://doi.org/10.3389/fninf.2022.883700

.. Ref GitHub
.. Ref examples

.. _Brian2: https://brian2.readthedocs.io/en/stable/index.html
.. _Brian2 standalone device documentation: https://brian2.readthedocs.io/en/stable/user/computation.html#standalone-code-generation
.. _Brian2 discourse forum: https://brian.discourse.group/
.. _issue tracker on GitHub: https://github.com/brian-team/brian2cuda/issues?q=is%3Aopen+is%3Aissue+label%3Abug+
