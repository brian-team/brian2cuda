Known issues
============

In addition to the issues noted below, you can refer to `our bug tracker on GitHub`_.

.. _our bug tracker on GitHub: https://github.com/brian-team/brian2cuda/issues?q=is%3Aopen+is%3Aissue+label%3Abug+

List of known issues:

.. contents::
    :local:
    :depth: 1

Known issues when using multiple ``run`` calls
----------------------------------------------

Changing the integration time step of ``Synapses`` with delays between ``run`` calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Changing the integration time step of `Synapses` objects with transmission
delays between successive ``run`` calls currently leads to the loss of spikes.
This is the case for spikes that are queued for effect application but haven't
been applied yet when the first ``run`` call terminates. See `Brian2CUDA issue
#136`_ for progress on this issue.

.. _Brian2CUDA issue #136: https://github.com/brian-team/brian2cuda/issues/136


Changing delays between ``run`` calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Changing the delay of ``Synapses`` objects between ``run`` calls currently
leads to the loss of spikes. This is the case when changing homogenenous delays
or when switching between homogeneous and heterogeneous delays (e.g.
``Synapses.delay = 'j*ms'`` before the first ``run`` call and ``Synapses.delay
= '1*ms'`` after the first ``run`` call). Changing heterogenenous delays
between ``run`` calls is not effected from this bug and should work as expected
(e.g. from ``synapses.delay = 'j*ms'`` to ``synapses.delay = '2*j*ms'``).
See `Brian2CUDA issue #302`_ for progress on this issue.

.. _Brian2CUDA issue #302: https://github.com/brian-team/brian2cuda/issues/302
