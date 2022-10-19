Known issues
============

In addition to the issues noted below, you can refer to `our bug tracker on GitHub`_.

.. _our bug tracker on GitHub: https://github.com/brian-team/brian2cuda/issues?q=is%3Aopen+is%3Aissue+label%3Abug+

List of known issues:

.. contents::
    :local:
    :depth: 2

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


Using a different integration time for ``Synapses`` and its source ``NeuronGroup``
----------------------------------------------------------------------------------
There is currently a bug when using ``Synapses`` with homogeneous delays and
choosing a different integration time step (``dt``) for any of its
``SynapticPathway`` and its associated source ``NeuronGroup``. This bug does
not occur when the delays are heterogenenous or when only the target
``NeuronGroup`` has a different clock. See `Brian2CUDA issue #222`_ for
progress on the issue. Any of the following examples has this bug::

    from brian2 import *

    group_different_dt = NeuronGroup(1, 'v:1', threshold='True', dt=2*defaultclock.dt)
    group_same_dt = NeuronGroup(1, 'v:1', threshold='True', dt=defaultclock.dt)

    # Bug: Source of pre->post synaptic pathway uses different dt than synapses
    #      and synapses have homogeneous delays
    synapses = Synapses(
        group_different_dt,
        group_same_dt,
        on_pre='v+=1',
        delay=1*ms,
        dt=defaultclock.dt
    )

    # No bug: Synapses have no delays
    synapses = Synapses(
        group_different_dt,
        group_same_dt,
        on_pre='v+=1',
        dt=defaultclock.dt
    )

    # No bug: Synapses have heterogeneous delays
    synapses = Synapses(
        group_different_dt,
        group_same_dt,
        on_pre='v+=1',
        dt=defaultclock.dt
    )
    synapses.delay = 'j*ms'

    # No bug: Source of pre->post synaptic pathway uses the same dt as synapses
    synapses = Synapses(
        group_same_dt,
        group_different_dt,
        on_post='v+=1',
        delay=1*ms,
        dt=defaultclock.dt
    )

.. _Brian2CUDA issue #222: https://github.com/brian-team/brian2cuda/issues/222


``SpikeMonitor`` and ``EventMonitor`` data is not sorted by indices
-------------------------------------------------------------------
In all Brian2 devices, ``SpikeMonitor`` and ``EventMonitor`` data is first
sorted by time and then by neuron index. In Brian2CUDA, the data is only sorted
by time but not always by index given a fixed time point. See `Brian2CUDA issue
#46`_ for progress on this issue.

.. _Brian2CUDA issue #46: https://github.com/brian-team/brian2cuda/issues/46


Single precision mode fails when using variable names with double digit and dot or scientific notations in name
---------------------------------------------------------------------------------------------------------------
In single precision mode (set via ``prefs.core.default_float_dtype``),
Brian2CUDA replaces floating point literals like ``.2``, ``1.`` or ``.4`` in generated code with
single precision versions ``1.2f``, ``1.f`` and ``.4f``. Under some
circumstances, the search/replace algorithm fails and performs a wrong string
replacement. This is the case e.g. for variable name with double digit and a
dot in their name, such as ``variable12.attribute`` or when variable names have
a substring that can be interpreted as a scientific number, e.g.
`variable28e2`, which has `28e2` as substring. If such a wrong replacement
occurs, compilation typically fails due to not declared variables. See
`Brian2CUDA issue #254`_ for progress on the issue.

.. _Brian2CUDA issue #254: https://github.com/brian-team/brian2cuda/issues/254
