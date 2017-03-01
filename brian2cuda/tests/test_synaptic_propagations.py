from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_allclose, assert_equal, assert_raises

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices

import brian2cuda


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_CudaSpikeQueue_push_outer_loop():

    # This test only works, if the CudaSpikeQueue::push() kernel is called with
    # <<<num_blocks, num_threads>>> as specified below
    # In each block for the last syn, we will enter the second loop cycle with a different delay.
    # In the example in spikequeue.h this triggers situations (a) for syn=512 and (b) for syn=1024.
    # syn                (0-512/3)   (512/3-1023) 1024
    # delay                     0              1     2
    # tid                (0-512/3)   (512/3-1023)    0
    # i                         0              0  1024
    # loop cycle                0              0     1
    # delay_start_idx           0            512   512

    num_blocks = 15
    num_threads = 1024
    N = (num_threads + 1) * num_blocks # each block gets 1025 synapses to push

    default_dt = defaultclock.dt
    delays = arange(N)%2 * default_dt

    inp = SpikeGeneratorGroup(1, [0], [0]*ms)
    G = NeuronGroup(N, 'v:1', threshold='v>1', reset='v=0')
    S = Synapses(inp, G, on_pre='v+=1.1')
    S.connect()
    S.delay = delays  # delays 0, dt
    mon = SpikeMonitor(G)

    run(3*default_dt)

    even_indices = np.arange(0,N,2)  # delay=0
    odd_indices = np.arange(1,N,2)  # delay=dt

    assert_allclose(S.delay[even_indices], 0)
    assert_allclose(S.delay[odd_indices], default_dt)
    # when issue 46 is done, remove np.sort
    assert_equal(np.sort(mon.i[mon.t==default_dt]), even_indices)
    assert_equal(np.sort(mon.i[mon.t==2*default_dt]), odd_indices)
    assert len(mon) == N


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_circular_eventspaces():
    # test postsynaptic effects in no_or_const_delay_mode are applied correctly

    default_dt = defaultclock.dt

    inp = SpikeGeneratorGroup(1, [0], [0]*ms)
    G = NeuronGroup(5, 'v:1', threshold='v>1', reset='v=0')
    # synapses with homogenous delays
    S0 = Synapses(inp, G, on_pre='v+=1.1', delay=0*ms)
    S0.connect(i=0, j=0)
    S1 = Synapses(inp, G, on_pre='v+=1.1', delay=2*default_dt)
    S1.connect(i=0, j=1)
    S2 = Synapses(inp, G, on_pre='v+=1.1', delay=5*default_dt)
    S2.connect(i=0, j=2)
    # synapse with heterogeneous delays
    S3 = Synapses(inp, G, on_pre='v+=1.1')
    S3.connect(i=0, j=[3, 4])
    S3.delay = 'j*default_dt'
    mon = SpikeMonitor(G)

    run(6*default_dt)

    # neurons should spike in the timestep after effect application
    assert_allclose(mon.t[mon.i[:] == 0], 1*default_dt)
    assert_allclose(mon.t[mon.i[:] == 1], 3*default_dt)
    assert_allclose(mon.t[mon.i[:] == 2], 6*default_dt)
    assert_allclose(mon.t[mon.i[:] == 3], 4*default_dt)
    assert_allclose(mon.t[mon.i[:] == 4], 5*default_dt)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_synaptic_effect_modes():
    # make sure on_pre pathway changing pre variables has the same mode as
    # on_post pathway changing post variables

    num_blocks = 15
    num_threads = 1024
    N = num_blocks * num_threads
    # all neurons spike every timestep
    neuron = NeuronGroup(1, 'v:1', threshold='True')
    group = NeuronGroup(N, 'v:1', threshold='True')
    # the on_pre pathway does not serialize in syanptic_effects == 'target' mode
    # with no_or_const_delay_mode
    S0 = Synapses(group, neuron, on_pre='v_post+=1', delay=0*ms)
    S0.connect()
    # the on_post pathway serializes entirely in synaptic_effects == 'source' mode
    S1 = Synapses(group, neuron, on_post='v_post-=1')
    S1.post.delay = 0*ms
    S1.connect()
    mon = SpikeMonitor(neuron, variables=['v'])

    run(2*defaultclock.dt)

    # each timestep there are the same number of + and - to neuron.v
    assert_allclose(mon.v[:], 0)
