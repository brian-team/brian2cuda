import pytest
from numpy.testing import assert_equal

from brian2 import *
from brian2.tests.utils import assert_allclose

import brian2cuda


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_multiple_runs_with_scalar_delay():

    inG = NeuronGroup(1, 'v : 1', threshold='True')
    G = NeuronGroup(1, 'v : 1')
    G.v[:] = 0
    S = Synapses(inG, G, on_pre='v += 1', delay=defaultclock.dt)
    S.connect()
    mon = StateMonitor(G, 'v', record=True)

    run(2*defaultclock.dt)
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    assert_equal(mon.v[:], [[0, 0, 1, 2]])


@pytest.mark.xfail(
    reason='See Brian2CUDA issue #302',
    condition="config.device == 'cuda_standalone'",
    raises=AssertionError
)
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_increasing_delay_scalar():

    inG = NeuronGroup(1, 'v : 1', threshold='True')
    G = NeuronGroup(1, 'v : 1')
    G.v[:] = 0
    S = Synapses(inG, G, on_pre='v += 1')
    S.connect()
    S.delay[:] = 1*defaultclock.dt
    mon = StateMonitor(G, 'v', record=True)

    run(1*defaultclock.dt)
    S.delay[:] = 2*defaultclock.dt
    run(5*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    # Note: The monitor runs first in each time step, hence only records variables from
    #       the previous time step (the spike in the first time step arrives in the
    #       second time step and is only recorded in the third time step
    # mon.v[i, t]
    assert_allclose(mon.v[:], [[0, 0, 1, 1, 2, 3]])


@pytest.mark.xfail(
    reason='See Brian2CUDA issue #302',
    condition="config.device == 'cuda_standalone'",
    raises=AssertionError
)
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_decreasing_delay_scalar():

    inG = NeuronGroup(1, 'v : 1', threshold='True')
    G = NeuronGroup(1, 'v : 1')
    G.v[:] = 0
    S = Synapses(inG, G, on_pre='v += 1')
    S.connect()
    S.delay[:] = 2*defaultclock.dt
    mon = StateMonitor(G, 'v', record=True)

    run(1*defaultclock.dt)
    S.delay[:] = 1*defaultclock.dt
    run(5*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    # Note: The monitor runs first in each time step, hence only records variables from
    #       the previous time step (the spike in the first time step arrives in the
    #       second time step and is only recorded in the third time step
    # mon.v[i, t]
    assert_allclose(mon.v[:], [[0, 0, 1, 1, 2, 3]])


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_multiple_runs_with_heterogeneous_delay():

    inG = NeuronGroup(1, 'v : 1', threshold='True')
    G = NeuronGroup(2, 'v : 1')
    G.v[:] = 0
    S = Synapses(inG, G, on_pre='v += 1')
    S.connect()
    S.delay[:] = array([0, 1]) * defaultclock.dt
    mon = StateMonitor(G, 'v', record=True)

    run(2*defaultclock.dt)
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    assert_equal(mon.v[0, :], [0, 1, 2, 3])
    assert_equal(mon.v[1, :], [0, 0, 1, 2])


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_increasing_delay_heterogeneous():

    inG = NeuronGroup(1, 'v : 1', threshold='True')
    G = NeuronGroup(2, 'v : 1')
    G.v[:] = 0
    S = Synapses(inG, G, on_pre='v += 1')
    S.connect()
    S.delay[:] = '1*j*dt'
    mon = StateMonitor(G, 'v', record=True)

    run(1*defaultclock.dt)
    S.delay[:] = '2*j*dt'
    run(5*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    # mon.v[i, t]
    assert_allclose(mon.v[0, :], [0, 1, 2, 3, 4, 5])
    assert_allclose(mon.v[1, :], [0, 0, 1, 1, 2, 3])


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_reducing_delay_heterogeneous():

    inG = NeuronGroup(1, 'v : 1', threshold='True')
    G = NeuronGroup(2, 'v : 1')
    G.v[:] = 0
    S = Synapses(inG, G, on_pre='v += 1')
    S.connect()
    S.delay[:] = '2*j*dt'
    mon = StateMonitor(G, 'v', record=True)

    run(1*defaultclock.dt)
    S.delay[:] = '1*j*dt'
    run(5*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    # mon.v[i, t]
    assert_allclose(mon.v[0, :], [0, 1, 2, 3, 4, 5])
    assert_allclose(mon.v[1, :], [0, 0, 0, 2, 3, 4])


@pytest.mark.xfail(
    reason='See Brian2CUDA issue #136',
    condition="config.device == 'cuda_standalone'",
    raises=AssertionError
)
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_changed_dt_spikes_in_queue_heterogenenous_delay():
    defaultclock.dt = .5*ms
    G1 = NeuronGroup(1, 'v:1', threshold='v>1', reset='v=0')
    G1.v = 1.1
    G2 = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0')
    S = Synapses(G1, G2, on_pre='v+=1.1')
    S.connect()
    S.delay = 'j*ms'
    mon = SpikeMonitor(G2)
    net = Network(G1, G2, S, mon)
    net.run(5*ms)
    defaultclock.dt = 1*ms
    net.run(3*ms)
    defaultclock.dt = 0.1*ms
    net.run(2*ms)

    device.build(direct_call=False, **device.build_options)
    # Spikes should have delays of 0, 1, 2, ... ms and always
    # trigger a spike one dt later
    expected = [0.5, 1.5, 2.5, 3.5, 4.5, # dt=0.5ms
                6, 7, 8, #dt = 1ms
                8.1, 9.1 #dt=0.1ms
                ] * ms
    assert_allclose(mon.t[:], expected)

if __name__ == '__main__':
    #test_changing_delay_scalar()
    test_changing_delay_heterogeneous()
