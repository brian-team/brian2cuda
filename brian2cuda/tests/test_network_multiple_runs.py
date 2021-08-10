import pytest
from numpy.testing import assert_equal

from brian2 import *
from brian2.tests.utils import assert_allclose

import brian2cuda


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_changing_delay_scalar():

    set_device('cuda_standalone', directory=None, build_on_run=False)
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

    # mon.v[i, t]
    assert_allclose(mon.v[:], [[0, 0, 1, 1, 2, 3]])


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_changing_delay_heterogeneous():

    set_device('cuda_standalone', directory=None, build_on_run=False)
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

if __name__ == '__main__':
    #test_changing_delay_scalar()
    test_changing_delay_heterogeneous()
