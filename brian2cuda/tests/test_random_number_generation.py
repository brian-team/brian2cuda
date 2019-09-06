from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_raises

from brian2 import *
from brian2.monitors.statemonitor import StateMonitor
from brian2.core.clocks import defaultclock
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices, set_device, device

import brian2cuda


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rand():

    set_device('cuda_standalone', directory=None)
    G = NeuronGroup(1000, 'dv/dt = rand() : second')
    mon = StateMonitor(G, 'v', record=True)

    run(3*defaultclock.dt)

    print(mon.v[:5, :])
    assert_raises(AssertionError, assert_equal, mon.v[:, -1], 0)
    assert_raises(AssertionError, assert_equal, mon.v[:, -2], mon.v[:, -1])


@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_random_number_generation_with_multiple_runs():

    set_device('cuda_standalone', directory=None, build_on_run=False)
    G = NeuronGroup(1000, 'dv/dt = rand() : second')
    mon = StateMonitor(G, 'v', record=True)

    run(1*defaultclock.dt)
    run(2*defaultclock.dt)
    device.build(direct_call=False, **device.build_options)

    assert_raises(AssertionError, assert_equal, mon.v[:, -1], 0)
    assert_raises(AssertionError, assert_equal, mon.v[:, -2], mon.v[:, -1])


# adapted for standalone mode from brian2/tests/test_neurongroup.py
@attr('standalone-compatible', 'multiple-runs')
def test_random_values_fixed_and_random():

    set_device('cuda_standalone', directory=None, build_on_run=False)
    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.1*xi/sqrt(ms) : 1')
    seed(13579)
    G.v = 'rand()'
    seed()
    mon = StateMonitor(G, 'v', record=True)
    run(2*defaultclock.dt)

    #G2 = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.1*xi/sqrt(ms) : 1')
    seed(13579)
    G.v = 'rand()'
    seed()
    #mon2 = StateMonitor(G2, 'v', record=True)
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    first_run_values = np.array(mon.v)

    # First time step should be identical
    assert_allclose(mon[:, 0], mon.v[:, 2])
    # Second should be different
    assert np.var(mon[:, 1] - mon.v[:, 3]) > 0


if __name__ == '__main__':
    test_random_number_generation_with_multiple_runs()
