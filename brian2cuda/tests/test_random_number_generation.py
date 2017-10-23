from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_allclose, assert_equal, assert_raises

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices

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


if __name__ == '__main__':
    test_rand()
