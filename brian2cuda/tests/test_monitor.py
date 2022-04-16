import pytest

from numpy.testing import assert_equal
from brian2 import *


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_state_monitor_more_threads_than_single_block():
    set_device("cuda_standalone", directory=None)
    n = 2000
    G = NeuronGroup(n, 'v:1')
    mon = StateMonitor(G, 'v', record=True)
    v_init = arange(n)
    G.v = v_init

    run(3 * defaultclock.dt)

    for t in range(3):
        assert_equal(mon.v[:, t], v_init)


if __name__ == '__main__':
    import brian2cuda
    set_device('cuda_standalone', directory=None)
    test_state_monitor_more_threads_than_single_block()
