import pytest

from numpy.testing import assert_allclose, assert_equal
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


@pytest.mark.standalone_compatible
def test_spike_monitor_subgroups_all_attr():
    G = NeuronGroup(6, '''do_spike : boolean''', threshold='do_spike')
    G.do_spike = [True, False, False, False, True, True]
    spikes_all = SpikeMonitor(G)
    spikes_1 = SpikeMonitor(G[:2])
    spikes_2 = SpikeMonitor(G[2:4])
    spikes_3 = SpikeMonitor(G[4:])

    run(defaultclock.dt)

    assert_allclose(spikes_all.i, [0, 4, 5])
    assert_allclose(spikes_all.t, [0, 0, 0]*ms)
    assert_allclose(spikes_all.count, [1, 0, 0, 0, 1, 1])
    assert_allclose(spikes_all.N, 3)

    assert_allclose(spikes_1.i, [0])
    assert_allclose(spikes_1.t, [0]*ms)
    assert_allclose(spikes_1.count, [1, 0])
    assert_allclose(spikes_1.N, 1)

    assert len(spikes_2.i) == 0
    assert len(spikes_2.t) == 0
    assert_allclose(spikes_2.count, [0, 0])
    assert spikes_2.N == 0

    assert_allclose(spikes_3.i, [0, 1])  # recorded spike indices are relative
    assert_allclose(spikes_3.t, [0, 0] * ms)
    assert_allclose(spikes_3.count, [1, 1])
    assert_allclose(spikes_3.N, 2)

if __name__ == '__main__':
    import brian2cuda
    set_device('cuda_standalone', directory=None)
    test_state_monitor_more_threads_than_single_block()
