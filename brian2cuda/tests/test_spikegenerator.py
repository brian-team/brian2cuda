import pytest
from numpy.testing import assert_equal

from brian2 import *
from brian2.tests.utils import assert_allclose
from brian2.utils.logger import catch_logs

import brian2cuda


# Copied from brian2.tests.test_spikegenerator (where it isn't 'standalone-compatible')
@pytest.mark.standalone_compatible
def test_spikegenerator_extreme_period():
    '''
    Basic test for `SpikeGeneratorGroup`.
    '''
    indices = np.array([0, 1, 2])
    times   = np.array([0, 1, 2]) * ms
    SG = SpikeGeneratorGroup(5, indices, times, period=1e6*second)
    s_mon = SpikeMonitor(SG)
    with catch_logs() as l:
        run(10*ms)

    assert_equal(s_mon.i, np.array([0, 1, 2]))
    assert_allclose(s_mon.t, [0, 1, 2]*ms)
    assert len(l) == 1 and l[0][1].endswith('spikegenerator_long_period')


# Adapted from brian2.tests.test_spikegenerator (where it isn't 'standalone-compatible')
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_spikegenerator_period_repeat():
    '''
    Basic test for `SpikeGeneratorGroup`.
    '''
    indices = np.zeros(10)
    times   = arange(0, 1, 0.1) * ms

    rec = np.rec.fromarrays([times, indices], names=['t', 'i'])
    rec.sort()
    sorted_times = np.ascontiguousarray(rec.t)*1000
    sorted_indices = np.ascontiguousarray(rec.i)
    SG = SpikeGeneratorGroup(1, indices, times, period=1*ms)
    net   = Network(SG)

    # XXX: not standalone-version:
    # s_mon = SpikeMonitor(SG)
    # net   = Network(SG, s_mon)
    # for idx in range(5):
    #     net.run(1*ms)
    #     assert (idx+1)*len(SG.spike_time) == s_mon.num_spikes

    # Here, I use seperate monitors due to SpikeMonitor bug (#221)
    monitors = []
    for idx in range(5):
        s_mon = SpikeMonitor(SG)
        monitors.append(s_mon)
        net.add(s_mon)
        net.run(1*ms)
        net.remove(s_mon)

    device.build(direct_call=False, **device.build_options)

    for idx in range(5):
        assert len(SG.spike_time) == monitors[idx].num_spikes


# Adapted from brian2.tests.test_spikegenerator (where it isn't 'standalone-compatible')
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_spikegenerator_rounding():
    # all spikes should fall into the first time bin
    indices = np.arange(100)
    times = np.linspace(0, 0.1, 100, endpoint=False)*ms
    SG = SpikeGeneratorGroup(100, indices, times, dt=0.1*ms)
    mon = SpikeMonitor(SG)
    net = Network(SG, mon)
    net.run(0.1*ms)

    # all spikes should fall in separate bins
    dt = 0.1*ms
    indices = np.zeros(10000)
    times = np.arange(10000)*dt
    SG = SpikeGeneratorGroup(1, indices, times, dt=dt)
    target = NeuronGroup(1, 'count : 1',
                         threshold='True', reset='count=0')  # set count to zero at every time step
    syn = Synapses(SG, target, on_pre='count+=1')
    syn.connect()
    # XXX: Using a new monitor here due to SpikeMonitor bug (#221)
    mon2 = StateMonitor(target, 'count', record=0, when='end')
    net = Network(SG, target, syn, mon2)
    # change the schedule so that resets are processed before synapses
    net.schedule = ['start', 'groups', 'thresholds', 'resets', 'synapses', 'end']
    net.run(10000*dt)

    device.build(direct_call=False, **device.build_options)

    assert_equal(mon.count, np.ones(100))
    assert_equal(mon2[0].count, np.ones(10000))


if __name__ == '__main__':
      test_spikegenerator_extreme_period()
