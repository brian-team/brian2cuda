import pytest

from brian2 import *
from brian2.tests.utils import assert_allclose
from brian2.devices.device import reinit_devices, set_device, reset_device

import brian2cuda

# CUDA version of brian2.tests.test_cpp_standalone.test_openmp_consistency
@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_stdp_example():
    previous_device = get_device()
    n_cells    = 100
    n_recorded = 10
    numpy.random.seed(42)
    taum       = 20 * ms
    taus       = 5 * ms
    Vt         = -50 * mV
    Vr         = -60 * mV
    El         = -49 * mV
    fac        = (60 * 0.27 / 10)
    gmax       = 20*fac
    dApre      = .01
    taupre     = 20 * ms
    taupost    = taupre
    dApost     = -dApre * taupre / taupost * 1.05
    dApost    *=  0.1*gmax
    dApre     *=  0.1*gmax

    connectivity = numpy.random.randn(n_cells, n_cells)
    sources = numpy.random.random_integers(0, n_cells-1, 10*n_cells)
    # Only use one spike per time step (to rule out that a single source neuron
    # has more than one spike in a time step)
    times = numpy.random.choice(numpy.arange(10*n_cells), 10*n_cells, replace=False)*ms
    v_init = Vr + numpy.random.rand(n_cells) * (Vt - Vr)

    eqs  = Equations('''
    dv/dt = (g-(v-El))/taum : volt
    dg/dt = -g/taus         : volt
    ''')

    results = {}

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()
        P = NeuronGroup(n_cells, model=eqs, threshold='v>Vt', reset='v=Vr', refractory=5 * ms)
        Q = SpikeGeneratorGroup(n_cells, sources, times)
        P.v = v_init
        P.g = 0 * mV
        S = Synapses(P, P,
                     model='''dApre/dt=-Apre/taupre : 1 (event-driven)
                              dApost/dt=-Apost/taupost : 1 (event-driven)
                              w : 1''',
                     on_pre='''g += w*mV
                               Apre += dApre
                               w = w + Apost''',
                     on_post='''Apost += dApost
                                w = w + Apre''')
        S.connect()

        S.w = fac*connectivity.flatten()

        T = Synapses(Q, P, model="w : 1", on_pre="g += w*mV")
        T.connect(j='i')
        T.w = 10*fac

        spike_mon = SpikeMonitor(P)
        rate_mon = PopulationRateMonitor(P)
        state_mon = StateMonitor(S, 'w', record=range(n_recorded), dt=0.1*second)
        v_mon = StateMonitor(P, 'v', record=range(n_recorded))

        run(0.2 * second, report='text')

        device.build(directory=None, with_output=False)

        results[devicename] = {}
        results[devicename]['w'] = state_mon.w
        results[devicename]['v'] = v_mon.v
        results[devicename]['s'] = spike_mon.num_spikes
        results[devicename]['r'] = rate_mon.rate[:]

    for key in ['w', 'v', 'r', 's']:
        assert_allclose(results['cpp_standalone'][key], results['cuda_standalone'][key])

    reset_device(previous_device)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_stdp_heterog_delays_example():
    previous_device = get_device()
    n_cells    = 100
    n_recorded = 10
    numpy.random.seed(42)
    taum       = 20 * ms
    taus       = 5 * ms
    Vt         = -50 * mV
    Vr         = -60 * mV
    El         = -49 * mV
    fac        = (60 * 0.27 / 10)
    gmax       = 20*fac
    dApre      = .01
    taupre     = 20 * ms
    taupost    = taupre
    dApost     = -dApre * taupre / taupost * 1.05
    dApost    *=  0.1*gmax
    dApre     *=  0.1*gmax

    connectivity = numpy.random.randn(n_cells, n_cells)
    sources = numpy.random.random_integers(0, n_cells-1, 10*n_cells)
    # Only use one spike per time step (to rule out that a single source neuron
    # has more than one spike in a time step)
    times = numpy.random.choice(numpy.arange(10*n_cells), 10*n_cells, replace=False)*ms
    v_init = Vr + numpy.random.rand(n_cells) * (Vt - Vr)

    eqs  = Equations('''
    dv/dt = (g-(v-El))/taum : volt
    dg/dt = -g/taus         : volt
    ''')

    delay = numpy.random.rand(n_cells**2) * 2 * ms

    results = {}

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()
        P = NeuronGroup(n_cells, model=eqs, threshold='v>Vt', reset='v=Vr', refractory=5 * ms)
        Q = SpikeGeneratorGroup(n_cells, sources, times)
        P.v = v_init
        P.g = 0 * mV
        S = Synapses(P, P,
                     model='''dApre/dt=-Apre/taupre : 1 (event-driven)
                              dApost/dt=-Apost/taupost : 1 (event-driven)
                              w : 1''',
                     on_pre='''g += w*mV
                               Apre += dApre
                               w = w + Apost''',
                     on_post='''Apost += dApost
                                w = w + Apre''')
        S.connect()

        S.delay = delay
        S.w = fac*connectivity.flatten()

        T = Synapses(Q, P, model="w : 1", on_pre="g += w*mV")
        T.connect(j='i')
        T.w = 10*fac

        spike_mon = SpikeMonitor(P)
        rate_mon = PopulationRateMonitor(P)
        state_mon = StateMonitor(S, 'w', record=range(n_recorded), dt=0.1*second)
        v_mon = StateMonitor(P, 'v', record=range(n_recorded))

        run(0.2 * second, report='text')

        device.build(directory=None, with_output=False)

        results[devicename] = {}
        results[devicename]['w'] = state_mon.w
        results[devicename]['v'] = v_mon.v
        results[devicename]['s'] = spike_mon.num_spikes
        results[devicename]['r'] = rate_mon.rate[:]

    for key in ['w', 'v', 'r', 's']:
        assert_allclose(results['cpp_standalone'][key], results['cuda_standalone'][key])

    reset_device(previous_device)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_sorted_indices_statemonitor():
    previous_device = get_device()

    results = {}

    n_cells = 5
    n_recorded = 10
    delay = np.arange(n_cells) * defaultclock.dt

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()
        P = NeuronGroup(n_cells, model='', threshold='True')
        S = Synapses(P, P,
                     model='''w : 1''',
                     on_pre='''w += 1''')
        S.connect(j='i')
        S.pre.delay = delay

        state_mon = StateMonitor(S, 'w', record=range(n_recorded))

        run(defaultclock.dt * (n_cells + 1))

        device.build(directory=None, with_output=False)
        results[devicename] = state_mon.w.astype(int)

    assert_allclose(results['cpp_standalone'].sum(axis=0), results['cuda_standalone'].sum(axis=0))
    assert_allclose(results['cpp_standalone'], results['cuda_standalone'])

    reset_device(previous_device)
